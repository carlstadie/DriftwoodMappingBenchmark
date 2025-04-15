import os
import glob
import json
import math
import time
import multiprocessing
from itertools import product
from datetime import timedelta

import h5py
import numpy as np
from tqdm import tqdm

import rasterio
import rasterio.warp
import rasterio.mask
import rasterio.merge
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
from rasterio.windows import Window, bounds

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # must be set before importing TF. for parallel predictions on one GPU
import tensorflow as tf
from core.optimizers import get_optimizer
from core.frame_info import image_normalize
from core.util import memory, raster_copy
from core.losses import accuracy, dice_coef, dice_loss, specificity, sensitivity, get_loss


def load_model(config):
    """Load a saved Tensorflow model into memory"""

    # Load and compile the model
    model = tf.keras.models.load_model(config.trained_model_path,
                                       custom_objects={'tversky': get_loss('tversky', config.tversky_alphabeta),
                                                       'dice_coef': dice_coef, 'dice_loss': dice_loss,
                                                       'accuracy': accuracy, 'specificity': specificity,
                                                       'sensitivity': sensitivity},
                                       compile=False)
    model.compile(optimizer=get_optimizer(config.optimizer_fn), loss=get_loss(config.loss_fn, config.tversky_alphabeta),
                  metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity])

    return model


def load_model_info():
    """Get and display config of the pre-trained model. Store model metadata as json in the output prediction folder."""

    # If no specific trained model was specified, use the most recent model
    if config.trained_model_path is None:
        model_fps = glob.glob(os.path.join(config.saved_models_dir, "*.h5"))
        config.trained_model_path = sorted(model_fps, key=lambda t: os.stat(t).st_mtime)[-1]

    # Print metadata from model training if available
    print(f"Loaded pretrained model from {config.trained_model_path} :")
    with h5py.File(config.trained_model_path, 'r') as model_file:
        if "custom_meta" in model_file.attrs:
            try:
                custom_meta = json.loads(model_file.attrs["custom_meta"].decode("utf-8"))
            except:
                custom_meta = json.loads(model_file.attrs["custom_meta"])
            print(custom_meta, "\n")

            # Save metadata in output prediction folder for future reference
            with open(os.path.join(config.prediction_output_dir, "model_custom_meta.json"), "a") as out_file:
                json.dump(custom_meta, out_file)
                out_file.write("\n\n")

            # Read patch size to use for prediction from model
            if config.prediction_patch_size is None:
                config.prediction_patch_size = custom_meta["patch_size"]


def get_images_to_predict():
    """ Get all input images to predict

    Either takes only the images specifically listed in a text file at config.to_predict_filelist,
    or all images in config.to_predict_dir with the correct prefix and file type
    """
    input_images = []
    if config.to_predict_filelist is not None and os.path.exists(config.to_predict_filelist):
        for line in open(config.to_predict_filelist):
            if os.path.isabs(line.strip()) and os.path.exists(line.strip()):                 # absolute paths
                input_images.append(line.strip())
            elif os.path.exists(os.path.join(config.to_predict_dir, line.strip())):          # relative paths
                input_images.append(os.path.join(config.to_predict_dir, line.strip()))

        print(f"Found {len(input_images)} images to predict listed in {config.to_predict_filelist}.")
    else:
        for root, dirs, files in os.walk(config.to_predict_dir):
            for file in files:
                if file.endswith(config.predict_images_file_type) and file.startswith(config.predict_images_prefix):
                    input_images.append(os.path.join(root, file))

        print(f"Found {len(input_images)} valid images to predict in {config.to_predict_dir}.")
    if len(input_images) == 0:
        raise Exception("No images to predict.")

    return sorted(input_images)


def merge_validity_masks(config, input_images):
    """Merge multiple configured validity masks (eg. land borders, water mask..) into a single mask file. """

    merged_validity_mask_fp = None
    if config.prediction_mask_fps is not None and len(config.prediction_mask_fps) > 0:

        # Initialise valid area as the combined extent of input images
        valid_area = gpd.GeoDataFrame({"geometry": [box(*rasterio.open(im).bounds) for im in input_images]},
                                      crs=rasterio.open(input_images[0]).crs).to_crs("EPSG:4326")

        # Overlay all validty masks, reading only the current validity area for faster loading
        for mask_fp in tqdm(config.prediction_mask_fps, desc=f"{'Merging validity masks':<25}", leave=False):
            df = gpd.read_file(mask_fp, mask=valid_area).to_crs("EPSG:4326")
            valid_area = gpd.overlay(valid_area, df, how="intersection", make_valid=False)

            if len(valid_area.geometry) == 0:
                raise Exception(f"No areas to predict. Prediction images have no overlap with validity mask {mask_fp}")

        # Write the merged validity mask to file, to be used in patch filtering and cropping
        merged_validity_mask_fp = os.path.join(config.prediction_output_dir, "validity_mask.gpkg")
        valid_area.to_file(merged_validity_mask_fp, driver="GPKG", crs="EPSG:4326")

    return merged_validity_mask_fp


def split_image_to_chunks(image_fp, output_file, config):
    """Split an image into smaller chunks for parallel processing, for lower memory usage and higher GPU utilisation.

    Setting  config.prediction_gridsize = (1, 1) means no splitting is done and the entire image is predicted at once.
    Returns a list of params used by predict_image() during parallel processing.
    """

    # Load validity mask if available
    validity_mask = None
    if config.validity_mask_fp is not None:
        validity_mask = gpd.read_file(config.validity_mask_fp)

    # Split image into grid of n_rows x n_cols chunks
    n_rows, n_cols = config.prediction_gridsize
    params = []
    with rasterio.open(image_fp) as raster:
        chunk_width, chunk_height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)

        # Create a list of chunk parameters used for parallel processing
        for i, j in product(range(n_rows), range(n_cols)):
            chunk_bounds = bounds(Window(chunk_width*j, chunk_height*i, chunk_width, chunk_height), raster.transform)

            # Exclude image chunks that are entirely outside valid area
            if validity_mask is None or np.any(validity_mask.intersects(box(*chunk_bounds))):
                params.append([image_fp, chunk_bounds, f"{output_file}_{chunk_width*j}_{chunk_height*i}.tif", config])

    return params


def get_patch_offsets(image, patch_width, patch_height, stride, validity_mask_fp=None):
    """Get a list of patch offsets based on image size, patch size and stride.

    If a validity mask is configured, patches outside the valid area are filtered out so they will not be predicted.
    """

    # Create iterator of all patch offsets, as tuples (x_off, y_off)
    patch_offsets = list(product(range(0, image.width, stride), range(0, image.height, stride)))

    # Optionally filter prediction area by a shapefile validity mask, with any mask intersecting patches not predicted
    if validity_mask_fp is not None:
        mask_polygons = gpd.read_file(validity_mask_fp, bbox=box(*image.bounds))
        offset_geom = [box(*bounds(Window(col_off, row_off, patch_width, patch_height), image.transform))
                        for col_off, row_off in patch_offsets]
        offsets_df = gpd.GeoDataFrame({"geometry": offset_geom, "col_off": list(zip(*patch_offsets))[0],
                                       "row_off": list(zip(*patch_offsets))[1]})
        offsets_df["unique_patch"] = offsets_df.index
        filtered_df = gpd.sjoin(offsets_df, mask_polygons, op="intersects", how="inner").drop_duplicates("unique_patch")
        patch_offsets = list(zip(filtered_df.col_off, filtered_df.row_off))

    return patch_offsets


def add_to_result(res, prediction, row, col, he, wi, operator='MAX'):
    """Add results of a patch to the total results of a larger area.

    The operator can be MIN (useful if there are too many false positives), or MAX (useful for tackling false negatives)
    """
    curr_value = res[row:row + he, col:col + wi]
    new_predictions = prediction[:he, :wi]
    if operator == 'MIN':
        curr_value[curr_value == -1] = 1  # For MIN case mask was initialised with -1, and replaced here to get min()
        resultant = np.fmin(curr_value, new_predictions)
    elif operator == 'MAX':
        resultant = np.fmax(curr_value, new_predictions)
    elif operator == 'MEAN':
        resultant = np.nanmean([curr_value, new_predictions], axis=0)
    else:  # operator == 'REPLACE':
        resultant = new_predictions
    res[row:row + he, col:col + wi] = resultant
    return res


def predict_using_model(model, batch, batch_pos, mask, operator):
    """Predict one batch of patches with tensorflow, and add result to the output prediction. """

    tm = np.stack(batch, axis=0)
    prediction = model.predict(tm)
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis=-1)
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = add_to_result(mask, p, row, col, he, wi, operator)
    return mask


def write_mask_to_disk(detected_mask, profile, output_fp, config):
    """Write the output prediction mask to a raster file"""

    # For non-float formats, convert predictions to 1/0 with a given threshold
    if config.output_dtype != "float32":
        detected_mask[detected_mask < config.prediction_threshold] = 0
        detected_mask[detected_mask >= config.prediction_threshold] = 1

    # Set format specific profile options
    profile.update(dtype=config.output_dtype, count=1, tiled=True, compress="LZW")
    if config.output_dtype == "uint8":
        profile.update(nodata=255)                            # for uint8, use 255 as nodata
    if config.output_dtype == "bool":
        profile.update(dtype="uint8", nbits=1, nodata=None)   # for binary geotiff, write as byte and pass NBITS=1

    # If we have a validity mask, mask by its cutline to get smooth edges in partially valid patches (no step blocks)
    if config.validity_mask_fp is not None:

        # We have to write to memory array first, because rasterio doesn't allow masking of an array directly..
        with rasterio.open(f"/vsimem/temp.tif", 'w', **profile) as out_ds:
            out_ds.write(detected_mask.astype(profile["dtype"]), 1)

        # Mask by valid areas
        with rasterio.open(f"/vsimem/temp.tif") as src:
            valid_areas = gpd.read_file(config.validity_mask_fp, bounds=src.bounds).geometry
            detected_mask, _ = rasterio.mask.mask(src, valid_areas, indexes=1)

    # Write prediction to file
    with rasterio.open(output_fp, 'w', **profile) as out_ds:
        out_ds.write(detected_mask.astype(profile["dtype"]), 1)


def predict_image(params):
    image_fp, image_bounds, out_fp, config = params

    # Load the model
    model = load_model(config)
    pos = min(int(multiprocessing.current_process().name[-1].replace('s', '0')), config.prediction_workers) + 1

    # For jp2 compressed files, we first decompress in memory to speed up prediction and resampling
    if image_fp.lower().endswith(".jp2"):
        raster_copy("/vsimem/decompressed.tif", image_fp, multi_core=True, bounds=image_bounds,
                    pbar=tqdm(total=100, position=pos, leave=False, desc=f"{'Decompressing jp2':<25}"))
        image_fp = "/vsimem/decompressed.tif"

    # Optionally resample the image in memory
    if config.resample_factor != 1:
        raster_copy("/vsimem/resampled.tif", image_fp, resample=config.resample_factor, bounds=image_bounds,
                    multi_core=True, pbar=tqdm(total=100, position=pos, leave=False,
                                               desc=f"Resampling x{config.resample_factor:<13}"))
        image_fp = "/vsimem/resampled.tif"

    # Get list of patch offsets to predict for this image
    img = rasterio.open(image_fp, tiled=True, blockxsize=256, blockysize=256, bounds=image_bounds)
    patch_width, patch_height = config.prediction_patch_size
    stride = config.prediction_stride
    offsets = get_patch_offsets(img, patch_width, patch_height, stride, config.validity_mask_fp)

    # Initialise mask to zeros, or -1 for MIN operator
    mask = np.zeros((img.height, img.width), dtype=np.float32)   # prediction is a float
    if config.prediction_operator == "MIN":
        mask = mask - 1

    batch, batch_pos = [], []
    big_window = Window(0, 0, img.width, img.height)
    for col_off, row_off in tqdm(offsets, position=pos, leave=False, desc=f"Predicting {len(offsets)}/"
                                 f"{math.ceil(img.width/stride)*math.ceil(img.height/stride)} patches..."):

        # Initialise patch with zero padding in case of corner images. size is based on number of channels
        patch = np.zeros((patch_height, patch_width, np.sum(config.channels_used)))

        # Load patch window from image, reading only necessary channels
        patch_window = Window(col_off=col_off, row_off=row_off, width=patch_width, height=patch_height).intersection(
            big_window)
        temp_im = img.read(list(np.where(config.channels_used)[0] + 1), window=patch_window)
        temp_im = np.transpose(temp_im, axes=(1, 2, 0))      # switch channel order for TF

        # Normalize the image along the width and height i.e. independently per channel. Ignore nodata for normalization
#        temp_im = image_normalize(temp_im, axis=(0, 1), nodata_val=img.nodatavals[0])

        # Add to batch list
        patch[:patch_window.height, :patch_window.width] = temp_im
        batch.append(patch)
        batch_pos.append((patch_window.col_off, patch_window.row_off, patch_window.width, patch_window.height))

        # Predict one batch at a time
        if len(batch) == config.prediction_batch_size:
            mask = predict_using_model(model, batch, batch_pos, mask, config.prediction_operator)
            batch, batch_pos = [], []

    # Run once more to process the last partial batch (when image not exactly divisible by N batches)
    if batch:
        mask = predict_using_model(model, batch, batch_pos, mask, config.prediction_operator)

    # Write the predicted mask to file
    write_mask_to_disk(mask, img.profile, out_fp, config)

    return out_fp


@memory(percentage=99)
def predict_all(conf):
    """Predict trees in all the files in the input image dir. """

    global config
    config = conf

    print("Starting prediction.")
    start = time.time()

    # Create folder for output predictions
    if config.prediction_output_dir is None:
        config.prediction_output_dir = os.path.join(config.predictions_base_dir, time.strftime('%Y%m%d-%H%M') + '_' + config.prediction_name)
    if not os.path.exists(config.prediction_output_dir):
        os.mkdir(config.prediction_output_dir)
    rasters_dir = os.path.join(config.prediction_output_dir, "rasters")
    if not os.path.exists(rasters_dir):
        os.mkdir(rasters_dir)

    # Load model info
    load_model_info()

    # Get list of images to analyse
    input_images = get_images_to_predict()

    # Combine possible multiple validity masks into one mask
    config.validity_mask_fp = merge_validity_masks(config, input_images)

    # Process all input images
    for image_fp in tqdm(input_images, desc=f"{'Analysing images':<25}", position=0):

        # Check if image has already been predicted
        output_file = os.path.join(rasters_dir, config.output_prefix +
                                   image_fp.split("/")[-1].replace(config.predict_images_file_type, ".tif"))
        if os.path.isfile(output_file) and not config.overwrite_analysed_files:
            print(f"File already analysed, skipping {image_fp}")
            continue

        print(f"\nAnalysing {image_fp}")
        t0 = time.time()

        # Split into several smaller image chunks
        params = split_image_to_chunks(image_fp, output_file, config)
        if len(params) == 0:
            print(f"No parts of the image intersect validity mask, skipping {image_fp}")
            continue

        # Process image chunks in parallel
        chunk_fps = []
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=config.prediction_workers) as pool:
            with tqdm(total=len(params), desc="Processing image chunks", position=1, leave=False) as pbar:
                for result in pool.imap_unordered(predict_image, params, chunksize=1):
                    pbar.update()
                    if result:
                        chunk_fps.append(result)
                pbar.update()

        # Merge chunks back into one output raster
        print(f"\nWriting raster to {output_file}")
        gdal.BuildVRT(f"/vsimem/merged.vrt", chunk_fps)
        options = ["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "NBITS=1" if config.output_dtype == "bool" else ""]
        gdal.Translate(output_file, f"/vsimem/merged.vrt", creationOptions=options)

        # Delete temp chunks
        for f in chunk_fps:
            os.remove(f)
        print(f"Processed {image_fp} in: {str(timedelta(seconds=time.time() - t0))}\n")

    print(f"Prediction completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
