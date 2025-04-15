import os
import time
from datetime import timedelta
import json

import numpy as np
from tqdm import tqdm

import rasterio
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box

from core.util import raster_copy
from core.frame_info import image_normalize

import scipy
import skimage.transform


def get_areas_and_polygons():
    """Read in the training rectangles and polygon shapefiles.

    Runs a spatial join on the two DBs, which assigns rectangle ids to all polygons in a column "index_right".
    """

    print("Reading training data shapefiles.. ", end="")
    start = time.time()

    # Read in areas and remove all columns except geometry
    areas = gpd.read_file(os.path.join(config.training_data_dir, config.training_area_fn))
    areas = areas.drop(columns=[c for c in areas.columns if c != "geometry"])

    # Read in polygons and remove all columns except geometry
    polygons = gpd.read_file(os.path.join(config.training_data_dir, config.training_polygon_fn))
    polygons = polygons.drop(columns=[c for c in polygons.columns if c != "geometry"])

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(polygons)} polygons in {len(areas)} areas.\n"
          f"Assigning polygons to areas..      ", end="")
    start = time.time()

    # Perform a spatial join operation to pre-index all polygons with the rectangle they are in
    polygons = gpd.sjoin(polygons, areas, op="intersects", how="inner")

    print(f"Done in {time.time()-start:.2f} seconds.")
    return areas, polygons


def get_images_with_training_areas(areas):
    """Get a list of input images and the training areas they cover.

    Returns a list of tuples with img path and its area ids, eg [(<img_path>, [0, 12, 17, 18]), (...), ...]
    """

    print("Assigning areas to input images..  ", end="")
    start = time.time()

    # Get all input image paths
    image_paths = []
    for root, dirs, files in os.walk(config.training_image_dir):
        for file in files:
            if file.startswith(config.train_image_prefix) and file.lower().endswith(config.train_image_type.lower()):
                image_paths.append(os.path.join(root, file))

    # Find the images that contain training areas
    images_with_areas = []
    for im in image_paths:

        # Get image bounds and CRS (Coordinate Reference System)
        with rasterio.open(im) as raster:
            im_bounds = box(*raster.bounds)
            image_crs = raster.crs  # Get the CRS of the image

        # Check if the image CRS matches the CRS of the training areas
        if image_crs != areas.crs:
        
            continue

        # Get training areas that are in this image
        areas_in_image = np.where(areas.envelope.intersects(im_bounds))[0]

        if len(areas_in_image) > 0:
            images_with_areas.append((im, [int(x) for x in list(areas_in_image)]))

    print(f"Done in {time.time()-start:.2f} seconds. Found {len(image_paths)} training "
          f"images of which {len(images_with_areas)} contain training areas.")

    return images_with_areas
    


def resolution_degrees2metres(xres_degrees, yres_degrees, latitude):
    """Calculate the resolution in degrees equivalent to a desired resolution in metres."""
    xres_metres = xres_degrees * (111320 * math.cos(math.radians(abs(latitude))))  # at equator 1°lon ~= 111.32 km
    yres_metres = yres_degrees * 110540  # and        1°lat ~= 110.54 km
    return xres_metres, yres_metres

def add_additional_band(image_fp, image_bounds, out_fp, new_band, pbar_pos=0):
    # image_fp, out_fp, coverband_fp, image_bounds, average_res_m = params
    pbar = tqdm(total=5, desc=f"{'Adding coverband...':<25}", leave=False, position=pbar_pos, disable=True)

    # Read window of source image
    with rasterio.open(image_fp) as image_ds:
        image_window = from_bounds(*image_bounds, image_ds.transform)
        img = image_ds.read(window=image_window)
        pbar.update()

        # Read window of new band
        with rasterio.open(new_band["source_fp"]) as src:
            band_index = new_band["source_band"] if "source_band" in new_band.keys() else 1
            new_band_img = src.read(band_index, window=from_bounds(*image_bounds, src.transform))
        pbar.update()

        # Mask new band invalid values [optional]
        if "maskvals" in new_band.keys() and len(new_band["maskvals"]) > 0:
            mask = np.isin(new_band_img, new_band["maskvals"])
            new_band_img[mask] = 0

        # Scale new band values [optional]
        if "scale_factor" in new_band.keys() and new_band["scale_factor"] is not None:
            new_band_img = new_band_img.astype(np.float32) * new_band["scale_factor"]
        pbar.update()

        # Resample new band values [optional]
        if "average_to_resolution_m" in new_band.keys() and new_band["average_to_resolution_m"] is not None:
            try:
                scale = resolution_degrees2metres(*image_ds.res, 0)[1] / new_band["average_to_resolution_m"]
                new_band_img = skimage.transform.rescale(new_band_img, scale=scale, order=0, mode='reflect')
            except Exception:
                print(new_band_img.shape)
                assert False
        pbar.update()

        # Ensure new band is same resolution as input bands
        new_band_img = skimage.transform.resize(new_band_img, img.shape[1:], order=0, mode='reflect')

        # Insert extra band into merged img
        merged_img = np.concatenate([img, [new_band_img]], axis=0)

        # Write output merged image to file
        profile = image_ds.profile
        profile["count"] = profile["count"] + 1
        profile["transform"] = image_ds.window_transform(image_window)
        profile["width"] = img.shape[2]
        profile["height"] = img.shape[1]
        # profile["dtype"] = "uint16"
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(merged_img.astype(profile["dtype"]))

    pbar.update()
    return out_fp

def get_vectorized_annotation(polygons, areas, area_id, xsize, ysize):
    """Get the annotation as a list of shapes with geometric properties (center, area).

    Each entry in the output dictionary corresponds to an annotation polygon and has the following info:
        - center: centroid of the polygon in pixel coordinates
        - area(m): surface covered by the polygon in meters
        - area(px): surface covered by the polygon in pixels on the output frame
        - pseudo_radius(m): radius of the circle with the same area as the polygon in meters
        - pseudo_radius(px): radius of the circle with the same area as the polygon in pixels on the output frame
        - geometry: list of polygon points in pixel coordinates
    """
    # Find tree centers and pseudo-radius for this area
    isinarea = polygons[polygons.within(box(*areas.bounds.iloc[area_id]))]

    # Explode geodf to avoid handling multipolygons
    isinarea = isinarea.explode(column="geometry", index_parts=False)

    # Convert to equal area projection to compute area
    isinarea_ea = isinarea.to_crs(epsg=6933)
    isinarea.loc[:, "area(m)"] = isinarea_ea.area
    isinarea.loc[:, "pseudo_radius(m)"] = isinarea_ea.area.apply(lambda x: np.sqrt(x / np.pi))

    # Get transform from pixel coordinates to real-world coordinates
    bounds = areas.iloc[[area_id]].to_crs(epsg=6933).bounds.iloc[0]
    trsfrm = rasterio.transform.from_bounds(*bounds, xsize, ysize)

    # Deduce ground resolution
    gr = np.mean([np.abs(trsfrm.a), np.abs(trsfrm.e)])
    isinarea.loc[:, "area(px)"] = isinarea["area(m)"] / (gr**2)
    isinarea.loc[:, "pseudo_radius(px)"] = isinarea["pseudo_radius(m)"] / gr

    # Invert to get transform from real world coordinates to pixel coordinates
    trsfrm = ~trsfrm
    trsfrm = [element for tupl in trsfrm.column_vectors for element in tupl]
    isinarea.loc[:, "geometry"] = isinarea_ea["geometry"].affine_transform(trsfrm[:6])
    isinarea.loc[:, "center"] = isinarea.centroid
    isinarea.loc[:, "center"] = isinarea["center"].apply(lambda p: (p.x, p.y))
    isinarea.loc[:, "geometry"] = isinarea["geometry"].apply(lambda x: list(x.exterior.coords))

    # Convert dataframe to dict
    isinarea.drop(labels=["index_right"], inplace=True, axis=1)
    isinarea = pd.DataFrame(isinarea)
    dic = isinarea.to_dict(orient="records")
    return dic

def preprocess_all(conf):
    """Run preprocessing for all training data."""

    global config
    config = conf

    print("Starting preprocessing.")
    start = time.time()
    
    #new_band = AdditionalInputBand(name="worldcover",image_source=ImageSource(file="/nfs/Other_data/WorldCover_10m_2020/ESA_WorldCover_10m.vrt"))

    # Create output folder
    output_dir = os.path.join(config.preprocessed_base_dir, time.strftime('%Y%m%d-%H%M')+'_'+config.preprocessed_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read in area and polygon shapefiles
    areas, polygons = get_areas_and_polygons()

    # Scan input images and find which images contain which training areas
    images_with_areas = get_images_with_training_areas(areas)

    # For each input image, get all training areas in the image
    for im_path, area_ids in tqdm(images_with_areas, "Processing images with training areas", position=1):

        # For each area, extract the image channels and write img and annotation channels to a merged file
        for area_id in tqdm(area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0):

            # Extract the part of input image that overlaps training area, with optional resampling
            extract_ds = raster_copy("/vsimem/extracted", im_path, mode="translate", bounds=areas.bounds.iloc[area_id],
                                     resample=config.resample_factor, bands=list(config.preprocessing_bands + 1))

            # Create new  raster with two extra bands for labels and boundaries (float because we normalise im to float)
            n_bands = len(config.preprocessing_bands)
            mem_ds = gdal.GetDriverByName("MEM").Create("", xsize=extract_ds.RasterXSize, ysize=extract_ds.RasterYSize,
                                                        bands=n_bands + 1, eType=gdal.GDT_Float32)
            mem_ds.SetProjection(extract_ds.GetProjection())
            mem_ds.SetGeoTransform(extract_ds.GetGeoTransform())

            # Normalise image bands of the extract and write into new raster
            for i in range(1, n_bands+1):

                mem_ds.GetRasterBand(i).WriteArray(extract_ds.GetRasterBand(i).ReadAsArray())
            # Write annotation polygons into second-last band       (GDAL only writes the polygons in the area bounds)
            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            gdal.Rasterize(mem_ds, polygons_fp, bands=[n_bands+1], burnValues=[1], allTouched=config.rasterize_borders)

            # Write extracted area to disk
            output_fp = os.path.join(output_dir, f"{area_id}.tif")
            gdal.GetDriverByName("GTiff").CreateCopy(output_fp, mem_ds, 0)

    if len(areas) > len(os.listdir(output_dir)):
        print(f"WARNING: Training images not found for {len(areas)-len(os.listdir(output_dir))} area(s)!")

    print(f"Preprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
