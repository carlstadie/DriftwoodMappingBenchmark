import json
import math
import os
import time
from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy
import skimage.transform
from osgeo import gdal
from shapely.geometry import box
from tqdm import tqdm

from core.frame_info import image_normalize
from core.util import raster_copy


def get_areas_and_polygons():
    """Read training rectangles and polygon shapefiles and pre-index polygons.

    The spatial join assigns rectangle ids to polygons in a column "index_right".
    """
    print("Reading training data shapefiles.. ", end="")
    start = time.time()

    # Read in areas and keep only geometry
    areas = gpd.read_file(os.path.join(config.training_data_dir, config.training_area_fn))
    areas = areas.drop(columns=[c for c in areas.columns if c != "geometry"])

    # Read in polygons and keep only geometry
    polygons = gpd.read_file(
        os.path.join(config.training_data_dir, config.training_polygon_fn)
    )
    polygons = polygons.drop(columns=[c for c in polygons.columns if c != "geometry"])

    print(
        f"Done in {time.time()-start:.2f} seconds. Found {len(polygons)} polygons in "
        f"{len(areas)} areas.\nAssigning polygons to areas..      ",
        end="",
    )
    start = time.time()

    # Pre-index polygons with their containing rectangle
    polygons = gpd.sjoin(polygons, areas, op="intersects", how="inner")

    print(f"Done in {time.time()-start:.2f} seconds.")
    return areas, polygons


def get_images_with_training_areas(areas):
    """Return list of (image_path, [area_ids]) for images overlapping training areas."""
    print("Assigning areas to input images..  ", end="")
    start = time.time()

    # Collect all candidate image paths
    image_paths = []
    for root, dirs, files in os.walk(config.training_image_dir):
        for fname in files:
            if (
                fname.startswith(config.train_image_prefix)
                and fname.lower().endswith(config.train_image_type.lower())
            ):
                image_paths.append(os.path.join(root, fname))

    # Filter images to those that actually contain training areas
    images_with_areas = []
    for im in image_paths:
        # Get image bounds and CRS
        with rasterio.open(im) as raster:
            im_bounds = box(*raster.bounds)
            image_crs = raster.crs

        # Only consider images whose CRS matches the areas CRS
        if image_crs != areas.crs:
            continue

        # Get training area ids overlapping this image
        areas_in_image = np.where(areas.envelope.intersects(im_bounds))[0]
        if len(areas_in_image) > 0:
            images_with_areas.append((im, [int(x) for x in list(areas_in_image)]))

    print(
        f"Done in {time.time()-start:.2f} seconds. Found {len(image_paths)} training "
        f"images of which {len(images_with_areas)} contain training areas."
    )
    return images_with_areas


def calculate_boundary_weights(polygons, scale):
    """Return boundary polygons between close polygons using scaled intersections.

    Each polygon is scaled around its centroid. Overlaps of scaled polygons
    define the boundaries. Optional difference with originals ensures
    boundaries do not cover labels.

    Needed only if we wnat to segment individual logs instead of just the patch.
    But first fnd mw the HiWi who labels that shit. Sorry.
    """
    # Scale polygons (factor should match resolution and target type)
    scaled_polys = gpd.GeoDataFrame(
        {"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin="center")}
    )

    # Intersections of scaled polygons are boundaries
    boundaries = []
    for i in range(len(scaled_polys)):
        nearby_polys = scaled_polys[
            scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)
        ]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(
                    scaled_polys.iloc[i].geometry.intersection(
                        nearby_polys.iloc[j].geometry
                    )
                )

    # Keep only Polygon geometries
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # Optional: remove overlaps with original polygons
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how="difference")
    if len(boundaries) == 0:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


def get_vectorized_annotation(polygons, areas, area_id, xsize, ysize):
    """Return list of per-polygon dicts (center, areas, pseudo-radii, geometry)."""
    # Select polygons that are within the area bounds
    isinarea = polygons[polygons.within(box(*areas.bounds.iloc[area_id]))]
    isinarea = isinarea.explode(column="geometry", index_parts=False)

    # Equal-area projection for area computations
    isinarea_ea = isinarea.to_crs(epsg=6933)
    isinarea.loc[:, "area(m)"] = isinarea_ea.area
    isinarea.loc[:, "pseudo_radius(m)"] = isinarea_ea.area.apply(
        lambda x: np.sqrt(x / np.pi)
    )

    # Pixel transform over requested raster size
    bounds = areas.iloc[[area_id]].to_crs(epsg=6933).bounds.iloc[0]
    trsfrm = rasterio.transform.from_bounds(*bounds, xsize, ysize)

    # Ground resolution in meters per pixel
    gr = np.mean([np.abs(trsfrm.a), np.abs(trsfrm.e)])
    isinarea.loc[:, "area(px)"] = isinarea["area(m)"] / (gr ** 2)
    isinarea.loc[:, "pseudo_radius(px)"] = isinarea["pseudo_radius(m)"] / gr

    # Invert to pixel coordinates and extract polygon exterior points
    trsfrm = ~trsfrm
    trsfrm = [element for tupl in trsfrm.column_vectors for element in tupl]
    isinarea.loc[:, "geometry"] = isinarea_ea["geometry"].affine_transform(trsfrm[:6])
    isinarea.loc[:, "center"] = isinarea.centroid
    isinarea.loc[:, "center"] = isinarea["center"].apply(lambda p: (p.x, p.y))
    isinarea.loc[:, "geometry"] = isinarea["geometry"].apply(
        lambda x: list(x.exterior.coords)
    )

    # Convert to a plain list of dicts
    isinarea.drop(labels=["index_right"], inplace=True, axis=1)
    isinarea = pd.DataFrame(isinarea)
    dic = isinarea.to_dict(orient="records")
    return dic


def resolution_degrees2metres(xres_degrees, yres_degrees, latitude):
    """Convert resolution in degrees to approximate meters at given latitude."""
    xres_metres = xres_degrees * (
        111320 * math.cos(math.radians(abs(latitude)))
    )
    yres_metres = yres_degrees * 110540
    return xres_metres, yres_metres


def add_additional_band(image_fp, image_bounds, out_fp, new_band, pbar_pos=0):
    """Add an auxiliary band to a raster by sampling a source file over given bounds."""
    pbar = tqdm(
        total=5,
        desc=f"{'Adding coverband...':<25}",
        leave=False,
        position=pbar_pos,
        disable=True,
    )

    # Read window of source image
    with rasterio.open(image_fp) as image_ds:
        image_window = rasterio.windows.from_bounds(*image_bounds, image_ds.transform)
        img = image_ds.read(window=image_window)
        pbar.update()

        # Read window of new band
        with rasterio.open(new_band["source_fp"]) as src:
            band_index = new_band["source_band"] if "source_band" in new_band else 1
            new_band_img = src.read(
                band_index, window=rasterio.windows.from_bounds(*image_bounds, src.transform)
            )
        pbar.update()

        # Mask invalid values [optional]
        if "maskvals" in new_band and len(new_band["maskvals"]) > 0:
            mask = np.isin(new_band_img, new_band["maskvals"])
            new_band_img[mask] = 0

        # Scale values [optional]
        if "scale_factor" in new_band and new_band["scale_factor"] is not None:
            new_band_img = new_band_img.astype(np.float32) * new_band["scale_factor"]
        pbar.update()

        # Resample to coarser resolution [optional]
        if (
            "average_to_resolution_m" in new_band
            and new_band["average_to_resolution_m"] is not None
        ):
            scale = (
                resolution_degrees2metres(*image_ds.res, 0)[1]
                / new_band["average_to_resolution_m"]
            )
            new_band_img = skimage.transform.rescale(
                new_band_img, scale=scale, order=0, mode="reflect"
            )
        pbar.update()

        # Ensure same resolution as input bands
        new_band_img = skimage.transform.resize(
            new_band_img, img.shape[1:], order=0, mode="reflect"
        )

        # Insert extra band
        merged_img = np.concatenate([img, [new_band_img]], axis=0)

        # Write output merged image to file
        profile = image_ds.profile
        profile["count"] = profile["count"] + 1
        profile["transform"] = image_ds.window_transform(image_window)
        profile["width"] = img.shape[2]
        profile["height"] = img.shape[1]
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(merged_img.astype(profile["dtype"]))

    pbar.update()
    return out_fp


def preprocess_all(conf):
    """Run preprocessing for all training data."""
    global config
    config = conf

    print("Starting preprocessing.")
    start = time.time()

    # Create output folder
    output_dir = os.path.join(
        config.preprocessed_base_dir,
        time.strftime("%Y%m%d-%H%M") + "_" + config.preprocessed_name,
    )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Read area and polygon shapefiles
    areas, polygons = get_areas_and_polygons()

    # Images that contain training areas
    images_with_areas = get_images_with_training_areas(areas)

    # For each input image, process overlapping training areas
    for im_path, area_ids in tqdm(
        images_with_areas, "Processing images with training areas", position=1
    ):
        # For each area, extract channels and write merged output
        for area_id in tqdm(
            area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0
        ):
            # Extract overlapping window (with optional resampling)
            extract_ds = raster_copy(
                "/vsimem/extracted",
                im_path,
                mode="translate",
                bounds=areas.bounds.iloc[area_id],
                resample=config.resample_factor,
                bands=list(config.preprocessing_bands + 1),
            )

            # Create new raster with an extra band for labels (float for normalization)
            n_bands = len(config.preprocessing_bands)
            mem_ds = gdal.GetDriverByName("MEM").Create(
                "",
                xsize=extract_ds.RasterXSize,
                ysize=extract_ds.RasterYSize,
                bands=n_bands + 1,
                eType=gdal.GDT_Float32,
            )
            mem_ds.SetProjection(extract_ds.GetProjection())
            mem_ds.SetGeoTransform(extract_ds.GetGeoTransform())

            # Normalize image bands (if desired) and write them
            for i in range(1, n_bands + 1):
                mem_ds.GetRasterBand(i).WriteArray(
                    extract_ds.GetRasterBand(i).ReadAsArray()
                )

            # Rasterize labels into the last band
            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            gdal.Rasterize(
                mem_ds,
                polygons_fp,
                bands=[n_bands + 1],
                burnValues=[1],
                allTouched=config.rasterize_borders,
            )

            if config.get_json:
                # Optionally write per-area annotation JSON
                dic = get_vectorized_annotation(
                    polygons, areas, area_id, extract_ds.RasterXSize, extract_ds.RasterYSize
                )
                output_fp = os.path.join(output_dir, f"{area_id}.json")
                with open(output_fp, "w") as fp:
                    json.dump(dic, fp)

            # Write extracted area to disk (tif with image+labels)
            output_fp = os.path.join(output_dir, f"{area_id}.tif")
            gdal.GetDriverByName("GTiff").CreateCopy(output_fp, mem_ds, 0)

    if len(areas) > len(os.listdir(output_dir)):
        print(
            f"WARNING: Training images not found for {len(areas)-len(os.listdir(output_dir))} areas!"
        )

    print(
        f"Preprocessing completed in "
        f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n"
    )
