import os
import math
import glob
import time
from datetime import timedelta

from tqdm import tqdm
import multiprocessing

import rasterio
import geopandas as gpd
from osgeo import ogr, gdal
from shapely.geometry import shape
from rasterio.windows import Window
from rasterio.features import shapes


def create_vector_vrt(vrt_out_fp, layer_fps, out_layer_name="trees", pbar=False):
    """Create an OGR virtual vector file.
    Concatenates several vector files in a single VRT file with OGRVRTUnionLayers.
    Layer file paths are stored as relative paths, to allow copying of the VRT file with all its data layers.
    """
    if len(layer_fps) == 0:
        return print(f"Warning! Attempt to create empty VRT file, skipping: {vrt_out_fp}")

    xml = f'<OGRVRTDataSource>\n' \
          f'    <OGRVRTUnionLayer name="{out_layer_name}">\n'
    for layer_fp in tqdm(layer_fps, desc="Creating VRT", disable=not pbar):
        shapefile = ogr.Open(layer_fp)
        layer = shapefile.GetLayer()
        relative_path = layer_fp.replace(f"{os.path.join(os.path.dirname(vrt_out_fp), '')}", "")
        xml += f'        <OGRVRTLayer name="{os.path.basename(layer_fp).split(".")[0]}">\n' \
               f'            <SrcDataSource relativeToVRT="1">{relative_path}</SrcDataSource>\n' \
               f'            <SrcLayer>{layer.GetName()}</SrcLayer>\n' \
               f'            <GeometryType>wkb{ogr.GeometryTypeToName(layer.GetGeomType())}</GeometryType>\n' \
               f'        </OGRVRTLayer>\n'
    xml += '    </OGRVRTUnionLayer>\n' \
           '</OGRVRTDataSource>\n'
    with open(vrt_out_fp, "w") as file:
        file.write(xml)


def polygonize_chunk(params):
    """Polygonize a single window chunk of the raster image."""

    raster_fp, out_fp, window = params
    polygons = []
    with rasterio.open(raster_fp) as src:
        raster_crs = src.crs
        for feature, _ in shapes(src.read(window=window), src.read(window=window), 4,
                                 src.window_transform(window)):
            polygons.append(shape(feature))
    if len(polygons) > 0:
        gpd.GeoDataFrame({"geometry": polygons}).to_file(out_fp, driver="GPKG", crs=raster_crs, layer="trees")
        return out_fp


def create_polygons(raster_dir, polygons_basedir):
    """Polygonize the raster to a vector polygons file.
    Because polygonization is slow and scales exponentially with image size, the raster is split into a grid of several
    smaller chunks, which are processed in parallel.
    As vector file merging is also very slow, the chunks are not merged into a single vector file, but instead linked in
    a virtual vector VRT file, which allows the viewing in QGIS as a single layer.
    """

    # Polygonise all raster predictions
    raster_fps = glob.glob(f"{raster_dir}/*.tif")
    for raster_fp in tqdm(raster_fps, desc="Polygonising raster predictions"):

        # Create a folder for the polygons VRT file, and a sub-folder for the actual gpkg data linked in the VRT
        prediction_name = os.path.splitext(os.path.basename(raster_fp))[0]
        polygons_dir = os.path.join(polygons_basedir, prediction_name)
        if os.path.exists(polygons_dir):
            print(f"Skipping, already processed {polygons_dir}")
            continue
        os.mkdir(polygons_dir)
        os.mkdir(os.path.join(polygons_dir, "vrtdata"))

        # Create a list of rasterio windows to split the image into a grid of smaller chunks
        chunk_windows = []
        n_rows, n_cols = config.postproc_gridsize
        with rasterio.open(raster_fp) as raster:
            width, height = math.ceil(raster.width / n_cols), math.ceil(raster.height / n_rows)
        for i in range(n_rows):
            for j in range(n_cols):
                out_fp = os.path.join(polygons_dir, "vrtdata", f"{prediction_name}_{width * j}_{height * i}.gpkg")
                chunk_windows.append([raster_fp, out_fp, Window(width * j, height * i, width, height)])

        # Polygonise image chunks in parallel
        polygon_fps = []
        with multiprocessing.Pool(processes=config.postproc_workers) as pool:
            with tqdm(total=len(chunk_windows), desc="Polygonising raster chunks", position=1, leave=False) as pbar:
                for out_fp in pool.imap_unordered(polygonize_chunk, chunk_windows):
                    if out_fp:
                        polygon_fps.append(out_fp)
                    pbar.update()

        # Merge all polygon chunks into one polygon VRT
        create_vector_vrt(os.path.join(polygons_dir, f"polygons_{prediction_name}.vrt"), polygon_fps)

    # Create giant VRT of all polygon VRTs
    merged_vrt_fp = os.path.join(polygons_basedir, f"all_polygons_{os.path.basename(config.postprocessing_dir)}.vrt")
    create_vector_vrt(merged_vrt_fp, glob.glob(f"{polygons_basedir}/*/*.vrt"))


def create_centroid_chunk(polygon_chunk_fp):
    """Create centroids for one chunk of polygons.
    The area in m2 is added as an attribute to the centroid file, using ST_Area(geom, proj_mode) with OTF reprojection.
    """

    # Copy polygons to a memory layer, because we need write access to create the CRS tables needed by ST_Area(_, _)
    # [The ST_Area(geom, proj_mode) function re-projects OTF and computes area in m2, allowing us to stay in 4326]
    polygons_ds = ogr.GetDriverByName('GPKG').Open(polygon_chunk_fp)
    memory_ds = ogr.GetDriverByName('GPKG').CreateDataSource('/vsimem/temp_polygons.gpkg')
    memory_ds.ExecuteSQL("SELECT InitSpatialMetadata(1)")
    memory_ds.CopyLayer(polygons_ds.GetLayerByName("trees"), "trees")

    # Convert to centroids and add area in m2 as an attribute. Use spherical approximation for area (->exact=false)
    centroids_layer = memory_ds.ExecuteSQL("SELECT ST_Centroid(geom) AS geom, ST_Area(geom, false) AS area "
                                           "FROM trees", dialect="sqlite")
    # Write to centroids file
    centroid_fp = polygon_chunk_fp.replace("polygons", "centroids")
    driver = ogr.GetDriverByName('GPKG')
    centroids_ds = driver.CreateDataSource(centroid_fp)
    centroids_ds.CopyLayer(centroids_layer, "trees")

    return centroid_fp


def create_centroids(polygons_basedir, centroids_basedir):
    """Convert polygons to centroids.
    Processing is done in parallel on the polygonised chunks, and centroid files are not merged but linked in a VRT.
    """

    # Process all polygon folders
    polygon_dirs = glob.glob(f"{polygons_basedir}/*/")
    for polygon_dir in tqdm(polygon_dirs, desc="Creating centroids shapefiles"):

        # Create a folder for the centroids VRT file, and a sub-folder for the actual gpkg data linked in the VRT
        prediction_name = os.path.basename(polygon_dir.rstrip("/"))
        centroids_dir = os.path.join(centroids_basedir, prediction_name)
        if os.path.exists(centroids_dir):
            print(f"Skipping, already processed {centroids_dir}")
            continue
        os.mkdir(centroids_dir)
        os.mkdir(os.path.join(centroids_dir, "vrtdata"))

        # Get all polygon chunk file paths
        polygon_fps = glob.glob(f"{polygon_dir}vrtdata/*.gpkg")

        # Convert polygonized chunks to centroid chunks in parallel
        centroid_fps = []
        with multiprocessing.Pool(processes=config.postproc_workers) as pool:
            with tqdm(total=len(polygon_fps), desc="Processing centroid chunks", position=1, leave=False) as pbar:
                for out_fp in pool.imap_unordered(create_centroid_chunk, polygon_fps):
                    centroid_fps.append(out_fp)
                    pbar.update()

        # Merge all centroid chunks into one centroids VRT
        create_vector_vrt(os.path.join(centroids_dir, f"centroids_{prediction_name}.vrt"), centroid_fps)

    # Create giant VRT of all polygon VRTs
    merged_vrt_fp = os.path.join(centroids_basedir, f"all_centroids_{os.path.basename(config.postprocessing_dir)}.vrt")
    create_vector_vrt(merged_vrt_fp, glob.glob(f"{centroids_basedir}/*/*.vrt"))


def resolution_metres2degrees(xres_metres, yres_metres, latitude):
    """Calculate the resolution in degrees equivalent to a desired resolution in metres."""

    xres_degrees = xres_metres / (111320 * math.cos(math.radians(abs(latitude))))    # at equator 1°lon ~= 111.32 km
    yres_degrees = yres_metres / 110540                                              # and        1°lat ~= 110.54 km
    return xres_degrees, yres_degrees


def create_density_raster(params):
    """Create a density map from a centroids file, at a given resolution.
    The raster has multiple bands, with first band including all trees and successive bands showing trees by crown area,
    as configured in area_limits.
    The centroid features are summed with gdal.Rasterize(), with a burn value chosen to always produce density per ha.
    The extent of the density raster is taken from the shapefile extent (not ideal but ok), and resolution calculated
    to match the target resolution in metres.
    """

    centroids_fp, out_fp, resolution_m, central_latitude, area_limits = params

    # Create output raster with one band per area group, and extent from shapefile size
    num_bands = len(area_limits) + 1
    centroids_ds = ogr.Open(centroids_fp)
    centroids_layer = centroids_ds.GetLayer("trees")
    x_res, y_res = resolution_metres2degrees(resolution_m, resolution_m, central_latitude)
    x_min, x_max, y_min, y_max = centroids_layer.GetExtent()
    width = round((x_max - x_min) / x_res)       # TO-DO figure out issue of edge pixels and missing/too much overlap
    height = round((y_max - y_min) / y_res)
    options = ["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "PREDICTOR=2"]
    out_raster_ds = gdal.GetDriverByName('GTiff').Create(out_fp, width, height, num_bands, gdal.GDT_Float32, options)
    out_raster_ds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    out_raster_ds.SetProjection(centroids_layer.GetSpatialRef().ExportToWkt())

    # Set burn value for gdal.Rasterize() such that density is always per hectar: number of trees per 10000m2
    burn_value = 10000 / resolution_m**2     # eg 1 for resolution 100m, 0.01 for 1km resolution, 100 for 10m resolution

    # Band 1: All areas
    gdal.Rasterize(out_raster_ds, centroids_fp, bands=[1], burnValues=[burn_value], add=True)

    # Bands 2 .. N-1: Specific area classes, between consecutive size limits from area_limits
    for i in range(len(area_limits) - 1):
        lower_lim, upper_lim = area_limits[i], area_limits[i + 1]
        gdal.Rasterize(out_raster_ds, centroids_fp, bands=[i + 2], burnValues=[burn_value], add=True,
                       SQLStatement=f"SELECT * FROM trees WHERE area >= {lower_lim} AND area < {upper_lim}")

    # Band N: last band is all areas bigger than max limit
    gdal.Rasterize(out_raster_ds, centroids_fp, bands=[len(area_limits) + 1], burnValues=[burn_value], add=True,
                   SQLStatement=f"SELECT * FROM trees WHERE area >= {area_limits[-1]}")
    out_raster_ds.GetRasterBand(len(area_limits) + 1).SetDescription(f"Areas > {area_limits[-1]} m2")

    return out_fp


def create_density_maps(centroids_dir, density_dir, temp_dir, resolutions, area_limits):
    """ Create density maps of trees per ha at different resolutions
    Tiles are processed in parallel, and then merged into one output map per resolution.
    """

    # Get centroid files
    centroids_fps = glob.glob(f"{centroids_dir}/*/*.vrt")
    if len(centroids_fps) == 0:
        print("No centroids found to create density maps")
        return

    # Find central latitude of centroid files, which we need to get resolution in ° for a given target resolution in m
    lowest_y_min, highest_y_max = 90, -90
    for centroids_fp in centroids_fps:
        centroids_ds = ogr.Open(centroids_fp)
        x_min, x_max, y_min, y_max = centroids_ds.GetLayer("trees").GetExtent()
        lowest_y_min, highest_y_max = min(y_min, lowest_y_min), max(y_max, highest_y_max)
    central_latitude = (lowest_y_min + highest_y_max) / 2

    # Create density rasters for all configured resolutions
    for resolution_m in resolutions:

        # Create output path and check if it's been processed before
        density_fp = os.path.join(density_dir,
                                  f"density_{resolution_m}m_{os.path.basename(config.postprocessing_dir)}.tif")
        if os.path.exists(density_fp):
            print(f"Skipping, {resolution_m}m map already exists at {density_fp}")
            continue

        # Process all centroid files in parallel
        temp_files = []
        params = [[fp, os.path.join(temp_dir, os.path.basename(fp)), resolution_m, central_latitude, area_limits]
                  for fp in centroids_fps]
        with multiprocessing.Pool(processes=config.postproc_workers) as pool:
            with tqdm(total=len(centroids_fps), desc=f"Creating {resolution_m}m density map") as pbar:
                for out_fp in pool.imap_unordered(create_density_raster, params):
                    temp_files.append(out_fp)
                    pbar.update()

        # Merge density rasters  of all tiles into one file per resolution
        gdal.BuildVRT(f"/vsimem/merged_{resolution_m}m.vrt", temp_files)
        options = ["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "PREDICTOR=2"]
        gdal.Translate(density_fp, f"/vsimem/merged_{resolution_m}m.vrt", creationOptions=options)

        # Add band names. Have to do it here because merging with gdal.Translate() loses the individual band names
        merged_ds = gdal.Open(density_fp)
        merged_ds.GetRasterBand(1).SetDescription("Density per ha, all crown sizes")
        for i in range(len(area_limits) - 1):
            merged_ds.GetRasterBand(i + 2).SetDescription(
                f"Density per ha, crown size {area_limits[i]}-{area_limits[i + 1]} m2")
        merged_ds.GetRasterBand(len(area_limits) + 1).SetDescription(
            f"Density per ha, crown size > {area_limits[-1]} m2")
        del merged_ds

        # Clean up temp dir
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))


def resample_raster(params):
    """ Resample a raster to a target resolution in degrees, with average resampling"""

    raster_fp, out_fp, x_res, y_res = params
    warp_options = dict(
        xRes=x_res,
        yRes=y_res,
        resampleAlg=gdal.GRA_Average,
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"],
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        warpMemoryLimit=1000000000,
        multithread=True
    )
    gdal.Warp(out_fp, raster_fp, **warp_options)
    return out_fp


def create_canopy_cover_maps(raster_dir, canopy_dir, temp_dir, resolutions):
    """Create raster maps of canopy cover at different resolutions
    Canopy cover is calculated as a fraction by taking pixel-wise average of binary prediction rasters.
    Tiles are processed in parallel, and then merged into one output map per resolution.
    """

    raster_fps = glob.glob(f"{raster_dir}/*.tif")
    if len(raster_fps) == 0:
        print("No rasters found to create canopy cover maps")
        return

    # Create a VRT of all raster prediction files to find central latitude,
    # which we need to compute resolution in degrees for a given target resolution in metres
    gdal.BuildVRT("/vsimem/all_prediction_rasters.vrt", raster_fps)
    all_rasters_ds = rasterio.open("/vsimem/all_prediction_rasters.vrt")
    central_latitude = (all_rasters_ds.bounds[3] + all_rasters_ds.bounds[1]) / 2

    # Create density raster for all configured resolutions
    for resolution_m in resolutions:

        # Create output path and check if it's been processed before
        canopy_fp = os.path.join(canopy_dir,
                                 f"canopy_cover_{resolution_m}m_{os.path.basename(config.postprocessing_dir)}.tif")
        if os.path.exists(canopy_fp):
            print(f"Skipping, {resolution_m}m map already exists at {canopy_fp}")
            continue

        # Process all rasters in parallel
        temp_files = []
        x_res, y_res = resolution_metres2degrees(resolution_m, resolution_m, central_latitude)
        params = [[fp, os.path.join(temp_dir, os.path.basename(fp)), x_res, y_res] for fp in raster_fps]
        with multiprocessing.Pool(processes=math.ceil(config.postproc_workers/2)) as pool:
            with tqdm(total=len(raster_fps), desc=f"Creating {resolution_m}m canopy cover map") as pbar:
                for out_fp in pool.imap_unordered(resample_raster, params):
                    temp_files.append(out_fp)
                    pbar.update()

        # Merge canopy cover rasters of all tiles into one file per resolution
        gdal.BuildVRT(f"/vsimem/merged_{resolution_m}m.vrt", temp_files)
        options = dict(
            outputType=gdal.GDT_Float32 if config.canopy_map_dtype == "float32" else gdal.GDT_Byte,
            scaleParams=[[0, 1, 0, 100]],
            creationOptions=["TILED=YES", "BIGTIFF=IF_SAFER", "COMPRESS=LZW", "PREDICTOR=2"]
        )
        gdal.Translate(canopy_fp, f"/vsimem/merged_{resolution_m}m.vrt", **options)

        # Clean up temp dir
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))


def postprocess_all(conf):
    """Run postprocessing on all prediction rasters"""

    global config
    config = conf

    print("Starting postprocessing.")
    start = time.time()

    # Get folder of predictions that will be postprocessed
    if config.postprocessing_dir is None:
        config.postprocessing_dir = os.path.join(config.predictions_base_dir, sorted(
            [f for f in os.listdir(config.predictions_base_dir)])[-1])

    print(f"Postprocessing predictions in {config.postprocessing_dir}")

    # Create sub-folders for the various outputs
    rasters_dir = os.path.join(config.postprocessing_dir, "rasters")
    polygons_dir = os.path.join(config.postprocessing_dir, "polygons")
    centroids_dir = os.path.join(config.postprocessing_dir, "centroids")
    densities_dir = os.path.join(config.postprocessing_dir, "density_rasters")
    canopies_dir = os.path.join(config.postprocessing_dir, "canopy_cover_rasters")
    temp_dir = os.path.join(config.postprocessing_dir, "temp")
    for dir_name in [rasters_dir, polygons_dir, centroids_dir, densities_dir, canopies_dir, temp_dir]:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

    # Create polygon shapefiles
    if config.create_polygons:
        create_polygons(rasters_dir, polygons_dir)

    # Create centroid shapefiles with area attributes
    if config.create_centroids:
        create_centroids(polygons_dir, centroids_dir)

    # Create density maps
    if config.create_density_maps:
        create_density_maps(centroids_dir, densities_dir, temp_dir, config.density_resolutions, config.area_thresholds)

    # Create canopy cover maps
    if config.create_canopy_cover_maps:
        create_canopy_cover_maps(rasters_dir, canopies_dir, temp_dir, config.canopy_resolutions)

    # Clean up temp dir
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"Postprocessing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")
