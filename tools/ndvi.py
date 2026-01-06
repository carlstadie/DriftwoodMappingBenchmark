import os
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
import shapely.geometry as sg
from tqdm import tqdm

from osgeo import gdal, ogr, osr


# ===================== CONFIG =====================

# Input PlanetScope scenes
INPUT_DIR = Path("/isipd/projects/p_planetdw/data/methods_test/training_images/AE/to_encode/test/")

# Output directory for NDVI / vegetation results
OUTPUT_DIR = Path("/isipd/projects/p_planetdw/data/methods_test/auxilliary_data/ndvi")

# Final merged vegetation GPKG
VEG_GPKG = OUTPUT_DIR.parent / "vegetation_DWOlliIsland.gpkg"
VEG_LAYER = "vegetation"

# Temporary directory for per-chunk vector files
TEMP_VEC_DIR = OUTPUT_DIR / "tmp_chunks"

# Training areas (AOIs)
TRAINING_AREAS_GPKG = Path(
    "/isipd/projects/p_planetdw/data/methods_test/training/training_areas.gpkg"
)
TRAINING_LAYER = None  # set to layer name if needed, None for default

# Bands and thresholds
RED_BAND = 3
NIR_BAND = 4
NDVI_THRESHOLD = 0.0
MIN_AREA = 1     # CRS units (e.g. m² if projected). 0 = no area filtering.

# Chunking & parallelism
CHUNK_SHAPE = 1024            # pixels per chunk
N_WORKERS = 16  # number of parallel workers

# Global AOI mask (shared READ-ONLY by workers)
AOI_MASK = None
AOI_MASK_SHAPE = None
AOI_MASK_INFO = ""


# ===================== NDVI MASK =====================

def vegetation_mask(red: np.ndarray, nir: np.ndarray, threshold: float) -> np.ndarray:
    """
    Return a uint8 mask (1 = vegetation, 0 = background) for NDVI >= threshold.

    Uses inequality:
        (nir - red) / (nir + red) >= thr
        <=> (nir - red) >= thr * (nir + red)  for denom > 0
    to avoid a division.
    """
    red = red.astype("float32", copy=False)
    nir = nir.astype("float32", copy=False)

    denom = nir + red
    num = nir - red

    valid = denom > 0.0
    veg = np.zeros(red.shape, dtype="uint8")
    if not valid.any():
        return veg

    thr_denom = threshold * denom[valid]
    veg_pixels = num[valid] >= thr_denom
    veg[valid] = veg_pixels.astype("uint8")
    return veg


# ===================== CHUNK WINDOWS =====================

def build_chunk_windows_for_aoi_bbox(src, aoi_scene):
    """
    Build a list of Window objects covering only the bounding box of the AOIs
    (not the whole raster), chunked by CHUNK_SHAPE.
    """
    # AOI bounds: (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = aoi_scene.total_bounds

    # Clamp AOI bounds to raster bounds
    rb = src.bounds
    minx = max(minx, rb.left)
    maxx = min(maxx, rb.right)
    miny = max(miny, rb.bottom)
    maxy = min(maxy, rb.top)

    # Convert bounds to a window in pixel coordinates
    full_window = from_bounds(
        minx, miny, maxx, maxy,
        transform=src.transform,
        width=src.width,
        height=src.height,
    ).round_offsets().round_lengths()

    row_start = int(full_window.row_off)
    col_start = int(full_window.col_off)
    row_stop = row_start + int(full_window.height)
    col_stop = col_start + int(full_window.width)

    windows = []
    for row in range(row_start, row_stop, CHUNK_SHAPE):
        h = min(CHUNK_SHAPE, row_stop - row)
        for col in range(col_start, col_stop, CHUNK_SHAPE):
            w = min(CHUNK_SHAPE, col_stop - col)
            windows.append(Window(col, row, w, h))

    return windows


# ===================== GDAL POLYGONIZE HELPER =====================

def polygonize_chunk_with_gdal(
    veg_array: np.ndarray,
    transform,
    crs_wkt: str,
    src_name: str,
    out_fp: str,
):
    """
    Use GDAL Polygonize to convert a uint8 veg_array (0/1) into polygons, written to a GPKG.

    Only polygons where value == 1 are kept, and MIN_AREA is applied (if > 0).
    Also sets a 'src' field with the input raster name.
    """
    h, w = veg_array.shape
    if veg_array.dtype != np.uint8:
        veg_array = veg_array.astype("uint8")

    # Skip if nothing is vegetated
    if not veg_array.any():
        return None

    # Create in-memory raster
    mem_drv = gdal.GetDriverByName("MEM")
    ras_ds = mem_drv.Create("", w, h, 1, gdal.GDT_Byte)
    ras_ds.SetGeoTransform(transform.to_gdal()) if hasattr(transform, "to_gdal") \
        else ras_ds.SetGeoTransform(transform)
    ras_ds.SetProjection(crs_wkt)

    band = ras_ds.GetRasterBand(1)
    band.WriteArray(veg_array)
    band.SetNoDataValue(0)  # non-veg is 0

    # Create output GPKG
    gpkg_drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(out_fp):
        gpkg_drv.DeleteDataSource(out_fp)
    vec_ds = gpkg_drv.CreateDataSource(out_fp)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)

    layer = vec_ds.CreateLayer("vegetation", srs=srs, geom_type=ogr.wkbPolygon)

    # Fields: value (pixel value) and src (scene name)
    fld_val = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(fld_val)
    fld_src = ogr.FieldDefn("src", ogr.OFTString)
    fld_src.SetWidth(128)
    layer.CreateField(fld_src)

    layer_defn = layer.GetLayerDefn()
    val_index = layer_defn.GetFieldIndex("value")
    src_index = layer_defn.GetFieldIndex("src")

    # Polygonize: use band as both src and mask -> non-zero pixels only
    gdal.Polygonize(band, band, layer, val_index, [], callback=None)

    # Filter features: keep only value == 1 and MIN_AREA, set src field
    fids_to_delete = []
    for feat in layer:
        val = feat.GetField(val_index)
        if val != 1:
            fids_to_delete.append(feat.GetFID())
            continue

        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            fids_to_delete.append(feat.GetFID())
            continue

        if MIN_AREA > 0 and geom.GetArea() < MIN_AREA:
            fids_to_delete.append(feat.GetFID())
            continue

        feat.SetField(src_index, src_name)
        layer.SetFeature(feat)

    for fid in fids_to_delete:
        layer.DeleteFeature(fid)

    layer.SyncToDisk()

    # Cleanup
    layer = None
    vec_ds = None
    band = None
    ras_ds = None

    # If layer ended up empty, remove file and return None
    ds_check = gpkg_drv.Open(out_fp, 0)
    if ds_check is not None and ds_check.GetLayer(0).GetFeatureCount() == 0:
        ds_check = None
        gpkg_drv.DeleteDataSource(out_fp)
        return None

    return out_fp


# ===================== WORKER =====================

def process_chunk(params):
    """
    Worker: compute NDVI + AOI mask in a chunk, polygonize with GDAL, write chunk GPKG.

    params: (tif_path_str, col_off, row_off, width, height, chunk_out_fp, crs_wkt)
    """
    global AOI_MASK, AOI_MASK_SHAPE

    tif_path_str, col, row, w, h, chunk_out_fp, crs_wkt = params
    window = Window(col, row, w, h)

    # Quick AOI check using shared AOI_MASK
    if AOI_MASK is None:
        raise RuntimeError("AOI_MASK is not set in worker process")

    r0 = int(row)
    r1 = int(row + h)
    c0 = int(col)
    c1 = int(col + w)

    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(AOI_MASK_SHAPE[0], r1)
    c1 = min(AOI_MASK_SHAPE[1], c1)

    aoi_sub = AOI_MASK[r0:r1, c0:c1]
    if aoi_sub.size == 0 or not aoi_sub.any():
        # No AOI pixels in this chunk: skip
        return None

    with rasterio.open(tif_path_str) as src:
        arrays = src.read(
            [RED_BAND, NIR_BAND],
            window=window,
            out_dtype="float32",
        )
        red, nir = arrays
        transform = src.window_transform(window)

    veg = vegetation_mask(red, nir, NDVI_THRESHOLD)

    # Restrict to AOI
    if aoi_sub.shape != veg.shape:
        # Crop AOI mask to veg shape if needed (edge alignment issues)
        aoi_sub = aoi_sub[:veg.shape[0], :veg.shape[1]]
    veg[aoi_sub == 0] = 0

    if not veg.any():
        return None

    src_name = os.path.basename(tif_path_str)
    out_fp = polygonize_chunk_with_gdal(
        veg,
        transform,
        crs_wkt,
        src_name,
        chunk_out_fp,
    )
    return out_fp


# ===================== PER-SCENE PROCESSING =====================

def process_scene(tif_path: Path, aoi_scene: gpd.GeoDataFrame):
    """
    For a single raster:
      - build AOI_MASK (global)
      - build chunk windows over AOI bbox
      - run per-chunk multiprocessing (GDAL polygonize)
      - return list of chunk GPKG paths
    """
    global AOI_MASK, AOI_MASK_SHAPE, AOI_MASK_INFO

    tif_path_str = str(tif_path)
    chunk_outputs = []

    with rasterio.open(tif_path_str) as src:
        crs = src.crs
        crs_wkt = crs.to_wkt()

        # Ensure AOIs in same CRS
        if aoi_scene.crs != crs:
            aoi_scene = aoi_scene.to_crs(crs)

        print(f"  - Raster size: {src.width} x {src.height} px")

        # Rasterize buffered AOIs as mask
        pixel_size = abs(src.transform.a)  # assume square, non-rotated
        buffer_dist = pixel_size * 0.5

        aoi_scene_buf = aoi_scene.copy()
        aoi_scene_buf["geometry"] = aoi_scene_buf.geometry.buffer(buffer_dist)

        shapes_for_rasterize = [
            (geom, 1)
            for geom in aoi_scene_buf.geometry
            if geom is not None and not geom.is_empty and geom.is_valid
        ]

        t_mask = time.time()
        AOI_MASK = rasterize(
            shapes=shapes_for_rasterize,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=True,   # be generous on edges
        )
        AOI_MASK_SHAPE = AOI_MASK.shape
        n_aoi_pixels = int(AOI_MASK.sum())
        AOI_MASK_INFO = f"AOI mask shape={AOI_MASK.shape}, AOI pixels={n_aoi_pixels}"
        print(f"  - Rasterized buffered AOIs to mask in {time.time() - t_mask:.2f}s ({AOI_MASK_INFO})")

        # Build windows over AOI bbox
        t0 = time.time()
        windows = build_chunk_windows_for_aoi_bbox(src, aoi_scene_buf)
        print(f"  - Built {len(windows)} chunk windows over AOI bbox in {time.time() - t0:.2f}s")

    if not windows:
        AOI_MASK = None
        AOI_MASK_SHAPE = None
        AOI_MASK_INFO = ""
        return []

    # Prepare params for workers
    scene_tmp_dir = TEMP_VEC_DIR / tif_path.stem
    scene_tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with rasterio.open(tif_path_str) as src:
        crs_wkt = src.crs.to_wkt()

    for i, w in enumerate(windows):
        chunk_out_fp = str(scene_tmp_dir / f"{tif_path.stem}_chunk_{i:05d}.gpkg")
        jobs.append(
            (tif_path_str, w.col_off, w.row_off, w.width, w.height, chunk_out_fp, crs_wkt)
        )

    print("  - Starting chunk multiprocessing...")
    t1 = time.time()
    with mp.Pool(processes=N_WORKERS) as pool:
        for out_fp in tqdm(
            pool.imap_unordered(process_chunk, jobs),
            total=len(jobs),
            desc=f"  Chunks {tif_path.name}",
            position=1,
            leave=False,
        ):
            if out_fp:
                chunk_outputs.append(out_fp)
    print(f"  - Finished chunks in {time.time() - t1:.2f}s")

    # Clear AOI mask for this scene
    AOI_MASK = None
    AOI_MASK_SHAPE = None
    AOI_MASK_INFO = ""

    if not chunk_outputs:
        print("  - No vegetation polygons found in AOI for this scene.")
    else:
        print(f"  - {len(chunk_outputs)} chunk vector files produced.")

    return chunk_outputs


# ===================== MERGING CHUNKS =====================

def merge_chunk_files(chunk_fps, merged_fp, layer_name):
    """
    Merge many GPKG chunk files into a single GPKG using GDAL/OGR.
    """
    if not chunk_fps:
        return

    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(merged_fp):
        drv.DeleteDataSource(merged_fp)

    out_ds = None
    out_layer = None

    for i, fp in enumerate(tqdm(chunk_fps, desc="Merging chunks")):
        src_ds = drv.Open(fp, 0)
        if src_ds is None:
            continue
        src_layer = src_ds.GetLayer(0)

        if out_ds is None:
            # Create destination
            out_ds = drv.CreateDataSource(str(merged_fp))
            srs = src_layer.GetSpatialRef()
            geom_type = src_layer.GetGeomType()
            out_layer = out_ds.CreateLayer(layer_name, srs=srs, geom_type=geom_type)

            # Copy fields
            src_defn = src_layer.GetLayerDefn()
            for fld_i in range(src_defn.GetFieldCount()):
                fld_defn = src_defn.GetFieldDefn(fld_i)
                out_layer.CreateField(fld_defn)

        out_defn = out_layer.GetLayerDefn()
        for feat in src_layer:
            out_feat = ogr.Feature(out_defn)
            out_feat.SetFrom(feat)
            out_layer.CreateFeature(out_feat)
            out_feat = None

        src_layer = None
        src_ds = None

    if out_layer is not None:
        out_layer.SyncToDisk()
    out_layer = None
    out_ds = None


# ===================== MAIN =====================

def main():
    print("=== Vegetation polygonization with GDAL (AOI-based, chunked, parallel) ===")
    print(f"Input rasters: {INPUT_DIR}")
    print(f"Training areas: {TRAINING_AREAS_GPKG}")
    print(f"Output GPKG: {VEG_GPKG} (layer='{VEG_LAYER}')")
    print(f"Chunk size: {CHUNK_SHAPE} px, workers: {N_WORKERS}")
    print("======================================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_VEC_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_AREAS_GPKG.exists():
        raise FileNotFoundError(f"Training areas GPKG not found: {TRAINING_AREAS_GPKG}")

    # Clean existing output
    if VEG_GPKG.exists():
        print(f"Removing existing GPKG: {VEG_GPKG}")
        VEG_GPKG.unlink()

    tif_paths = sorted(INPUT_DIR.glob("*.tif"))
    if not tif_paths:
        print(f"No .tif files found in {INPUT_DIR}")
        return

    stats = {"processed": 0, "skipped": 0, "no_aoi": 0}
    expected_crs = None
    training_gdf = None
    training_sindex = None
    all_chunk_files = []

    for idx, tif_path in enumerate(tqdm(tif_paths, desc="Scenes", position=0)):
        stats["processed"] += 1
        print(f"\n=== Scene {idx + 1}/{len(tif_paths)}: {tif_path.name} ===")

        with rasterio.open(tif_path) as src:
            crs = src.crs

            if expected_crs is None:
                expected_crs = crs

                # Load training areas
                if TRAINING_LAYER is None:
                    training_gdf = gpd.read_file(TRAINING_AREAS_GPKG)
                else:
                    training_gdf = gpd.read_file(TRAINING_AREAS_GPKG, layer=TRAINING_LAYER)

                if training_gdf.empty:
                    raise ValueError(f"No features found in training areas: {TRAINING_AREAS_GPKG}")

                if training_gdf.crs != expected_crs:
                    print("Reprojecting training areas to match raster CRS...")
                    training_gdf = training_gdf.to_crs(expected_crs)

                training_sindex = training_gdf.sindex
                print(f"Loaded {len(training_gdf)} training polygons in CRS {training_gdf.crs}")

            elif crs != expected_crs:
                raise ValueError(f"CRS mismatch in {tif_path.name}: expected {expected_crs}, got {crs}")

            # Raster bbox geometry
            scene_geom = sg.box(*src.bounds)

            # Candidate AOIs via spatial index
            possible_idx = list(training_sindex.intersection(scene_geom.bounds))
            if not possible_idx:
                stats["no_aoi"] += 1
                print("  - No AOIs intersect raster bbox (by index), skipping.")
                continue

            aoi_scene = training_gdf.iloc[possible_idx]
            aoi_scene = aoi_scene[aoi_scene.intersects(scene_geom)]

            if aoi_scene.empty:
                stats["no_aoi"] += 1
                print("  - No AOIs intersect raster geometry, skipping.")
                continue

            print(f"  - {len(aoi_scene)} training polygons overlap this scene.")

        # Process scene with GDAL polygonize
        chunk_files = process_scene(tif_path, aoi_scene)
        if not chunk_files:
            stats["skipped"] += 1
        else:
            all_chunk_files.extend(chunk_files)

    # Merge all chunk GPKGs into final one
    if all_chunk_files:
        print("\nMerging chunk GPKGs into final vegetation.gpkg...")
        merge_chunk_files(all_chunk_files, VEG_GPKG, VEG_LAYER)
        print(f"Final GPKG written to {VEG_GPKG}")
    else:
        print("\nNo chunk files produced; final GPKG will be empty / missing.")

    # Optional: clean up temp chunk directory
    # import shutil
    # shutil.rmtree(TEMP_VEC_DIR, ignore_errors=True)

    print(
        f"\nDone.\n"
        f"  Rasters processed: {stats['processed']}\n"
        f"  Scenes with AOI but no veg: {stats['skipped']}\n"
        f"  Scenes with no AOI overlap: {stats['no_aoi']}\n"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
