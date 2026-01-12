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
from shapely.ops import unary_union
from tqdm import tqdm

from core.frame_info import image_normalize
from core.util import raster_copy

# Reuse the same split logic as training
from core.split_frames import split_dataset, summarize_positive_rates


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
    """Return boundary polygons between close polygons using scaled intersections."""
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
    isinarea.loc[:, "area(px)"] = isinarea["area(m)"] / (gr**2)
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
    xres_metres = xres_degrees * (111320 * math.cos(math.radians(abs(latitude))))
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
                band_index,
                window=rasterio.windows.from_bounds(*image_bounds, src.transform),
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
            scale = resolution_degrees2metres(*image_ds.res, 0)[1] / new_band[
                "average_to_resolution_m"
            ]
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


# ============================
# helpers (split + chips)
# ============================

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_split_indices_from_aalist(
    split_list_path: str,
    stems_sorted: list,
) -> tuple[list, list, list]:
    """
    Load a predefined split from an aa_frames_list.json-like file and map it
    onto the *current* stems_sorted ordering.

    Supports optional explicit id list in the JSON (area_ids/frame_ids/stems/ids).
    If absent, assumes indices already refer to the current ordering.
    """
    print("Loading split...")
    with open(split_list_path, "r") as f:
        obj = json.load(f)

    train_idx = obj.get("training_frames", obj.get("train", []))
    val_idx = obj.get("validation_frames", obj.get("val", []))
    test_idx = obj.get("testing_frames", obj.get("test", []))

    # Optional explicit mapping list (index -> area_id)
    id_list = None
    for k in ("area_ids", "frame_ids", "stems", "ids"):
        if k in obj and isinstance(obj[k], list) and len(obj[k]) > 0:
            id_list = [int(x) for x in obj[k]]
            break

    # If there's an id_list, indices refer to positions in id_list; map to current stems
    if id_list is not None:
        stem_to_pos = {int(s): int(i) for i, s in enumerate(stems_sorted)}

        def _map(idxs):
            out = []
            for i in idxs:
                i = int(i)
                if i < 0 or i >= len(id_list):
                    continue
                area_id = int(id_list[i])
                if area_id in stem_to_pos:
                    out.append(stem_to_pos[area_id])
            return out

        tr = _map(train_idx)
        va = _map(val_idx)
        te = _map(test_idx)
    else:
        # Otherwise assume indices already match current ordering; just clip to range
        n = len(stems_sorted)

        def _clip(idxs):
            out = []
            for i in idxs:
                i = int(i)
                if 0 <= i < n:
                    out.append(i)
            return out

        tr = _clip(train_idx)
        va = _clip(val_idx)
        te = _clip(test_idx)

    print(
        f"[SPLIT][LOAD] Using predefined split from {split_list_path} | "
        f"train={len(tr)} val={len(va)} test={len(te)}"
    )
    return tr, va, te


def _write_split_json(frames_json: str, train_idx: list, val_idx: list, test_idx: list) -> None:
    frame_split = {
        "training_frames": list(map(int, train_idx)),
        "validation_frames": list(map(int, val_idx)),
        "testing_frames": list(map(int, test_idx)),
    }
    _ensure_dir(os.path.dirname(frames_json))
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)


def _chip_params_from_config():
    px = float(getattr(config, "pixel_size_m", 0.15))
    ps = (16,16) #getattr(config, "patch_size", (256, 256))
    p = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)

    chip_size_m = float(getattr(config, "chip_size_m", p * px))
    chip_pad_m = float(getattr(config, "chip_pad_m", 0.25 * chip_size_m))
    cluster_buffer_m = float(getattr(config, "chip_cluster_buffer_m", 0.50 * chip_size_m))
    chip_max_size_m = float(getattr(config, "chip_max_size_m", 2.00 * chip_size_m))
    neg_chips_per_area = int(getattr(config, "neg_chips_per_area", 1))

    return chip_size_m, chip_pad_m, cluster_buffer_m, chip_max_size_m, neg_chips_per_area


def _random_negative_boxes_in_area_ea(area_ea_geom, polys_ea: gpd.GeoDataFrame, chip_size_m: float, n: int, rng):
    """
    Sample up to n random boxes inside *area polygon* (EA CRS) that do not intersect polygons.
    Ensures each box is fully within the training area geometry (no leakage).
    """
    if n <= 0:
        return []

    minx, miny, maxx, maxy = area_ea_geom.bounds
    half = 0.5 * chip_size_m

    polys_union = unary_union(list(polys_ea.geometry)) if len(polys_ea) > 0 else None

    out = []
    tries = 0
    max_tries = 200 * max(1, n)

    while len(out) < n and tries < max_tries:
        tries += 1

        if (maxx - minx) > chip_size_m:
            cx = float(rng.uniform(minx + half, maxx - half))
        else:
            cx = float((minx + maxx) / 2)

        if (maxy - miny) > chip_size_m:
            cy = float(rng.uniform(miny + half, maxy - half))
        else:
            cy = float((miny + maxy) / 2)

        b = box(cx - half, cy - half, cx + half, cy + half)

        if not b.within(area_ea_geom):
            continue

        if polys_union is not None and b.intersects(polys_union):
            continue

        out.append(b)

    return out


def _merge_overlapping_boxes_ea(boxes_ea):
    """Iteratively merge any overlapping boxes (positive-area overlap) into union bboxes."""
    boxes = list(boxes_ea)
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            merged = False
            for j in range(len(boxes)):
                b = boxes[j]
                inter = a.intersection(b)
                if not inter.is_empty and float(getattr(inter, "area", 0.0)) > 0.0:
                    u = a.union(b)
                    a = box(*u.bounds)
                    boxes.pop(j)
                    changed = True
                    merged = True
                    break
            if merged:
                boxes.append(a)
            else:
                out.append(a)
        boxes = out
    return boxes


def _safe_box_within_area_ea(b_ea, area_ea):
    """Ensure axis-aligned bbox is fully within area geometry; shrink via intersection bounds if needed."""
    if b_ea.within(area_ea):
        return b_ea
    clipped = b_ea.intersection(area_ea)
    if clipped.is_empty:
        return None
    cand = box(*clipped.bounds)
    if cand.within(area_ea):
        return cand
    return None


def _pos_boxes_from_label_components(
    src: rasterio.io.DatasetReader,
    area_geom_crs,
    *,
    chip_pad_m: float,
    chip_max_size_m: float,
    chip_cc_dilate_m: float,
):
    """
    Build non-overlapping, variable-sized POS chip boxes from connected components of the label band.

    Returns a list of boxes in EPSG:6933 (meters), guaranteed non-overlapping and within area geometry.
    """
    lab = src.read(src.count)
    pos = (np.asarray(lab) > 0)

    if not np.any(pos):
        return []

    px_m = float(getattr(config, "pixel_size_m", 0.15)) * float(getattr(config, "resample_factor", 1))
    dilate_px = int(max(0, round(float(chip_cc_dilate_m) / max(px_m, 1e-12))))

    if dilate_px > 0:
        pos = scipy.ndimage.binary_dilation(pos, iterations=dilate_px)

    lbl, n = scipy.ndimage.label(pos)
    if n <= 0:
        return []

    objs = scipy.ndimage.find_objects(lbl)
    bboxes_crs = []
    for sl in objs:
        if sl is None:
            continue
        r0, r1 = int(sl[0].start), int(sl[0].stop)
        c0, c1 = int(sl[1].start), int(sl[1].stop)
        if r1 <= r0 or c1 <= c0:
            continue

        win = rasterio.windows.Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        b = rasterio.windows.bounds(win, transform=src.transform)
        bboxes_crs.append(box(*b))

    if len(bboxes_crs) == 0:
        return []

    # Convert all to equal-area once (fast)
    bboxes_ea = gpd.GeoSeries(bboxes_crs, crs=src.crs).to_crs(epsg=6933).tolist()
    area_ea = gpd.GeoSeries([area_geom_crs], crs=src.crs).to_crs(epsg=6933).iloc[0]

    padded = []
    for b in bboxes_ea:
        bb = box(*b.bounds).buffer(float(chip_pad_m))
        inter = bb.intersection(area_ea)
        if inter.is_empty:
            continue
        cand = box(*inter.bounds)
        cand = _safe_box_within_area_ea(cand, area_ea)
        if cand is None:
            continue
        padded.append(cand)

    if len(padded) == 0:
        return []

    # Ensure no overlaps after padding by merging overlaps
    merged = _merge_overlapping_boxes_ea(padded)

    # Enforce max size without introducing overlap (handled elsewhere)
    out = []
    for m in merged:
        out.append(m)

    return out


def _write_chip_from_area_raster(src: rasterio.io.DatasetReader, bounds_crs, out_fp: str):
    """Crop chip from already-written area raster (includes label band). Returns label array (last band)."""
    rb = src.bounds
    minx = max(float(bounds_crs[0]), float(rb.left))
    miny = max(float(bounds_crs[1]), float(rb.bottom))
    maxx = min(float(bounds_crs[2]), float(rb.right))
    maxy = min(float(bounds_crs[3]), float(rb.top))
    if minx >= maxx or miny >= maxy:
        return None

    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None

    data = src.read(window=win)
    if data.size == 0 or data.shape[1] <= 0 or data.shape[2] <= 0:
        return None

    prof = src.profile.copy()
    prof.update(
        {
            "height": int(data.shape[1]),
            "width": int(data.shape[2]),
            "transform": src.window_transform(win),
        }
    )

    with rasterio.open(out_fp, "w", **prof) as dst:
        dst.write(data)

    return data[-1]


def preprocess_all(conf):
    """Run preprocessing for all training data."""
    global config
    config = conf

    print("Starting preprocessing.")
    start = time.time()

    # One timestamp for BOTH outputs (areas + chips)
    run_stamp = time.strftime("%Y%m%d-%H%M") + "_" + config.run_name

    # Large rasterized areas: config.preprocessed_dir / time_runname
    output_dir = os.path.join(config.preprocessed_base_dir, run_stamp)
    _ensure_dir(output_dir)

    # Chips: config.training_data_dir / time_runname
    chips_dir = os.path.join(config.training_data_base_dir, run_stamp)
    _ensure_dir(chips_dir)

    # Read area and polygon shapefiles
    areas, polygons = get_areas_and_polygons()

    # Images that contain training areas
    images_with_areas = get_images_with_training_areas(areas)

    # Remember which source image was used for each area (needed for chips)
    area_to_im = {}

    # Track which area rasters actually got written
    written_area_ids = set()

    # For each input image, process overlapping training areas
    for im_path, area_ids in tqdm(
        images_with_areas, "Processing images with training areas", position=1
    ):
        for area_id in tqdm(
            area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0
        ):
            out_fp = os.path.join(output_dir, f"{area_id}.tif")

            # Avoid duplicate work if an area intersects multiple images
            if os.path.exists(out_fp):
                written_area_ids.add(int(area_id))
                if int(area_id) not in area_to_im:
                    area_to_im[int(area_id)] = im_path
                continue

            extract_ds = raster_copy(
                "/vsimem/extracted",
                im_path,
                mode="translate",
                bounds=areas.bounds.iloc[area_id],
                resample=config.resample_factor,
                bands=list(config.preprocessing_bands + 1),
            )

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

            for i in range(1, n_bands + 1):
                mem_ds.GetRasterBand(i).WriteArray(
                    extract_ds.GetRasterBand(i).ReadAsArray()
                )

            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            gdal.Rasterize(
                mem_ds,
                polygons_fp,
                bands=[n_bands + 1],
                burnValues=[1],
                allTouched=config.rasterize_borders,
            )

            if config.get_json:
                dic = get_vectorized_annotation(
                    polygons, areas, area_id, extract_ds.RasterXSize, extract_ds.RasterYSize
                )
                json_fp = os.path.join(output_dir, f"{area_id}.json")
                with open(json_fp, "w") as fp:
                    json.dump(dic, fp)

            gdal.GetDriverByName("GTiff").CreateCopy(out_fp, mem_ds, 0)

            written_area_ids.add(int(area_id))
            if int(area_id) not in area_to_im:
                area_to_im[int(area_id)] = im_path

            mem_ds = None
            extract_ds = None

    if len(areas) > len([f for f in os.listdir(output_dir) if f.lower().endswith(".tif")]):
        print(
            f"WARNING: Training images not found for "
            f"{len(areas)-len([f for f in os.listdir(output_dir) if f.lower().endswith('.tif')])} areas!"
        )

    # ---- build frames list from written rasters (for split_dataset stratification) ----
    written_area_stems = sorted([int(x) for x in written_area_ids])

    class _FrameStub:
        def __init__(self, annotations):
            self.annotations = annotations

    frames = []
    for sid in written_area_stems:
        fp = os.path.join(output_dir, f"{sid}.tif")
        if not os.path.exists(fp):
            continue
        with rasterio.open(fp) as ds:
            lab = ds.read(ds.count)  # last band
        frames.append(_FrameStub(lab))

    frames_json = os.path.join(output_dir, "aa_frames_list.json")

    # Optional predefined split list path
    split_list_path = getattr(config, "split_list_path", None)
    if split_list_path is not None and str(split_list_path).strip() != "" and os.path.exists(str(split_list_path)):
        tr_idx, va_idx, te_idx = _load_split_indices_from_aalist(str(split_list_path), written_area_stems)
        _write_split_json(frames_json, tr_idx, va_idx, te_idx)
    else:
        tr_idx, va_idx, te_idx = split_dataset(
            frames,
            frames_json,
            test_size=float(getattr(config, "test_ratio", 0.2)),
            val_size=float(getattr(config, "val_ratio", 0.2)),
            n_bins=int(getattr(config, "split_n_bins", 5)),
            random_state=int(getattr(config, "split_random_state", 1337)),
            stratify_by_positives=True,
        )

    # Print training-like distribution stats
    stats = summarize_positive_rates(
        frames,
        {"train": tr_idx, "val": va_idx, "test": te_idx},
    )

    def _fmt(s):
        return (
            f"{s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | "
            f"{s['min']:.3f}..{s['max']:.3f}  (n={s['n']})"
        )

    print("\n[DATA][STATS] positive-rate % by frame - mean | median | std | min..max  (n)")
    print(f"  train: {_fmt(stats['train'])}")
    print(f"    val: {_fmt(stats['val'])}")
    print(f"   test: {_fmt(stats['test'])}")

    # Build area_id -> split mapping (for chip inheritance)
    split_of_area = {}
    for i in tr_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "train"
    for i in va_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "val"
    for i in te_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "test"
    for aid in written_area_stems:
        if int(aid) not in split_of_area:
            split_of_area[int(aid)] = "train"

    # ============================
    # chips (inherit split from parent area)
    # ============================
    make_chips = bool(getattr(config, "make_chip_dataset", True))
    if make_chips:
        chip_size_m, chip_pad_m, cluster_buffer_m, chip_max_size_m, neg_chips_per_area = _chip_params_from_config()
        rng = np.random.default_rng(int(getattr(config, "split_random_state", 1337)) + 17)

        polys = polygons.copy()
        try:
            polys["geometry"] = polys["geometry"].buffer(0)
        except Exception:
            pass

        chip_id = 0
        chip_split = {}

        # Dilation distance for grouping components (overrideable). Keep moderate by default.
        chip_cc_dilate_m = float(getattr(config, "chip_cc_dilate_m", min(10.0, 0.25 * chip_size_m)))

        # Focus areas (optional but required for this chip mode): vector polygons that define chip footprints.
        # Path is joined from config.training_data_dir and config.focus_areas.
        focus_fp = os.path.join(config.training_data_dir, getattr(config, "focus_areas"))
        if not os.path.exists(focus_fp):
            raise FileNotFoundError(f"Focus areas file not found: {focus_fp}")

        focus_areas = gpd.read_file(focus_fp)
        focus_areas = focus_areas.drop(columns=[c for c in focus_areas.columns if c != "geometry"])
        if focus_areas.crs is None:
            focus_areas = focus_areas.set_crs(areas.crs)
        if focus_areas.crs != areas.crs:
            focus_areas = focus_areas.to_crs(areas.crs)

        # Assign each focus polygon to one or more training areas (index_right = area_id)
        focus_areas = gpd.sjoin(focus_areas, areas, op="intersects", how="inner")

        # Pre-group focus geometries by area for fast lookup in the chip loop
        focus_by_area = {}
        for _rid, _row in focus_areas.iterrows():
            _aid = int(_row["index_right"])
            _geom = _row.geometry
            if _geom is None or _geom.is_empty:
                continue
            focus_by_area.setdefault(_aid, []).append(_geom)

        for area_id in tqdm(written_area_stems, desc="Building chips (no-overlap CC)", position=0):
            # Keep original guard (but we no longer use src_im; we crop from the area raster)
            if int(area_id) not in area_to_im:
                continue

            split_name = split_of_area.get(int(area_id), "train")

            area_raster_fp = os.path.join(output_dir, f"{area_id}.tif")
            if not os.path.exists(area_raster_fp):
                continue

            # Training area geometry in CRS (same as rasters)
            area_geom_crs = areas.iloc[int(area_id)].geometry

            # POS + NEG chips are made disjoint by tracking occupied EA boxes
            occupied_ea = []

            with rasterio.open(area_raster_fp) as src:
                # --- POS chips: direct crop of focus areas (vector polygons) ---
                # Focus polygons are in areas CRS; we clip them to the training area to avoid leakage,
                # then crop their *bounding boxes* from the already-rasterized area TIFF.
                focus_geoms = focus_by_area.get(int(area_id), [])

                # Convert area geometry to equal-area once for safe box-within-area checks
                try:
                    area_geom_src = gpd.GeoSeries([area_geom_crs], crs=areas.crs).to_crs(src.crs).iloc[0]
                except Exception:
                    area_geom_src = area_geom_crs
                area_ea = gpd.GeoSeries([area_geom_src], crs=src.crs).to_crs(epsg=6933).iloc[0]

                pos_boxes_ea = []
                for fg in focus_geoms:
                    if fg is None or fg.is_empty:
                        continue

                    # Clip focus polygon by training area in vector domain (areas CRS)
                    fg_clip = fg.intersection(area_geom_crs)
                    if fg_clip is None or fg_clip.is_empty:
                        continue

                    # Project to raster CRS for bounds extraction
                    fg_src = gpd.GeoSeries([fg_clip], crs=areas.crs).to_crs(src.crs).iloc[0]
                    b_src = box(*fg_src.bounds)

                    # Convert bbox to equal-area and ensure it is fully within the training area (no leakage)
                    b_ea = gpd.GeoSeries([b_src], crs=src.crs).to_crs(epsg=6933).iloc[0]
                    b_ea = _safe_box_within_area_ea(b_ea, area_ea)
                    if b_ea is None:
                        continue

                    pos_boxes_ea.append(b_ea)

                # Merge overlaps so final chips are non-overlapping within each training area
                if len(pos_boxes_ea) > 1:
                    pos_boxes_ea = _merge_overlapping_boxes_ea(pos_boxes_ea)

                # Write POS chips (crop from area raster)
                for b_ea in pos_boxes_ea:
                    b_crs = gpd.GeoSeries([b_ea], crs="EPSG:6933").to_crs(src.crs).iloc[0]
                    out_fp = os.path.join(chips_dir, f"{chip_id}.tif")
                    _ = _write_chip_from_area_raster(src, b_crs.bounds, out_fp)
                    chip_split[int(chip_id)] = split_name
                    chip_id += 1
                    occupied_ea.append(b_ea)

                occupied_union = unary_union(occupied_ea) if len(occupied_ea) > 0 else None

                # --- NEG chips: random, fully within training area, label-free, and non-overlapping with POS/other NEG ---
                if neg_chips_per_area > 0:
                    # We'll sample in equal-area (meters) for size correctness
                    area_ea = gpd.GeoSeries([area_geom_crs], crs=src.crs).to_crs(epsg=6933).iloc[0]
                    minx, miny, maxx, maxy = area_ea.bounds

                    # Enforce square NEG chips of fixed pixel size (default 512x512)
                    # Compute pixel size (m) from area raster transform
                    try:
                        px_m_x = abs(float(src.transform.a))
                        px_m_y = abs(float(src.transform.e))
                        px_m = float((px_m_x + px_m_y) / 2.0)
                    except Exception:
                        px_m = float(getattr(config, "pixel_size_m", 0.15))

                    neg_chip_px = int(getattr(config, "neg_chip_pixels", 512))
                    half = 0.5 * neg_chip_px * px_m

                    n_done = 0
                    tries = 0
                    max_tries = 400 * max(1, int(neg_chips_per_area))

                    while n_done < int(neg_chips_per_area) and tries < max_tries:
                        tries += 1

                        width = maxx - minx
                        height = maxy - miny

                        # Need enough room for a full square of side 2*half
                        if width <= 2 * half or height <= 2 * half:
                            cx = 0.5 * (minx + maxx)
                            cy = 0.5 * (miny + maxy)
                        else:
                            cx = float(rng.uniform(minx + half, maxx - half))
                            cy = float(rng.uniform(miny + half, maxy - half))


                        # Square NEG chip box sized to produce ~neg_chip_px pixels per side
                        nb = box(cx - half, cy - half, cx + half, cy + half)

                        # Must be fully within training area to prevent leakage
                        if not nb.within(area_ea):
                            continue

                        # No overlap with any existing chip (POS or previously accepted NEG)
                        if occupied_union is not None:
                            inter = nb.intersection(occupied_union)
                            if not inter.is_empty and float(getattr(inter, "area", 0.0)) > 0.0:
                                continue

                        # Convert to raster CRS and crop; then verify label-free
                        nb_crs = gpd.GeoSeries([nb], crs="EPSG:6933").to_crs(src.crs).iloc[0]
                        out_fp = os.path.join(chips_dir, f"{chip_id}.tif")
                        lab = _write_chip_from_area_raster(src, nb_crs.bounds, out_fp)
                        if lab is None:
                            try:
                                os.remove(out_fp)
                            except Exception:
                                pass
                            continue
                        if np.any(np.asarray(lab) > 0):
                            # Not truly negative; remove and retry
                            try:
                                os.remove(out_fp)
                            except Exception:
                                pass
                            continue

                        chip_split[int(chip_id)] = split_name
                        chip_id += 1
                        n_done += 1

                        # Update occupied set (keeps all chips disjoint)
                        occupied_ea.append(nb)
                        occupied_union = unary_union(occupied_ea)

        # Write aa_frames_list.json for chips (inherit split -> index lists)
        train_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "train"]
        val_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "val"]
        test_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "test"]
        _write_split_json(os.path.join(chips_dir, "aa_frames_list.json"), train_idx, val_idx, test_idx)

    print(
        f"Preprocessing completed in "
        f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n"
    )


# Global config holder
config = None
