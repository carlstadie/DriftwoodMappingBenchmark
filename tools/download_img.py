from pathlib import Path
import re
import json
import math
import hashlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely import wkb as _wkb
import ee

# ==========================
# CONFIG
# ==========================
AOI_FOLDER = r"/isipd/projects/p_planetdw/data/methods_test/aois"
DRIVE_FOLDER = "S2_GEE_Exports"                     # Google Drive folder name

DATE_WINDOW_DAYS = 45
CLOUD_THRESH_PROP = 80.0                            # scene-level CLOUDY_PIXEL_PERCENTAGE filter
PER_TILE_MAX_AOI_CLOUD = 0.30                       # 0..1; skip a tile if best image still cloudier than this. set None to disable

# whole-scene export (no per-pixel blending)
BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]                    # export order (RGB+NIR)
SCL_CLOUD_CLASSES = [3, 8, 9, 10]                   # shadow, cloud medium, cloud high, cirrus

MAX_PIXELS = 1e13
EXPORT_REGION_SIMPLIFY_M = 10                       # meters; light simplification of export region
SUMMARY_CSV = "S2_GEE_export_summary.csv"           # written next to AOI folder

print(AOI_FOLDER)

# ==========================
# Helpers (naming, AOIs & UTM)
# ==========================
def _sanitize_for_description(s: str) -> str:
    # Only allow: letters, digits, ".", ",", ":", ";", "_", "-" (EE requirement)
    return re.sub(r"[^A-Za-z0-9\.\,\:\;\_\-]+", "_", s)

def _sanitize_for_filename(s: str) -> str:
    # Be conservative for filenames too
    return re.sub(r"[^A-Za-z0-9\.\_\-]+", "_", s)

def _truncate(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[:maxlen].rstrip("._-")

def _short_id(s: str, n: int = 8) -> str:
    # Stable short id from any string (e.g., system:index)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:n].upper()

def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def create_aois(input_folder, target_crs="EPSG:4326"):
    input_folder = Path(input_folder)
    files = sorted([f for pat in ("*.geojson", "*.gpkg") for f in input_folder.glob(pat)])
    if not files:
        raise FileNotFoundError("No .geojson or .gpkg files found in AOI_FOLDER")

    gdfs = []
    for file in files:
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            if file.suffix.lower() == ".geojson":
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                print(f"WARNING: {file.name} has no CRS; skipping.")
                continue
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)

        parts = file.stem.split("_")
        gdf["filename"]   = file.name
        gdf["region"]     = parts[0] if len(parts) > 0 else None
        gdf["target"]     = parts[1] if len(parts) > 1 else None
        gdf["date_raw"]   = parts[2] if len(parts) > 2 else None
        gdf["resolution"] = parts[3] if len(parts) > 3 else None
        gdf["tile"]       = parts[4] if len(parts) > 4 else None
        gdfs.append(gdf)

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)
    merged["date"] = pd.to_datetime(
        merged["date_raw"].where(merged["date_raw"].astype(str).str.fullmatch(r"\d{8}")),
        format="%Y%m%d", errors="coerce"
    )
    cols = ["filename", "region", "target", "date", "resolution", "tile", "geometry"]
    merged = merged[[c for c in cols if c in merged.columns]].copy()

    dissolved = merged.dissolve(by="target", as_index=True).reset_index()

    # centroid for UTM guess
    tmp_proj = dissolved.to_crs("EPSG:3413")
    cent = tmp_proj.geometry.centroid.to_crs("EPSG:4326")
    dissolved["centroid_lon"] = cent.x
    dissolved["centroid_lat"] = cent.y
    dissolved["utm_epsg"] = dissolved.apply(
        lambda r: utm_epsg_from_lonlat(r["centroid_lon"], r["centroid_lat"]), axis=1
    )
    return dissolved

def date_window(aoi_date, days=DATE_WINDOW_DAYS):
    if pd.isna(aoi_date):
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=365)
    else:
        start = (aoi_date - pd.Timedelta(days=days))
        end   = (aoi_date + pd.Timedelta(days=days))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ==========================
# Robust Shapely → EE geometry
# ==========================
_MAX_COORDS = 60000
_SIMPLIFY_DEG = 1e-4  # ~11 m at equator

def _drop_z(geom):
    return _wkb.loads(_wkb.dumps(geom, output_dimension=2))

def _poly_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(_poly_parts(g))
        return parts
    return []

def _coord_count(geom):
    total = 0
    for poly in _poly_parts(geom):
        total += len(poly.exterior.coords)
        for ring in poly.interiors:
            total += len(ring.coords)
    return total

def _to_valid_simple_2d_polygon_or_multipolygon(geom):
    if geom.is_empty:
        raise ValueError("AOI is empty.")
    g = make_valid(geom).buffer(0)
    g = _drop_z(g)
    parts = _poly_parts(g)
    if not parts:
        raise ValueError("AOI has no polygonal area after cleaning.")
    g2 = MultiPolygon(parts) if len(parts) > 1 else parts[0]
    if _coord_count(g2) > _MAX_COORDS:
        g2 = make_valid(g2.simplify(_SIMPLIFY_DEG, preserve_topology=True)).buffer(0)
        parts2 = _poly_parts(g2)
        if not parts2:
            raise ValueError("AOI lost area after simplification.")
        g2 = MultiPolygon(parts2) if len(parts2) > 1 else parts2[0]
    return g2

def to_ee_geometry(geom):
    g2 = _to_valid_simple_2d_polygon_or_multipolygon(geom)
    gj = json.loads(json.dumps(mapping(g2)))
    return ee.Geometry(gj, proj="EPSG:4326", geodesic=False, evenOdd=True)

def to_ee_geometry_safe(geom):
    try:
        return to_ee_geometry(geom)
    except Exception as e:
        print(f"[to_ee_geometry_safe] primary failed: {e}  → using bounding box.")
        minx, miny, maxx, maxy = geom.bounds
        return ee.Geometry.Rectangle([minx, miny, maxx, maxy],
                                     proj="EPSG:4326", geodesic=False)

def simplify_for_export(region, meters=EXPORT_REGION_SIMPLIFY_M):
    if not meters or meters <= 0:
        return region
    return (region.transform("EPSG:3857", 1)
                  .simplify(meters)
                  .transform("EPSG:4326", 1))

# ==========================
# GEE init
# ==========================
def gee_init():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# ==========================
# AOI Cloud metric + rank key (with null-safe defaults)
# ==========================
def compute_aoi_cloud_frac(img, region):
    scl = img.select("SCL")
    is_cloud = ee.Image(0)
    for c in SCL_CLOUD_CLASSES:
        is_cloud = is_cloud.Or(scl.eq(c))
    stats = is_cloud.rename("c").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20, bestEffort=True, maxPixels=MAX_PIXELS
    )
    return ee.Number(stats.get("c"))

def _coalesce_number(val, fallback):
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val, None), fallback, val))

def add_rank_properties(col, region, aoi_date):
    target_date = ee.Date(aoi_date.strftime("%Y-%m-%d")) if aoi_date is not None else None

    def _with_props(im):
        ac = compute_aoi_cloud_frac(im, region)
        ac_filled = _coalesce_number(ac, 1.0)
        t  = ee.Date(im.get("system:time_start"))
        dd = t.difference(target_date, "day").abs() if target_date else ee.Number(9999)
        dd_filled = _coalesce_number(dd, 9999)
        scene_cloud = _coalesce_number(ee.Number(im.get("CLOUDY_PIXEL_PERCENTAGE")), 100)
        rank_key = ac_filled.multiply(1e6).add(dd_filled.multiply(1e3)).add(scene_cloud)
        tile = ee.String(ee.Algorithms.If(im.get("MGRS_TILE"), im.get("MGRS_TILE"), "TNA"))
        return (im
                .set("aoi_cloud", ac)
                .set("aoi_cloud_filled", ac_filled)
                .set("date_delta_days", dd)
                .set("date_delta_days_filled", dd_filled)
                .set("scene_cloud_filled", scene_cloud)
                .set("rank_key", rank_key)
                .set("tile_id", tile)
                .set("date_str", t.format("YYYY-MM-dd")))

    return col.map(_with_props)

# ==========================
# Build collection (NO per-pixel mixing)
# ==========================
def build_s2_collection(region, start_date, end_date, aoi_date):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESH_PROP))
           .select(BANDS + ["SCL"]))
    return add_rank_properties(col, region, aoi_date)

# ==========================
# Select whole best scene per tile (no mosaicking)
# ==========================
def select_best_per_tile(col, per_tile_max_cloud=None):
    tiles = ee.List(col.aggregate_array("tile_id")).distinct()

    def _best_for_tile(tile):
        tc = col.filter(ee.Filter.eq("tile_id", tile)).sort("rank_key")
        best = ee.Image(tc.first())
        return best.set("tile_id", tile)

    best_list = tiles.map(_best_for_tile)
    best_col = ee.ImageCollection.fromImages(best_list)
    if per_tile_max_cloud is not None:
        best_col = best_col.filter(ee.Filter.lte("aoi_cloud_filled", per_tile_max_cloud))
    return best_col

# ==========================
# Export (ENTIRE SCENE) with safe names
# ==========================
def export_scene(img, description, file_prefix):
    export_region = simplify_for_export(img.geometry(), EXPORT_REGION_SIMPLIFY_M)
    crs = img.select(BANDS[0]).projection().crs()
    task = ee.batch.Export.image.toDrive(
        image = img.select(BANDS).multiply(0.0001).toFloat(),
        description = description,          # <= 100 chars, EE-safe
        folder = DRIVE_FOLDER,
        fileNamePrefix = file_prefix,       # can be longer; also sanitized
        region = export_region,
        crs = crs,
        scale = 10,
        maxPixels = MAX_PIXELS,
        fileFormat = "GeoTIFF"
    )
    task.start()
    try:
        crs_str = crs.getInfo()
    except Exception:
        crs_str = "image CRS"
    print(f"[Export] Started: {file_prefix} → Drive/{DRIVE_FOLDER}  (CRS={crs_str}, 10 m)")
    return task

# ==========================
# Summary building (null-safe)
# ==========================
def summarize_selected_images(selected_col, region, target, utm, start_date, end_date, task_name):
    def per_image_feature(im):
        scl = im.select("SCL")
        is_cloud = ee.Image(0)
        for c in SCL_CLOUD_CLASSES:
            is_cloud = is_cloud.Or(scl.eq(c))
        keep = is_cloud.Not().And(scl.gt(0))
        clear_area = ee.Image.pixelArea().updateMask(keep).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=20, bestEffort=True, maxPixels=MAX_PIXELS
        ).get("area")
        clear_area = _coalesce_number(clear_area, 0)

        props = ee.Dictionary({
            "AOI": target,
            "UTM": utm,
            "ExportTask": task_name,
            "WindowStart": start_date,
            "WindowEnd": end_date,
            "SystemIndex": im.get("system:index"),
            "Date": im.get("date_str"),
            "SceneCloudPct": im.get("scene_cloud_filled"),
            "AOICloudFrac": _coalesce_number(im.get("aoi_cloud"), 1.0),
            "DateDeltaDays": im.get("date_delta_days_filled"),
            "MGRS": im.get("tile_id"),
            "RankKey": im.get("rank_key"),
            "ClearArea_m2": clear_area
        })
        return ee.Feature(None, props)

    fc = ee.FeatureCollection(selected_col.map(per_image_feature))
    feats = fc.getInfo()["features"]

    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({
            "AOI": p.get("AOI"),
            "UTM": p.get("UTM"),
            "ExportTask": p.get("ExportTask"),
            "WindowStart": p.get("WindowStart"),
            "WindowEnd": p.get("WindowEnd"),
            "SystemIndex": p.get("SystemIndex"),
            "Date": p.get("Date"),
            "SceneCloudPct": p.get("SceneCloudPct"),
            "AOICloudFrac": p.get("AOICloudFrac"),
            "DateDeltaDays": p.get("DateDeltaDays"),
            "MGRS": p.get("MGRS"),
            "RankKey": p.get("RankKey"),
            "ClearArea_m2": p.get("ClearArea_m2")
        })
    return rows

# ==========================
# Main
# ==========================
def main():
    gee_init()
    aois = create_aois(AOI_FOLDER)
    print(f"[AOIs] {len(aois)} dissolved AOIs")

    all_rows = []
    tasks = []

    for _, row in aois.iterrows():
        target = row.get("target") or "unknown"
        geom = row.geometry
        utm = row.get("utm_epsg") or "EPSG:32633"
        aoi_date = row.get("date") if pd.notna(row.get("date")) else None
        start, end = date_window(row.get("date"), DATE_WINDOW_DAYS)
        region = to_ee_geometry_safe(geom)

        print(f"\n=== {target} | window={start}..{end} | {utm} ===")

        col = build_s2_collection(region, start, end, aoi_date)
        best_col = select_best_per_tile(col, PER_TILE_MAX_AOI_CLOUD)

        size = best_col.size().getInfo()
        img_list = best_col.toList(size)

        for i in range(size):
            im = ee.Image(img_list.get(i))
            date_str = im.get("date_str").getInfo()
            tile_id  = im.get("tile_id").getInfo()
            sys_idx  = im.get("system:index").getInfo()

            # Short, EE-safe description (≤100 chars)
            desc_raw = f"{target}_{tile_id}_{date_str}"
            desc = _truncate(_sanitize_for_description(desc_raw), 100)

            # Filename prefix (sanitized; can be longer)
            date_tag = (row.get("date").strftime("%Y%m%d") if pd.notna(row.get("date")) else "nodate")
            sid = _short_id(sys_idx, 8)
            fname_raw = f"{target}_{tile_id}_{date_str}_{date_tag}_{sid}_S2L2A_SR_10m_fullscene"
            file_prefix = _truncate(_sanitize_for_filename(fname_raw), 180)

            task = export_scene(im, desc, file_prefix)
            tasks.append(task)

        rows = summarize_selected_images(best_col, region, target, utm, start, end, f"{target}_fullscene")
        rows = sorted(rows, key=lambda r: (r["AOI"], r["MGRS"], r["DateDeltaDays"], r["AOICloudFrac"]))
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = Path(AOI_FOLDER).parent / SUMMARY_CSV
    df.to_csv(out_csv, index=False)
    print(f"\n[Summary] Wrote selected-image summary: {out_csv}")
    if not df.empty:
        print(df.head(12).to_string(index=False))

    print(f"\nSubmitted {len(tasks)} export task(s). Open https://code.earthengine.google.com/tasks to monitor.")

if __name__ == "__main__":
    main()
