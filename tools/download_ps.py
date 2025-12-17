"""
PlanetScope downloader/compositor following the SAME logic as your Sentinel-2 GEE script:
- date window per AOI
- rank scenes by (cloud proxy -> date distance -> scene cloud)
- pick best whole scene per "tile" (Planet: strip_id) for crisp borders
- mosaic whole scenes (no per-pixel mixing)
- fill ONLY missing coverage using cloud-masked (UDM2) scenes, closest-in-time wins
- write GeoTIFF + CSV summary

Auth:
  Either run:   planet auth login
  or set env:   PL_API_KEY=xxxx  (depending on your SDK auth setup)
See Planet SDK user guide examples for search/orders usage.
"""

from __future__ import annotations

import os
import re
import math
import json
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely import wkb as _wkb

from planet import Planet, data_filter, order_request


# ==========================
# CONFIG (mirror your S2 code style)
# ==========================
AOI_FOLDER = r"/path/to/aois/study_area"          # same idea as your AOI_FOLDER
OUTPUT_ROOT = r"/path/to/output/planet_exports"  # local output (Planet doesn't export to Drive like GEE)

DATE_WINDOW_DAYS = 90
CATALOG_CLOUD_MAX = 0.80          # Data API "cloud_cover" is 0..1 for many PSScene records (fallback to 1..100 handled)
PER_STRIP_MAX_AOI_CLOUD = 0.30    # like your PER_TILE_MAX_AOI_CLOUD, but using clear_percent proxy (see below)
MAX_CANDIDATES = 60               # cap catalog hits per AOI to avoid huge orders

# Planet product bundle (keep a consistent band-count across scenes)
# 4-band SR+UDM2: analytic_sr_udm2  (B,G,R,NIR)  -- recommended stable default
# 8-band SR+UDM2: analytic_8b_sr_udm2            -- only use if you KNOW you have 8-band consistently
PRODUCT_BUNDLE = "analytic_sr_udm2"
FALLBACK_BUNDLE = ""  # e.g. "analytic_sr_udm2" or leave empty; avoid mixing 4b/8b in one mosaic

# Output grid (to mimic your S2 10m export, keep this at 10 to avoid massive 3m rasters)
OUT_RES_M = 10.0

# PlanetScope SR scale: reflectance DN scaled by 10,000 -> multiply by 0.0001
SR_SCALE = 10000.0

# UDM2 masking behavior for the *gap fill* scenes
UDM2_CLEAR_BAND = 1
UDM2_CONF_BAND = 7
UDM2_UNUSABLE_BAND = 8
UDM2_CONF_MIN = 50  # 0..100; raise to be stricter

SUMMARY_CSV = "PSScene_planet_export_summary.csv"


# ==========================
# Helpers (AOIs & UTM) — same spirit as your script
# ==========================
def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _safe_str(s, fallback="unknown"):
    s = str(s) if s is not None else ""
    s = s.strip()
    return s if s else fallback


def _sanitize_for_fs(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def create_aois(input_folder: str, target_crs="EPSG:4326") -> gpd.GeoDataFrame:
    """
    Same logic as your create_aois():
      filename like: region_target_YYYYMMDD_resolution_tile.(geojson/gpkg)
      adds centroid+utm_epsg and a per-target running aoi_idx
    """
    input_folder = Path(input_folder)
    files = sorted([f for pat in ("*.geojson", "*.gpkg") for f in input_folder.glob(pat)])
    if not files:
        raise FileNotFoundError(f"No .geojson or .gpkg files found in {input_folder}")

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
        gdf["filename"] = file.name
        gdf["region"] = parts[0] if len(parts) > 0 else None
        gdf["target"] = parts[1] if len(parts) > 1 else None
        gdf["date_raw"] = parts[2] if len(parts) > 2 else None
        gdf["resolution"] = parts[3] if len(parts) > 3 else None
        gdf["tile"] = parts[4] if len(parts) > 4 else None
        gdfs.append(gdf)

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)
    merged["date"] = pd.to_datetime(
        merged["date_raw"].where(merged["date_raw"].astype(str).str.fullmatch(r"\d{8}")),
        format="%Y%m%d",
        errors="coerce",
    )

    cols = ["filename", "region", "target", "date", "resolution", "tile", "geometry"]
    merged = merged[[c for c in cols if c in merged.columns]].copy()
    merged["target"] = merged["target"].apply(lambda x: _safe_str(x, "unknown"))

    tmp_proj = merged.to_crs("EPSG:3857")
    cent = tmp_proj.geometry.centroid.to_crs("EPSG:4326")
    merged["centroid_lon"] = cent.x
    merged["centroid_lat"] = cent.y
    merged["utm_epsg"] = merged.apply(
        lambda r: utm_epsg_from_lonlat(r["centroid_lon"], r["centroid_lat"]), axis=1
    )

    merged = merged.sort_values(["target"]).reset_index(drop=True)
    merged["aoi_idx"] = merged.groupby("target").cumcount() + 1
    return merged


def date_window(aoi_date, days=DATE_WINDOW_DAYS) -> Tuple[str, str]:
    if pd.isna(aoi_date):
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=365)
    else:
        start = (aoi_date - pd.Timedelta(days=days))
        end = (aoi_date + pd.Timedelta(days=days))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ==========================
# Robust Shapely -> Planet clip geometry (similar to your EE-safe geom)
# Clip tool has constraints (no holes, <1500 vertices, etc.), so we simplify.
# ==========================
_MAX_VERTICES_PLANET = 1500  # clip tool guidance
_SIMPLIFY_M_START = 5.0

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

def _strip_holes(poly: Polygon) -> Polygon:
    return Polygon(poly.exterior)

def _vertex_count(geom) -> int:
    n = 0
    for p in _poly_parts(geom):
        n += len(p.exterior.coords)
    return n

def to_planet_clip_geom_safe(geom, max_vertices=_MAX_VERTICES_PLANET) -> Dict[str, Any]:
    """
    Returns GeoJSON Polygon/MultiPolygon:
      - 2D
      - valid
      - no interior rings (holes removed)
      - simplified to <= max_vertices (approx) in EPSG:3857 meters
    If it still fails, falls back to bbox Polygon.
    """
    try:
        g = make_valid(geom).buffer(0)
        g = _drop_z(g)
        parts = _poly_parts(g)
        if not parts:
            raise ValueError("AOI has no polygonal area after cleaning.")

        parts = [_strip_holes(p) for p in parts]
        g2 = MultiPolygon(parts) if len(parts) > 1 else parts[0]

        # Simplify in meters until vertex count is under threshold
        tol = _SIMPLIFY_M_START
        while _vertex_count(g2) > max_vertices:
            gdf = gpd.GeoSeries([g2], crs="EPSG:4326").to_crs("EPSG:3857")
            g_s = gdf.iloc[0].simplify(tol, preserve_topology=True)
            g_s = make_valid(g_s).buffer(0)
            parts_s = _poly_parts(g_s)
            if not parts_s:
                raise ValueError("AOI lost area after simplification.")
            parts_s = [_strip_holes(p) for p in parts_s]
            g2 = MultiPolygon(parts_s) if len(parts_s) > 1 else parts_s[0]
            g2 = gpd.GeoSeries([g2], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
            tol *= 1.5
            if tol > 200:
                break

        return json.loads(json.dumps(mapping(g2)))
    except Exception as e:
        print(f"[to_planet_clip_geom_safe] failed: {e} -> using bbox")
        minx, miny, maxx, maxy = geom.bounds
        bbox = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])
        return json.loads(json.dumps(mapping(bbox)))


# ==========================
# Planet ranking logic (analogous to your add_rank_properties + best-per-tile)
# ==========================
def _parse_acquired(item: Dict[str, Any]) -> Optional[pd.Timestamp]:
    s = (item.get("properties") or {}).get("acquired")
    if not s:
        return None
    return pd.to_datetime(s, utc=True, errors="coerce")

def _get_strip_id(item: Dict[str, Any]) -> str:
    props = item.get("properties") or {}
    return _safe_str(props.get("strip_id") or props.get("satellite_id") or item.get("id"), "unknown")

def _scene_cloud_pct(item: Dict[str, Any]) -> float:
    props = item.get("properties") or {}
    # commonly 0..1: cloud_cover; sometimes 0..100: cloud_percent
    if "cloud_cover" in props and props["cloud_cover"] is not None:
        v = float(props["cloud_cover"])
        return v * 100.0 if v <= 1.0 else v
    if "cloud_percent" in props and props["cloud_percent"] is not None:
        return float(props["cloud_percent"])
    return 100.0

def _aoi_cloud_frac_proxy(item: Dict[str, Any]) -> float:
    """
    Planet catalog often has clear_percent per scene (not AOI-specific).
    We use (1 - clear_percent/100) as a proxy for your AOI cloud fraction.
    If missing, default to 1.0 (worst), like your null-safe fill behavior.
    """
    props = item.get("properties") or {}
    if props.get("clear_percent") is None:
        return 1.0
    cp = float(props["clear_percent"])
    cp = max(0.0, min(100.0, cp))
    return 1.0 - (cp / 100.0)

def _date_delta_days(item: Dict[str, Any], target_date: Optional[pd.Timestamp]) -> float:
    if target_date is None:
        return 9999.0
    t = _parse_acquired(item)
    if t is None or pd.isna(t):
        return 9999.0
    return float(abs((t - target_date).total_seconds()) / 86400.0)

def add_rank_key(items: List[Dict[str, Any]], aoi_date: Optional[pd.Timestamp]) -> List[Dict[str, Any]]:
    out = []
    target_date = pd.to_datetime(aoi_date, utc=True) if aoi_date is not None else None
    for it in items:
        ac = _aoi_cloud_frac_proxy(it)              # ~ your aoi_cloud_filled
        dd = _date_delta_days(it, target_date)      # ~ your date_delta_days_filled
        sc = _scene_cloud_pct(it)                   # ~ your scene_cloud_filled
        rank_key = ac * 1e6 + dd * 1e3 + sc         # same structure as your script
        it2 = dict(it)
        it2["_rank"] = {
            "aoi_cloud_frac_proxy": ac,
            "date_delta_days": dd,
            "scene_cloud_pct": sc,
            "rank_key": rank_key,
            "strip_id": _get_strip_id(it),
            "acquired": str((_parse_acquired(it) or "")),
        }
        out.append(it2)
    return out

def select_best_per_strip(items_ranked: List[Dict[str, Any]],
                          per_strip_max_cloud: Optional[float]) -> List[Dict[str, Any]]:
    """
    Analog to your select_best_per_tile(): one best scene per strip_id by min rank_key.
    Optionally drop strips whose best aoi_cloud_frac_proxy > threshold.
    """
    by_strip: Dict[str, List[Dict[str, Any]]] = {}
    for it in items_ranked:
        sid = it["_rank"]["strip_id"]
        by_strip.setdefault(sid, []).append(it)

    best = []
    for sid, lst in by_strip.items():
        lst2 = sorted(lst, key=lambda x: x["_rank"]["rank_key"])
        b = lst2[0]
        if per_strip_max_cloud is not None:
            if float(b["_rank"]["aoi_cloud_frac_proxy"]) > float(per_strip_max_cloud):
                continue
        best.append(b)
    # deterministic ordering (optional)
    best = sorted(best, key=lambda x: x["_rank"]["rank_key"])
    return best


# ==========================
# Planet search + order + download
# ==========================
def planet_search_psscene(pl: Planet,
                          aoi_geom_geojson: Dict[str, Any],
                          start_date: str,
                          end_date: str) -> List[Dict[str, Any]]:
    """
    Data API search using SDK:
      permission_filter + geometry_filter + date_range_filter + range_filter(cloud_cover)
    """
    # Catalog uses datetimes; make end exclusive-ish by adding 1 day
    start_dt = pd.to_datetime(start_date).to_pydatetime()
    end_dt = pd.to_datetime(end_date).to_pydatetime()

    # try cloud_cover as 0..1 (range_filter expects the raw field scale)
    sfilter = data_filter.and_filter([
        data_filter.permission_filter(),
        data_filter.geometry_filter(aoi_geom_geojson),
        data_filter.date_range_filter("acquired", gte=start_dt, lt=end_dt),
        data_filter.range_filter("cloud_cover", lt=CATALOG_CLOUD_MAX),
    ])

    items = []
    for it in pl.data.search(["PSScene"], search_filter=sfilter, limit=MAX_CANDIDATES):
        items.append(it)
    return items

@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def build_planet_order(item_ids: List[str],
                       aoi_clip_geom: Dict[str, Any],
                       utm_epsg: str,
                       name: str) -> Dict[str, Any]:
    """
    Orders API request:
      - product bundle: PRODUCT_BUNDLE (e.g. analytic_sr_udm2)
      - tools: clip + reproject + file_format(COG)
    """
    tools = []

    # clip tool (geometry constraints apply)
    tools.append(order_request.clip_tool(aoi=aoi_clip_geom))

    # reproject tool (force same grid across scenes)
    tools.append(order_request.reproject_tool(projection=utm_epsg, resolution=OUT_RES_M))

    # make per-scene outputs COGs (final mosaic we write as GeoTIFF)
    tools.append(order_request.file_format_tool(file_format="COG"))

    prod_kwargs = dict(
        item_ids=item_ids,
        item_type="PSScene",
        product_bundle=PRODUCT_BUNDLE,
    )
    if FALLBACK_BUNDLE:
        prod_kwargs["fallback_bundle"] = FALLBACK_BUNDLE

    request = order_request.build_request(
        name=name,
        products=[order_request.product(**prod_kwargs)],
        tools=tools
    )
    return request

def download_order_to(pl: Planet, order: Dict[str, Any], out_dir: Path) -> Path:
    """
    Wait + download. SDK usually downloads into a folder under CWD; we chdir into out_dir.
    Returns the directory that contains downloaded files.
    """
    order_id = order["id"]
    pl.orders.wait(order_id)
    with chdir(out_dir):
        pl.orders.download_order(order_id, overwrite=True)
    # assets typically land under out_dir/order_id/...
    return out_dir

def discover_downloaded_assets(root_dir: Path, item_ids: List[str]) -> Dict[str, Dict[str, Path]]:
    """
    Find SR GeoTIFF + UDM2 GeoTIFF per item_id by filename heuristics.
    """
    all_tifs = list(root_dir.rglob("*.tif")) + list(root_dir.rglob("*.tiff"))
    out: Dict[str, Dict[str, Path]] = {iid: {} for iid in item_ids}

    for p in all_tifs:
        name = p.name.lower()
        for iid in item_ids:
            if iid in p.name:
                if "udm2" in name:
                    out[iid]["udm2"] = p
                else:
                    # prefer SR / analytic in name; otherwise first non-udm2 tif
                    if "sr" in name or "analytic" in name:
                        out[iid]["sr"] = p
                    else:
                        out[iid].setdefault("sr", p)

    # drop incomplete
    out2 = {iid: d for iid, d in out.items() if ("sr" in d and "udm2" in d)}
    missing = [iid for iid in item_ids if iid not in out2]
    if missing:
        print(f"[WARN] Missing SR+UDM2 files for {len(missing)} item(s): {missing[:5]}{'...' if len(missing)>5 else ''}")
    return out2


# ==========================
# Raster compositing (whole-scene mosaic + gap fill with UDM2)
# ==========================
def _read_sr_as_float(src: rasterio.DatasetReader) -> np.ndarray:
    arr = src.read().astype(np.float32)  # (bands, rows, cols)
    arr *= (1.0 / SR_SCALE)
    return arr

def _udm2_clear_mask(udm2_ds: rasterio.DatasetReader) -> np.ndarray:
    clear = udm2_ds.read(UDM2_CLEAR_BAND).astype(np.uint8)
    conf = udm2_ds.read(UDM2_CONF_BAND).astype(np.float32)
    unus = udm2_ds.read(UDM2_UNUSABLE_BAND).astype(np.uint16)
    mask = (clear == 1) & (conf >= float(UDM2_CONF_MIN)) & (unus == 0)
    return mask  # (rows, cols) bool

def _ensure_match_grid(src_ds: rasterio.DatasetReader,
                       template_ds: rasterio.DatasetReader,
                       data: np.ndarray) -> np.ndarray:
    """
    If transforms/shapes differ, reproject each band onto template grid.
    data: (bands, rows, cols) for src_ds
    """
    if (src_ds.transform == template_ds.transform and
        src_ds.crs == template_ds.crs and
        src_ds.width == template_ds.width and
        src_ds.height == template_ds.height):
        return data

    out = np.zeros((data.shape[0], template_ds.height, template_ds.width), dtype=np.float32)
    for b in range(data.shape[0]):
        reproject(
            source=data[b],
            destination=out[b],
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            resampling=Resampling.bilinear,
        )
    return out

def mosaic_whole_scenes_last_wins(sr_paths: List[Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Whole-scene mosaic like your best_col.mosaic(): later scenes overwrite earlier ones.
    Uses each scene's internal nodata mask as coverage indicator (treated as 0 after scaling).
    """
    if not sr_paths:
        raise ValueError("No SR rasters to mosaic.")

    with rasterio.open(sr_paths[0]) as tmpl:
        profile = tmpl.profile.copy()
        base = _read_sr_as_float(tmpl)
        base_valid = np.any(base != 0.0, axis=0)

        for p in sr_paths[1:]:
            with rasterio.open(p) as ds:
                arr = _read_sr_as_float(ds)
                arr = _ensure_match_grid(ds, tmpl, arr)
                valid = np.any(arr != 0.0, axis=0)
                # last wins where valid
                base[:, valid] = arr[:, valid]
                base_valid = base_valid | valid

        profile.update(
            dtype="float32",
            count=base.shape[0],
            nodata=0.0,
            compress="deflate",
            tiled=True,
            bigtiff="if_safer",
        )
        return base, profile

def gap_fill_inplace(base: np.ndarray,
                     base_profile: Dict[str, Any],
                     fill_pairs: List[Tuple[Path, Path]]):
    """
    Mimics your build_gap_fill_image()+unmask():
      - identify gaps where base has no data
      - iterate fill scenes sorted so "closest-in-time wins" (we apply last wins by ordering)
      - only write into gaps
      - for fill scenes: mask by UDM2 (clear/conf/unusable)
    """
    gap = ~np.any(base != 0.0, axis=0)  # gaps where all bands are zero
    if not gap.any():
        return

    # Template for grid matching
    with rasterio.open(fill_pairs[0][0]) as tmpl:
        for sr_path, udm2_path in fill_pairs:
            if not gap.any():
                break

            with rasterio.open(sr_path) as sr_ds, rasterio.open(udm2_path) as udm2_ds:
                sr = _read_sr_as_float(sr_ds)
                sr = _ensure_match_grid(sr_ds, tmpl, sr)

                clear = _udm2_clear_mask(udm2_ds)
                if (udm2_ds.transform != tmpl.transform or udm2_ds.width != tmpl.width or udm2_ds.height != tmpl.height):
                    # reproject mask onto template if needed (nearest)
                    clear2 = np.zeros((tmpl.height, tmpl.width), dtype=np.uint8)
                    reproject(
                        source=clear.astype(np.uint8),
                        destination=clear2,
                        src_transform=udm2_ds.transform,
                        src_crs=udm2_ds.crs,
                        dst_transform=tmpl.transform,
                        dst_crs=tmpl.crs,
                        resampling=Resampling.nearest,
                    )
                    clear = clear2.astype(bool)

                valid = np.any(sr != 0.0, axis=0)
                fill_ok = clear & valid & gap
                if fill_ok.any():
                    base[:, fill_ok] = sr[:, fill_ok]
                    gap[fill_ok] = False


def write_geotiff(path: Path, data: np.ndarray, profile: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.update(driver="GTiff")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(np.float32))


# ==========================
# Main
# ==========================
def main():
    pl = Planet()  # assumes planet auth login or configured credentials

    aois = create_aois(AOI_FOLDER)
    print(f"[AOIs] {len(aois)} individual AOI(s) (no dissolve)")

    out_root = Path(OUTPUT_ROOT)
    all_rows = []

    for _, row in aois.iterrows():
        target = _safe_str(row.get("target"), "unknown")
        target_safe = _sanitize_for_fs(target)
        aoi_idx = int(row.get("aoi_idx") or 1)

        geom = row.geometry
        utm = row.get("utm_epsg") or "EPSG:32633"
        aoi_date = row.get("date") if pd.notna(row.get("date")) else None
        start, end = date_window(row.get("date"), DATE_WINDOW_DAYS)

        # Planet clip geometry (simplified/validated)
        aoi_clip = to_planet_clip_geom_safe(geom)

        print(f"\n=== {target} [#{aoi_idx}] | window={start}..{end} | {utm} ===")

        # 1) Search catalog
        items = planet_search_psscene(pl, aoi_clip, start, end)
        if not items:
            print("[Search] No PSScene items found.")
            continue

        # 2) Rank + best-per-strip
        items_ranked = add_rank_key(items, aoi_date)
        best = select_best_per_strip(items_ranked, PER_STRIP_MAX_AOI_CLOUD)

        if not best:
            print("[Select] No strips passed PER_STRIP_MAX_AOI_CLOUD; try relaxing threshold.")
            continue

        # Build fill list: remaining items sorted so "closest-in-time wins" LAST,
        # matching your `.sort(date_delta_days_filled, False)` then mosaic :contentReference[oaicite:9]{index=9}
        remaining = [it for it in items_ranked if it["id"] not in {b["id"] for b in best}]
        remaining_sorted = sorted(remaining, key=lambda x: x["_rank"]["date_delta_days"], reverse=True)

        # We download best + a limited number of fill candidates
        fill_keep = remaining_sorted[: min(len(remaining_sorted), MAX_CANDIDATES)]
        download_list = best + fill_keep
        item_ids = [it["id"] for it in download_list]

        # 3) Order (clip + reproject + COG)
        date_tag = (row.get("date").strftime("%Y%m%d") if pd.notna(row.get("date")) else "nodate")
        order_name = f"{target_safe}_{aoi_idx:03d}_PSScene_{date_tag}_{utm}_res{int(OUT_RES_M)}m"
        request = build_planet_order(item_ids, aoi_clip, utm, name=order_name)

        aoi_out_dir = out_root / f"{target_safe}_{aoi_idx:03d}"
        print(f"[Order] Creating order with {len(item_ids)} item(s): {order_name}")
        order = pl.orders.create_order(request)

        print(f"[Order] Waiting + downloading (quota usage may apply) ...")
        download_order_to(pl, order, aoi_out_dir)

        # 4) Discover downloaded SR+UDM2 files
        assets = discover_downloaded_assets(aoi_out_dir, item_ids)
        if not assets:
            print("[Download] Could not find any SR+UDM2 pairs after download.")
            continue

        # 5) Base mosaic: best whole scenes only
        best_ids = [it["id"] for it in best if it["id"] in assets]
        best_sr_paths = [assets[iid]["sr"] for iid in best_ids]
        base, profile = mosaic_whole_scenes_last_wins(best_sr_paths)

        # 6) Gap fill: only where base has no coverage, using UDM2 clear mask
        fill_pairs = []
        for it in remaining_sorted:
            iid = it["id"]
            if iid in assets:
                fill_pairs.append((assets[iid]["sr"], assets[iid]["udm2"]))
        if fill_pairs:
            gap_fill_inplace(base, profile, fill_pairs)

        # 7) Write output
        out_tif = aoi_out_dir / f"{order_name}_mosaic.tif"
        write_geotiff(out_tif, base, profile)
        print(f"[Write] {out_tif}")

        # 8) Summary rows (like your selected-image summary)
        for it in best:
            r = it["_rank"]
            all_rows.append({
                "AOI": target,
                "AOI_Index": aoi_idx,
                "UTM": utm,
                "WindowStart": start,
                "WindowEnd": end,
                "ItemID": it["id"],
                "Acquired": r["acquired"],
                "StripID": r["strip_id"],
                "SceneCloudPct": r["scene_cloud_pct"],
                "AOICloudFracProxy": r["aoi_cloud_frac_proxy"],
                "DateDeltaDays": r["date_delta_days"],
                "RankKey": r["rank_key"],
                "Output": str(out_tif),
            })

    # Write summary CSV
    df = pd.DataFrame(all_rows)
    out_csv = Path(OUTPUT_ROOT) / SUMMARY_CSV
    df.to_csv(out_csv, index=False)
    print(f"\n[Summary] Wrote: {out_csv}")
    if not df.empty:
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
