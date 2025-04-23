import os
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Pool

def sample_raster_at_points(args):
    raster_dir, group = args
    group = group.copy()
    raster_files = [f for f in os.listdir(raster_dir) if f.endswith('.tif')]

    for raster in raster_files:
        raster_path = os.path.join(raster_dir, raster)
        with rasterio.open(raster_path) as src:
            src_bounds = box(*src.bounds)

            if src.crs != group.crs:
                continue

            points_in_image = group[group.intersects(src_bounds)]

            if points_in_image.empty:
                continue

            sampled_values = []
            with rasterio.open(raster_path) as src:
                for idx, point in points_in_image.iterrows():
                    row, col = src.index(point.geometry.x, point.geometry.y)
                    value = src.read(1)[row, col]
                    sampled_values.append((idx, value))

            for idx, val in sampled_values:
                group.at[idx, raster] = val

    return group


# File paths
point_file = '/isipd/projects/p_planetdw/data/methods_test/training_images/random_points.gpkg'
target_areas = '/isipd/projects/p_planetdw/data/auxilliary/footprints_macs/footprints_macs_for_analysis.gpkg'
raster_dir = '/isipd/projects/p_planetdw/data/methods_test/training_images/MACS'
output_dir = '/isipd/projects/p_planetdw/data/methods_test/'

# Read and align data
points = gpd.read_file(point_file, layer='Assigned_CRS')
areas = gpd.read_file(target_areas)

if points.crs != areas.crs:
    points = points.to_crs(areas.crs)

points = gpd.sjoin(points, areas[['target_area', 'geometry']], how='left', op='within')

# Group points by target_area
groups = [group for _, group in points.groupby('target_area')]

# Prepare arguments as tuples
args = [(raster_dir, group) for group in groups]
print(args)

# Process in parallel
with Pool(processes=4) as pool:
    results = list(tqdm(pool.imap_unordered(sample_raster_at_points, args),
                        total=len(args), desc="Processing groups"))


# Combine results
sampled_points = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))

# melt the gdf so that names of target areas are a column not coloumn heads
sampled_points = sampled_points.melt(id_vars=['geometry'], var_name='target_area', value_name='value')


tas = sampled_points['target_area'].unique()

plt.figure(figsize=(10, 6))

for ta in tas:
    sampled_points[sampled_points['target_area'] == ta]['value'].hist(
        bins=50, alpha=0.5, label=ta
    )

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "histograms.png"))

