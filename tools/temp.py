import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

ds = xr.open_dataset("/isipd/projects/p_planetdw/cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_1748335467728.nc")

# Preview the variables
print(ds)

# Select a single time slice (first time step) and depth if applicable
uo = ds['uo'].isel(time=0).squeeze()
vo = ds['vo'].isel(time=0).squeeze()

lat = ds['latitude']
lon = ds['longitude']

# Create meshgrid
lon2d, lat2d = np.meshgrid(lon, lat)
uo_vals = uo.values
vo_vals = vo.values

# Flatten and filter NaNs
points = []
for i in range(lat.size):
    for j in range(lon.size):
        u = uo_vals[i, j]
        v = vo_vals[i, j]
        if not np.isnan(u) and not np.isnan(v):
            pt = Point(lon[j].item(), lat[i].item())
            points.append({
                'geometry': pt,
                'uo': u,
                'vo': v,
                'magnitude': np.sqrt(u**2 + v**2),
                'angle': np.degrees(np.arctan2(v, u))
            })

gdf = gpd.GeoDataFrame(points, crs='EPSG:4326')

gdf.to_file("/isipd/projects/p_planetdw/velocity_vectors.geojson", driver="GeoJSON")
# Or to shapefile
# gdf.to_file("velocity_vectors.shp")
