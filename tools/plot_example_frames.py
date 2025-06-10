import rasterio
import rasterio.windows
import rasterio.features
import matplotlib.pyplot as plt
import os
from shapely.geometry import box
import geopandas as gpd
import numpy as np
from PIL import Image

image_dir = '/isipd/projects/p_planetdw/data/dw_detection/examplevis'
prediction_path = '/isipd/projects/p_planetdw/data/dw_detection/examplevis/dw_larger100m2_connected.gpkg'
center_path = '/isipd/projects/p_planetdw/data/dw_detection/examplevis/centroids.gpkg'
output_dir = '/isipd/projects/p_planetdw/data/dw_detection/examplevis/example_frames'

def percentile_stretch(img, lower=2, upper=98):
    stretched = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        band = img[i]
        p2 = np.percentile(band, lower)
        p98 = np.percentile(band, upper)
        if p98 > p2:
            stretched[i] = np.clip(255 * (band - p2) / (p98 - p2), 0, 255).astype(np.uint8)
        else:
            stretched[i] = np.zeros_like(band, dtype=np.uint8)
    return stretched

def plot_example_frames(image_dir, prediction_path, center_path, output_dir, num_frames=20, extent=1000):
    centers = gpd.read_file(center_path)
    predictions = gpd.read_file(prediction_path)

    if centers.empty:
        print("No centers found.")
        return
    if predictions.empty:
        print("No predictions found.")
        return

    frames = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jp2')]

    if not image_files:
        print("No .jp2 images found.")
        return

    for idx, center in enumerate(centers.geometry):
        print(f"\nProcessing center {idx + 1}/{len(centers)}")

        for image_name in image_files:
            image_path = os.path.join(image_dir, image_name)
            with rasterio.open(image_path) as src:
                if centers.crs != src.crs:
                    center_transformed = gpd.GeoSeries([center], crs=centers.crs).to_crs(src.crs).iloc[0]
                else:
                    center_transformed = center

                cx, cy = center_transformed.x, center_transformed.y
                bbox = box(cx - extent, cy - extent, cx + extent, cy + extent)
                img_bounds = box(*src.bounds)
                if not bbox.intersects(img_bounds):
                    print("No intersection with image.")
                    continue

                print(f"Reading image: {image_name}")

                row_start, col_start = src.index(bbox.bounds[0], bbox.bounds[3])
                row_stop, col_stop = src.index(bbox.bounds[2], bbox.bounds[1])

                row_start, row_stop = max(0, row_start), min(src.height, row_stop)
                col_start, col_stop = max(0, col_start), min(src.width, col_stop)

                window = rasterio.windows.Window.from_slices((row_start, row_stop), (col_start, col_stop))
                img = src.read(window=window)

                pred = predictions.to_crs(src.crs)
                pred_crop = pred[pred.geometry.intersects(bbox)]
                if pred_crop.empty:
                    print("No prediction overlap.")
                    continue

                height, width = img.shape[1:]
                pred_raster = np.zeros((height, width), dtype=np.uint8)
                transform = src.window_transform(window)

                for geom in pred_crop.geometry:
                    if geom.is_empty:
                        continue
                    if geom.geom_type == 'Polygon':
                        geom_iter = [geom]
                    elif geom.geom_type == 'MultiPolygon':
                        geom_iter = list(geom.geoms)
                    else:
                        continue

                    for g in geom_iter:
                        try:
                            mask = rasterio.features.rasterize(
                                [(g, 1)],
                                out_shape=(height, width),
                                transform=transform,
                                fill=0,
                                dtype=np.uint8
                            )
                            print(f"Rasterized geometry with {mask.sum()} pixels")
                            pred_raster = np.maximum(pred_raster, mask)
                        except Exception as e:
                            print(f"Skipping geometry due to error: {e}")

                if pred_raster.sum() == 0:
                    print("Empty prediction mask after rasterization.")
                    continue

                # Optional debug image for rasterized mask
                debug_path = os.path.join(output_dir, f'debug_pred_mask_{idx}.png')
                Image.fromarray((pred_raster * 255).astype(np.uint8)).save(debug_path)
                print(f"Saved debug prediction mask to {debug_path}")

                img = np.concatenate((img, pred_raster[np.newaxis, :, :]), axis=0)
                frames.append(img)

                print("Frame added.")
                if len(frames) >= num_frames:
                    break
        if len(frames) >= num_frames:
            break

    if not frames:
        print("No frames were generated.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        print(f"Saving frame {i} with shape {frame.shape}")

        rgb_stretched = percentile_stretch(frame[[2, 1, 0]])  # Assuming PlanetScope order: R=2, G=1, B=0
        rgb = np.moveaxis(rgb_stretched, 0, -1)

        pred_mask = frame[-1]
        yellow_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
        yellow_overlay[..., 0] = 255
        yellow_overlay[..., 1] = 51
        yellow_overlay[..., 2] = 255
        yellow_overlay[..., 3] = np.where(pred_mask > 0, 255, 0).astype(np.uint8)

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.imshow(yellow_overlay, interpolation='none')
        #plt.title(f'Frame {i}')
        plt.axis('off')
        out_path = os.path.join(output_dir, f'frame_{i}.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved to {out_path}")

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    plot_example_frames(image_dir, prediction_path, center_path, output_dir)
    print(f"Example frames (if any) saved to {output_dir}")
