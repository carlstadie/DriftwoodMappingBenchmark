import rasterio
import numpy as np
import os
import tqdm

img_dir = r'/isipd/projects/p_planetdw/data/methods_test/training_data/MACS/20250429-1208_MACS_test_utm8'

img_paths = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
appendix = '_8'

#delete all images with 4 bands
def delete_images_with_4_bands(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            img = src.read()
            num_bands = img.shape[0]
            if num_bands == 4:
                os.remove(img_path)
                print(f"Deleted {img_path}")
            else:
                print(f"Kept {img_path} with {num_bands} bands")

#delete images where the first 4 bands include exaytly the value 0
def delete_images_with_0_in_first_4_bands(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            bands = src.read(range(1, 5))
            if np.any(bands == 0):
                os.remove(img_path)
                print(f"Deleted {img_path}")

def append_test_to_img_filename(img_dir, appendix):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        new_img_path = img_path.replace('.tif', f'{appendix}.tif')
        os.rename(img_path, new_img_path)
        print(f"Renamed {img_path} to {new_img_path}")

def delete_images_with_only_0_in_last_band(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            last_band = src.read(src.count)
            if np.all(last_band == 0):
                os.remove(img_path)
                print(f"Deleted {img_path}")


if __name__ == "__main__":
    append_test_to_img_filename(img_dir, appendix)
    delete_images_with_0_in_first_4_bands(img_dir)
    delete_images_with_only_0_in_last_band(img_dir)
    