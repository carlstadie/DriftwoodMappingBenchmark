import os
import glob
import torch
import time
import numpy as np
import rasterio
import tempfile
import pandas as pd
import shutil
import geosam
from geosam import ImageEncoder
from tqdm import tqdm
from multiprocessing import Process, current_process

# Config
checkpoint_path = '/isipd/projects/p_planetdw/data/methods_test/training_images/SAM_checkpoints/sam_vit_l_0b3195.pth'
image_folder = '/isipd/projects/p_planetdw/data/methods_test/training_images/MACS'
feature_dir = '/isipd/projects/p_planetdw/data/methods_test/training_images/MACS/encodings'


GPUS = [0]

def float32_to_uint16(image):
    """Convert float32 image to uint16 by shifting by 10000 """
    image = image* 10000
    image = np.clip(image, 0, 65535)  # Clip to valid range for uint16
    image = image.astype(np.uint16)  # Convert to uint16

    return image

def encode_images_on_gpu(gpu_id, image_paths):
    torch.cuda.set_device(gpu_id)
    print(f"[{current_process().name}] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")

    # Tell ImageEncoder which GPU to use
    img_encoder = ImageEncoder(checkpoint_path, gpu_id=gpu_id)

    for image_path in tqdm(image_paths, desc=f"GPU {gpu_id}"):
        try:
            start = time.time()
            img_encoder.encode_image(str(image_path), feature_dir)
            print(f"[GPU {gpu_id}] Encoded {os.path.basename(image_path)} in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] Failed on {image_path}: {e}")


def chunk_list(lst, n):
    """Split list `lst` into `n` roughly equal chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

if __name__ == "__main__":
    # Get all .tif image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")))
    print(f"[INFO] Found {len(image_paths)} images in {image_folder}")
    print(f"[INFO] Available GPUs: {len(GPUS)}")

    temp_image_folder = tempfile.mkdtemp(prefix="converted_")

    # check if first image in image path is float32
    with rasterio.open(image_paths[0]) as src:
        dtype = src.dtypes[0]

    if len(GPUS) == 0:
        raise RuntimeError("No CUDA GPUs available.")

    # Divide images among GPUs
    chunks = chunk_list(image_paths, len(GPUS))

    # Launch one process per GPU
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = Process(target=encode_images_on_gpu, args=(gpu_id, chunk))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    #find all csv files in feature_dir and subdirs
    csv_files = glob.glob(os.path.join(feature_dir, "**", "*.csv"), recursive=True)
    for csv_file in csv_files:
        #open file as pd.df
        df = pd.read_csv(csv_file)
        #check if colum res is not a string
        if df['res'].dtype == 'object':
            #will be something like '(3,3)', write only the first number as float
            df['res'] = df['res'].apply(lambda x: float(x.split(',')[0].replace('(', '')))
            #save the df back to csv
            df.to_csv(csv_file, index=False)

    print("[INFO] All encoding complete.")
