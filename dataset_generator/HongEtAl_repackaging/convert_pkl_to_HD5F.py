import os
import pickle
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random


def process_pkl_file(args):
    category, pkl_path, IMG_SIZE = args
    image_name = os.path.splitext(os.path.basename(pkl_path))[0]
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error processing file {pkl_path}: {e}")
        raise  # Propagate the exception
    # Convert (x_min, x_max, y_min, y_max) to (center_x, center_y, width, height)
    center_x = ((data["bbox_x_min"] + data["bbox_x_max"]) / 2.0) / IMG_SIZE
    center_y = ((data["bbox_y_min"] + data["bbox_y_max"]) / 2.0) / IMG_SIZE
    width = (data["bbox_x_max"] - data["bbox_x_min"]) / IMG_SIZE
    height = (data["bbox_y_max"] - data["bbox_y_min"]) / IMG_SIZE

    key = f"{category}/{image_name}"
    return key, [center_x, center_y, width, height]


def main():
    # PARAMETERS
    DATASET_DIR = "path/to/DicarloAsGenai_public"  # Root folder of the repackaged public image dataset.
    CATEGORY_MAPPING_PATH = os.path.join(DATASET_DIR, "categories.pkl")
    IMG_SIZE = 256  # Image size used for the repackaged DiCarlo images.
    THREADS = 16

    # Load category mapping (created by CLUSTER_generate_category_indexing.py)
    with open(CATEGORY_MAPPING_PATH, "rb") as f:
        cat_to_idx = pickle.load(f)

    # Collect tasks for all categories and images.
    tasks = []
    for category in sorted(os.listdir(DATASET_DIR)):
        category_path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(category_path):
            continue

        bboxes_dir = os.path.join(category_path, "bboxes")
        if not os.path.isdir(bboxes_dir):
            continue

        for filename in sorted(os.listdir(bboxes_dir)):
            if filename.endswith(".pkl"):
                pkl_path = os.path.join(bboxes_dir, filename)
                tasks.append((category, pkl_path, IMG_SIZE))

    print(f"Found {len(tasks)} pickle files to process.")

    image_keys = []
    bboxes = []

    # random shuffle so we don't read/write from the same category folder with multiple processes, which can bottleneck
    random.shuffle(tasks)

    start_time = time.time()
    # Use ThreadPoolExecutor to parallelize I/O operations.
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(process_pkl_file, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            key, bbox = future.result()
            image_keys.append(key)
            bboxes.append(bbox)
            if i % 100 == 0:  # Change 100 to whatever interval you prefer.
                print(f"Processed {i} files...", flush=True)

    elapsed = time.time() - start_time
    print(f"Processed all files in {elapsed:.2f} seconds.")

    # Convert lists to numpy arrays.
    image_keys_np = np.array(image_keys, dtype='S')
    bboxes_np = np.array(bboxes, dtype=np.float32)

    # Create additional arrays for category indices and human readable category names.
    # Each key is of the form "category/image_name". We split the key to get the category.
    category_indices = []
    category_names = []
    # image_keys_np is an array of bytes; decode each to a string.
    for key in image_keys:
        # key is already a string if you used the decoding step.
        category = key.split("/")[0]
        category_names.append(category)
        if category not in cat_to_idx:
            print(f"Error: Category '{category}' not found in the category mapping.", flush=True)
            raise KeyError(f"Category '{category}' not found in category mapping")
        else:
            category_indices.append(cat_to_idx[category])

    category_indices_np = np.array(category_indices, dtype=np.int32)
    category_names_np = np.array(category_names, dtype='S')  # stored as bytes

    # Save the consolidated data into an HDF5 file.
    output_filename = "labels.h5"
    with h5py.File(os.path.join(DATASET_DIR, output_filename), "w") as hf:
        hf.create_dataset("image_keys", data=image_keys_np)
        hf.create_dataset("bboxes", data=bboxes_np)
        hf.create_dataset("category_indices", data=category_indices_np)
        hf.create_dataset("category_names", data=category_names_np)
    print(f"Saved consolidated bounding box and category data to {output_filename}")

    # Compute mean and standard deviation over the bboxes (center_x, center_y, width, height)
    bbox_mean = np.mean(bboxes_np, axis=0)
    bbox_std = np.std(bboxes_np, axis=0)

    # Save these statistics to a text file with explicit labels.
    stats_filename = os.path.join(DATASET_DIR, "bbox_stats.txt")
    with open(stats_filename, "w") as f:
        f.write(f"center_x_mean: {bbox_mean[0]}\n")
        f.write(f"center_x_std: {bbox_std[0]}\n\n")
        f.write(f"center_y_mean: {bbox_mean[1]}\n")
        f.write(f"center_y_std: {bbox_std[1]}\n\n")
        f.write(f"width_mean: {bbox_mean[2]}\n")
        f.write(f"width_std: {bbox_std[2]}\n\n")
        f.write(f"height_mean: {bbox_mean[3]}\n")
        f.write(f"height_std: {bbox_std[3]}\n")
    print(f"Saved bounding box stats to {stats_filename}", flush=True)


if __name__ == '__main__':
    main()
