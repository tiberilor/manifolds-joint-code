# `dataset_generator`

This folder contains the minimal scripts needed to generate our image dataset and package its annotations.

We generated three such datasets:

- **train**: 9000 images per category, used to train the CNNs
- **validation**: 1000 images per category, used to validate the CNNs during training
- **test**: 10000 images per category, used to generate neural representations and evaluate linear decoders

## Sample dataset

A small random sample of the dataset (70 images per category) is provided in `dataset_sample/`. It follows the same structure as the full dataset and includes example images, bounding-box `.pkl` files, and the packaged annotation files (`categories.pkl`, `labels.h5`, `bbox_stats.txt`).

To visualize sample images, use `BROWSE_dataset.py`.

## What the final dataset looks like

The final dataset is a folder with one subfolder per category. Each category folder contains:

- an `images/` folder with `.jpg` images
- a `bboxes/` folder with one `.pkl` file per image (these .pkl files are a byproduct of the image generation process and are not strictly needed once the final dataset has been generated and packaged as explained below)

After post-processing, the dataset root also contains:

- `categories.pkl`: mapping from category name to integer index
- `labels.h5`: consolidated category and bounding-box annotations
- `bbox_stats.txt`: mean and standard deviation of the bounding-box variables

Each image has a matching `.pkl` file with the same base name. The `.pkl` file stores the bounding box in **pixel units** on a `512 x 512` image, in the format:

- `bbox_x_min`
- `bbox_x_max`
- `bbox_y_min`
- `bbox_y_max`

The `labels.h5` file stores the bounding boxes in **normalized units** (divided by 512), in the format:

- `center_x`
- `center_y`
- `width`
- `height`

It also stores the category name and category index for each image.

## Procedure

### Step 1: generate images
Run the image-generation script.

This produces, for each generated image:

- a `.jpg` image file
- a matching `.pkl` file with the bounding box in pixel units (`x_min`, `x_max`, `y_min`, `y_max`)

### Step 2: generate `categories.pkl`
Run the category-index script.

This creates `categories.pkl`, which maps each category folder name to an integer index.

### Step 3: generate `labels.h5`
Run the annotation-packaging script.

This reads all `.pkl` bounding-box files, converts them to normalized `(center_x, center_y, width, height)` format, and writes:

- `labels.h5`
- `bbox_stats.txt`

## Scripts

### 1. Image-generation script
This is the main dataset-generation script.

**Purpose.**  
Generate dataset images together with per-image bounding-box annotations.

**How inputs are provided.**  
The main inputs are specified as editable variables inside the script, not through `argparse`. In particular, the user must set:
- the path to `Objects365_categories.pkl`
- the path to the CerberusDet code
- the path to the CerberusDet weights
- the target dataset output folder

The script also takes a `job_id` argument from the command line. This is used only to make output filenames unique across parallel jobs.

**What it does.**  
It processes all categories listed in `Objects365_categories.pkl`. For each category, it creates the category folder if it does not already exist, together with its `images/` and `bboxes/` subfolders. It then keeps generating data for that category until the target number of images is reached. The target number is determined by the generation parameters set inside the script.

The script is resumable: if a category already contains some generated images, it counts how many are already present and generates only the missing number.

**Input.**  
- `Objects365_categories.pkl`, which specifies the categories to generate and the accepted detector labels for each category
- the required generation and detection models
- the target dataset output folder

**Output.**  
For each accepted sample, the script writes:
- a `.jpg` image file in `images/`
- a matching `.pkl` file in `bboxes/`

The `.pkl` file stores the bounding box in pixel units on a `512 x 512` image, in the format:
- `bbox_x_min`
- `bbox_x_max`
- `bbox_y_min`
- `bbox_y_max`

Image filenames are unique and include the category name, the job id, and an image index.

In practice, we generated the full dataset by launching 10 parallel jobs on A100 GPUs. Because categories were processed in randomized order, overlap across jobs was usually limited. When some categories ended up with more images than needed, we removed the excess afterwards.

### 2. `GENERATE_category_indexing.py`
This script creates `categories.pkl`.

**Purpose.**  
Define a fixed mapping from category names to integer indices.

**How inputs are provided.**  
The dataset path is specified as an editable variable inside the script, not through `argparse`.

**Input.**  
A dataset directory containing one subfolder per category.

**Output.**  
A file `categories.pkl` stored in the dataset root. It maps each category folder name to an integer index. Categories are sorted alphabetically before indices are assigned.

### 3. `CONVERT_pkl_to_HD5F.py`
This script creates `labels.h5` and `bbox_stats.txt`.

**Purpose.**  
Convert the per-image `.pkl` bounding-box annotations into a single consolidated annotation file for the dataset.

**How inputs are provided.**  
The dataset path and the path to `categories.pkl` are specified as editable variables inside the script, not through `argparse`.

**Input.**  
- the dataset directory containing category folders and per-image `.pkl` files
- `categories.pkl`

**Output.**  
- `labels.h5`, containing:
  - `image_keys`
  - `bboxes`
  - `category_indices`
  - `category_names`
- `bbox_stats.txt`, containing the mean and standard deviation of the bounding-box variables

In `labels.h5`, bounding boxes are stored in normalized units (divided by 512), in the format:
- `center_x`
- `center_y`
- `width`
- `height`

## Other content

### `Objects365_categories.pkl`
This file contains the category specification used by the image-generation script: generation names and accepted detector labels for each category.

### `print_objects365_categories.py`
This is a small utility script that prints the contents of `Objects365_categories.pkl`.
