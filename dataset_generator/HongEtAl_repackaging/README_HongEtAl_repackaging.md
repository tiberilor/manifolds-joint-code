# `HongEtAl_repackaging`

This folder contains scripts to repackage the **public** Hong et al. / DiCarlo lab dataset into the same general format as our image dataset in `GenerativeAI_image_dataset`.

The goal is to reorganize the public Hong et al. data so that it can be used smoothly in the same downstream analysis pipeline.

## What this produces

The repackaged data mirror the structure of the synthetic dataset:

- one folder per category
- inside each category:
  - `images/` with `.jpg` images
  - `bboxes/` with one `.pkl` bounding-box file per image
- a `categories.pkl` file mapping category names to integer indices
- a `labels.h5` file containing category labels and normalized bounding-box annotations
- a `bbox_stats.txt` file with bounding-box summary statistics

For the image-dataset structure and annotation conventions, see the README in `../GenerativeAI_image_dataset`, which should be taken as the main reference.

In addition to the image dataset, the repackaging also creates a parallel dataset of neural responses, organized by category, for either IT or V4.

## Procedure

### Step 1
Run the main repackaging script for the public Hong et al. dataset.

This:
- loads the public BrainIO assembly
- extracts one brain region (`IT` or `V4`)
- extracts the category names, sorts them alphabetically, and saves `categories.pkl`
- copies the corresponding images into category folders
- writes one bounding-box `.pkl` file per image
- writes one per-category neural-response `.pkl` file

### Step 2
Run the `labels.h5` generation script on the repackaged image dataset.

This creates:
- `labels.h5`
- `bbox_stats.txt`

## Requirements

You need access to the **public** Hong et al. dataset through the BrainIO / Brain-Score ecosystem.

In practice, this means you should have the required Python packages installed, including at least:

- `brainio_collection`
- `numpy`
- `Pillow`
- `h5py`

Depending on your environment, you may also need packages imported by the repackaging script, such as:

- `xarray`
- `pandas`
- `scipy`
- `brainscore_vision`

You must also set the image path and output paths inside the scripts before running them.