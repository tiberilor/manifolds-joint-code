# `computing_other_quantities`

This folder contains the scripts used to compute additional quantities reported in the paper that are not part of the main linear decoding / theory pipeline.

## Contents

### 1. CKA script
CLUSTER_compute_online_cka_two_models.py computes **CKA** between the representations of two models (or two representation sources) on the same dataset. It supports both the standard biased estimator and the debiased estimator, and saves the sufficient statistics together with the final CKA values.

**Main inputs**
- checkpoint for model 1
- checkpoint for model 2
- layer for model 1
- layer for model 2
- dataset folder
- results folder name

**Optional inputs**
- random projection
- random neuron subsampling
- category subsampling

**Expected dataset input**
- a image/representations dataset folder in the same format used elsewhere in this codebase

**Output**
- one `.pkl` results file containing the CKA values and the accumulated sufficient statistics

---

### 2. Global alignment script
CLUSTER_compute_alignment.py computes the **global mutual manifold alignment** quantity \(A\) from category covariance matrices. It first computes one covariance matrix per category, then evaluates all pairwise alignments and their summary statistics.

**Main inputs**
- checkpoint
- layer
- dataset folder
- results folder name

**Optional inputs**
- random projection

**Expected dataset input**
- a image/representations dataset folder in the same format used elsewhere in this codebase


**Output**
- one `.pkl` results file containing:
  - per-category covariance matrices
  - per-category participation ratios and related norms
  - pairwise alignments
  - global mean and standard deviation of the alignment measures

---

### 3. Public neural Brain-Score script
CLUSTER_compute_brainscore_WITH_CLEANUP.py computes **Brain-Score alignment to public neural benchmarks** across a list of candidate layers, and identifies the best layer for each region.

**What must be filled inside the script**
- `EXTRA_SYS_PATHS`: add any paths needed so that `torch.load(...)` can resolve the model class when loading checkpoints
- `MODEL_PATHS`: map each `model_id` to a checkpoint path, or to a path whose basename is `resnet50` for the ImageNet-pretrained baseline
- optionally edit `CANDIDATE_LAYERS` if you want a different list of tested layers

**Command-line inputs**
- `--model_id`
- `--results_folder_name`

**Expected model input**
- a checkpoint containing `ckpt["model"]`
- optionally `ckpt["transform"]`
- or the special `resnet50` baseline path

**Output**
- one `.pkl` results file containing:
  - score for each layer and region
  - best layer by region
  - best score by region
  - average neural score across regions when available

## Notes

The CKA and global-alignment scripts follow the same general pattern as the scripts in `linear_decoding_and_theory`: they load a dataset, extract representations from a selected layer, and save a single results file. The Brain-Score script instead uses a small user-edited configuration section inside the script to specify which models to score. 