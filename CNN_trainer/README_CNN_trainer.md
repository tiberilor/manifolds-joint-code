# `CNN_trainer`

This folder contains the scripts needed to train the CNNs used in the paper.

It covers the four network variants:

- **C**: classification only
- **R**: regression only
- **CR**: joint classification and regression
- **CRloc**: classification plus category-specific regression

## Contents

### 1. `CLUSTER_train_model.py`
Main training script.

It:
- loads the training and validation datasets
- builds the selected network
- trains the model
- evaluates on the validation set each epoch
- saves checkpoints and training history

### 2. `LIB_model.py`
Model definition.

It defines the ResNet-based architecture used for all four network variants.

### 3. `GPU_SRUN_train_model.sh`
Example launcher script.

It shows how to run the training script with the settings used for the models reported in the paper.
All other arguments that are not explicitly set in the .sh can be left at their defaults unless you want a different setup.
Before running the example `.sh`, update these paths:

- the conda activation path
- the path to `CLUSTER_train_model.py`
- the training dataset path
- the validation dataset path
- the output directory

## External dependency

This folder does **not** contain the dataloader.

The training script imports `DatasetSDXL` from `LIB_dataloader.py`, which is located elsewhere in this codebase and is documented separately.

## Outputs

Training saves outputs inside a subfolder named after `job_id` under `save_dir`.

The main outputs are:

- epoch checkpoints
- `*_latest.pth`
- `*_BEST.pth`
- `*_final.pth`
- `*_performance_history.json`

If a matching `*_latest.pth` checkpoint already exists, training resumes automatically.
