# Additional feature analyses

These are modified versions of the training, dataloading, regression, and theory scripts in the main codebase. They extend the analysis to bounding-box area, mean sRGB luminance, local sRGB luminance contrast, and mean HSV saturation.

Run `compute_additional_labels.py` once for each SDXL dataset, then select the desired regression targets with `--target_features`. Other arguments and outputs follow the corresponding scripts in the main codebase.
