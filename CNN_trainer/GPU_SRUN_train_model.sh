#!/bin/bash

#SBATCH --cpus-per-task=16

#SBATCH -t 3-00:00 # time in days-hours-minutes

#SBATCH --mem=200G

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

# Load software modules and source conda environment
source path/to/conda/bin/activate stable_diffusion

# RUN PROGRAM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

script_string="path/to/CLUSTER_train_model.py \\
#--job_id NetworkCR \\
#
# MODEL
--resnet_version resnet50 \\
# uncomment desired active heads (and per_class_bbox flag) depending on the desired network type (CR, C, R, or CRloc):
# for CR:
--active_heads True True True True True \\
# for C:
#--active_heads True False False False False \\
# for R:
#--active_heads False True True True True \\
# for CRloc:
#--active_heads True True True True True \\
# --per_class_bbox \\
# UNCOMMENT OPTION BELOW FOR NETWORK CRloc:
# --per_class_bbox \\
#
# DATA
--train_data_dir path/to/SDXL_dataset_train \\
--val_data_dir path/to/SDXL_dataset_validation \\
#
# TRAINING
--freeze_schedule 1:layer4 2:layer3 3:layer2 \\
--batch_size 208 \\
--epochs 40 \\
--optimizer SGD \\
--regression_loss SmoothL1 \\
#
# SYSTEM
--num_workers 16 \\
--save_dir path/to/result_dir \\
--save_every 1 \\
--time_per_epoch \\
# commented line \\"

# Use awk to filter out lines starting with #
#filtered_script=$(echo "$script_string" | awk '!/^ *#/')
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "srun --cpus-per-task $SLURM_CPUS_PER_TASK --gres=gpu:1 python $filtered_script"

