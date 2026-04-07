#!/bin/bash

# SBATCH OPTIONS

#SBATCH --cpus-per-task=16

#SBATCH -t 0-08:00 # time in days-hours-minutes

#SBATCH --mem=200G

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

# Load software modules and source conda environment
source path/to/conda/bin/activate stable_diffusion

# RUN PROGRAM

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

script_string="path/to/CLUSTER_compute_cross_validated_regression.py \\
--job_id $SLURM_JOB_ID \\
--layer backbone.layer4 \\
--checkpoint path/to/model/checkpoint.pth \\
--dataset_folder path/to/dataset/folder/SDXL_dataset_test \\
--results_folder_name path/to/results/folder \\
--random_projection_size 2048 \\
--random_projection_seed 1 \\
#--random_subsample_size 168 \\
#--random_subsample_seed 1 \\
--l2_min_exp -10 \\
--l2_max_exp 4 \\
--n_l2 40 \\
#
# CV SPLITS: Select based on the dataset
# FOR SDXL DATASET: 200 img per batch. 200 / 10 = 20; 200 - 20 = 180; 180 / 20 = 9 ok
#--n_cv_splits 10 \\
#--n_l2_cv_splits 20 \\
# FOR DICARLO IMAGES PUBLIC DATASET: 200 img per batch. outer_test =  200/20 = 10; outer_train = 200 - 10 = 190; inner_test = 190/19 = 10 OK
#--n_cv_splits 20 \\
#--n_l2_cv_splits 19 \\
#
--outer_split_method kfold \\
--inner_split_method kfold \\
# FOR SDXL DATASET: 10000 images per category so 200 is a good dividend
# FOR DICARLO PUBLIC: 400 images per category so 200 is a good dividend
--batch_size 200 \\
--num_workers 16 \\
# commented line \\"

# Use awk to filter out lines starting with #
#filtered_script=$(echo "$script_string" | awk '!/^ *#/')
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "srun --cpus-per-task $SLURM_CPUS_PER_TASK --gres=gpu:1 python $filtered_script"

