#!/bin/bash

# SBATCH OPTIONS

#SBATCH --cpus-per-task=16

#SBATCH -t 3-00:00 # time in days-hours-minutes

#SBATCH --mem=200G

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

# Load software modules and source conda environment
source path/to/conda/bin/activate stable_diffusion

# RUN PROGRAM

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

script_string="path/to/CLUSTER_local_cv_regression_GAMMA_ONLY.py \\
--job_id $SLURM_JOB_ID \\
--layer backbone \\
--checkpoint path/to/model/checkpoint.pth \\
--dataset_folder path/to/dataset/folder/SDXL_dataset_test \\
--results_folder_name path/to/results/folder \\
#--random_projection_size 2048 \\
#--random_projection_seed 1 \\
# GOOD OVERALL RANGE OF GAMMAS. Run the script for all of these three lists of gammas
#--gammas 0.0000001 0.0000002 0.0000003 0.0000006 0.000001 0.000002 0.000003 0.000006 0.00001 0.00002 0.00003 0.00006 \\
#--gammas 0.0001 0.0002 0.0003 0.0006 0.001 0.002 0.003 0.006 0.01 0.02 0.03 0.06 \\
#--gammas 0.1 0.2 0.3 0.6 1.0 2.0 3.0 6.0 10.0 20.0 30.0 60.0 \\
#
# CV SPLITS: Select based on the dataset
# FOR SDXL DATASET: 200 img per batch. 200 / 10 = 20; 200 - 20 = 180; 180 / 10 = 18 ok
--n_cv_splits 10 \\
--n_gamma_cv_splits 10 \\
# FOR DICARLO IMAGES PUBLIC DATASET: 200 img per batch. outer_test =  200/20 = 10; outer_train = 200 - 10 = 190; inner_test = 190/19 = 10 OK
#--n_cv_splits 20 \\
#--n_gamma_cv_splits 19 \\
#
--outer_split_method kfold \\
--inner_split_method kfold \\
--num_workers 16 \\
# FOR SDXL DATASET: 10000 images per category so 200 is a good dividend
# FOR DICARLO PUBLIC: 400 images per category so 200 is a good dividend
--batch_size 200 \\
# commented line \\"

# Use awk to filter out lines starting with #
#filtered_script=$(echo "$script_string" | awk '!/^ *#/')
filtered_script=$(echo "$script_string" | awk '!/^ *#/' | sed -e ':a' -e 'N' -e '$!ba' -e 's/\\$//')

#echo "$filtered_script"

# Run the Python script using eval
eval "srun --cpus-per-task $SLURM_CPUS_PER_TASK --gres=gpu:1 python $filtered_script"
