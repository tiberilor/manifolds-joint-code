# Repository structure

This repository contains the code to reproduce the results in the paper.

Dataset generation, CNN training, and linear decoding analyses were run on an HPC cluster using NVIDIA A100 GPUs. Saved result files can instead be analyzed and plotted on the local machine.

## Environments
Two Conda environment files are provided:
- `environment-local.yml`: for local-machine use
- `environment-cluster.yml`: for HPC-cluster use

## Folders

- `dataset_generator/`  
  Scripts for generating the image datasets used in the paper, including both the main generative-AI dataset and a repackaged version of the Hong et al. image set.

- `CNN_trainer/`  
  Scripts for training the CNN models used in the paper.

- `linear_decoding_and_theory/`  
  Scripts for computing linear decoding results and the quantities needed to evaluate the theory.

- `computing_other_quantities/`  
  Scripts for computing additional quantities used in the paper, such as mutual manifold alignment, CKA, Brain-Score.

- `plotting_scripts/`  
  Minimal plotting scripts showing how to reproduce the main quantities and figures of the paper starting from the saved result files.

## Utility libraries

- `LIB_dataloader.py`  
  Utility functions for loading datasets and building dataloaders used by many scripts across the repository.

- `LIB_plot_utility.py`  
  Utility functions shared by many plotting scripts, which perofrom much of the computation of the plotted quantities.

## Installation guide

Create the Conda environment with:

```bash
/usr/bin/time -p conda env create -n manifolds-joint-code -f <environment.yml> -y
```

Replace `<environment.yml>` with either:
- `environment-local.yml` for local-machine use
- `environment-cluster.yml` for HPC-cluster use

On our local machine, creating the environment from `environment-local.yml` took approximately 2.5 minutes when the required Conda and pip packages were already present in the local cache. A fresh installation on a new machine may take longer because packages may need to be downloaded.

The code was tested on Linux x86_64 systems:
- Local machine: Ubuntu 24.04.4 LTS (x86_64)
- HPC cluster: Rocky Linux 8.10 (Green Obsidian) (x86_64)

### Additional requirement for dataset generation

For dataset generation, CerberusDet must also be installed separately, since it is not included in the Conda environment file.

To set up CerberusDet:

Clone the repository:
```bash
mkdir -p <your_desired_CerberusDet_repository_path>
cd <your_desired_CerberusDet_repository_path>
git clone https://github.com/ai-forever/CerberusDet.git
```

Download the pretrained checkpoint to a separate folder:
```bash
mkdir -p <your_desired_weights_path>
wget -O <your_desired_weights_path>/voc_obj365_full_bs_best.pt \
  https://huggingface.co/iitolstykh/cerberusdet-yolov8x-voc-o365-full/resolve/main/voc_obj365_full_bs_best.pt
```

For compatibility with recent PyTorch versions, modify `experimental.py` so that the checkpoint can be loaded:
```bash
sed -i 's/torch.load(ckpt_path, map_location=map_location)/torch.load(ckpt_path, map_location=map_location, weights_only=False)/' \
<your_desired_CerberusDet_repository_path>/CerberusDet/cerberusdet/models/experimental.py
```

After setting up CerberusDet, edit the dataset-generation script so that the user-configurable path variables point to:
- the local CerberusDet repository
- the local CerberusDet checkpoint file

The dataset-generation code was tested with CerberusDet at commit `f5c34eaf45ae14ff5f8683c821a2ab241449f087`.

## DEMO

The code and instructions provided in this repository allow reproduction of the complete pipeline end-to-end. However, dataset generation, CNN training, and large-scale analyses of neural representations require substantial storage and compute resources. In this work, these steps were run on an HPC cluster using NVIDIA A100 GPUs and are not practical to demo on a local machine.

For a lightweight demo that can be run on a local machine, we provide on [Figshare](https://doi.org/10.6084/m9.figshare.33491980) a version of this codebase supplemented with some of the generated data. This version additionally contains:

- a small sample of the generated image dataset (70 images per category), together with a dataset-visualization script
- a subset of saved outputs from the large-scale analyses, including linear-decoding results and measured manifold-geometry quantities
- plotting scripts that load these saved results, evaluate the theory, and reproduce selected figures from the manuscript
- the trained checkpoints used for the three main-text CNNs: classification-only network C, joint classification-and-regression network CR, and regression-only network R

For a lightweight demonstration of the analysis code, see the `plotting_scripts/` folder. These scripts use the provided sample results to reproduce representative figures from the paper without rerunning the full computational pipeline. On our local machine, each demo script completed in approximately 1–10 minutes.

The sample images and their visualization script are also included directly in the current GitHub repository, under `dataset_generator/GenerativeAI_image_dataset/`.