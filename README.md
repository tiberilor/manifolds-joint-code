# Repository structure

This repository contains the code to reproduce the results in the paper

The repository includes an `environment.yml` file capturing the main conda environment used for most scripts in this repo. Some components require additional manual installation that is not included in this environment file, in particular:
- CerberusDet, used by the image-generation pipeline;
- BrainIO / Brain-Score packages, used only by the neural benchmark / repackaging scripts.

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