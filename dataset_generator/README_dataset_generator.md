# `dataset_generator`

This folder contains code for preparing the datasets used in the paper.

## Subfolders

### `GenerativeAI_image_dataset`
Code and documentation for generating the synthetic image dataset used in the main analyses, together with bounding-box annotations and packaged label files.

### `HongEtAl_repackaging`
Code for repackaging the **public** Hong et al. / DiCarlo lab dataset into the same general structure as the `GenerativeAI_image_dataset`, so that the same downstream analysis pipeline can be applied with minimal changes.

For details on the synthetic dataset structure, see the README inside `GenerativeAI_image_dataset`. The Hong et al. repackaged dataset is organized to closely mirror that format.