# `linear_decoding_and_theory`

This folder contains the scripts for the linear decoding analyses and for computing the theory quantities used in the paper.

## Main scripts

The main scripts in this folder are:

1. **classification**  
   Runs the linear one-vs-rest classification analysis. CLUSTER_classification.py

2. **global regression**  
   Runs global linear regression of the bbox variables. CLUSTER_compute_cross_validated_regression.py

3. **global regression on locally centered manifolds**  
   Runs global regression after centering each category manifold locally. This is used to compute the centroid contribution. CLUSTER_global_cv_regression_FULL_DATA_fluctuations_only.py

4. **local regression with shared gamma**  
   Runs the local linear regression with a shared regularization parameter `gamma`. CLUSTER_local_cv_regression_GAMMA_ONLY.py

5. **evaluate theory quantities**  
   Computes the theory quantities from the data, using as input a result file from the local regression script for one specific gamma value. CLUSTER_compute_regression_performance_CV_LOCAL_RULE.py

6. **evaluate theory quantities on generative model of manifolds**  
   Computes the theory quantities on manifolds generated from the fitted generative model, as used in the SI theory validation. CLUSTER_resampled_manifold_theory.py

## Example shell scripts

Example launchers are provided in the subfolder `example_sh`.

- There is an example `.sh` for **classification**. GPU_SRUN_classification.sh
- There is an example `.sh` for **global regression**. GPU_SRUN_cv_regression.sh
- There is an example `.sh` for **local regression with shared gamma**. GPU_SRUN_local_cv_regression_GAMMA_ONLY.sh

For these scripts, any parameter that is **not** explicitly set in the example `.sh` can be left at its default value.

For **global regression on locally centered manifolds**, usage is completely analogous to the example `.sh` for global regression.

For the two **theory** scripts, there is typically no separate configuration to set by hand beyond choosing the input file and results folder, because they inherit the relevant configuration from the selected local-regression result file.

## Common options used in most scripts

Some options are common across most of the scripts in this folder.

### `--layer`
Selects the representation layer to decode from, for example `backbone`, `backbone.layerX`, `backbone.layerX.Y`. With X=1,2,3,4 and being the main layer number similarly Y being the sublayer number

### `--random_projection_size`, `--random_projection_seed`
Apply a random projection to the representations before decoding.

### `--random_subsample_size`, `--random_subsample_seed`
Subsample neural units.

### `--manifold_subsample_size`, `--manifold_subsample_seed`
Subsample the number of categories/manifolds.

These are the main shared options that matter for reproducing the decoding and theory analyses in the paper.

---

## Detailed script description

## 1. Classification

This script runs the linear one-vs-rest classification analysis.

Use the example `.sh` in `example_sh` as a template.

The output is a single result file containing the classification results across categories and cross-validation splits.

## 2. Global regression

This script runs global linear regression of the bbox variables.

Use the example `.sh` in `example_sh` as a template.

The output is a single result file containing the global regression results, including the regression errors and related quantities for the four bbox features.

## 3. Global regression on locally centered manifolds

This script is used to compute global regression after centering each category manifold locally.

Its usage is completely analogous to the example `.sh` provided for global regression.

The output is a single result file, analogous to the global-regression output, but computed on locally centered manifolds. This result is used to isolate the centroid contribution.

## 4. Local regression with shared gamma

This script runs the local linear regression with a shared regularization parameter `gamma`.

Use the example `.sh` in `example_sh` as a template. It shows the standard launch setup and also includes a good list of `gamma` values that was typically a wide enough range to find the optimal gamma.

The output is **one result file per gamma value**.

For selecting the gamma that gives the best **local cross-validated performance**, this is usually done at the plotting stage using plotting code provided elsewhere in the codebase.

## 5. Evaluate theory quantities

This script computes the theory quantities on the real representations.

Its main input is **a result file from the local regression script for one specific gamma value**.

In practice, the workflow is:

1. Run **local regression with shared gamma** for a range of gamma values.
2. Use the plotting script in this folder to determine which gamma should be used for the theory computation.
3. Run the theory script using the result file corresponding to that selected gamma value.

The plotting script in this folder (PLOT_optimal_gamma.py) takes a folder containing the local-regression result files for different gamma values, plus the bbox feature of interest, and identifies the gamma to use for the theory computation.

The theory script then loads the selected local-regression result file and copies the relevant configuration from it, so there is usually no additional configuration to set manually beyond choosing that input file.

The output is a single theory-results file containing the quantities needed to evaluate the theory.

## 6. Evaluate theory quantities on generative model of manifolds

This script computes the theory quantities on manifolds sampled from the fitted generative model, as used in the SI theory-validation analysis.

Its main input is **a theory-results file produced by the real-data theory script**.

In practice, the workflow is:

1. Run **local regression with shared gamma** for a range of gamma values.
2. Use the plotting script in this folder to determine which gamma should be used for the theory computation.
3. Run **evaluate theory quantities** using the result file from the local regression script corresponding to that selected gamma value.
4. Run this generative-model theory script using the theory-results file produced in the previous step.

This script loads that theory-results file and reuses the theory quantities stored there, in particular the local rules and related quantities needed to define the generative-model resampling procedure. It also copies the relevant configuration from that file, so it usually does not require separate manual configuration beyond selecting the input theory-results file and the target bbox feature.