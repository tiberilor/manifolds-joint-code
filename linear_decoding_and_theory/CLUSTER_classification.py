"""
CV one-vs-rest Least-Squares Classification (deterministic, streamable)

This is the **classification twin** of my ridge REGRESSION script, with the **same style**:
- Same accumulators: covariance/means/io_covariance/label_mean/label_variance.
- Same split mechanics (outer + inner), same per-batch accumulation, same centering rules
  (test centered w.r.t. train means, including label_mean and io_covariance centering).
- Same random projection (Hadamard+subsample) and random neuron subsampling.
- Same **global dimensionality reduction block placed exactly where it was**.

Differences:
- Targets are **binary 0/1** for **one-vs-rest** classification: for classifier “feature = category_c”,
  y = 1 on samples from category_c, else 0. This lets us keep label moments and io_covariance
  **identical** to the regression code path: only positives contribute non-zero sums.
- Solve closed-form LS for each λ (both inner and outer), compute **bias** b = mu_y − wᵀ mu_x.
- 0/1 error cannot be computed from moments → **second pass** to tally errors for **all λ**
  (inner tests for λ selection, outer tests for reporting). Deterministic.

Classification decision:  ŷ = 1{ wᵀ x + b ≥ 0.5 }  (since y∈{0,1}).

The results dict mirrors the regression one but stores **classification errors** instead of MSE,
and additionally includes per-λ outer error curves.
"""

import os
import argparse
import pickle
from einops import einsum
import torch
import numpy as np
import random
import copy
import math
from hadamard_transform import hadamard_transform, next_power_of_2, pad_to_power_of_2, is_a_power_of_2
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import time

# IMPORT DATALOADERS / MODELS like in the original
sys.path.append('../../codebase_v5')

# -> AvgPool_SDXLdataset model
sys.path.append('../ANN_models/AvgPool_SDXLdataset')
import LIB_model

from functools import partial
from pathlib import Path

import math

# ---------------------------
# argparse (kept identical)
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="0000",
                    help="A name/ID for this run (used for naming the results).")
parser.add_argument("--layer", type=str,
                    default="backbone",
                    help="layer from which to extract the features.")
parser.add_argument("--checkpoint", type=str,
                    default="./checkpoint.pth",
                    help="Path to the model checkpoint file.")
parser.add_argument("--dataset_folder", type=str,
                    default="./",
                    help="Path to the dataset folder.")

parser.add_argument("--results_folder_name", type=str,
                    default="./",
                    help="Name of the folder in the checkpoint folder where we will store the results.")
parser.add_argument("--random_projection_size", type=int, default=None,
                    help="The size of the random projection. Default None corresponds to no projection.")
parser.add_argument("--random_projection_seed", type=int, default=1,
                    help="Random seed value. E.g. used for generating the random projection")

# Random neuron subsampling (same)
parser.add_argument("--random_subsample_size", type=int, default=None,
                    help="Number of neurons to keep via random subsampling. Default None = no subsampling.")
parser.add_argument("--random_subsample_seed", type=int, default=1,
                    help="Random seed used to pick the subsampled neuron indices.")

parser.add_argument("--n_cv_splits", type=int, default=1, help="cv splits over which performance is cross-validated")
parser.add_argument("--n_l2_cv_splits", type=int, default=1, help="inner cv splits over which the best l2 is selected")

parser.add_argument("--l2_min_exp", type=float, default=-6, help="min base-10 exponent for λ (λ=10^exp)")
parser.add_argument("--l2_max_exp", type=float, default=2,  help="max base-10 exponent for λ (λ=10^exp)")
parser.add_argument("--n_l2", type=int, default=9, help="number of λs between l2_min_exp and l2_max_exp (inclusive)")

parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images.")
parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
parser.add_argument("--classifier_chunk_size", type=int, default=1001,
                    help="Number of classifiers to score at once in 2nd pass (vectorized across λ too).")


# Split controls (identical)
parser.add_argument("--outer_split_method", type=str, choices=["kfold", "random"], default="kfold",
                    help="‘kfold’: equal folds; ‘random’: random train/test splits")
parser.add_argument("--inner_split_method", type=str, choices=["kfold", "random"], default="kfold",
                    help="‘kfold’: equal folds; ‘random’: random train/test splits")
parser.add_argument("--outer_test_ratio", type=float, default=0.2,
                    help="When using --split_method random, fraction of data to hold out as test")
parser.add_argument("--inner_test_ratio", type=float, default=0.2,
                    help="When using --split_method random, fraction of data to hold out as test")
parser.add_argument("--random_seed_splits", type=int, default=43, help="Seed for random splits")

# Global dimensionality reduction (kept in the SAME place in code flow)
parser.add_argument("--global_dimensionality_reduction_style", type=str, default="",
                    help="percentage|dimensions|'' (none)")
parser.add_argument("--global_dimensionality_reduction_strength", type=float, default=100,
                    help="Percent variance to keep if style=percentage (e.g., 90)")
parser.add_argument("--global_dimensionality_reduction_dimensions", type=float, default=2048,
                    help="Dims to keep if style=dimensions")
parser.add_argument("--global_dimensionality_reduction_file", type=str, default="./boh.pkl",
                    help="Path to the file containing the global stats.")

args = parser.parse_args()


# --------------------------------
# helpers (kept identical in style)
# --------------------------------
def print_args(args):
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:20s}: {value!r}")
    print("=========================", flush=True)


def seed_everything(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_bbox_stats_from_loader(dataloader):
    ds = getattr(dataloader, "dataset", None)
    if hasattr(ds, "dataset"): ds = ds.dataset
    mean = getattr(ds, "bbox_mean", None)
    std  = getattr(ds, "bbox_std", None)
    return mean, std


def get_splits(n_items, method="kfold", n_splits=5, test_ratio=0.2, seed=None):
    """
    Return a list of (train_idx, test_idx) **per batch**, identical to the original.
    """
    indices = list(range(n_items))
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed)
    if method == "kfold":
        split_size = int(math.ceil(n_items / n_splits))
        folds = []
        for fold in range(n_splits):
            start = fold * split_size
            end = min(start + split_size, n_items)
            test_idx = torch.tensor(indices[start:end], dtype=torch.long)
            train_idx = torch.tensor(indices[:start] + indices[end:], dtype=torch.long)
            folds.append((train_idx, test_idx))
        return folds
    elif method == "random":
        folds = []
        test_size = int(math.floor(n_items * test_ratio))
        for _ in range(n_splits):
            perm = indices.copy(); random.shuffle(perm)
            test_idx = torch.tensor(perm[:test_size], dtype=torch.long)
            train_idx = torch.tensor(perm[test_size:], dtype=torch.long)
            folds.append((train_idx, test_idx))
        return folds
    else:
        raise ValueError(f"Unknown split method {method}")


def convert_tensors_to_numpy(d):
    if isinstance(d, dict):
        return {k: convert_tensors_to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    else:
        return d


def accumulate(old, addition):
    """
    Same semantics: if old is None -> clone/addition; else old + addition.
    """
    if old is None:
        return addition.clone() if hasattr(addition, "clone") else copy.deepcopy(addition)
    else:
        return old + addition


# ---- global dim-red helpers (unchanged) ----
def explained_variance_ratio(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    if eigs.numel() == 0 or dims_kept <= 0:
        return torch.tensor(0.0, dtype=eigs.dtype, device=eigs.device)
    eigs_sorted = torch.sort(eigs, descending=True).values
    k = min(int(dims_kept), eigs_sorted.numel())
    return eigs_sorted[:k].sum() / eigs_sorted.sum()


def explained_variance_percent(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    return explained_variance_ratio(eigs, dims_kept) * 100


def explained_variance_dim(eigs, threshold=0.9):
    eigs_sorted = torch.sort(eigs, descending=True).values
    cumulative = torch.cumsum(eigs_sorted, dim=0)
    total = torch.sum(eigs_sorted)
    explained_ratio = cumulative / total
    idx = (explained_ratio >= threshold).nonzero(as_tuple=False)
    return (idx[0].item() + 1) if idx.numel() > 0 else len(eigs_sorted)


# --------------
# print & device
# --------------
print_args(args)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


# --------------
# dataset/model
# --------------
dataset_path = Path(args.dataset_folder)
if dataset_path.is_file() or dataset_path.suffix.lower() == ".h5":
    dataset_folder_name = dataset_path.parent.name
else:
    dataset_folder_name = dataset_path.name

# λ grid
l2_strengths = np.logspace(args.l2_min_exp, args.l2_max_exp, num=args.n_l2, base=10.0).tolist()
print(f"Sampling l2 strengths: {l2_strengths}", flush=True)

# model & transform (same as original)
if os.path.basename(os.path.normpath(args.checkpoint)).lower() == "resnet50":
    model = LIB_model.ResNetBBoxModel(resnet_version="resnet50").to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
elif dataset_folder_name in ["representations_DicarloAsGenai_public_IT", "representations_DicarloAsGenai_public_V4"]:
    model = None
    transform = None
    print("\nRepresentations dataset detected. Setting model to None\n")
else:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ckpt["model"].to(device); model.eval()
    transform = ckpt["transform"]

print(f"Dataset: {dataset_folder_name}", flush=True)

# dataset factory (kept same)
if dataset_folder_name in ["SDXL_dataset_test", "SDXL_dataset_train", "SDXL_dataset_validation",
                           "DicarloAsGenai_public"]:
    from LIB_dataloader import CategoryDatasetSDXL
    CategoryDataset = partial(CategoryDatasetSDXL, dataset_folder=args.dataset_folder, transform=transform)
    # NOTE: features will be overwritten to categories later for classification

elif dataset_folder_name in ["representations_DicarloAsGenai_public_IT", "representations_DicarloAsGenai_public_V4"]:
    from LIB_dataloader import CategoryDatasetRepresentation
    CategoryDataset = partial(CategoryDatasetRepresentation, dataset_folder=args.dataset_folder, transform=transform)

else:
    print(f"ERROR: dataset {dataset_folder_name} is not implemented", flush=True)
    exit(1)

# categories (kept same)
# retrieve categories (sorted for determinism)
all_categories = [d for d in os.listdir(args.dataset_folder)
                  if os.path.isdir(os.path.join(args.dataset_folder, d))]
all_categories.sort()  # deterministic order

# DEBUG START: uncomment this to test quickly on fewer categories
# all_categories = all_categories[0:3]
# DEBUG END

categories = all_categories

n_categories = len(categories)

# IMPORTANT: in classification, the “features” are the **categories**.
# Keep the same variable name to minimize changes downstream.
features = categories

# ------------------------------
# FIRST PASS accumulators (same)
# ------------------------------
fjlt_idx = None
fjlt_D = None
subsample_idx = None

# label stats min/max (kept; trivial for 0/1 but we record them)
# TODO: remove below
# max_label_test = {feature: {split: -float("inf") for split in range(args.n_cv_splits)} for feature in features}
# min_label_test = {feature: {split:  float("inf") for split in range(args.n_cv_splits)} for feature in features}
# max_label_train = {feature: {split: -float("inf") for split in range(args.n_cv_splits)} for feature in features}
# min_label_train = {feature: {split:  float("inf") for split in range(args.n_cv_splits)} for feature in features}

# define the results dict skeleton as in regression code (fields adapted for classification)
results = {
    "args": vars(args).copy(),

    # store some spectra/coefs for outer train (like your script)
    # TODO: remove below
    # "outer_train_covariance_eigenvalues": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    # "outer_train_io_covariance_coefficients": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},

    # inner classification error stats per λ
    "inner_balacc_test_avg": {feature: {split: {l2: None for l2 in l2_strengths}
                          for split in range(args.n_cv_splits)} for feature in features},

    "inner_balacc_test_std": {feature: {split: {l2: None for l2 in l2_strengths}
                          for split in range(args.n_cv_splits)} for feature in features},

    # best λ per feature/split
    "best_l2_strength": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},

    # TP,P,TN,N at the best λ (per feature/split)
    "outer_counts_best": {feature: {split: {"TP": None, "P": None, "TN": None, "N": None}
                        for split in range(args.n_cv_splits)} for feature in features},

    "balanced_accuracy": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},

    "outer_balanced_accuracy_per_l2": {feature: {split: {l2: None for l2 in l2_strengths}
                                   for split in range(args.n_cv_splits)} for feature in features},

    # summarize across splits
    "balanced_accuracy_avg": {feature: None for feature in features},
    "balanced_accuracy_std": {feature: None for feature in features},

    "labels_variance_full": {feature: None for feature in features},
}

# Sufficient statistics dictionaries (identical structure)
full = {
    "covariance": None,
    "mean": None,
    "n_images": None,
    "io_covariance": {feature: None for feature in features},
    "label_variance": {feature: None for feature in features},
    "label_mean": {feature: None for feature in features},
}
train = {
    "covariance": {split: None for split in range(args.n_cv_splits)},
    "mean": {split: None for split in range(args.n_cv_splits)},
    "n_images": {split: None for split in range(args.n_cv_splits)},
    "io_covariance": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    "label_variance": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    "label_mean": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
}
test = copy.deepcopy(train)

inner_train = {
    "covariance": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
    "mean":       {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
    "n_images":   {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
    "io_covariance": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
    "label_variance": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
    "label_mean": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
}
inner_test = copy.deepcopy(inner_train)

# -------------------------------
# Global dim-red
# -------------------------------
global_projector = None
global_projection_mean = None
global_eigenvalues = None
global_eigenvectors = None
global_d = None
global_explained_variance = None

if args.global_dimensionality_reduction_style != "":
    with open(args.global_dimensionality_reduction_file, "rb") as f:
        global_stats = pickle.load(f)
    global_eigenvalues = torch.tensor(global_stats["covariance_eigenvalues"], device=device, dtype=torch.float64)  # sorted desc
    global_eigenvectors = torch.tensor(global_stats["covariance_eigenvectors"], device=device, dtype=torch.float64)  # cols = eigenvectors (desc)
    global_projection_mean = torch.tensor(global_stats["mean"], device=device, dtype=torch.float64)
    if args.global_dimensionality_reduction_style == "dimensions":
        global_d = int(args.global_dimensionality_reduction_dimensions)
        global_explained_variance = explained_variance_percent(global_eigenvalues, global_d)
    elif args.global_dimensionality_reduction_style == "percentage":
        global_explained_variance = args.global_dimensionality_reduction_strength
        global_d = explained_variance_dim(global_eigenvalues, threshold=(args.global_dimensionality_reduction_strength/100))
    else:
        global_d = -1
        global_explained_variance = 100.0
    print(f"Initializing global projector. Keeping first {global_d} dimensions, or {global_explained_variance}% of the variance")
    global_projector = global_eigenvectors[:, :global_d]
# END global dim-red load

# =====================================================
# FIRST PASS: stream features and accumulate moments
# =====================================================
for category in categories:
    print(f"Processing category: {category}", flush=True)
    tcat0 = time.perf_counter()

    # category-specific dataset/dataloader (same pattern)
    dataset = CategoryDataset(category=category)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,  # deterministic given seed
                            num_workers=args.num_workers,
                            pin_memory=True)

    for batch in dataloader:
        # NOTE: gt_bboxes exist but are unused for classification; kept in signature for parity
        images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch
        images = images.to(device)

        # NEW - support pixels layer option
        # Retrieve reps (supports --layer=pixels to use raw image vectors)
        if args.layer.lower() == "pixels":
            reps = images.flatten(1)
            # DEBUG:
            print(f"Reps size: {reps.size()}")
            # END DEBUG
        elif model is not None:
            with torch.no_grad():
                reps = model.forward_features(images, module_name=args.layer)  # shape [batch size, # neurons]
        else:
            reps = images.to(device)
        # END NEW

        reps = reps.to(dtype=torch.float64)

        # random neuron subsampling (same)
        if (subsample_idx is None) and (args.random_subsample_size is not None):
            seed_everything(args.random_subsample_seed)
            N_orig = reps.size(1)
            if args.random_subsample_size > N_orig:
                raise ValueError(f"--random_subsample_size ({args.random_subsample_size}) > neurons ({N_orig}).")
            subsample_idx = torch.randperm(N_orig, device=device)[:args.random_subsample_size]
        if args.random_subsample_size is not None:
            reps = reps[:, subsample_idx]

        # random projection (Hadamard + subsample) (same)
        if (fjlt_D is None) and (fjlt_idx is None) and (args.random_projection_size is not None):
            seed_everything(args.random_projection_seed)
            N_orig = reps.size(1)
            N_power2 = N_orig if is_a_power_of_2(N_orig) else next_power_of_2(N_orig)
            D = (torch.randint(0, 2, (N_power2,), device=device) * 2 - 1).to(torch.float64)  # ±1
            idx = torch.randperm(N_power2, device=device)[:args.random_projection_size]
            fjlt_D = D; fjlt_idx = idx
        if args.random_projection_size is not None:
            reps = pad_to_power_of_2(reps)
            reps = hadamard_transform(reps * fjlt_D[None, :])
            reps = reps[:, fjlt_idx] * np.sqrt(reps.size(1) / args.random_projection_size)

        # Global projection (APPLIED HERE, like your script)
        if args.global_dimensionality_reduction_style != "":
            reps = (reps - global_projection_mean) @ global_projector

        # outer cv loop (per batch)
        batch_size = reps.size(0)
        splits = get_splits(
            n_items=batch_size,
            method=args.outer_split_method,
            n_splits=args.n_cv_splits,
            test_ratio=args.outer_test_ratio,
            seed=args.random_seed_splits,
        )
        for split, (train_idx, test_idx) in enumerate(splits):
            reps_train = reps[train_idx]
            reps_test = reps[test_idx]

            # Train accumulators
            n_images_train = reps_train.size(0)
            train["n_images"][split] = accumulate(train["n_images"][split], n_images_train)
            cov_train_addition = einsum(reps_train, reps_train, "i d, i e -> d e")
            train["covariance"][split] = accumulate(train["covariance"][split], cov_train_addition)
            mean_train_addition = torch.sum(reps_train, dim=0)
            train["mean"][split] = accumulate(train["mean"][split], mean_train_addition)

            # Test accumulators
            n_images_test = reps_test.size(0)
            test["n_images"][split] = accumulate(test["n_images"][split], n_images_test)
            cov_test_addition = einsum(reps_test, reps_test, "i d, i e -> d e")
            test["covariance"][split] = accumulate(test["covariance"][split], cov_test_addition)
            mean_test_addition = torch.sum(reps_test, dim=0)
            test["mean"][split] = accumulate(test["mean"][split], mean_test_addition)

            # ======= feature-specific pieces (classification 0/1) =======
            # We only do work when feature == category.
            feature = category

            # For the classifier “feature”, all items in this loader are POSITIVES (y=1):
            # label sum (== y) and label^2 sum (== y since 0/1) are both counts.
            labels_train_sum = float(n_images_train)
            labels_test_sum = float(n_images_test)

            # update label means (sum of y)
            train["label_mean"][feature][split] = accumulate(train["label_mean"][feature][split],
                                                             torch.as_tensor(labels_train_sum, dtype=torch.float64, device=reps.device))
            test["label_mean"][feature][split] = accumulate(test["label_mean"][feature][split],
                                                            torch.as_tensor(labels_test_sum, dtype=torch.float64, device=reps.device))

            # update label variance accumulator with y^2 (== y for 0/1)
            train["label_variance"][feature][split] = accumulate(train["label_variance"][feature][split],
                                                                 torch.as_tensor(labels_train_sum, dtype=torch.float64, device=reps.device))
            test["label_variance"][feature][split] = accumulate(test["label_variance"][feature][split],
                                                                torch.as_tensor(labels_test_sum, dtype=torch.float64, device=reps.device))

            # update io_covariance with E[x*y] numerator:
            # since y=1 for all items here, this is just sum of reps.
            io_train_addition = mean_train_addition
            io_test_addition = mean_test_addition
            train["io_covariance"][feature][split] = accumulate(train["io_covariance"][feature][split], io_train_addition)
            test["io_covariance"][feature][split] = accumulate(test["io_covariance"][feature][split], io_test_addition)

            # ===== inner splits inside TRAIN (unchanged mechanics) =====
            inner_batch_size = reps_train.size(0)
            inner_splits = get_splits(
                n_items=inner_batch_size,
                method=args.inner_split_method,
                n_splits=args.n_l2_cv_splits,
                test_ratio=args.inner_test_ratio,
                seed=args.random_seed_splits + split,
            )
            for inner_split, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
                inner_reps_train = reps_train[inner_train_idx]
                inner_reps_test = reps_train[inner_test_idx]

                inner_n_images_train = inner_reps_train.size(0)
                inner_n_images_test = inner_reps_test.size(0)

                inner_train["n_images"][split][inner_split] = accumulate(inner_train["n_images"][split][inner_split],
                                                                         inner_n_images_train)
                inner_test["n_images"][split][inner_split] = accumulate(inner_test["n_images"][split][inner_split],
                                                                        inner_n_images_test)

                inner_cov_train_addition = einsum(inner_reps_train, inner_reps_train, "i d, i e -> d e")
                inner_cov_test_addition  = einsum(inner_reps_test,  inner_reps_test,  "i d, i e -> d e")
                inner_train["covariance"][split][inner_split] = accumulate(inner_train["covariance"][split][inner_split],
                                                                           inner_cov_train_addition)
                inner_test["covariance"][split][inner_split] = accumulate(inner_test["covariance"][split][inner_split],
                                                                           inner_cov_test_addition)

                inner_mean_train_addition = torch.sum(inner_reps_train, dim=0)
                inner_mean_test_addition = torch.sum(inner_reps_test,  dim=0)
                inner_train["mean"][split][inner_split] = accumulate(inner_train["mean"][split][inner_split],
                                                                     inner_mean_train_addition)
                inner_test["mean"][split][inner_split] = accumulate(inner_test["mean"][split][inner_split],
                                                                     inner_mean_test_addition)

                # feature loop, only positives for feature == category
                feature = category
                # label sums are just counts (y=1 for these items)
                inner_labels_train_sum = float(inner_n_images_train)
                inner_labels_test_sum = float(inner_n_images_test)

                inner_train["label_mean"][feature][split][inner_split] = accumulate(
                    inner_train["label_mean"][feature][split][inner_split],
                    torch.as_tensor(inner_labels_train_sum, dtype=torch.float64, device=reps.device))
                inner_test["label_mean"][feature][split][inner_split] = accumulate(
                    inner_test["label_mean"][feature][split][inner_split],
                    torch.as_tensor(inner_labels_test_sum, dtype=torch.float64, device=reps.device))

                inner_train["label_variance"][feature][split][inner_split] = accumulate(
                    inner_train["label_variance"][feature][split][inner_split],
                    torch.as_tensor(inner_labels_train_sum, dtype=torch.float64, device=reps.device))
                inner_test["label_variance"][feature][split][inner_split] = accumulate(
                    inner_test["label_variance"][feature][split][inner_split],
                    torch.as_tensor(inner_labels_test_sum, dtype=torch.float64, device=reps.device))

                inner_train["io_covariance"][feature][split][inner_split] = accumulate(
                    inner_train["io_covariance"][feature][split][inner_split],
                    inner_mean_train_addition)
                inner_test["io_covariance"][feature][split][inner_split] = accumulate(
                    inner_test["io_covariance"][feature][split][inner_split],
                    inner_mean_test_addition)

        # FULL sums (like your “update full only once” trick)
        full["n_images"] = accumulate(full["n_images"], reps.size(0))
        full["covariance"] = accumulate(full["covariance"], einsum(reps, reps, "i d, i e -> d e"))
        full["mean"] = accumulate(full["mean"], torch.sum(reps, dim=0))
        # for classification, only the current feature=category has y=1 on this loader:
        feat = category
        full["label_mean"][feat] = accumulate(full["label_mean"][feat],     torch.as_tensor(reps.size(0), dtype=torch.float64, device=reps.device))
        full["label_variance"][feat] = accumulate(full["label_variance"][feat], torch.as_tensor(reps.size(0), dtype=torch.float64, device=reps.device))
        full["io_covariance"][feat] = accumulate(full["io_covariance"][feat],  torch.sum(reps, dim=0))

    # --- timing for FIRST PASS on this category
    tcat1 = time.perf_counter()
    elapsed = tcat1 - tcat0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\tFirst pass finished for {category}. "
          f"Time: {h:02d}h or {m:02d}m or {s:02d}s",
          flush=True)


# -------------------------
# Normalize & Center (same)
# -------------------------
# divide by counts
for split in range(args.n_cv_splits):
    train["covariance"][split] /= train["n_images"][split]
    test["covariance"][split] /= test["n_images"][split]
    train["mean"][split] /= train["n_images"][split]
    test["mean"][split] /= test["n_images"][split]
    for feature in features:
        train["label_mean"][feature][split] /= train["n_images"][split]
        test["label_mean"][feature][split] /= test["n_images"][split]
        train["label_variance"][feature][split] /= train["n_images"][split]
        test["label_variance"][feature][split] /= test["n_images"][split]
        train["io_covariance"][feature][split] /= train["n_images"][split]
        test["io_covariance"][feature][split] /= test["n_images"][split]
# full (use split 0 convention)
full["covariance"] /= full["n_images"]
full["mean"] /= full["n_images"]
for feature in features:
    full["label_mean"][feature] /= full["n_images"]
    full["label_variance"][feature] /= full["n_images"]
    full["io_covariance"][feature] /= full["n_images"]

# subtract means from covariances (test centered w.r.t. train means)
for split in range(args.n_cv_splits):
    mu_tr = train["mean"][split]; mu_te = test["mean"][split]
    train["covariance"][split] -= torch.outer(mu_tr, mu_tr)
    test["covariance"][split] += (torch.outer(mu_tr, mu_tr)
                                  - torch.outer(mu_tr, mu_te)
                                  - torch.outer(mu_te, mu_tr))
    for feature in features:
        # labels: keep SAME formulas as in your script
        train["label_variance"][feature][split] -= train["label_mean"][feature][split] ** 2
        test["label_variance"][feature][split] += (train["label_mean"][feature][split] ** 2
                                                   - 2*train["label_mean"][feature][split]*test["label_mean"][feature][split]
                                                   )
        train["io_covariance"][feature][split] -= train["mean"][split] * train["label_mean"][feature][split]
        test["io_covariance"][feature][split] += (train["mean"][split] * train["label_mean"][feature][split]
                                                  - train["mean"][split] * test["label_mean"][feature][split]
                                                  - test["mean"][split] * train["label_mean"][feature][split])
# full centering
full["covariance"] -= torch.outer(full["mean"], full["mean"])
for feature in features:
    full["label_variance"][feature] -= full["label_mean"][feature] ** 2
    full["io_covariance"][feature] -= full["mean"] * full["label_mean"][feature]

# inner: divide and center (same pattern)
for split in range(args.n_cv_splits):
    for inner_split in range(args.n_l2_cv_splits):
        inner_train["covariance"][split][inner_split] /= inner_train["n_images"][split][inner_split]
        inner_test["covariance"][split][inner_split] /= inner_test["n_images"][split][inner_split]
        inner_train["mean"][split][inner_split] /= inner_train["n_images"][split][inner_split]
        inner_test["mean"][split][inner_split] /= inner_test["n_images"][split][inner_split]
        for feature in features:
            inner_train["label_mean"][feature][split][inner_split] /= inner_train["n_images"][split][inner_split]
            inner_test["label_mean"][feature][split][inner_split] /= inner_test["n_images"][split][inner_split]
            inner_train["label_variance"][feature][split][inner_split] /= inner_train["n_images"][split][inner_split]
            inner_test["label_variance"][feature][split][inner_split] /= inner_test["n_images"][split][inner_split]
            inner_train["io_covariance"][feature][split][inner_split] /= inner_train["n_images"][split][inner_split]
            inner_test["io_covariance"][feature][split][inner_split] /= inner_test["n_images"][split][inner_split]

        # center (test w.r.t. train)
        mu_in_tr = inner_train["mean"][split][inner_split]
        mu_in_te = inner_test["mean"][split][inner_split]
        inner_train["covariance"][split][inner_split] -= torch.outer(mu_in_tr, mu_in_tr)
        inner_test["covariance"][split][inner_split] += ((torch.outer(mu_in_tr, mu_in_tr)
                                                          - torch.outer(mu_in_tr, mu_in_te))
                                                          - torch.outer(mu_in_te, mu_in_tr))
        for feature in features:
            inner_train["label_variance"][feature][split][inner_split] -= inner_train["label_mean"][feature][split][inner_split] ** 2
            inner_test["label_variance"][feature][split][inner_split] += (
                inner_train["label_mean"][feature][split][inner_split] ** 2
                - 2*inner_test["label_mean"][feature][split][inner_split]*inner_train["label_mean"][feature][split][inner_split])

            inner_train["io_covariance"][feature][split][inner_split] -= inner_train["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split]
            inner_test["io_covariance"][feature][split][inner_split] += (
                (inner_train["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split]
                 - inner_test["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split])
                - inner_train["mean"][split][inner_split] * inner_test["label_mean"][feature][split][inner_split]
            )


# -------------------------------------------------------
# Fit models (closed form) for ALL λ (inner and outer)
# -------------------------------------------------------
def eye_like(C):
    return torch.eye(C.size(0), device=C.device, dtype=C.dtype)


# store (w, b) like this to reuse in the second pass
inner_models = {feature: {split: {inner: {l2: (None, None) for l2 in l2_strengths}
                          for inner in range(args.n_l2_cv_splits)}
                for split in range(args.n_cv_splits)} for feature in features}
outer_models = {feature: {split: {l2: (None, None) for l2 in l2_strengths}
                for split in range(args.n_cv_splits)} for feature in features}


# helper: fit w,b from centered moments (same d, new b)
def fit_wb(Cov_x, mu_x, n_tot, io_cov_centered, label_mean_centered_unused, l2):
    """
    io_cov_centered = E[(x-mu_x)(y-mu_y)] (already centered above),
    label means are available separately for b.
    """
    I = eye_like(Cov_x)
    try:
        w = torch.linalg.solve(Cov_x + l2 * I, io_cov_centered)
    except RuntimeError:
        return None, None
    # b = mu_y - wᵀ mu_x  (use the *train* means)
    b = None  # caller fills b since it knows mu_y
    return w, b


# DO REGRESSION
for feature in features:
    print(f"Fitting inner/outer models for feature(classifier): {feature}", flush=True)
    t0 = time.perf_counter()  # start timing
    for split in range(args.n_cv_splits):
        # inner models for all λ and inner_splits
        for inner_split in range(args.n_l2_cv_splits):
            C_tr = inner_train["covariance"][split][inner_split]
            mux = inner_train["mean"][split][inner_split]
            muy = inner_train["label_mean"][feature][split][inner_split]
            d_tr = inner_train["io_covariance"][feature][split][inner_split]
            for l2 in l2_strengths:
                w, b_none = fit_wb(C_tr, mux, inner_train["n_images"][split][inner_split], d_tr, None, l2)
                if w is not None:
                    b64 = muy - (w @ mux)                                 # compute bias at high precision
                    w32 = w.to(torch.float32).contiguous().cpu().pin_memory()
                    b32 = b64.to(torch.float32).contiguous().cpu().pin_memory()
                    inner_models[feature][split][inner_split][l2] = (w32, b32)
                else:
                    inner_models[feature][split][inner_split][l2] = (None, None)


        # outer models for all λ
        C_tr = train["covariance"][split]
        mux = train["mean"][split]
        muy = train["label_mean"][feature][split]
        d_tr = train["io_covariance"][feature][split]

        for l2 in l2_strengths:
            w, b_none = fit_wb(C_tr, mux, train["n_images"][split], d_tr, None, l2)
            if w is not None:
                b64 = muy - (w @ mux)
                w32 = w.to(torch.float32).contiguous().cpu().pin_memory()
                b32 = b64.to(torch.float32).contiguous().cpu().pin_memory()
                outer_models[feature][split][l2] = (w32, b32)
            else:
                outer_models[feature][split][l2] = (None, None)

    t1 = time.perf_counter()  # stop timing
    elapsed = t1 - t0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\tDone. Time: {h:02d}h {m:02d}m {s:02d}s  (~{elapsed / 3600.0:.2f} h, {elapsed / 60.0:.2f} min)",
          flush=True)


# =====================================================
# SECOND PASS
# =====================================================
# =====================================================
# PRE-STACK CLASSIFIER BANKS (FP32, CPU-pinned) FOR 2ND PASS
# One bank per OUTER split, and per (split, inner_split) for INNER.
# =====================================================
l2_list = list(l2_strengths)
outer_bank = {split: None for split in range(args.n_cv_splits)}
inner_bank = {split: {inner: None for inner in range(args.n_l2_cv_splits)}
              for split in range(args.n_cv_splits)}

# OUTER banks
for split in range(args.n_cv_splits):
    W_cols, b_cols, col_k_idx, col_l2 = [], [], [], []
    for k_idx, k in enumerate(features):
        mdl = outer_models[k][split]
        for l2 in l2_list:
            w_cpu, b_cpu = mdl[l2]
            if w_cpu is None:
                continue
            # Already FP32, contiguous, pinned from the save step
            W_cols.append(w_cpu)
            b_cols.append(b_cpu)
            col_k_idx.append(k_idx)
            col_l2.append(l2)
    W_cpu = torch.stack(W_cols, dim=1).contiguous().pin_memory()    # [D, M]
    b_cpu = torch.stack(b_cols, dim=0).contiguous().pin_memory()    # [M]
    outer_bank[split] = {
        "W_cpu": W_cpu,
        "b_cpu": b_cpu,
        "col_k_idx": torch.tensor(col_k_idx, dtype=torch.int64),    # [M]
        "col_l2": col_l2,                                           # python list of λ keys
    }

# INNER banks
for split in range(args.n_cv_splits):
    for inner in range(args.n_l2_cv_splits):
        W_cols, b_cols, col_k_idx, col_l2 = [], [], [], []
        for k_idx, k in enumerate(features):
            mdl = inner_models[k][split][inner]
            for l2 in l2_list:
                w_cpu, b_cpu = mdl[l2]
                if w_cpu is None:
                    continue
                W_cols.append(w_cpu)
                b_cols.append(b_cpu)
                col_k_idx.append(k_idx)
                col_l2.append(l2)
        W_cpu = torch.stack(W_cols, dim=1).contiguous().pin_memory()
        b_cpu = torch.stack(b_cols, dim=0).contiguous().pin_memory()
        inner_bank[split][inner] = {
            "W_cpu": W_cpu,
            "b_cpu": b_cpu,
            "col_k_idx": torch.tensor(col_k_idx, dtype=torch.int64),
            "col_l2": col_l2,
        }


# allocate counters
# INNER: per feature → split → λ → [inner_split] buckets
inner_TP = {
    c: {split: {l2: [0 for _ in range(args.n_l2_cv_splits)] for l2 in l2_strengths}
        for split in range(args.n_cv_splits)}
    for c in features
}
inner_P  = copy.deepcopy(inner_TP)
inner_TN = copy.deepcopy(inner_TP)
inner_N  = copy.deepcopy(inner_TP)

# OUTER: per feature → split → λ
outer_TP = {feature: {split: {l2: 0 for l2 in l2_strengths} for split in range(args.n_cv_splits)} for feature in features}
outer_P = copy.deepcopy(outer_TP)
outer_TN = copy.deepcopy(outer_TP)
outer_N = copy.deepcopy(outer_TP)


# run second pass identical to first (projection/subsampling/dim-red and splits identical)
for category in categories:
    print(f"Second pass (errors) category: {category}", flush=True)
    tcat0 = time.perf_counter()

    dataset = CategoryDataset(category=category)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    for batch in dataloader:
        images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch
        images = images.to(device)

        # NEW - support pixels layer option
        # Retrieve reps (supports --layer=pixels to use raw image vectors)
        if args.layer.lower() == "pixels":
            reps = images.flatten(1)
            # DEBUG:
            print(f"Reps size: {reps.size()}")
            # END DEBUG
        elif model is not None:
            with torch.no_grad():
                reps = model.forward_features(images, module_name=args.layer)  # shape [batch size, # neurons]
        else:
            reps = images.to(device)
        # END NEW
        reps = reps.to(dtype=torch.float32)

        if args.random_subsample_size is not None:
            reps = reps[:, subsample_idx]
        if args.random_projection_size is not None:
            reps = pad_to_power_of_2(reps)
            reps = hadamard_transform(reps * fjlt_D.to(reps.dtype)[None, :])
            reps = reps[:, fjlt_idx] * math.sqrt(reps.size(1) / args.random_projection_size)
        if args.global_dimensionality_reduction_style != "":
            reps = (reps - global_projection_mean.to(reps.dtype)) @ global_projector.to(reps.dtype)

        batch_size = reps.size(0)
        splits = get_splits(
            n_items=batch_size,
            method=args.outer_split_method,
            n_splits=args.n_cv_splits,
            test_ratio=args.outer_test_ratio,
            seed=args.random_seed_splits,
        )
        for split, (train_idx, test_idx) in enumerate(splits):
            reps_train = reps[train_idx]
            reps_test = reps[test_idx]

            # OUTER test errors: for each classifier(feature) k,
            # these samples are positives iff k == current category; otherwise negatives
            # OUTER test errors using pre-stacked bank
            bank = outer_bank[split]
            if "W" not in bank:
                bank["W"] = bank["W_cpu"].to(device, non_blocking=True)   # FP32 on GPU
                bank["b"] = bank["b_cpu"].to(device, non_blocking=True)

            with torch.no_grad():
                scores = reps_test @ bank["W"] + bank["b"]          # [B, M]
                y_pred = (scores >= 0.5)                            # bool [B, M]
                B = reps_test.size(0)
                tp_per_col = y_pred.sum(dim=0).tolist()
                tn_per_col = (~y_pred).sum(dim=0).tolist()

            col_k_idx_list = bank["col_k_idx"].tolist()             # [M]
            col_l2_list = bank["col_l2"]                            # [M]
            for j, k_idx in enumerate(col_k_idx_list):
                k = features[k_idx]
                l2 = col_l2_list[j]
                if k == category:
                    outer_TP[k][split][l2] += int(tp_per_col[j])
                    outer_P[k][split][l2]  += B
                else:
                    outer_TN[k][split][l2] += int(tn_per_col[j])
                    outer_N[k][split][l2]  += B


            # INNER test errors: inner splits live inside reps_train
            inner_batch_size = reps_train.size(0)
            inner_splits = get_splits(
                n_items=inner_batch_size,
                method=args.inner_split_method,
                n_splits=args.n_l2_cv_splits,
                test_ratio=args.inner_test_ratio,
                seed=args.random_seed_splits + split,
            )
            for inner_split, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
                inner_reps_test = reps_train[inner_test_idx]
                bank = inner_bank[split][inner_split]
                if "W" not in bank:
                    bank["W"] = bank["W_cpu"].to(device, non_blocking=True)
                    bank["b"] = bank["b_cpu"].to(device, non_blocking=True)

                with torch.no_grad():
                    scores = inner_reps_test @ bank["W"] + bank["b"]  # [B, M]
                    y_pred = (scores >= 0.5)
                    B = inner_reps_test.size(0)
                    tp_per_col = y_pred.sum(dim=0).tolist()
                    tn_per_col = (~y_pred).sum(dim=0).tolist()

                col_k_idx_list = bank["col_k_idx"].tolist()
                col_l2_list = bank["col_l2"]
                for j, k_idx in enumerate(col_k_idx_list):
                    k = features[k_idx]
                    l2 = col_l2_list[j]
                    if k == category:
                        inner_TP[k][split][l2][inner_split] += int(tp_per_col[j])
                        inner_P[k][split][l2][inner_split] += B
                    else:
                        inner_TN[k][split][l2][inner_split] += int(tn_per_col[j])
                        inner_N[k][split][l2][inner_split] += B

    # --- timing for SECOND PASS on this category
    tcat1 = time.perf_counter()
    elapsed = tcat1 - tcat0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"\tSecond pass finished for {category}. "
          f"Time: {h:02d}h or {m:02d}m or {s:02d}s",
          flush=True)


# ---------------------------------------------
# Aggregate errors, select λ*, and summarize
# ---------------------------------------------
for feature in features:
    print(f"Feature: {feature}", flush=True)
    for split in range(args.n_cv_splits):
        print(f"\tComputing split: {split}", flush=True)
        # inner balanced-accuracy means/std per λ
        inner_means = {}
        for l2 in l2_strengths:
            vals = []
            for inner_idx in range(args.n_l2_cv_splits):
                P = inner_P[feature][split][l2][inner_idx]
                N = inner_N[feature][split][l2][inner_idx]
                TP = inner_TP[feature][split][l2][inner_idx]
                TN = inner_TN[feature][split][l2][inner_idx]
                if (P > 0) and (N > 0):
                    bal = 0.5 * (TP / P + TN / N)
                    vals.append(bal)
            mean_val = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            std_val = float(np.std(vals)) if len(vals) > 0 else float("nan")
            results["inner_balacc_test_avg"][feature][split][l2] = mean_val
            results["inner_balacc_test_std"][feature][split][l2] = std_val
            inner_means[l2] = mean_val
            print(f"\t\t\tl2: {l2}, bal-acc: {mean_val} +- {std_val}", flush=True)

        finite_items = [(l2, v) for l2, v in inner_means.items() if np.isfinite(v)]
        best_l2 = max(finite_items, key=lambda kv: kv[1])[0] if len(finite_items) > 0 else l2_strengths[0]
        results["best_l2_strength"][feature][split] = best_l2
        print(f"\t\tBest l2: {best_l2}", flush=True)

        # Fill outer per-λ curve and outer error at best λ
        # per-λ outer balanced accuracy
        for l2 in l2_strengths:
            P = outer_P[feature][split][l2]
            N = outer_N[feature][split][l2]
            TP = outer_TP[feature][split][l2]
            TN = outer_TN[feature][split][l2]
            if (P > 0) and (N > 0):
                bal = 0.5 * (TP / P + TN / N)
            else:
                bal = None
            results["outer_balanced_accuracy_per_l2"][feature][split][l2] = bal

        # best-λ outer balanced accuracy + store counts for later
        bl2 = best_l2
        P = outer_P[feature][split][bl2]
        N = outer_N[feature][split][bl2]
        TP = outer_TP[feature][split][bl2]
        TN = outer_TN[feature][split][bl2]
        bal = (0.5 * (TP / P + TN / N)) if (P > 0 and N > 0) else None
        results["balanced_accuracy"][feature][split] = bal
        results["outer_counts_best"][feature][split] = {"TP": int(TP), "P": int(P), "TN": int(TN), "N": int(N)}

        print(f"\t\tbalanced acc (outer, best λ): {results['balanced_accuracy'][feature][split]}", flush=True)

# summarize over splits
for feature in features:
    vals = [results["balanced_accuracy"][feature][split]
            for split in range(args.n_cv_splits)
            if results["balanced_accuracy"][feature][split] is not None]
    if len(vals) == 0:
        results["balanced_accuracy_avg"][feature] = None
        results["balanced_accuracy_std"][feature] = None
    else:
        results["balanced_accuracy_avg"][feature] = float(np.mean(vals))
        results["balanced_accuracy_std"][feature] = float(np.std(vals))

# store labels variance “full”
for feature in features:
    results["labels_variance_full"][feature] = full["label_variance"][feature]

# -----------------------
# Save & pretty print
# -----------------------
results_numpy = convert_tensors_to_numpy(results)

# bbox normalization information (if available)
# (we fetch from the last dataloader; same as original)
normalization_bbox_mean, normalization_bbox_std = get_bbox_stats_from_loader(dataloader)
results_numpy["normalization_bbox_mean"] = normalization_bbox_mean
results_numpy["normalization_bbox_std"] = normalization_bbox_std

# global dim-red bookkeeping
if args.global_dimensionality_reduction_style != "":
    results_numpy["args"]["global_dimensionality_reduction_dimensions"] = global_d
    results_numpy["args"]["global_dimensionality_reduction_strength"] = float(global_explained_variance)

print("\n### SUMMARY (balanced accuracy) ###\n")
for feature in features:
    avg = results_numpy["balanced_accuracy_avg"][feature]
    std = results_numpy["balanced_accuracy_std"][feature]
    print(f"Feature(classifier): {feature}")
    print(f"\tbalanced accuracy (outer, best λ) avg: {avg}")
    print(f"\tbalanced accuracy (outer, best λ) std: {std}")

checkpoint_dir = os.path.dirname(args.checkpoint)
results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
os.makedirs(results_dir, exist_ok=True)

# naming style preserved (+rnd_sub/+rnd_proj/+global strings)
rnd_sub_string = (f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
                  if args.random_subsample_size is not None else "")
rnd_proj_string = (f"_RndProj_size{args.random_projection_size}_seed{args.random_projection_seed}"
                   if args.random_projection_size is not None else "")
if args.global_dimensionality_reduction_style != "":
    global_dim_red_string = (f"_GlobalRed_{args.global_dimensionality_reduction_style}"
                             f"_Dim{global_d}_KV{global_explained_variance}")
else:
    global_dim_red_string = ""

dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))
results_file = os.path.join(
    results_dir,
    f"{args.job_id}_CVclassification_{dataset_name}_{args.layer}"
    f"_nOutSplit{args.n_cv_splits}_nInSplit{args.n_l2_cv_splits}"
    + rnd_sub_string + rnd_proj_string + global_dim_red_string + ".pkl"
)

with open(results_file, 'wb') as f:
    pickle.dump(results_numpy, f)

print("Classification concluded. Exit", flush=True)
exit()
