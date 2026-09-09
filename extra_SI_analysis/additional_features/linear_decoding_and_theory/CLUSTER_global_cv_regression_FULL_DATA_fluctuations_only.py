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
from networkx.algorithms.centrality import katz_centrality_numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDITIONAL_FEATURES_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(ADDITIONAL_FEATURES_DIR, "CNN_trainer")
sys.path.insert(0, ADDITIONAL_FEATURES_DIR)
sys.path.insert(0, MODEL_DIR)
import LIB_model
import json
import hashlib
from torch.utils.data import Subset


DEFAULT_BBOX_FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
ADDITIONAL_TARGET_FEATURES = [
    "bbox_area",
    "mean_srgb_luma",
    "local_srgb_luma_contrast",
    "mean_hsv_saturation",
]
ALL_TARGET_FEATURES = DEFAULT_BBOX_FEATURES + ADDITIONAL_TARGET_FEATURES

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
parser.add_argument(
    "--target_features",
    nargs="+",
    default=None,
    choices=ALL_TARGET_FEATURES,
    help="SDXL regression targets, in the exact order to decode. "
         "Defaults to the four bounding-box features.",
)

parser.add_argument("--results_folder_name", type=str,
                    default="./",
                    help="Name of the folder in the checkpoint folder where we will store the results.")
parser.add_argument("--manifold_subsample_size", type=int, default=None,
                    help="Number of categories to keep via random subsampling. Default None = no subsampling.")
parser.add_argument("--manifold_subsample_seed", type=int, default=1,
                    help="Random seed used to pick the subsampled categories.")
parser.add_argument("--manifold_subsampling_grouping_json", type=str, default=None,
                    help="Path to a JSON file with entries like {'category': ..., 'macro': ...}. "
                         "If set, category subsampling is balanced across macro groups.")

parser.add_argument("--exclude_categories_Q", type=int, default=0,
                    help="If >0, define a deterministic set of Q held-out categories (sampled from the on-disk list).")
parser.add_argument("--exclude_categories_seed", type=int, default=1,
                    help="Seed for sampling the held-out categories.")
parser.add_argument("--exclude_categories_eval_mode", type=str, default="none",
                    choices=["none", "only_excluded", "exclude_excluded"],
                    help="How to use the held-out categories at eval time: "
                         "'none' (ignore), 'only_excluded' (ONLY held-out), "
                         "'exclude_excluded' (ALL EXCEPT held-out).")

parser.add_argument("--random_projection_size", type=int, default=None,
                    help="The size of the random projection. Default None corresponds to no projection.")
parser.add_argument("--random_projection_seed", type=int, default=1,
                    help="Random seed value. E.g. used for generating the random projection")
parser.add_argument("--random_subsample_size", type=int, default=None,
                    help="Number of neurons to keep via random subsampling. Default None = no subsampling.")
parser.add_argument("--random_subsample_seed", type=int, default=1,
                    help="Random seed used to pick the subsampled neuron indices.")
parser.add_argument(
    "--skip_random_subsampling_or_projection_when_size_match_original",
    action="store_true",
    help="If set, skip random subsampling and/or random projection when the requested size equals the current number of neurons."
)
parser.add_argument("--n_cv_splits", type=int, default=1, help="cv splits over which performance is cross-validated")
parser.add_argument("--n_l2_cv_splits", type=int, default=1, help="inner cv splits over which the best l2 is selected")

parser.add_argument(
    "--l2_min_exp", type=float, default=-6,
    help="Minimum base-10 exponent for λ (i.e. λ=10^l2_min_exp)."
)
parser.add_argument(
    "--l2_max_exp", type=float, default=2,
    help="Maximum base-10 exponent for λ (i.e. λ=10^l2_max_exp)."
)
parser.add_argument(
    "--n_l2", type=int, default=9,
    help="Number of λ values to sample between l2_min_exp and l2_max_exp (inclusive)."
)
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images.")
parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
parser.add_argument(
    "--outer_split_method",
    type=str,
    choices=["kfold", "random"],
    default="kfold",
    help="‘kfold’: equal‐sized folds; ‘random’: random train/test splits"
)
parser.add_argument(
    "--inner_split_method",
    type=str,
    choices=["kfold", "random"],
    default="kfold",
    help="‘kfold’: equal‐sized folds; ‘random’: random train/test splits"
)
parser.add_argument(
    "--outer_test_ratio",
    type=float,
    default=0.2,
    help="When using --split_method random, fraction of data to hold out as test"
)
parser.add_argument(
    "--inner_test_ratio",
    type=float,
    default=0.2,
    help="When using --split_method random, fraction of data to hold out as test"
)
parser.add_argument(
    "--random_seed_splits",
    type=int,
    default=43,
    help="Seed for random splits"
)
parser.add_argument("--global_dimensionality_reduction_style", type=str,
                    default="",
                    help="string identifier describing the dimensionality reduction type. Options are 'percentage', 'dimensions', or 'neurons'. "
                         "Default is no dimensionality reduction")
parser.add_argument("--global_dimensionality_reduction_strength", type=float, default=100,
                    help="Percentage of total covariance to explain (e.g., 90 for 90%). Only applicable to style percentage")
parser.add_argument("--global_dimensionality_reduction_dimensions", type=int, default=2048,
                    help="Number of dimensions to keep. Only applicable to style dimensions")
parser.add_argument("--global_dimensionality_reduction_file", type=str,
                    default="./boh.pkl",
                    help="Path to the file containing the global statistics.")
parser.add_argument("--n_subsampled_images", type=int, default=None,
                    help="If set, the maximum number of images to use PER CATEGORY.")
parser.add_argument("--images_subsampling_seed", type=int, default=1,
                    help="Random seed for per-category image subsampling.")

parser.add_argument('--augment_dihedral', action='store_true',
                    help="If set, augment each image with the 8 dihedral (D4) transforms (4 rotations x optional horizontal flip). "
                         "Splits are done per-base image to avoid leakage.")

args = parser.parse_args()

AUGMENT_GROUP_SIZE = 8 if getattr(args, "augment_dihedral", False) else 1


def print_args(args):
    """
    Nicely prints out every command-line argument and its value.
    """
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:20s}: {value!r}")
    print("=========================", flush=True)


def get_bbox_stats_from_loader(dataloader):
    """
    If the underlying dataset has attributes `bbox_mean` and `bbox_std`,
    returns them as a tuple of torch.Tensor (or numpy.ndarray).
    Otherwise, returns (None, None).
    """
    # most DataLoader subclasses expose the dataset as `.dataset`
    ds = getattr(dataloader, "dataset", None)
    # if you ever wrap with Subset or ConcatDataset, you can unwrap like this:
    if hasattr(ds, "dataset"):
        ds = ds.dataset

    mean = getattr(ds, "bbox_mean", None)
    std  = getattr(ds, "bbox_std", None)
    return mean, std


def get_splits(n_items, method="kfold", n_splits=5, test_ratio=0.2, seed=None, group_size=None):
    """
    Returns a list of (train_idx, test_idx) tuples, each of which is a torch.LongTensor.

    If group_size > 1, splits are done in "base index" space (n_items/group_size),
    then expanded so that all items in each contiguous block of size=group_size
    stay together (prevents augmentation leakage).
    """
    if group_size is None:
        group_size = AUGMENT_GROUP_SIZE
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError(f"Invalid group_size={group_size}")

    if n_items % group_size != 0:
        raise ValueError(
            f"n_items={n_items} not divisible by group_size={group_size}. "
            "If using augmentation, ensure dataset ordering is grouped in fixed blocks."
        )

    n_groups = n_items // group_size
    indices = list(range(n_groups))

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    def expand(group_idx: torch.Tensor) -> torch.Tensor:
        if group_size == 1:
            return group_idx.to(dtype=torch.long)
        group_idx = group_idx.to(dtype=torch.long)
        offsets = torch.arange(group_size, dtype=torch.long)
        return (group_idx[:, None] * group_size + offsets[None, :]).reshape(-1)

    if method == "kfold":
        split_size = int(math.ceil(n_groups / n_splits))
        folds = []
        for fold in range(n_splits):
            start = fold * split_size
            end = min(start + split_size, n_groups)
            test_g = torch.tensor(indices[start:end], dtype=torch.long)
            train_g = torch.tensor(indices[:start] + indices[end:], dtype=torch.long)
            folds.append((expand(train_g), expand(test_g)))
        return folds

    elif method == "random":
        folds = []
        test_size = int(math.floor(n_groups * test_ratio))
        for _ in range(n_splits):
            perm = indices.copy()
            random.shuffle(perm)
            test_g = torch.tensor(perm[:test_size], dtype=torch.long)
            train_g = torch.tensor(perm[test_size:], dtype=torch.long)
            folds.append((expand(train_g), expand(test_g)))
        return folds

    else:
        raise ValueError(f"Unknown split method {method}")


def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible experiments.

    Args:
        seed (int): the seed to use for all libraries (default: 42).
    """
    # 1. Python built-ins
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. CuDNN backend settings for reproducibility
    #    - deterministic: only deterministic algorithms
    #    - benchmark: disable auto-benchmark to avoid non-deterministic selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_tensors_to_numpy(d):
    if isinstance(d, dict):
        return {key: convert_tensors_to_numpy(value) for key, value in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.cpu().numpy()  # Convert tensor to numpy (using .cpu() if it's on GPU)
    else:
        return d  # Leave non-tensor items unchanged


def accumulate(old, addition):
    """
    If `old` is None, return a clone of `addition`;
    otherwise return old + addition.
    For non-Tensors (e.g. ints) we deepcopy instead of clone.
    """
    if old is None:
        if hasattr(addition, "clone"):
            return addition.clone()
        else:
            # e.g. addition is int, float, list, dict, …
            return copy.deepcopy(addition)
    else:
        return old + addition


def explained_variance_ratio(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    """
    Returns the fraction of total variance explained by the top `dims_kept` eigenvalues.

    Args:
        eigs (torch.Tensor): 1D tensor of eigenvalues (assumed non-negative).
        dims_kept (int): Number of top eigenvalues/components kept.

    Returns:
        torch.Tensor: Scalar tensor with the explained variance ratio in [0, 1].
                      If total variance is 0, returns NaN.
    """
    if eigs.numel() == 0 or dims_kept <= 0:
        return torch.tensor(0.0, dtype=eigs.dtype, device=eigs.device)

    eigs_sorted = torch.sort(eigs, descending=True).values
    k = min(int(dims_kept), eigs_sorted.numel())

    explained = eigs_sorted[:k].sum()
    total = eigs_sorted.sum()

    # If total == 0, this yields NaN rather than raising.
    return explained / total


def explained_variance_percent(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    """
    Same as above but as a percentage (0–100).
    """
    return explained_variance_ratio(eigs, dims_kept) * 100


def explained_variance_dim(eigs, threshold=0.9):
    """
    Computes the number of top eigenvalues needed to explain a given
    fraction (threshold) of the total variance.

    Args:
        eigs (torch.Tensor): 1D tensor of eigenvalues (assumed non-negative).
        threshold (float): Fraction of total variance to be explained (e.g., 0.9 for 90%).

    Returns:
        int: Number of top eigenvalues needed to reach the threshold.
    """
    eigs_sorted = torch.sort(eigs, descending=True).values
    cumulative = torch.cumsum(eigs_sorted, dim=0)
    total = torch.sum(eigs_sorted)
    explained_ratio = cumulative / total

    # Find the first index where explained_ratio >= threshold
    indices = (explained_ratio >= threshold).nonzero(as_tuple=False)
    if indices.numel() == 0:
        return len(eigs_sorted)  # If threshold is never reached, return full dimensionality
    return indices[0].item() + 1


# Print arguments
print_args(args)

# retrieve device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

from pathlib import Path
dataset_path = Path(args.dataset_folder)
# If the user gives an .h5 file, use its parent folder name.
# Otherwise (directory case), use the directory name itself.
if dataset_path.is_file() or dataset_path.suffix.lower() == ".h5":
    dataset_folder_name = dataset_path.parent.name
else:
    dataset_folder_name = dataset_path.name

# define the l2 strengths to use

# build λ grid spaced logarithmically from 10^min_exp to 10^max_exp
l2_strengths = np.logspace(
    args.l2_min_exp,
    args.l2_max_exp,
    num=args.n_l2,
    base=10.0
).tolist()

print(f"Sampling l2 strengths: {l2_strengths}", flush=True)

# load the model and transforms
if os.path.basename(os.path.normpath(args.checkpoint)).lower() == "resnet50":
    # instantiate the model from scratch, which contains a pretrained resnet50 backbone
    model = LIB_model.ResNetBBoxModel(
        resnet_version="resnet50",
        # we can leave the default heads here; we only need forward_features()
    ).to(device)
    model.eval()

    # manually define the transform
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
    model = ckpt["model"].to(device)
    model.eval()
    transform = ckpt["transform"]

from functools import partial

print(f"Dataset: {dataset_folder_name}", flush=True)
is_sdxl_dataset = dataset_folder_name in [
    "SDXL_dataset_test",
    "SDXL_dataset_train",
    "SDXL_dataset_validation",
]

if is_sdxl_dataset or dataset_folder_name == "DicarloAsGenai_public":
    features = (
        list(args.target_features)
        if is_sdxl_dataset and args.target_features is not None
        else list(DEFAULT_BBOX_FEATURES)
    )
    from LIB_dataloader import CategoryDatasetSDXL
    CategoryDataset = partial(
        CategoryDatasetSDXL,
        dataset_folder=args.dataset_folder,
        transform=transform,
        augment_dihedral=args.augment_dihedral,
        target_features=args.target_features if is_sdxl_dataset else None,
    )

elif dataset_folder_name in [
    "representations_DicarloAsGenai_public_IT",
    "representations_DicarloAsGenai_public_V4",
]:
    from LIB_dataloader import CategoryDatasetRepresentation
    CategoryDataset = partial(
        CategoryDatasetRepresentation,
        dataset_folder=args.dataset_folder,
        transform=transform,
    )
    features = list(DEFAULT_BBOX_FEATURES)

else:
    raise ValueError(f"Dataset {dataset_folder_name} is not implemented.")

if args.target_features is not None and not is_sdxl_dataset:
    raise ValueError("--target_features is supported only for SDXL datasets.")

print(f"Regression target features: {features}", flush=True)
# retrieve categories (sorted for determinism)
all_categories = [
    p.name
    for p in (dataset_path if dataset_path.is_dir() else dataset_path.parent).iterdir()
    if p.is_dir() and p.name.lower() != "crashes"
]
all_categories.sort()

excluded_categories = []
if args.exclude_categories_Q is not None and args.exclude_categories_Q > 0:
    if args.exclude_categories_Q >= len(all_categories):
        raise ValueError(
            f"--exclude_categories_Q={args.exclude_categories_Q} must be < number of categories "
            f"({len(all_categories)}) found on disk."
        )
    rng = random.Random(args.exclude_categories_seed)
    excluded_categories = rng.sample(all_categories, args.exclude_categories_Q)
    excluded_categories.sort()

if args.exclude_categories_eval_mode == "only_excluded":
    if not excluded_categories:
        raise ValueError("exclude_categories_eval_mode='only_excluded' requires --exclude_categories_Q > 0")
    all_categories = excluded_categories
elif args.exclude_categories_eval_mode == "exclude_excluded":
    if not excluded_categories:
        raise ValueError("exclude_categories_eval_mode='exclude_excluded' requires --exclude_categories_Q > 0")
    excluded_set = set(excluded_categories)
    all_categories = [c for c in all_categories if c not in excluded_set]

if args.exclude_categories_eval_mode != "none":
    print(f"[Category holdout] mode={args.exclude_categories_eval_mode} | "
          f"Q={args.exclude_categories_Q} seed={args.exclude_categories_seed} | "
          f"excluded={excluded_categories}", flush=True)


if args.manifold_subsample_size is not None:
    seed_everything(args.manifold_subsample_seed)

    # never request more than what exists
    K = min(args.manifold_subsample_size, len(all_categories))

    if args.manifold_subsampling_grouping_json is not None:
        # ----- load grouping (expects list of dicts with 'category' and 'macro') -----
        with open(args.manifold_subsampling_grouping_json, "r") as f:
            grouping = json.load(f)  # e.g. [{"category": "cat", "macro": "ANIMALS & WILDLIFE", ...}, ...]
        cat2macro = {row["category"]: row["macro"] for row in grouping}

        # ----- build per-macro pools, but only for categories present on disk -----
        macros_to_cats = {}
        for c in all_categories:
            m = cat2macro.get(c, None)
            if m is None:
                continue  # skip categories not present in the JSON mapping
            macros_to_cats.setdefault(m, []).append(c)

        if len(macros_to_cats) == 0:
            raise ValueError("No categories from the grouping JSON match the categories found on disk.")

        # ----- deterministic per-macro shuffles (prefix stability comes from fixed order + fixed seed) -----
        for m in macros_to_cats:
            random.shuffle(macros_to_cats[m])

        # deterministic macro visitation order
        macros = sorted(macros_to_cats.keys())

        # ----- round-robin across macros; on exhaustion, draw from any other non-empty macro at random -----
        selected = []
        step = 0
        while len(selected) < K:
            target_macro = macros[step % len(macros)]
            if macros_to_cats[target_macro]:
                selected.append(macros_to_cats[target_macro].pop(0))
            else:
                non_empty = [m for m in macros if macros_to_cats[m]]
                if not non_empty:
                    break  # nothing left
                # choose another macro at random (still deterministic due to seed)
                m_pick = random.choice(non_empty)
                selected.append(macros_to_cats[m_pick].pop(0))
            step += 1

        all_categories = selected
    else:
        # original global random picking (keeps the prefix-stability property)
        random.shuffle(all_categories)  # deterministic given the seed
        all_categories = all_categories[:K]


categories = all_categories

n_categories = len(categories)

# Define quantities dictionary
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
    "mean": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
    "n_images": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
    "io_covariance": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
    "label_variance": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
    "label_mean": {feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)} for feature in features},
}

inner_test = copy.deepcopy(inner_train)

fjlt_idx = None
fjlt_D = None

subsample_idx = None

max_label_test = {feature: {split: -float("inf") for split in range(args.n_cv_splits)} for feature in features}
min_label_test = {feature: {split:  float("inf") for split in range(args.n_cv_splits)} for feature in features}
max_label_train = {feature: {split: -float("inf") for split in range(args.n_cv_splits)} for feature in features}
min_label_train = {feature: {split:  float("inf") for split in range(args.n_cv_splits)} for feature in features}

# load global dimensionality reduction
if args.global_dimensionality_reduction_style != "":
    with open(args.global_dimensionality_reduction_file, "rb") as f:
        global_stats = pickle.load(f)
    global_eigenvalues = torch.tensor(global_stats["covariance_eigenvalues"], device=device, dtype=torch.float64)  # ordered from largest to smallest eigenvalue
    global_eigenvectors = torch.tensor(global_stats["covariance_eigenvectors"], device=device, dtype=torch.float64)  # shape [component, eigenvalue number], ordered from largest to smallest eigenvalue
    global_projection_mean = torch.tensor(global_stats["mean"], device=device, dtype=torch.float64)
    if args.global_dimensionality_reduction_style == "dimensions":
        global_d = args.global_dimensionality_reduction_dimensions
        global_explained_variance = explained_variance_percent(global_eigenvalues, global_d)
        global_projector = global_eigenvectors[:, :global_d]
    elif args.global_dimensionality_reduction_style == "percentage":
        global_explained_variance = args.global_dimensionality_reduction_strength
        global_d = explained_variance_dim(global_eigenvalues, threshold=(args.global_dimensionality_reduction_strength/100))
        global_projector = global_eigenvectors[:, :global_d]
    elif args.global_dimensionality_reduction_style == "neurons":
        # Per-neuron variance is the diagonal of C = V diag(λ) V^T.
        # Using einsum explicitly: diag(C)_neuron = Σ_eig λ[eig] * V[neuron, eig]^2
        neuron_var = einsum(global_eigenvectors ** 2, global_eigenvalues,
                            "neuron eig, eig -> neuron")
        # Pick top-K neurons by variance
        global_d = min(args.global_dimensionality_reduction_dimensions, neuron_var.numel())
        top_var, top_idx = torch.topk(neuron_var, k=global_d, largest=True, sorted=True)
        # % of total variance (trace(C)) captured by the selected neurons — for logging
        total_var = neuron_var.sum()
        global_explained_variance = (top_var.sum() / total_var * 100.0).item() if total_var > 0 else float("nan")
        # Build a selector projector (N × K): selects those neurons
        N = neuron_var.numel()
        global_projector = torch.zeros((N, global_d), dtype=torch.float64, device=device)
        global_projector[top_idx, torch.arange(global_d, device=device)] = 1.0
    else:
        global_d = -1
        global_explained_variance = 100.0
        global_projector = global_eigenvectors[:, :global_d]
    print(f"Initializing global projector. Keeping first {global_d} dimensions, or {global_explained_variance}% of the variance")


# loop over the categories: generate representations and do regression
for category in categories:
    print(f"Processing category: {category}", flush=True)

    # define the dataset and dataloader for that specific category
    dataset = CategoryDataset(category=category)  # <— only category here

    # Optional per-category image subsampling (base-image-aware if augmentation is enabled)
    if args.n_subsampled_images is not None:
        N = len(dataset)
        group_size = AUGMENT_GROUP_SIZE
        if group_size < 1:
            raise ValueError(f"Invalid AUGMENT_GROUP_SIZE={group_size}")
        if N % group_size != 0:
            raise ValueError(
                f"len(dataset)={N} not divisible by AUGMENT_GROUP_SIZE={group_size}. "
                "If you enabled augmentation, make sure the dataset returns grouped blocks of size 8."
            )

        N_base = N // group_size
        k_base = max(0, min(int(args.n_subsampled_images), N_base))
        if k_base < N_base:
            # Derive a per-category seed so choices don't depend on loop order
            cat_key = f"{args.images_subsampling_seed}:{category}".encode("utf-8")
            cat_seed = int.from_bytes(hashlib.sha256(cat_key).digest()[:4], "big")
            rng = np.random.RandomState(cat_seed)

            subset_base = rng.choice(np.arange(N_base), size=k_base, replace=False)
            subset_base.sort()  # keep original (non-shuffled) dataloader order

            if group_size == 1:
                subset_idx = subset_base
            else:
                subset_idx = (subset_base[:, None] * group_size + np.arange(group_size)[None, :]).reshape(-1)

            dataset = Subset(dataset, subset_idx.tolist())

    # -> define the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # define local dictionaries
    local_full = {
        "covariance": None,
        "mean": None,
        "n_images": None,
        "io_covariance": {feature: None for feature in features},
        "label_variance": {feature: None for feature in features},
        "label_mean": {feature: None for feature in features},
    }

    local_train = {
        "covariance": {split: None for split in range(args.n_cv_splits)},
        "mean": {split: None for split in range(args.n_cv_splits)},
        "n_images": {split: None for split in range(args.n_cv_splits)},
        "io_covariance": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
        "label_variance": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
        "label_mean": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    }

    local_test = copy.deepcopy(local_train)

    local_inner_train = {
        "covariance": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in
                       range(args.n_cv_splits)},
        "mean": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
        "n_images": {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)},
        "io_covariance": {
            feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)}
            for feature in features},
        "label_variance": {
            feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)}
            for feature in features},
        "label_mean": {
            feature: {split: {inner: None for inner in range(args.n_l2_cv_splits)} for split in range(args.n_cv_splits)}
            for feature in features},
    }

    local_inner_test = copy.deepcopy(local_inner_train)

    # compute running quantities looping over batches
    for batch in dataloader:
        # retrieve elements of the batch
        images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch

        gt_bboxes = gt_bboxes.to(device)
        gt_category_idxs = gt_category_idxs.to(device)
        images = images.to(device)

        if args.layer.lower() == "pixels":
            reps = images.flatten(1)
        elif model is not None:
            with torch.no_grad():
                reps = model.forward_features(images, module_name=args.layer)
        else:
            reps = images.to(device)

        reps = reps.to(dtype=torch.float64)

        do_random_subsample = (args.random_subsample_size is not None) and not (
            args.skip_random_subsampling_or_projection_when_size_match_original
            and (args.random_subsample_size == reps.size(1))
        )

        if (subsample_idx is None) and do_random_subsample:
            seed_everything(args.random_subsample_seed)
            N_orig = reps.size(1)
            if args.random_subsample_size > N_orig:
                raise ValueError(
                    f"--random_subsample_size ({args.random_subsample_size}) cannot exceed number of neurons ({N_orig}).")
            # pick K distinct neurons uniformly at random (without replacement)
            subsample_idx = torch.randperm(N_orig, device=device)[:args.random_subsample_size]

        # apply random neuron subsampling if requested
        if do_random_subsample:
            reps = reps[:, subsample_idx]

        # define the random projector if not yet defined, and if random projection is required
        do_random_projection = (args.random_projection_size is not None) and not (
                args.skip_random_subsampling_or_projection_when_size_match_original
                and (args.random_projection_size == reps.size(1))
        )

        if (fjlt_D is None) and (fjlt_idx is None) and do_random_projection:
            # print("Generating the random projection...", flush=True)
            seed_everything(args.random_projection_seed)

            N_orig = reps.size(1)  # old length
            if is_a_power_of_2(N_orig):
                N_power2 = N_orig
            else:
                N_power2 = next_power_of_2(N_orig)

            # ±1 diagonal of length **N_power2**
            D = (torch.randint(0, 2, (N_power2,), device=device) * 2 - 1).to(torch.float64)
            # k random output coordinates drawn from the **padded** length
            idx = torch.randperm(N_power2, device=device)[:args.random_projection_size]
            # store for reuse
            fjlt_D = D
            fjlt_idx = idx

        # project data if random projection is required
        if do_random_projection:
            # print("Performing the random projection...", flush=True)

            # (a) zero-pad the data to a power-of-two once
            reps = pad_to_power_of_2(reps)
            # (b) apply D
            reps_d = reps * fjlt_D[None, :]
            # (c) Hadamard (already orthonormal, no extra 1/√n needed)
            reps_h = hadamard_transform(reps_d)
            # (d) subsample + scale by √(n/k) with n = padded length
            reps = reps_h[:, fjlt_idx] * np.sqrt(N_power2 / args.random_projection_size)

        # do global projection if required
        if args.global_dimensionality_reduction_style != "":
            reps = (reps - global_projection_mean) @ global_projector

        # <editor-fold desc="DUPDATE FULL">
        # update n_images
        reps_full = reps
        n_images_full = reps_full.size(0)
        local_full["n_images"] = accumulate(local_full["n_images"], n_images_full)

        # update covariance
        cov_full_addition = einsum(reps_full, reps_full, "image neuron1, image neuron2 -> neuron1 neuron2")
        local_full["covariance"] = accumulate(local_full["covariance"], cov_full_addition)

        # update mean
        mean_full_addition = torch.sum(reps_full, dim=0)
        local_full["mean"] = accumulate(local_full["mean"], mean_full_addition)

        # update feature-specific quantities
        for feat_idx, feature in enumerate(features):
            labels = gt_bboxes[:, feat_idx].clone()  # shape [batch size]
            # convert to float64
            labels_full = labels.to(dtype=torch.float64)

            # update label mean
            labels_mean_full_addition = torch.sum(labels_full)
            local_full["label_mean"][feature] = accumulate(local_full["label_mean"][feature],
                                                                   labels_mean_full_addition)

            # update labels variance
            labels_var_full_addition = torch.sum(labels_full ** 2)
            local_full["label_variance"][feature] = accumulate(
                local_full["label_variance"][feature],
                labels_var_full_addition)

            # update io_covariance
            io_full_addition = einsum(reps_full, labels_full, "image neuron, image -> neuron")
            local_full["io_covariance"][feature] = accumulate(local_full["io_covariance"][feature],
                                                                      io_full_addition)
        # END UPDATE FULL
        # </editor-fold>

        # outer cv loop
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

            # update n_images
            n_images_train = reps_train.size(0)
            local_train["n_images"][split] = accumulate(local_train["n_images"][split], n_images_train)

            n_images_test = reps_test.size(0)
            local_test["n_images"][split] = accumulate(local_test["n_images"][split], n_images_test)

            # update covariance
            cov_train_addition = einsum(reps_train, reps_train, "image neuron1, image neuron2 -> neuron1 neuron2")
            local_train["covariance"][split] = accumulate(local_train["covariance"][split], cov_train_addition)

            cov_test_addition = einsum(reps_test, reps_test, "image neuron1, image neuron2 -> neuron1 neuron2")
            local_test["covariance"][split] = accumulate(local_test["covariance"][split], cov_test_addition)

            # update mean
            mean_train_addition = torch.sum(reps_train, dim=0)
            local_train["mean"][split] = accumulate(local_train["mean"][split], mean_train_addition)

            mean_test_addition = torch.sum(reps_test, dim=0)
            local_test["mean"][split] = accumulate(local_test["mean"][split], mean_test_addition)

            # update feature-specific quantities
            for feat_idx, feature in enumerate(features):
                labels = gt_bboxes[:, feat_idx].clone()  # shape [batch size]
                # convert to float64
                labels = labels.to(dtype=torch.float64)

                # divide into train and test
                labels_train = labels[train_idx]
                labels_test = labels[test_idx]

                # update max labels
                max_train = torch.max(labels_train)
                min_train = torch.min(labels_train)
                max_test = torch.max(labels_test)
                min_test = torch.min(labels_test)
                max_label_train[feature][split] = max(max_label_train[feature][split], max_train)
                min_label_train[feature][split] = min(min_label_train[feature][split], min_train)
                max_label_test[feature][split] = max(max_label_test[feature][split], max_test)
                min_label_test[feature][split] = min(min_label_test[feature][split], min_test)

                # update label mean
                labels_mean_train_addition = torch.sum(labels_train)
                local_train["label_mean"][feature][split] = accumulate(local_train["label_mean"][feature][split],
                                                                       labels_mean_train_addition)

                labels_mean_test_addition = torch.sum(labels_test)
                local_test["label_mean"][feature][split] = accumulate(local_test["label_mean"][feature][split],
                                                                      labels_mean_test_addition)

                # update labels variance
                labels_var_train_addition = torch.sum(labels_train ** 2)
                local_train["label_variance"][feature][split] = accumulate(local_train["label_variance"][feature][split],
                                                                           labels_var_train_addition)

                labels_var_test_addition = torch.sum(labels_test ** 2)
                local_test["label_variance"][feature][split] = accumulate(local_test["label_variance"][feature][split],
                                                                          labels_var_test_addition)

                # update io_covariance
                io_train_addition = einsum(reps_train, labels_train, "image neuron, image -> neuron")
                local_train["io_covariance"][feature][split] = accumulate(local_train["io_covariance"][feature][split],
                                                                          io_train_addition)

                io_test_addition = einsum(reps_test, labels_test, "image neuron, image -> neuron")
                local_test["io_covariance"][feature][split] = accumulate(local_test["io_covariance"][feature][split],
                                                                         io_test_addition)

            # loop over inner splits (those needed to determine l2)
            # outer cv loop
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

                # update n_images
                inner_n_images_train = inner_reps_train.size(0)
                local_inner_train["n_images"][split][inner_split] = accumulate(local_inner_train["n_images"][split][inner_split], inner_n_images_train)

                inner_n_images_test = inner_reps_test.size(0)
                local_inner_test["n_images"][split][inner_split] = accumulate(local_inner_test["n_images"][split][inner_split], inner_n_images_test)

                # update covariance
                inner_cov_train_addition = einsum(inner_reps_train, inner_reps_train, "image neuron1, image neuron2 -> neuron1 neuron2")
                local_inner_train["covariance"][split][inner_split] = accumulate(local_inner_train["covariance"][split][inner_split], inner_cov_train_addition)

                inner_cov_test_addition = einsum(inner_reps_test, inner_reps_test, "image neuron1, image neuron2 -> neuron1 neuron2")
                local_inner_test["covariance"][split][inner_split] = accumulate(local_inner_test["covariance"][split][inner_split], inner_cov_test_addition)

                # update mean
                inner_mean_train_addition = torch.sum(inner_reps_train, dim=0)
                local_inner_train["mean"][split][inner_split] = accumulate(local_inner_train["mean"][split][inner_split], inner_mean_train_addition)

                inner_mean_test_addition = torch.sum(inner_reps_test, dim=0)
                local_inner_test["mean"][split][inner_split] = accumulate(local_inner_test["mean"][split][inner_split], inner_mean_test_addition)

                # update feature-specific quantities
                for feat_idx, feature in enumerate(features):
                    # define outer labels
                    labels = gt_bboxes[:, feat_idx].clone()  # shape [batch size]
                    # convert to float64
                    labels = labels.to(dtype=torch.float64)
                    # divide into train and test
                    labels_train = labels[train_idx]
                    labels_test = labels[test_idx]
                    # define inner labels
                    inner_labels_train = labels_train[inner_train_idx]
                    inner_labels_test = labels_train[inner_test_idx]

                    # update label mean
                    inner_labels_mean_train_addition = torch.sum(inner_labels_train)
                    local_inner_train["label_mean"][feature][split][inner_split] = accumulate(local_inner_train["label_mean"][feature][split][inner_split],
                                                                                              inner_labels_mean_train_addition)

                    inner_labels_mean_test_addition = torch.sum(inner_labels_test)
                    local_inner_test["label_mean"][feature][split][inner_split] = accumulate(local_inner_test["label_mean"][feature][split][inner_split],
                                                                                             inner_labels_mean_test_addition)

                    # update labels variance
                    inner_labels_var_train_addition = torch.sum(inner_labels_train ** 2)
                    local_inner_train["label_variance"][feature][split][inner_split] = accumulate(local_inner_train["label_variance"][feature][split][inner_split],
                                                                                                  inner_labels_var_train_addition)

                    inner_labels_var_test_addition = torch.sum(inner_labels_test ** 2)
                    local_inner_test["label_variance"][feature][split][inner_split] = accumulate(local_inner_test["label_variance"][feature][split][inner_split],
                                                                                                 inner_labels_var_test_addition)

                    # update io_covariance
                    inner_io_train_addition = einsum(inner_reps_train, inner_labels_train, "image neuron, image -> neuron")
                    local_inner_train["io_covariance"][feature][split][inner_split] = accumulate(local_inner_train["io_covariance"][feature][split][inner_split],
                                                                                                 inner_io_train_addition)

                    inner_io_test_addition = einsum(inner_reps_test, inner_labels_test, "image neuron, image -> neuron")
                    local_inner_test["io_covariance"][feature][split][inner_split] = accumulate(local_inner_test["io_covariance"][feature][split][inner_split],
                                                                                                inner_io_test_addition)

    # -----------------------------------------------------------------------------
    # Local (per-category) centering using FULL-category means (single-pass).
    # We accumulated raw (uncentered) sums for each subset; now transform them into
    # sums over δx = x - m_full and δy = y - ybar_full, where m_full and ybar_full
    # are computed from the full category.
    # -----------------------------------------------------------------------------

    # Full-category means (per category)
    N_full = local_full["n_images"]
    m_full = local_full["mean"] / N_full
    ybar_full = {feature: (local_full["label_mean"][feature] / N_full) for feature in features}

    # --- center FULL statistics (δ mean is exactly 0) ---
    Sx_full = local_full["mean"].clone()
    local_full["covariance"] += (
            N_full * torch.outer(m_full, m_full)
            - torch.outer(m_full, Sx_full)
            - torch.outer(Sx_full, m_full)
    )

    for feature in features:
        Sy_full = local_full["label_mean"][feature].clone()
        yb = ybar_full[feature]
        local_full["io_covariance"][feature] += (
                N_full * m_full * yb
                - m_full * Sy_full
                - yb * Sx_full
        )
        local_full["label_variance"][feature] += (
                N_full * (yb ** 2)
                - 2 * yb * Sy_full
        )
        # update first moments
        local_full["label_mean"][feature] -= N_full * yb

    local_full["mean"] -= N_full * m_full

    # --- center OUTER split statistics ---
    for split in range(args.n_cv_splits):
        Ntr = local_train["n_images"][split]
        Nte = local_test["n_images"][split]

        Sx_tr = local_train["mean"][split].clone()
        Sx_te = local_test["mean"][split].clone()

        local_train["covariance"][split] += (
                Ntr * torch.outer(m_full, m_full)
                - torch.outer(m_full, Sx_tr)
                - torch.outer(Sx_tr, m_full)
        )
        local_test["covariance"][split] += (
                Nte * torch.outer(m_full, m_full)
                - torch.outer(m_full, Sx_te)
                - torch.outer(Sx_te, m_full)
        )

        # update first moments (needed later for global intercept centering)
        local_train["mean"][split] -= Ntr * m_full
        local_test["mean"][split] -= Nte * m_full

        for feature in features:
            yb = ybar_full[feature]
            Sy_tr = local_train["label_mean"][feature][split].clone()
            Sy_te = local_test["label_mean"][feature][split].clone()

            local_train["io_covariance"][feature][split] += (
                    Ntr * m_full * yb
                    - m_full * Sy_tr
                    - yb * Sx_tr
            )
            local_test["io_covariance"][feature][split] += (
                    Nte * m_full * yb
                    - m_full * Sy_te
                    - yb * Sx_te
            )

            local_train["label_variance"][feature][split] += (
                    Ntr * (yb ** 2)
                    - 2 * yb * Sy_tr
            )
            local_test["label_variance"][feature][split] += (
                    Nte * (yb ** 2)
                    - 2 * yb * Sy_te
            )

            # update first moments
            local_train["label_mean"][feature][split] -= Ntr * yb
            local_test["label_mean"][feature][split] -= Nte * yb

    # --- center INNER split statistics ---
    for split in range(args.n_cv_splits):
        for inner_split in range(args.n_l2_cv_splits):
            Ntr = local_inner_train["n_images"][split][inner_split]
            Nte = local_inner_test["n_images"][split][inner_split]

            Sx_tr = local_inner_train["mean"][split][inner_split].clone()
            Sx_te = local_inner_test["mean"][split][inner_split].clone()

            local_inner_train["covariance"][split][inner_split] += (
                    Ntr * torch.outer(m_full, m_full)
                    - torch.outer(m_full, Sx_tr)
                    - torch.outer(Sx_tr, m_full)
            )
            local_inner_test["covariance"][split][inner_split] += (
                    Nte * torch.outer(m_full, m_full)
                    - torch.outer(m_full, Sx_te)
                    - torch.outer(Sx_te, m_full)
            )

            # update first moments
            local_inner_train["mean"][split][inner_split] -= Ntr * m_full
            local_inner_test["mean"][split][inner_split] -= Nte * m_full

            for feature in features:
                yb = ybar_full[feature]
                Sy_tr = local_inner_train["label_mean"][feature][split][inner_split].clone()
                Sy_te = local_inner_test["label_mean"][feature][split][inner_split].clone()

                local_inner_train["io_covariance"][feature][split][inner_split] += (
                        Ntr * m_full * yb
                        - m_full * Sy_tr
                        - yb * Sx_tr
                )
                local_inner_test["io_covariance"][feature][split][inner_split] += (
                        Nte * m_full * yb
                        - m_full * Sy_te
                        - yb * Sx_te
                )

                local_inner_train["label_variance"][feature][split][inner_split] += (
                        Ntr * (yb ** 2)
                        - 2 * yb * Sy_tr
                )
                local_inner_test["label_variance"][feature][split][inner_split] += (
                        Nte * (yb ** 2)
                        - 2 * yb * Sy_te
                )

                # update first moments
                local_inner_train["label_mean"][feature][split][inner_split] -= Ntr * yb
                local_inner_test["label_mean"][feature][split][inner_split] -= Nte * yb

    # UPDATE GLOBAL RESULTS DICTIONARIES
    full["covariance"] = accumulate(full["covariance"], local_full["covariance"])
    full["mean"] = accumulate(full["mean"], local_full["mean"])
    full["n_images"] = accumulate(full["n_images"], local_full["n_images"])

    for feature in features:
        full["io_covariance"][feature] = accumulate(full["io_covariance"][feature], local_full["io_covariance"][feature])
        full["label_variance"][feature] = accumulate(full["label_variance"][feature], local_full["label_variance"][feature])
        full["label_mean"][feature] = accumulate(full["label_mean"][feature], local_full["label_mean"][feature])

    for split in range(args.n_cv_splits):
        train["covariance"][split] = accumulate(train["covariance"][split], local_train["covariance"][split])
        train["mean"][split] = accumulate(train["mean"][split], local_train["mean"][split])
        train["n_images"][split] = accumulate(train["n_images"][split], local_train["n_images"][split])
        test["covariance"][split] = accumulate(test["covariance"][split], local_test["covariance"][split])
        test["mean"][split] = accumulate(test["mean"][split], local_test["mean"][split])
        test["n_images"][split] = accumulate(test["n_images"][split], local_test["n_images"][split])
        for feature in features:
            train["io_covariance"][feature][split] = accumulate(train["io_covariance"][feature][split],
                                                        local_train["io_covariance"][feature][split])
            train["label_variance"][feature][split] = accumulate(train["label_variance"][feature][split],
                                                         local_train["label_variance"][feature][split])
            train["label_mean"][feature][split] = accumulate(train["label_mean"][feature][split], local_train["label_mean"][feature][split])
            test["io_covariance"][feature][split] = accumulate(test["io_covariance"][feature][split],
                                                        local_test["io_covariance"][feature][split])
            test["label_variance"][feature][split] = accumulate(test["label_variance"][feature][split],
                                                         local_test["label_variance"][feature][split])
            test["label_mean"][feature][split] = accumulate(test["label_mean"][feature][split], local_test["label_mean"][feature][split])

    for split in range(args.n_cv_splits):
        for inner_split in range(args.n_l2_cv_splits):
            inner_train["covariance"][split][inner_split] = accumulate(inner_train["covariance"][split][inner_split], local_inner_train["covariance"][split][inner_split])
            inner_train["mean"][split][inner_split] = accumulate(inner_train["mean"][split][inner_split], local_inner_train["mean"][split][inner_split])
            inner_train["n_images"][split][inner_split] = accumulate(inner_train["n_images"][split][inner_split], local_inner_train["n_images"][split][inner_split])
            inner_test["covariance"][split][inner_split] = accumulate(inner_test["covariance"][split][inner_split], local_inner_test["covariance"][split][inner_split])
            inner_test["mean"][split][inner_split] = accumulate(inner_test["mean"][split][inner_split], local_inner_test["mean"][split][inner_split])
            inner_test["n_images"][split][inner_split] = accumulate(inner_test["n_images"][split][inner_split], local_inner_test["n_images"][split][inner_split])
            for feature in features:
                inner_train["io_covariance"][feature][split][inner_split] = accumulate(inner_train["io_covariance"][feature][split][inner_split],
                                                                    local_inner_train["io_covariance"][feature][split][inner_split])
                inner_train["label_variance"][feature][split][inner_split] = accumulate(inner_train["label_variance"][feature][split][inner_split],
                                                                     local_inner_train["label_variance"][feature][split][inner_split])
                inner_train["label_mean"][feature][split][inner_split] = accumulate(inner_train["label_mean"][feature][split][inner_split],
                                                                 local_inner_train["label_mean"][feature][split][inner_split])
                inner_test["io_covariance"][feature][split][inner_split] = accumulate(inner_test["io_covariance"][feature][split][inner_split],
                                                                   local_inner_test["io_covariance"][feature][split][inner_split])
                inner_test["label_variance"][feature][split][inner_split] = accumulate(inner_test["label_variance"][feature][split][inner_split],
                                                                    local_inner_test["label_variance"][feature][split][inner_split])
                inner_test["label_mean"][feature][split][inner_split] = accumulate(inner_test["label_mean"][feature][split][inner_split],
                                                                local_inner_test["label_mean"][feature][split][inner_split])


# divide by total number of images
for split in range(args.n_cv_splits):
    train["covariance"][split] /= train["n_images"][split]
    test["covariance"][split] /= test["n_images"][split]

    train["mean"][split] /= train["n_images"][split]
    test["mean"][split] /= test["n_images"][split]

    # update full as well, but only for split 0
    if split == 0:
        full["covariance"] /= full["n_images"]
        full["mean"] /= full["n_images"]

    for feature in features:
        train["label_mean"][feature][split] /= train["n_images"][split]
        test["label_mean"][feature][split] /= test["n_images"][split]

        train["label_variance"][feature][split] /= train["n_images"][split]
        test["label_variance"][feature][split] /= test["n_images"][split]

        train["io_covariance"][feature][split] /= train["n_images"][split]
        test["io_covariance"][feature][split] /= test["n_images"][split]

        # update full as well, but only for split 0
        if split == 0:
            full["label_mean"][feature] /= full["n_images"]
            full["label_variance"][feature] /= full["n_images"]
            full["io_covariance"][feature] /= full["n_images"]

# subtract means from covariances
# Note: the test covariances must have the train means subtracted!
# Note: this is not needed for train set as it's means are zero,
# but we keep it for clarity of the logic in the general case in which we do not fit just the fluctuations

for split in range(args.n_cv_splits):
    train["covariance"][split] -= torch.outer(train["mean"][split], train["mean"][split])
    test["covariance"][split] += (torch.outer(train["mean"][split], train["mean"][split])
                                  - torch.outer(test["mean"][split], train["mean"][split])
                                  - torch.outer(train["mean"][split], test["mean"][split]))

    # update full as well, but only for split 0
    if split == 0:
        full["covariance"] -= torch.outer(full["mean"], full["mean"])

    for feature in features:
        train["label_variance"][feature][split] -= train["label_mean"][feature][split]**2
        test["label_variance"][feature][split] += (train["label_mean"][feature][split] ** 2
                                                   - 2*train["label_mean"][feature][split]*test["label_mean"][feature][split])

        train["io_covariance"][feature][split] -= train["mean"][split] * train["label_mean"][feature][split]
        test["io_covariance"][feature][split] += (train["mean"][split] * train["label_mean"][feature][split]
                                                  - test["mean"][split] * train["label_mean"][feature][split]
                                                  - train["mean"][split] * test["label_mean"][feature][split])

        if split == 0:
            full["label_variance"][feature] -= full["label_mean"][feature] ** 2
            full["io_covariance"][feature] -= full["mean"] * full["label_mean"][feature]

# REPEAT The same as the above for the inner splits
# divide by total number of images
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
# subtract means from covariances
# Note: the test covariances must have the train means subtracted!
for split in range(args.n_cv_splits):
    for inner_split in range(args.n_l2_cv_splits):
        inner_train["covariance"][split][inner_split] -= torch.outer(inner_train["mean"][split][inner_split], inner_train["mean"][split][inner_split])
        inner_test["covariance"][split][inner_split] += (torch.outer(inner_train["mean"][split][inner_split], inner_train["mean"][split][inner_split])
                                                         - torch.outer(inner_train["mean"][split][inner_split], inner_test["mean"][split][inner_split])
                                                         - torch.outer(inner_test["mean"][split][inner_split], inner_train["mean"][split][inner_split]))

        for feature in features:
            inner_train["label_variance"][feature][split][inner_split] -= inner_train["label_mean"][feature][split][inner_split]**2
            inner_test["label_variance"][feature][split][inner_split] += (inner_train["label_mean"][feature][split][inner_split] ** 2
                                                                          - 2*inner_train["label_mean"][feature][split][inner_split]*inner_test["label_mean"][feature][split][inner_split])

            inner_train["io_covariance"][feature][split][inner_split] -= inner_train["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split]
            inner_test["io_covariance"][feature][split][inner_split] += (inner_train["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split]
                                                                         - inner_test["mean"][split][inner_split] * inner_train["label_mean"][feature][split][inner_split]
                                                                         - inner_train["mean"][split][inner_split] * inner_test["label_mean"][feature][split][inner_split])

# retrieve dataset normalization parameters for storage
normalization_bbox_mean, normalization_bbox_std = get_bbox_stats_from_loader(dataloader)

if normalization_bbox_mean is None:
    bbox_mean_dict = {feature: None for feature in features}
    bbox_std_dict  = {feature: None for feature in features}
else:
    bbox_mean_dict = {feature: normalization_bbox_mean[i] for i, feature in enumerate(features)}
    bbox_std_dict  = {feature: normalization_bbox_std[i] for i, feature in enumerate(features)}

# define results dictionary
args_dictionary = vars(args).copy()

results = {
    "args": args_dictionary,
    "inner_mse_test_avg": {feature: {split: {l2: None for l2 in l2_strengths}
                                 for split in range(args.n_cv_splits)} for feature in features},
    "inner_mse_test_std": {feature: {split: {l2: None for l2 in l2_strengths}
                                     for split in range(args.n_cv_splits)} for feature in features},
    "inner_normalized_mse_test_avg": {feature: {split: {l2: None for l2 in l2_strengths}
                                         for split in range(args.n_cv_splits)} for feature in features},
    "inner_normalized_mse_test_std": {feature: {split: {l2: None for l2 in l2_strengths}
                                         for split in range(args.n_cv_splits)} for feature in features},
    "best_l2_strength": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    "mse_test": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    "normalized_mse_test": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},
    "labels_variance": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},  # of test set for the given split
    "labels_max": max_label_test,  # of test set for the given split (see commented empty definition below)
    "labels_min": min_label_test,  # of test set for the given split (see commented empty definition below)
    # "labels_max": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},  # of test set for the given split
    # "labels_min": {feature: {split: None for split in range(args.n_cv_splits)} for feature in features},  # of test set for the given split
    "mse_test_avg": {feature: None for feature in features},
    "mse_test_std": {feature: None for feature in features},
    "normalized_mse_test_avg": {feature: None for feature in features},
    "normalized_mse_test_std": {feature: None for feature in features},
    "labels_variance_full": {feature: None for feature in features},  # for the whole dataset
    "labels_max_full": {feature: None for feature in features},  # for the whole dataset
    "labels_min_full": {feature: None for feature in features},  # for the whole dataset
    "normalization_bbox_mean": bbox_mean_dict,  # the bbox mean used by the dataloader for normalization
    "normalization_bbox_std": bbox_std_dict,  # the bbox std used by the dataloader for normalization
}

# compute the labels max and min for the whole dataset
for feature in features:
    results["labels_max_full"][feature] = torch.max(max_label_test[feature][0], max_label_train[feature][0])
    results["labels_min_full"][feature] = torch.min(min_label_test[feature][0], min_label_train[feature][0])

# DO THE CV REGRESSION
for feature in features:
    print(f"Feature: {feature}")
    outer_mses = []
    outer_normalized_mses = []
    for split in range(args.n_cv_splits):
        print(f"\tComputing split: {split}")
        inner_mses_avg = []
        for l2 in l2_strengths:
            inner_mses = []
            inner_mses_normalized = []
            for inner_split in range(args.n_l2_cv_splits):
                # DO THE REGRESSION
                C_tr = inner_train["covariance"][split][inner_split]  # [D×D]
                d_tr = inner_train["io_covariance"][feature][split][inner_split]  # [D]

                # build λ·I with the same dtype/device as C_tr
                I = torch.eye(C_tr.size(0),
                              device=C_tr.device,
                              dtype=C_tr.dtype)

                # solve (C_tr + λI) w = d_tr
                w = torch.linalg.solve(C_tr + l2 * I, d_tr)

                # now compute the test-MSE
                C_te = inner_test["covariance"][split][inner_split]
                d_te = inner_test["io_covariance"][feature][split][inner_split]
                y2 = inner_test["label_variance"][feature][split][inner_split]

                in_mse = (
                        (w @ C_te @ w)
                        - 2 * (w @ d_te)
                        + y2
                )

                # compute actual
                actual_labels_variance = (inner_test["label_variance"][feature][split][inner_split]
                                          - inner_train["label_mean"][feature][split][inner_split] ** 2
                                          + 2 * inner_test["label_mean"][feature][split][inner_split] *
                                          inner_train["label_mean"][feature][split][inner_split]
                                          - inner_test["label_mean"][feature][split][inner_split] ** 2)
                in_mse_normalized = in_mse / actual_labels_variance

                inner_mses.append(in_mse.item())
                inner_mses_normalized.append(in_mse_normalized.item())
            mean_inner_mses = np.mean(inner_mses)
            inner_mses_avg.append(mean_inner_mses)
            std_inner_mses = np.std(inner_mses)
            mean_inner_mses_normalized = np.mean(inner_mses_normalized)
            std_inner_mses_normalized = np.std(inner_mses_normalized)
            results["inner_mse_test_avg"][feature][split][l2] = mean_inner_mses
            results["inner_mse_test_std"][feature][split][l2] = std_inner_mses
            results["inner_normalized_mse_test_avg"][feature][split][l2] = mean_inner_mses_normalized
            results["inner_normalized_mse_test_std"][feature][split][l2] = std_inner_mses_normalized
            print(f"\t\t\tl2: {l2}, mse: {mean_inner_mses} +- {std_inner_mses}, nmse: {mean_inner_mses_normalized} +- {std_inner_mses_normalized}")
        # select the best l2
        best_l2_idx = np.argmin(inner_mses_avg)
        best_l2 = l2_strengths[best_l2_idx]
        results["best_l2_strength"][feature][split] = best_l2
        print(f"\t\tBest l2: {best_l2}")

        # NOW DO THE ACTUAL REGRESSION
        C_tr = train["covariance"][split]  # [D×D]
        d_tr = train["io_covariance"][feature][split]  # [D]

        # build λ·I with the same dtype/device as C_tr
        I = torch.eye(C_tr.size(0),
                      device=C_tr.device,
                      dtype=C_tr.dtype)

        # solve (C_tr + λI) w = d_tr
        w = torch.linalg.solve(C_tr + best_l2 * I, d_tr)

        # now compute the test-MSE
        C_te = test["covariance"][split]
        d_te = test["io_covariance"][feature][split]
        y2 = test["label_variance"][feature][split]

        outer_mse = (
                (w @ C_te @ w)
                - 2 * (w @ d_te)
                + y2
        )

        # compute the actual labels variance (i.e. centered w.r.t. test)
        actual_labels_variance = (test["label_variance"][feature][split]
                                  - train["label_mean"][feature][split] ** 2
                                  + 2 * train["label_mean"][feature][split] * test["label_mean"][feature][split]
                                  - test["label_mean"][feature][split] ** 2)
        outer_normalized_mse = outer_mse / actual_labels_variance

        # update results dictionary
        results["labels_variance"][feature][split] = actual_labels_variance
        results["mse_test"][feature][split] = outer_mse
        results["normalized_mse_test"][feature][split] = outer_normalized_mse

        outer_mses.append(outer_mse.item())
        outer_normalized_mses.append(outer_normalized_mse.item())

        print(f"\t\tmse: {outer_mse}, nmse: {outer_normalized_mse}", flush=True)

    # compute the summary statistics
    outer_mse_avg = np.mean(outer_mses)
    outer_mse_std = np.std(outer_mses)
    outer_normalized_mse_avg = np.mean(outer_normalized_mses)
    outer_normalized_mse_std = np.std(outer_normalized_mses)

    # update results dictionary
    results["labels_variance_full"][feature] = full["label_variance"][feature]
    results["mse_test_avg"][feature] = outer_mse_avg
    results["mse_test_std"][feature] = outer_mse_std
    results["normalized_mse_test_avg"][feature] = outer_normalized_mse_avg
    results["normalized_mse_test_std"][feature] = outer_normalized_mse_std

# STORE RESULTS

results_numpy = convert_tensors_to_numpy(results)

if args.global_dimensionality_reduction_style != "":
    results_numpy["args"]["global_dimensionality_reduction_dimensions"] = global_d
    results_numpy["args"]["global_dimensionality_reduction_strength"] = global_explained_variance

# print results summary
print("\n### SUMMARY ###\n")
for feature in features:
    print(f"Feature: {feature}")
    outer_mse_avg = results_numpy["mse_test_avg"][feature]
    outer_mse_std = results_numpy["mse_test_std"][feature]

    print("\tMSE summary:")
    print(f"\tmse (normalized by full labels variance): {outer_mse_avg/results_numpy['labels_variance_full'][feature]} "
          f"+- {outer_mse_std/results_numpy['labels_variance_full'][feature]}")
    print(f"\tmse: {outer_mse_avg} +- {outer_mse_std}")
    mse_lab_var_norm_avg = results_numpy["normalized_mse_test_avg"][feature]
    mse_lab_var_norm_std = results_numpy["normalized_mse_test_std"][feature]
    print(f"\tmse (label variance normalized): {mse_lab_var_norm_avg} +- {mse_lab_var_norm_std}")
    dat_rng = results_numpy["labels_max_full"][feature] - results_numpy["labels_min_full"][feature]
    mse_dat_rng_norm_avg = outer_mse_avg / (dat_rng ** 2)
    mse_dat_rng_norm_std = outer_mse_std / (dat_rng ** 2)
    print(f"\tmse (dataset range normalized): {mse_dat_rng_norm_avg} +- {mse_dat_rng_norm_std}")
    std = results_numpy["normalization_bbox_std"][feature]
    if std is None:
        mse_full_rng_norm_avg = mse_full_rng_norm_std = None
    else:
        mse_full_rng_norm_avg = (std ** 2) * outer_mse_avg
        mse_full_rng_norm_std = (std ** 2) * outer_mse_std
    print(f"\tmse (full range normalized): {mse_full_rng_norm_avg} +- {mse_full_rng_norm_std}")

    print(f"\tEstimated pearson: {np.sqrt(1-mse_lab_var_norm_avg)}")

    print("\tError summary (sqrtMSE):")
    print(f"\terror: {np.sqrt(outer_mse_avg)} +- {np.sqrt(outer_mse_std)}")
    print(f"\terror (label variance normalized): {np.sqrt(mse_lab_var_norm_avg)} +- {np.sqrt(mse_lab_var_norm_std)}")
    print(f"\terror (dataset range normalized): {np.sqrt(mse_dat_rng_norm_avg)} +- {np.sqrt(mse_dat_rng_norm_std)}")
    if std is None:
        print("\terror (full range normalized): None")
    else:
        print(f"\terror (full range normalized): {np.sqrt(mse_full_rng_norm_avg)} +- {np.sqrt(mse_full_rng_norm_std)}")


# get the directory containing the checkpoint file
checkpoint_dir = os.path.dirname(args.checkpoint)
# build the path to the new results folder
results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
# create it (and any missing parents), no error if it already exists
os.makedirs(results_dir, exist_ok=True)

if args.random_subsample_size is not None:
    rnd_sub_string = f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
else:
    rnd_sub_string = ""

if args.random_projection_size is not None:
    rnd_proj_string = f"_RndProj_size{args.random_projection_size}_seed{args.random_projection_seed}"
else:
    rnd_proj_string = ""

if args.global_dimensionality_reduction_style != "":
    global_dim_red_string = f"_GlobalRed_{args.global_dimensionality_reduction_style}_Dim{global_d}_KV{global_explained_variance}"
else:
    global_dim_red_string = ""

dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))

results_file = os.path.join(results_dir, f"{args.job_id}_CVregressionL2fluctuations_{dataset_name}_{args.layer}_nOutSplit{args.n_cv_splits}_nInSplit{args.n_l2_cv_splits}" + rnd_sub_string + rnd_proj_string + global_dim_red_string + ".pkl")

with open(results_file, 'wb') as file:
    pickle.dump(results_numpy, file)

print("Regression concluded. Exit", flush=True)

exit()
