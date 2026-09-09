import os
import argparse
import pickle
from einops import einsum
import torch
import numpy as np
import random
import hashlib
import h5py
from networkx.algorithms.centrality import katz_centrality_numpy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image
import sys
from hadamard_transform import hadamard_transform, next_power_of_2, pad_to_power_of_2, is_a_power_of_2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADDITIONAL_FEATURES_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(ADDITIONAL_FEATURES_DIR, "CNN_trainer")
sys.path.insert(0, ADDITIONAL_FEATURES_DIR)
sys.path.insert(0, MODEL_DIR)
import LIB_model
import json


DEFAULT_BBOX_FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
ADDITIONAL_TARGET_FEATURES = [
    "bbox_area",
    "mean_srgb_luma",
    "local_srgb_luma_contrast",
    "mean_hsv_saturation",
]
ALL_TARGET_FEATURES = DEFAULT_BBOX_FEATURES + ADDITIONAL_TARGET_FEATURES


def get_features_from_cv_rule(local_rule_dict, merged_args):
    target_features = getattr(merged_args, "target_features", None)
    if target_features is not None:
        return list(target_features)

    local_results = local_rule_dict.get("local", {})
    if local_results:
        first_category = next(iter(local_results.values()))
        regression_vectors = first_category.get("regression_vector", {})
        if regression_vectors:
            return list(regression_vectors.keys())

    return list(DEFAULT_BBOX_FEATURES)


def validate_local_rule_features(local_rule_dict, categories, features):
    local_results = local_rule_dict.get("local", {})
    missing_categories = [category for category in categories if category not in local_results]
    if missing_categories:
        raise ValueError(
            "The CV local-rule file is missing categories used by this theory run. "
            f"First missing categories: {missing_categories[:10]}"
        )

    for category in categories:
        regression_vectors = local_results[category].get("regression_vector", {})
        missing_features = [feature for feature in features if feature not in regression_vectors]
        if missing_features:
            raise ValueError(
                f"The CV local-rule file is missing regression vectors for category={category}: "
                f"{missing_features}"
            )


parser = argparse.ArgumentParser(description="Extract features and predictions for dataset images.")
parser.add_argument("--job_id", type=str, default="0000",
                    help="A name/ID for this run (used for naming the results).")
parser.add_argument("--results_folder_name", type=str,
                    default="./",
                    help="Name of the folder in the checkpoint folder where we will store the results.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
parser.add_argument("--cv_local_rule_file", type=str,
                    default="./checkpoint.pth",
                    help="Path to the file containing the cv local rule.")


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


# merge saved args and current args. Current args oveeride anything in saved args with the same key
args_new = parser.parse_args()

# load data dictionary for local linear rules
with open(args_new.cv_local_rule_file, "rb") as f:
    local_dict = pickle.load(f)

saved_args = local_dict["args"]
args = argparse.Namespace(**{**saved_args, **vars(args_new)})

# --- Back-compatibility: ensure new optional args exist on older cv-rule files ---
for k in ["random_subsample_size", "random_subsample_seed"]:
    if not hasattr(args, k):
        setattr(args, k, None)

# --- Back-compatibility: ensure optional args exist on older results ---
defaults = {
    "local_dimensionality_reduction_style": "",  # "" | "PR" | "percentage"
    "local_dimensionality_reduction_strength": 100,  # only used if style == "percentage"
}
for k, v in defaults.items():
    if not hasattr(args, k):
        setattr(args, k, v)

# Back-compat defaults for older result files
defaults = {
    "global_dimensionality_reduction_style": "",  # match producer default
    "global_dimensionality_reduction_strength": 100.0,
    "global_dimensionality_reduction_dimensions": 2048,
    "global_dimensionality_reduction_file": "./boh.pkl",
}
for k, v in defaults.items():
    if not hasattr(args, k):
        setattr(args, k, v)

# --- Back-compatibility: ensure manifold-subsampling args exist on older files ---
for k, v in {
    "manifold_subsample_size": None,
    "manifold_subsample_seed": 1,
    "manifold_subsampling_grouping_json": None,
}.items():
    if not hasattr(args, k):
        setattr(args, k, v)

# --- Back-compatibility: ensure per-category image subsampling args exist on older cv-rule files ---
if not hasattr(args, "n_subsampled_images"):
    args.n_subsampled_images = None
if not hasattr(args, "images_subsampling_seed"):
    args.images_subsampling_seed = 1

if not hasattr(args, "augment_dihedral"):
    args.augment_dihedral = False

if not hasattr(args, "target_features"):
    args.target_features = None

AUGMENT_GROUP_SIZE = 8 if getattr(args, "augment_dihedral", False) else 1


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


def participation_ratio(eigs: torch.Tensor) -> int:
    pr = (eigs.sum() ** 2) / eigs.pow(2).sum()  # still a 0-d torch.Tensor
    return int(torch.round(pr).item())  # round in torch, then .item()


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


def print_args(args):
    """
    Nicely prints out every command-line argument and its value.
    """
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:20s}: {value!r}")
    print("=========================", flush=True)

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

features_from_cv_rule = get_features_from_cv_rule(local_dict, args)
unsupported_features = [feature for feature in features_from_cv_rule if feature not in ALL_TARGET_FEATURES]
if unsupported_features:
    raise ValueError(f"Unsupported target features found in CV local-rule file: {unsupported_features}")

is_sdxl_dataset = dataset_folder_name in [
    "SDXL_dataset_test",
    "SDXL_dataset_train",
    "SDXL_dataset_validation",
]

if is_sdxl_dataset or dataset_folder_name == "DicarloAsGenai_public":
    features = list(features_from_cv_rule) if is_sdxl_dataset else list(DEFAULT_BBOX_FEATURES)
    if not is_sdxl_dataset and features_from_cv_rule != DEFAULT_BBOX_FEATURES:
        raise ValueError("Additional target features are supported only for SDXL datasets.")

    from LIB_dataloader import CategoryDatasetSDXL
    CategoryDataset = partial(
        CategoryDatasetSDXL,
        dataset_folder=args.dataset_folder,
        transform=transform,
        augment_dihedral=args.augment_dihedral,
        target_features=features if is_sdxl_dataset else None,
    )

elif dataset_folder_name in [
    "representations_DicarloAsGenai_public_IT",
    "representations_DicarloAsGenai_public_V4",
]:
    if features_from_cv_rule != DEFAULT_BBOX_FEATURES:
        raise ValueError("Additional target features are supported only for SDXL datasets.")

    from LIB_dataloader import CategoryDatasetRepresentation
    CategoryDataset = partial(
        CategoryDatasetRepresentation,
        dataset_folder=args.dataset_folder,
        transform=transform,
    )
    features = list(DEFAULT_BBOX_FEATURES)

else:
    raise ValueError(f"Dataset {dataset_folder_name} is not implemented.")

print(f"Theory target features: {features}", flush=True)

# retrieve categories (sorted for determinism)
all_categories = [
    p.name
    for p in (dataset_path if dataset_path.is_dir() else dataset_path.parent).iterdir()
    if p.is_dir() and p.name.lower() != "crashes"
]
all_categories.sort()

if args.manifold_subsample_size is not None:
    seed_everything(args.manifold_subsample_seed)

    # never request more than what exists
    K = min(args.manifold_subsample_size, len(all_categories))

    grouping_json = getattr(args, "manifold_subsampling_grouping_json", None)
    if grouping_json is not None:
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

validate_local_rule_features(local_dict, categories, features)

# Define results dictionaries
# define global regression results dictionary
res_glob = {"covariance": None,
            "covariance_fluctuations": None,  # computed using only the fluctuations part, i.e. C1=[<deltax deltax^T>]
            "covariance_eigenvalues": None,
            "covariance_eigenvectors": None,
            "linear_rule_vs_PC_overlap": {},
            "io_covariance": {feature: None for feature in features},
            "io_covariance_fluctuations": {feature: None for feature in features},
            # computed using only the fluctuations part, i.e. C1=[<deltay deltax>]
            "io_covariance_linearized": {feature: None for feature in features},
            "io_covariance_linearized_fluctuations": {feature: None for feature in features},
            # computed using only the fluctuations part, i.e. C1=[<deltay deltax>]
            # REMOVED: stuff below not needed anymore. Commented it
            # "covariance_parallel": {feature: None for feature in features},
            # "covariance_perp": {feature: None for feature in features},
            # "covariance_cross": {feature: None for feature in features},
            # # "covariance_center": {feature: None for feature in features},
            # "io_covariance_parallel": {feature: None for feature in features},
            # "io_covariance_cross": {feature: None for feature in features},
            # END REMOVED
            # "io_covariance_center": {feature: None for feature in features},
            "regression_error": {},
            "normalized_regression_error": {},
            "regression_error_gap": {},
            "normalized_regression_error_gap": {},
            # this is normalized by the labels variance, NOT the linearized labels variance
            "regression_vector": {},
            "labels_variance": {feature: None for feature in features},  # i.e. [<y_mu^2>]
            "labels_variance_fluctuations": {feature: None for feature in features},
            # computed only using the fluctuations part, i.e. [<deltay_mu^2>]
            "linearized_labels_variance": {feature: None for feature in features},  # i.e. [<\hat{y}_mu^2>]
            "linearized_labels_variance_fluctuations": {feature: None for feature in features},
            # computed only using the fluctuations part, i.e. [<deltay_mu^2>]
            "variance_along_local_linear_rules": {feature: 0.0 for feature in features},  # i.e. [sigma_1_mu^2]
            "mean_regression_direction": {feature: None for feature in features},
            "mean_regression_direction_norm": {},  # this is c, not c^2
            "mean_regression_direction_unit_vector": {},  # i.e. u_1
            "total_variance": None,
            "variance_pearson": {},  # i.e. rho, (Not rho**2 !!!)
            "linear_rule_scale_mean": {},  # i.e. n
            "linear_rule_scale_var": {},  # i.e. sigma^2_scale
            }

# define results dictionary. Keys nesting order: category->feature
results = {"covariance_eigenvalues": {},
           "linear_rule_vs_PC_overlap": {category: {} for category in categories},
           "centroid": {category: None for category in categories},
           "centroid_norm": {},
           "label_mean": {category: {feature: None for feature in features} for category in categories},
           # local mean of the globally centered labels
           "label_variance": {category: {feature: None for feature in features} for category in categories},
           # variance of LOCALLY CENTERED labels, i.e. <deltay**2>
           "linearized_label_variance": {category: {} for category in categories},
           # variance of LOCALLY CENTERED labels, i.e. <deltay**2>
           "io_covariance": {category: {feature: None for feature in features} for category in categories},
           "regression_error": {category: {} for category in categories},
           "normalized_regression_error": {category: {} for category in categories},
           "regression_vector": {category: {} for category in categories},
           "regression_vector_norm": {category: {} for category in categories},  # added in v2
           "bias": {category: {} for category in categories},
           "variance_along_linear_rule": {category: {} for category in categories},
           "variances_orthogonal": {category: {} for category in categories},
           # added in v2: eigenvalues of covariance on subspace orthogonal to linear rule
           "covariance_cross_terms": {category: {} for category in categories},  # added in v2 (i.e. Sigma_1d)
           "total_variance": {category: {} for category in categories},
           "linear_rule_scale": {category: {} for category in categories},  # (i.e. n_1^mu)
           "linear_rule_scale_projected": {category: {} for category in categories},  # (i.e. n_1^mu (u^mu_1^T u_1)
           "local_vs_common_linear_rule_overlap": {category: {} for category in categories},  # (i.e. (u^mu_1^T u_1)
           }

n_global_images = 0

# define running quantities that are not defined in the global dictionary
global_reps_mean = None
global_labels_mean = {feature: None for feature in features}
fjlt_idx = None
fjlt_D = None

subsample_idx = None

# load global dimensionality reduction
if args.global_dimensionality_reduction_style != "":
    with open(args.global_dimensionality_reduction_file, "rb") as f:
        global_stats = pickle.load(f)
    global_eigenvalues = torch.tensor(global_stats["covariance_eigenvalues"], device=device,
                                      dtype=torch.float64)  # ordered from largest to smallest eigenvalue
    global_eigenvectors = torch.tensor(global_stats["covariance_eigenvectors"], device=device,
                                       dtype=torch.float64)  # shape [component, eigenvalue number], ordered from largest to smallest eigenvalue
    global_projection_mean = torch.tensor(global_stats["mean"], device=device, dtype=torch.float64)
    if args.global_dimensionality_reduction_style == "dimensions":
        global_d = args.global_dimensionality_reduction_dimensions
        global_explained_variance = explained_variance_percent(global_eigenvalues, global_d)
        global_projector = global_eigenvectors[:, :global_d]
    elif args.global_dimensionality_reduction_style == "percentage":
        global_explained_variance = args.global_dimensionality_reduction_strength
        global_d = explained_variance_dim(global_eigenvalues,
                                          threshold=(args.global_dimensionality_reduction_strength / 100))
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
    print(
        f"Initializing global projector. Keeping first {global_d} dimensions, or {global_explained_variance}% of the variance")

# loop over the categories: generate representations and do regression
for category in categories:
    print(f"Processing category: {category}", flush=True)

    dataset = CategoryDataset(category=category)  # <— only category here

    # Optional per-category image subsampling (no dataloader edits needed)
    if args.n_subsampled_images is not None:
        N = len(dataset)
        k = max(0, min(int(args.n_subsampled_images), N))
        if k < N:
            # Derive a per-category seed so choices don't depend on loop order
            cat_key = f"{args.images_subsampling_seed}:{category}".encode("utf-8")
            cat_seed = int.from_bytes(hashlib.sha256(cat_key).digest()[:4], "big")
            rng = np.random.RandomState(cat_seed)

            subset_idx = rng.choice(np.arange(N), size=k, replace=False)
            subset_idx.sort()  # keep original (non-shuffled) dataloader order
            dataset = Subset(dataset, subset_idx.tolist())

        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True)

        # define running quantities that are not defined in the local dictionary
        local_covariance = None

        # compute running quantities looping over batches
        n_local_images = 0
        for batch in dataloader:
            images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch
            n_local_images += gt_bboxes.size(0)
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

        if (subsample_idx is None) and (args.random_subsample_size is not None):
            seed_everything(args.random_subsample_seed)
            N_orig = reps.size(1)
            if args.random_subsample_size > N_orig:
                raise ValueError(f"--random_subsample_size ({args.random_subsample_size}) > neurons ({N_orig}).")
            subsample_idx = torch.randperm(N_orig, device=device)[:args.random_subsample_size]
        if args.random_subsample_size is not None:
            reps = reps[:, subsample_idx]

        # define the random projector if not yet defined, and if random projection is required
        if (fjlt_D is None) and (fjlt_idx is None) and (args.random_projection_size is not None):
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
        if args.random_projection_size is not None:
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

        # <editor-fold desc="Update local quantities">
        # update local covariance
        local_covariance_addition = einsum(reps, reps, "image neuron1, image neuron2 -> neuron1 neuron2")
        local_covariance = local_covariance_addition.clone() \
            if local_covariance is None \
            else local_covariance + local_covariance_addition

        # update centroid
        centroid_addition = torch.sum(reps, dim=0)  # size [# neurons]
        results["centroid"][category] = centroid_addition.clone() \
            if results["centroid"][category] is None \
            else results["centroid"][category] + centroid_addition

        # update feature-specific quantities
        for feat_idx, feature in enumerate(features):
            labels = gt_bboxes[:, feat_idx]  # shape [batch size]
            # convert to float64
            labels = labels.to(dtype=torch.float64)

            # update labels mean
            labels_mean_addition = torch.sum(labels)
            results["label_mean"][category][feature] = labels_mean_addition.clone() \
                if results["label_mean"][category][feature] is None \
                else results["label_mean"][category][feature] + labels_mean_addition

            # update labels variance
            labels_var_addition = torch.sum(labels ** 2)
            results["label_variance"][category][feature] = labels_var_addition.clone() \
                if results["label_variance"][category][feature] is None \
                else results["label_variance"][category][feature] + labels_var_addition

            # update io_covariance
            io_addition = einsum(reps, labels, "image neuron, image -> neuron")
            results["io_covariance"][category][feature] = io_addition.clone() \
                if results["io_covariance"][category][feature] is None \
                else results["io_covariance"][category][feature] + io_addition
        # </editor-fold>

    n_global_images += n_local_images

    # divide local quantities by number of samples
    local_covariance /= n_local_images
    results["centroid"][category] /= n_local_images
    for feature in features:
        results["label_mean"][category][feature] /= n_local_images
        results["label_variance"][category][feature] /= n_local_images
        results["io_covariance"][category][feature] /= n_local_images

    # center the covariance, io_covariance, variance of locally centered labels
    local_covariance -= torch.outer(results["centroid"][category], results["centroid"][category])
    for feature in features:
        results["io_covariance"][category][feature] -= (
                results["centroid"][category] * results["label_mean"][category][feature])
        results["label_variance"][category][feature] -= results["label_mean"][category][feature] ** 2

    # compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(local_covariance)
    # Sort in descending order
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # do local dimensionality reduction if required
    style = args.local_dimensionality_reduction_style
    thresh = args.local_dimensionality_reduction_strength / 100
    if style != "":
        # determine the dimensionality
        if style == "PR":
            d = participation_ratio(eigenvalues)
        elif style == "percentage":
            d = explained_variance_dim(eigenvalues, threshold=thresh)
        else:
            print(f"ERROR: dimensionality reduction style <{style}> is not implemented", flush=True)
            exit(1)

        # Take the top-d eigenvectors
        V = eigenvectors[:, :d]  # shape (N, d)
        eigenvalues[d:] = 0.0  # everything beyond the top-d gets set to zero

        # Build the projector onto span{V}
        local_projector = V @ V.T  # shape (N, N)

        # project stuff
        local_covariance = local_projector @ local_covariance @ local_projector
        for feature in features:
            results["io_covariance"][category][feature] = local_projector @ results["io_covariance"][category][feature]

    results["covariance_eigenvalues"][category] = eigenvalues

    # <editor-fold desc="update global quantities">
    # NOTE 1: we need to re-add the local means and then subtract later the global mean
    # NOTE 2: we multiply by n_local_images and the divide later by n_global_images
    # update global covariance
    glob_cov_update = (local_covariance + torch.outer(results["centroid"][category],
                                                      results["centroid"][category])) * n_local_images
    res_glob["covariance"] = glob_cov_update.clone() \
        if res_glob["covariance"] is None \
        else res_glob["covariance"] + glob_cov_update

    glob_cov_fluctuations_update = local_covariance * n_local_images
    res_glob["covariance_fluctuations"] = glob_cov_fluctuations_update.clone() \
        if res_glob["covariance_fluctuations"] is None \
        else res_glob["covariance_fluctuations"] + glob_cov_fluctuations_update

    # update global representations mean
    global_reps_mean = results["centroid"][category] * n_local_images \
        if global_reps_mean is None \
        else global_reps_mean + (results["centroid"][category] * n_local_images)

    # update feature-specific quantities
    for feature in features:
        # update global labels mean
        global_labels_mean[feature] = results["label_mean"][category][feature] * n_local_images \
            if global_labels_mean[feature] is None \
            else global_labels_mean[feature] + (results["label_mean"][category][feature] * n_local_images)

        # update global labels variance
        lab_var_update = (results["label_variance"][category][feature] + (
                    results["label_mean"][category][feature] ** 2)) * n_local_images
        res_glob["labels_variance"][feature] = lab_var_update.clone() \
            if res_glob["labels_variance"][feature] is None \
            else res_glob["labels_variance"][feature] + lab_var_update

        # update global labels variance
        lab_var_fluctuations_update = results["label_variance"][category][feature] * n_local_images
        res_glob["labels_variance_fluctuations"][feature] = lab_var_fluctuations_update.clone() \
            if res_glob["labels_variance_fluctuations"][feature] is None \
            else res_glob["labels_variance_fluctuations"][feature] + lab_var_fluctuations_update

        # update io_covariance
        io_cov_update = (results["io_covariance"][category][feature] +
                         (results["centroid"][category] * results["label_mean"][category][feature])) * n_local_images
        res_glob["io_covariance"][feature] = io_cov_update.clone() \
            if res_glob["io_covariance"][feature] is None \
            else res_glob["io_covariance"][feature] + io_cov_update

        # update io_covariance
        io_cov_fluctuations_update = results["io_covariance"][category][feature] * n_local_images
        res_glob["io_covariance_fluctuations"][feature] = io_cov_fluctuations_update.clone() \
            if res_glob["io_covariance_fluctuations"][feature] is None \
            else res_glob["io_covariance_fluctuations"][feature] + io_cov_fluctuations_update

    # </editor-fold>

    # <editor-fold desc="compute actual local quantities and derived quantities">
    # compute the pseudo-inverse
    pseudo_inverse = torch.linalg.pinv(local_covariance, hermitian=True)

    # compute total variance (added with v3)
    results["total_variance"][category] = torch.trace(local_covariance)

    for feature in features:
        # compute regression vector
        # K^+ b, where K is covariance and b is io-covariance. Here + is the pseudo-inverse
        results["regression_vector"][category][feature] = torch.tensor(
            local_dict["local"][category]["regression_vector"][feature], dtype=results["centroid"][category].dtype,
            device=results["centroid"][category].device)

        # compute regression error
        regression_error = (
                results["regression_vector"][category][feature] @ local_covariance @
                results["regression_vector"][category][feature]
                + results["label_variance"][category][feature]
                - 2 * results["regression_vector"][category][feature] @ results["io_covariance"][category][feature])

        results["regression_error"][category][feature] = regression_error

        # compute normalized regression error
        results["normalized_regression_error"][category][feature] = (regression_error
                                                                     / results["label_variance"][category][feature])
        print(f"\tNormalized regression error: {results['normalized_regression_error'][category][feature]}", flush=True)

        # compute linear_rule vs PCs overlap
        w_direction = results["regression_vector"][category][feature] / torch.linalg.vector_norm(
            results["regression_vector"][category][feature])
        # eigenvector are organized as [components, eigenvector/eigenvalue number]
        overlaps = einsum(w_direction, eigenvectors, "neuron, neuron eig -> eig")
        results["linear_rule_vs_PC_overlap"][category][feature] = overlaps

        # compute variance along linear rule
        results["variance_along_linear_rule"][category][feature] = (
                w_direction @ local_covariance @ w_direction)

        # compute linearized labels variance
        results["linearized_label_variance"][category][feature] = (
                results["regression_vector"][category][feature] @ local_covariance @
                results["regression_vector"][category][feature])

        # update total variance along linear rules
        res_glob["variance_along_local_linear_rules"][feature] += (
                results["variance_along_linear_rule"][category][feature] / n_categories)

        # update common_regression_direction
        if res_glob["mean_regression_direction"][feature] is None:
            res_glob["mean_regression_direction"][feature] = w_direction.clone() / n_categories
        else:
            res_glob["mean_regression_direction"][feature] += w_direction / n_categories

        # Compute quantities necessary for finite number of manifolds errors
        # <editor-fold desc="compute intermediate steps">
        N = local_covariance.shape[0]

        # Step 1: Construct an orthonormal basis Q with w_direction as the first column.
        Q = torch.empty((N, N), dtype=local_covariance.dtype, device=local_covariance.device)
        Q[:, 0] = w_direction

        # For the remaining N-1 columns, generate random vectors and orthogonalize them w.r.t. w_direction.
        rand_mat = torch.randn(N, N - 1, dtype=local_covariance.dtype, device=local_covariance.device)
        for i in range(N - 1):
            vec = rand_mat[:, i]
            # Remove component along w_direction.
            vec = vec - torch.dot(vec, w_direction) * w_direction
            rand_mat[:, i] = vec
        # Orthonormalize the remaining columns via QR decomposition.
        Q_rest, _ = torch.linalg.qr(rand_mat)
        Q[:, 1:] = Q_rest

        # Step 2: Rotate the covariance matrix into the basis Q.
        C_rot = Q.t() @ local_covariance @ Q
        Sigma_1d = C_rot[0, 1:]  # Cross terms: shape (N-1,)
        C22 = C_rot[1:, 1:]  # Lower-right block, shape (N-1, N-1).

        # Step 3: Diagonalize C22.
        # torch.linalg.eigh returns eigenvalues in ascending order.
        eigenvalues, V = torch.linalg.eigh(C22)
        # Optionally, if you prefer descending order, flip them:
        eigenvalues = eigenvalues.flip(0)
        V = V.flip(1)
        sigma_d_sq = eigenvalues  # These are the diagonal entries (variances) in the orthogonal complement.

        # Step 4: Rotate the cross terms into the eigenbasis of C22.
        sigma_1d_rotated = Sigma_1d @ V  # Now these are the cross terms in the basis where C22 is diagonal.
        # </editor-fold>
        results["variances_orthogonal"][category][feature] = sigma_d_sq
        results["covariance_cross_terms"][category][feature] = sigma_1d_rotated
        results["regression_vector_norm"][category][feature] = torch.linalg.vector_norm(
            results["regression_vector"][category][feature])

        # REMOVED: STUFF BELOW NOT NEEDED ANYMORE. COMMENTED
        # Computation of covariance/io_covariance parallel, perpendicular, cross.

        # # compute projectors
        # w_projector = torch.outer(w_direction, w_direction)
        # I = torch.eye(w_direction.shape[0], device=w_direction.device, dtype=w_direction.dtype)
        # orthogonal_projector = I - w_projector
        #
        # # note: we re-multiply by n_local_images since this needs to go update the global, so we only divide at the end
        # # by the n_global_images
        # cov_parallel = (w_projector @ local_covariance @ w_projector) * n_local_images
        # cov_perp = (orthogonal_projector @ local_covariance @ orthogonal_projector) * n_local_images
        # cov_cross = (w_projector @ local_covariance @ orthogonal_projector) * n_local_images
        # cov_cross += torch.t(cov_cross.clone())
        # io_cov_parallel = w_projector @ local_covariance @ results["regression_vector"][category][feature] * n_local_images
        # io_cov_cross = orthogonal_projector @ local_covariance @ results["regression_vector"][category][feature] * n_local_images
        #
        # # update running mean for global
        #
        # if res_glob["covariance_parallel"][feature] is None:
        #     res_glob["covariance_parallel"][feature] = cov_parallel.clone()
        # else:
        #     res_glob["covariance_parallel"][feature] += cov_parallel
        #
        # if res_glob["covariance_perp"][feature] is None:
        #     res_glob["covariance_perp"][feature] = cov_perp.clone()
        # else:
        #     res_glob["covariance_perp"][feature] += cov_perp
        #
        # if res_glob["covariance_cross"][feature] is None:
        #     res_glob["covariance_cross"][feature] = cov_cross.clone()
        # else:
        #     res_glob["covariance_cross"][feature] += cov_cross
        #
        # if res_glob["io_covariance_cross"][feature] is None:
        #     res_glob["io_covariance_cross"][feature] = io_cov_cross.clone()
        # else:
        #     res_glob["io_covariance_cross"][feature] += io_cov_cross
        #
        # if res_glob["io_covariance_parallel"][feature] is None:
        #     res_glob["io_covariance_parallel"][feature] = io_cov_parallel.clone()
        # else:
        #     res_glob["io_covariance_parallel"][feature] += io_cov_parallel
        # END REMOVED

    # </editor-fold>

    # update linearized global quantities
    # Note1: we multiply by the n_local_images since we later divide by n_global_images
    # Note2: we add back the local labels mean (still without subtraction of global labels mean)
    # and later subtract the global labels mean
    for feature in features:
        lin_lab_var_update = ((results["regression_vector"][category][feature] @
                               local_covariance @
                               results["regression_vector"][category][feature])
                              + results["label_mean"][category][feature] ** 2) * n_local_images
        if res_glob["linearized_labels_variance"][feature] is None:
            res_glob["linearized_labels_variance"][feature] = lin_lab_var_update.clone()
        else:
            res_glob["linearized_labels_variance"][feature] += lin_lab_var_update

    for feature in features:
        lin_lab_var_fluctuations_update = (results["regression_vector"][category][feature] @
                                           local_covariance @
                                           results["regression_vector"][category][feature]) * n_local_images
        if res_glob["linearized_labels_variance_fluctuations"][feature] is None:
            res_glob["linearized_labels_variance_fluctuations"][feature] = lin_lab_var_fluctuations_update.clone()
        else:
            res_glob["linearized_labels_variance_fluctuations"][feature] += lin_lab_var_fluctuations_update

    # Note: though linearized io_cov and io_cov are provably the same,
    # we compute it explicitly because here we are using the l2 regularized linear rule
    for feature in features:
        io_cov_linearized_update = (((results["regression_vector"][category][feature] @ local_covariance)
                                     + (results["label_mean"][category][feature] * results["centroid"][category]))
                                    * n_local_images)
        if res_glob["io_covariance_linearized"][feature] is None:
            res_glob["io_covariance_linearized"][feature] = io_cov_linearized_update.clone()
        else:
            res_glob["io_covariance_linearized"][feature] += io_cov_linearized_update

    for feature in features:
        io_cov_linearized_fluctuations_update = ((results["regression_vector"][category][feature] @ local_covariance)
                                                 * n_local_images)
        if res_glob["io_covariance_linearized_fluctuations"][feature] is None:
            res_glob["io_covariance_linearized_fluctuations"][feature] = io_cov_linearized_fluctuations_update.clone()
        else:
            res_glob["io_covariance_linearized_fluctuations"][feature] += io_cov_linearized_fluctuations_update

# divide by total number of images
res_glob["covariance"] /= n_global_images
res_glob["covariance_fluctuations"] /= n_global_images
global_reps_mean /= n_global_images
for feature in features:
    global_labels_mean[feature] /= n_global_images
    res_glob["io_covariance"][feature] /= n_global_images
    res_glob["io_covariance_fluctuations"][feature] /= n_global_images
    res_glob["io_covariance_linearized"][feature] /= n_global_images
    res_glob["io_covariance_linearized_fluctuations"][feature] /= n_global_images
    res_glob["labels_variance"][feature] /= n_global_images
    res_glob["labels_variance_fluctuations"][feature] /= n_global_images
    res_glob["linearized_labels_variance"][feature] /= n_global_images
    res_glob["linearized_labels_variance_fluctuations"][feature] /= n_global_images
    # REMOVED: STUFF BELOW NOT NEEDED ANYMORE, COMMENTING IT
    # res_glob["covariance_parallel"][feature] /= n_global_images
    # res_glob["covariance_perp"][feature] /= n_global_images
    # res_glob["covariance_cross"][feature] /= n_global_images
    # res_glob["io_covariance_cross"][feature] /= n_global_images
    # res_glob["io_covariance_parallel"][feature] /= n_global_images
    # END REMOVED

# center the covariance, io_covariance, labels variance
res_glob["covariance"] -= torch.outer(global_reps_mean, global_reps_mean)
for feature in features:
    res_glob["io_covariance"][feature] -= global_reps_mean * global_labels_mean[feature]
    res_glob["io_covariance_linearized"][feature] -= global_reps_mean * global_labels_mean[feature]
    res_glob["labels_variance"][feature] -= global_labels_mean[feature] ** 2
    res_glob["linearized_labels_variance"][feature] -= global_labels_mean[feature] ** 2

# globally center local quantities
for category in categories:
    results["centroid"][category] -= global_reps_mean
    results["centroid_norm"][category] = torch.linalg.vector_norm(results["centroid"][category])
    for feature in features:
        results["label_mean"][category][feature] -= global_labels_mean[feature]
        results["bias"][category][feature] = (results["label_mean"][category][feature] -
                                              einsum(results["centroid"][category],
                                                     results["regression_vector"][category][feature],
                                                     "neuron, neuron ->"))

# <editor-fold desc="Compute actual global quantities">
# compute the pseudo-inverse of global covariance
pseudo_inverse = torch.linalg.pinv(res_glob["covariance"], hermitian=True)

# compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eigh(res_glob["covariance"])
res_glob["covariance_eigenvalues"] = eigenvalues
res_glob["covariance_eigenvectors"] = eigenvectors

# compute total variance
res_glob["total_variance"] = torch.trace(res_glob["covariance"])

print("Results of global regression:", flush=True)
for feature in features:
    print(f"Feature: {feature}", flush=True)
    # # compute regression vector
    # # K^+ b, where K is covariance and b is io-covariance. Here + is the pseudo-inverse
    res_glob["regression_vector"][feature] = pseudo_inverse @ res_glob["io_covariance"][feature]

    # compute normalized regression error
    res_glob["normalized_regression_error"][feature] = \
        (1 -
         einsum(res_glob["io_covariance"][feature], pseudo_inverse, res_glob["io_covariance"][feature],
                "neuron1, neuron1 neuron2, neuron2 ->") / res_glob["labels_variance"][feature])
    print(f"Normalized MSE, global: {res_glob['normalized_regression_error'][feature]}",
          flush=True)
    print(
        f"Normalized regression error (sqrt MSE), global: {torch.sqrt(res_glob['normalized_regression_error'][feature])}",
        flush=True)
    print(f"Pearson coefficient: {torch.sqrt(1 - res_glob['normalized_regression_error'][feature])}", flush=True)

    # compute regression error
    res_glob["regression_error"][feature] = (res_glob["normalized_regression_error"][feature] *
                                             res_glob["labels_variance"][feature])

    # compute linear_rule vs PCs overlap (added with v2)
    w_direction = res_glob["regression_vector"][feature] / torch.norm(res_glob["regression_vector"][feature])
    # eigenvector are organized as [components, eigenvector/eigenvalue number]
    overlaps = einsum(w_direction, res_glob["covariance_eigenvectors"], "neuron, neuron eig -> eig")
    res_glob["linear_rule_vs_PC_overlap"][feature] = overlaps

    # compute global-local error gap

    # compute regression error gap
    res_glob["regression_error_gap"][feature] = \
        (res_glob["linearized_labels_variance"][feature] -
         einsum(res_glob["io_covariance_linearized"][feature], pseudo_inverse,
                res_glob["io_covariance_linearized"][feature],
                "neuron1, neuron1 neuron2, neuron2 ->"))

    # compute normalized regression error gap
    res_glob["normalized_regression_error_gap"][feature] = (res_glob["regression_error_gap"][feature]
                                                            / res_glob["labels_variance"][feature])

    print(f"Normalized MSE gap, global: {res_glob['normalized_regression_error_gap'][feature]}",
          flush=True)
# </editor-fold>

# <editor-fold desc="compute additional theory quantities">
for feature in features:
    # compute strength of common regression direction (added with v3)
    res_glob["mean_regression_direction_norm"][feature] = (
        torch.linalg.vector_norm(res_glob["mean_regression_direction"][feature]))

    res_glob["mean_regression_direction_unit_vector"][feature] = (
            res_glob["mean_regression_direction"][feature] / res_glob["mean_regression_direction_norm"][feature])

    # compute pearson coefficient
    # -> gather stds in lists
    lin_label_stds = []
    linear_rule_stds = []
    for category in categories:
        lin_label_stds.append(np.sqrt(results["linearized_label_variance"][category][feature].item()))
        linear_rule_stds.append(np.sqrt(results["variance_along_linear_rule"][category][feature].item()))
    lin_label_stds = np.array(lin_label_stds)
    linear_rule_stds = np.array(linear_rule_stds)
    # -> compute pearson
    numerator = np.dot(lin_label_stds, linear_rule_stds) / np.shape(lin_label_stds)[0]
    denominator = np.sqrt(res_glob["variance_along_local_linear_rules"][feature].item() *
                          res_glob["linearized_labels_variance"][feature].item())
    res_glob["variance_pearson"][feature] = numerator / denominator

    # compute linear rule scale and related quantities
    projections = []
    for category in categories:
        # compute
        local_norm = torch.norm(results["regression_vector"][category][feature])
        local_scale = 1 / local_norm
        local_direction = results["regression_vector"][category][feature] / local_norm
        global_direction = res_glob["mean_regression_direction_unit_vector"][feature]
        overlap = torch.dot(local_direction, global_direction)
        projected_scale = local_scale * overlap
        # store
        projections.append(projected_scale)
        results["linear_rule_scale"][category][feature] = local_scale
        results["linear_rule_scale_projected"][category][feature] = projected_scale
        results["local_vs_common_linear_rule_overlap"][category][feature] = overlap
    projections = torch.stack(projections)
    res_glob["linear_rule_scale_mean"][feature] = torch.mean(projections)
    res_glob["linear_rule_scale_var"][feature] = torch.var(projections, unbiased=False)

# </editor-fold>

# STORE
# Ensure gamma exists for backward compatibility (set to 0 if missing)
if not hasattr(args, "gamma") or args.gamma is None:
    args.gamma = 0

args_dictionary = vars(args).copy()

final_results = {"local": results, "global": res_glob, "args": args_dictionary}

final_results_numpy = convert_tensors_to_numpy(final_results)

# Save the dictionary to a .pkl file

# get the directory containing the checkpoint file
checkpoint_dir = os.path.dirname(args.checkpoint)
# build the path to the new results folder
results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
# create it (and any missing parents), no error if it already exists
os.makedirs(results_dir, exist_ok=True)

if args.local_dimensionality_reduction_style != "":
    dim_red_string = f"_LocalRed_{args.local_dimensionality_reduction_style}"
    if args.local_dimensionality_reduction_style == "percentage":
        dim_red_string += f"_{args.local_dimensionality_reduction_strength}"
else:
    dim_red_string = ""

if args.random_projection_size is not None:
    rnd_proj_string = f"_RndProj_size{args.random_projection_size}_seed{args.random_projection_seed}"
else:
    rnd_proj_string = ""

rnd_sub_string = (f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
                  if args.random_subsample_size is not None else "")

manifolds_sub_string = (f"_ManSub_size{args.manifold_subsample_size}_seed{args.manifold_subsample_seed}"
                        if args.manifold_subsample_size is not None else "")

if args.global_dimensionality_reduction_style != "":
    global_dim_red_string = f"_GlobalRed_{args.global_dimensionality_reduction_style}_Dim{global_d}_KV{global_explained_variance}"
else:
    global_dim_red_string = ""

dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))

results_file = os.path.join(results_dir,
                            f"{args.job_id}_TheoryRegression_{dataset_name}_{args.layer}" + dim_red_string + rnd_proj_string + rnd_sub_string + manifolds_sub_string + global_dim_red_string + f"_gamma{args.gamma}" + ".pkl")

with open(results_file, 'wb') as file:
    pickle.dump(final_results_numpy, file)

print("Regression concluded. Exit", flush=True)

exit()
