import os
import argparse
import pickle
import tempfile
from pathlib import Path
from functools import partial
from random import Random
import hashlib
import json
import random
import warnings

import numpy as np
import torch
from einops import einsum
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from hadamard_transform import hadamard_transform, next_power_of_2, pad_to_power_of_2, is_a_power_of_2

# IMPORT DATALOADERS
# add path, import is done later
import sys
sys.path.append('../../codebase_v5')
# IMPORT MODELS
# -> AvgPool_SDXLdataset
sys.path.append('../ANN_models/AvgPool_SDXLdataset')
import LIB_model

ALL_BBOX_FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]


parser = argparse.ArgumentParser(
    description=(
        "Resample the local centered manifold fluctuations using the theory file produced by the current pipeline, "
        "and store a theory-results file that matches the current real-data theory format as closely as possible."
    )
)
parser.add_argument("--job_id", type=str, default="0000",
                    help="A name/ID for this run (used for naming the results).")
parser.add_argument("--results_folder_name", type=str, default="./",
                    help="Folder, relative to the checkpoint directory, where the output will be stored.")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use ('cuda' or 'cpu').")
parser.add_argument("--original_theory_file", type=str, required=True,
                    help="Path to the current theory-results file on real data. This file provides both the pipeline args and the local rules.")
parser.add_argument("--feature", type=str, required=True, choices=ALL_BBOX_FEATURES,
                    help="Single bbox feature to process in this run.")
parser.add_argument("--manifolds_resampling_seed", type=int, default=1,
                    help="Master seed used to derive one deterministic resampling seed per category.")
parser.add_argument("--subsample_sizes", type=int, nargs="*", default=[],
                    help="Optional list of manifold subset sizes to evaluate in the same run, e.g. --subsample_sizes 8 32 64.")
parser.add_argument("--n_independent_subsamplings", type=int, default=0,
                    help="How many independent nested subsampling runs to generate.")
parser.add_argument("--subsampling_seed_offset", type=int, default=1,
                    help="First seed used for the nested manifold subsampling runs. Run seeds are offset + 0, 1, ..., n-1.")
parser.add_argument("--temporary_folder_name", type=str, default=None,
                    help="Deprecated and ignored. Temporary subset accumulators are always created under <results_dir>/temp.")


# -------------------------------
# Generic helper utilities
# -------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch in the same way as the current pipeline."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def convert_tensors_to_numpy(d):
    """Recursively convert torch tensors to NumPy, while leaving existing NumPy arrays and memmaps untouched."""
    if isinstance(d, dict):
        return {key: convert_tensors_to_numpy(value) for key, value in d.items()}
    if isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    return d



def participation_ratio(eigs: torch.Tensor) -> int:
    pr = (eigs.sum() ** 2) / eigs.pow(2).sum()
    return int(torch.round(pr).item())



def explained_variance_ratio(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    if eigs.numel() == 0 or dims_kept <= 0:
        return torch.tensor(0.0, dtype=eigs.dtype, device=eigs.device)
    eigs_sorted = torch.sort(eigs, descending=True).values
    k = min(int(dims_kept), eigs_sorted.numel())
    explained = eigs_sorted[:k].sum()
    total = eigs_sorted.sum()
    return explained / total



def explained_variance_percent(eigs: torch.Tensor, dims_kept: int) -> torch.Tensor:
    return explained_variance_ratio(eigs, dims_kept) * 100



def explained_variance_dim(eigs, threshold=0.9):
    eigs_sorted = torch.sort(eigs, descending=True).values
    cumulative = torch.cumsum(eigs_sorted, dim=0)
    total = torch.sum(eigs_sorted)
    explained_ratio = cumulative / total
    indices = (explained_ratio >= threshold).nonzero(as_tuple=False)
    if indices.numel() == 0:
        return len(eigs_sorted)
    return indices[0].item() + 1



def print_args(args):
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:32s}: {value!r}")
    print("=========================", flush=True)


def category_seeds(all_categories, master_seed, bits=32):
    """Derive one deterministic seed per category. This keeps the per-category resampling independent of loop order."""
    rng = Random(master_seed)
    limit = 1 << bits
    return {cat: rng.randrange(limit) for cat in all_categories}



def safe_unit_vector(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x)
    if norm <= eps:
        out = torch.zeros_like(x)
        out[0] = 1.0
        return out
    return x / norm



def safe_unit_vector_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= eps:
        out = np.zeros_like(x)
        out[0] = 1.0
        return out
    return x / norm


# -------------------------------
# Resampling helpers copied from the old resampling script
# -------------------------------

def rotation_matrix_from_a_to_b(a, b):
    """Return an orthogonal matrix that rotates unit vector a onto unit vector b."""
    dot = torch.dot(a, b)

    if dot < -0.999999:
        arbitrary = torch.zeros_like(a)
        arbitrary[0] = 1.0
        if torch.abs(a[0]) > 0.9 and a.numel() > 1:
            arbitrary[1] = 1.0
        v = arbitrary - a * torch.dot(a, arbitrary)
        v = v / torch.linalg.vector_norm(v)
        return torch.eye(a.size(0), device=a.device, dtype=a.dtype) - 2.0 * torch.outer(v, v)

    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
    v1 = a
    v2 = b - dot * a
    v2 = v2 / torch.linalg.vector_norm(v2)

    I = torch.eye(a.size(0), device=a.device, dtype=a.dtype)
    R = I + (torch.cos(theta) - 1) * (torch.outer(v1, v1) + torch.outer(v2, v2)) \
        + torch.sin(theta) * (torch.outer(v2, v1) - torch.outer(v1, v2))
    return R



def random_rotation_around_axis(new_direction):
    """Return a random orthogonal transformation that leaves new_direction unchanged."""
    n = new_direction.shape[0]
    device = new_direction.device
    dtype = new_direction.dtype

    if n == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)

    e1 = torch.zeros_like(new_direction)
    e1[0] = 1.0

    if torch.allclose(new_direction, e1, atol=1e-6):
        A = torch.eye(n, device=device, dtype=dtype)
    else:
        v = new_direction - e1
        v = v / torch.linalg.vector_norm(v)
        A = torch.eye(n, device=device, dtype=dtype) - 2.0 * torch.outer(v, v)

    random_matrix = torch.randn(n - 1, n - 1, device=device, dtype=dtype)
    Q, R_qr = torch.linalg.qr(random_matrix)
    d = torch.sign(torch.diag(R_qr))
    d[d == 0] = 1.0
    Q *= d.unsqueeze(0)

    B = torch.eye(n, device=device, dtype=dtype)
    B[1:, 1:] = Q
    return A.t() @ B @ A


# -------------------------------
# Accumulator helpers
# -------------------------------

def make_empty_global_like(feature):
    """Create an empty global-results dictionary with the same schema as the current real-data theory script."""
    return {
        "covariance": None,
        "covariance_fluctuations": None,
        "covariance_eigenvalues": None,
        "covariance_eigenvectors": None,
        "linear_rule_vs_PC_overlap": {},
        "io_covariance": {feature: None},
        "io_covariance_fluctuations": {feature: None},
        "io_covariance_linearized": {feature: None},
        "io_covariance_linearized_fluctuations": {feature: None},
        "regression_error": {},
        "normalized_regression_error": {},
        "regression_error_gap": {},
        "normalized_regression_error_gap": {},
        "regression_vector": {},
        "labels_variance": {feature: None},
        "labels_variance_fluctuations": {feature: None},
        "linearized_labels_variance": {feature: None},
        "linearized_labels_variance_fluctuations": {feature: None},
        "variance_along_local_linear_rules": {feature: 0.0},
        "mean_regression_direction": {feature: None},
        "mean_regression_direction_norm": {},
        "mean_regression_direction_unit_vector": {},
        "total_variance": None,
        "variance_pearson": {},
        "linear_rule_scale_mean": {},
        "linear_rule_scale_var": {},
    }



def make_empty_local_results(categories, feature):
    """Create the local-results dictionary, using the same keys as the current theory script."""
    return {
        "n_images": {},
        "covariance_eigenvalues": {},
        "linear_rule_vs_PC_overlap": {category: {} for category in categories},
        "centroid": {category: None for category in categories},
        "centroid_norm": {},
        "label_mean": {category: {feature: None} for category in categories},
        "label_variance": {category: {feature: None} for category in categories},
        "linearized_label_variance": {category: {} for category in categories},
        "io_covariance": {category: {feature: None} for category in categories},
        "regression_error": {category: {} for category in categories},
        "normalized_regression_error": {category: {} for category in categories},
        "regression_vector": {category: {} for category in categories},
        "regression_vector_norm": {category: {} for category in categories},
        "bias": {category: {} for category in categories},
        "variance_along_linear_rule": {category: {} for category in categories},
        "variances_orthogonal": {category: {} for category in categories},
        "covariance_cross_terms": {category: {} for category in categories},
        "total_variance": {category: None for category in categories},
        "linear_rule_scale": {category: {} for category in categories},
        "linear_rule_scale_projected": {category: {} for category in categories},
        "local_vs_common_linear_rule_overlap": {category: {} for category in categories},
    }



def make_in_memory_accumulator(dim: int):
    """Running sums for the full-category global block. These are centered-only by construction."""
    return {
        "n_images": 0,
        "covariance_sum": np.zeros((dim, dim), dtype=np.float64),
        "io_covariance_sum": np.zeros(dim, dtype=np.float64),
        "io_covariance_linearized_sum": np.zeros(dim, dtype=np.float64),
        "labels_variance_sum": 0.0,
        "linearized_labels_variance_sum": 0.0,
    }



def update_in_memory_accumulator(acc, cov_update, io_update, io_lin_update, lab_var_update, lin_lab_var_update, n_local_images):
    acc["n_images"] += int(n_local_images)
    acc["covariance_sum"] += cov_update
    acc["io_covariance_sum"] += io_update
    acc["io_covariance_linearized_sum"] += io_lin_update
    acc["labels_variance_sum"] += float(lab_var_update)
    acc["linearized_labels_variance_sum"] += float(lin_lab_var_update)



def create_disk_backed_subset_accumulators(tmp_root: Path, subset_definitions, dim: int):
    """
    Create one disk-backed accumulator per subset.

    The point of this design is to keep the script memory-safe even when many subset-global
    covariance matrices are requested in the same run. We still only keep one category's local
    covariance in memory at a time, exactly like the current real-data theory script.
    """
    accs = {}
    for key, meta in subset_definitions.items():
        subset_dir = tmp_root / key
        subset_dir.mkdir(parents=True, exist_ok=True)
        accs[key] = {
            "size": meta["size"],
            "seed": meta["seed"],
            "categories": meta["categories"],
            "n_images": 0,
            "covariance_sum": np.memmap(subset_dir / "covariance_sum.dat", mode="w+", dtype=np.float64, shape=(dim, dim)),
            "io_covariance_sum": np.memmap(subset_dir / "io_covariance_sum.dat", mode="w+", dtype=np.float64, shape=(dim,)),
            "io_covariance_linearized_sum": np.memmap(subset_dir / "io_covariance_linearized_sum.dat", mode="w+", dtype=np.float64, shape=(dim,)),
            "labels_variance_sum": 0.0,
            "linearized_labels_variance_sum": 0.0,
        }
        accs[key]["covariance_sum"][:] = 0.0
        accs[key]["io_covariance_sum"][:] = 0.0
        accs[key]["io_covariance_linearized_sum"][:] = 0.0
        accs[key]["covariance_sum"].flush()
        accs[key]["io_covariance_sum"].flush()
        accs[key]["io_covariance_linearized_sum"].flush()
    return accs



def update_disk_backed_accumulator(acc, cov_update, io_update, io_lin_update, lab_var_update, lin_lab_var_update, n_local_images):
    acc["n_images"] += int(n_local_images)
    acc["covariance_sum"][:] += cov_update
    acc["io_covariance_sum"][:] += io_update
    acc["io_covariance_linearized_sum"][:] += io_lin_update
    acc["labels_variance_sum"] += float(lab_var_update)
    acc["linearized_labels_variance_sum"] += float(lin_lab_var_update)



def build_nested_subsamples(categories, requested_sizes, n_runs, seed_offset=1):
    """
    Build nested category subsets.

    For a fixed run seed, the size-64 subset contains the size-32 subset, which contains the
    size-8 subset, because every subset is just a longer prefix of the same seeded permutation.
    """
    sizes = sorted({int(s) for s in requested_sizes if int(s) > 0 and int(s) < len(categories)})
    if not sizes or n_runs <= 0:
        return {}, {category: [] for category in categories}

    subset_definitions = {}
    membership_map = {category: [] for category in categories}

    for run_idx in range(n_runs):
        run_seed = seed_offset + run_idx
        permuted_categories = list(categories)
        Random(run_seed).shuffle(permuted_categories)

        for size in sizes:
            chosen = permuted_categories[:size]
            key = f"size_{size}_seed_{run_seed}"
            subset_definitions[key] = {
                "size": size,
                "seed": run_seed,
                "categories": chosen,
            }
            for category in chosen:
                membership_map[category].append(key)

    return subset_definitions, membership_map



def finalize_global_from_accumulator(acc, local_results, categories_subset, feature):
    """
    Turn a centered-only running accumulator into the final global block.

    Because this script intentionally tests only the fluctuation theory, we never re-add local
    centroids or local label means. As a consequence, the fluctuation and non-fluctuation global
    objects are identical here, and the centroid contribution is exactly zero by construction.
    """
    if acc["n_images"] <= 0:
        raise ValueError("Cannot finalize a global block with zero images.")

    n_images = float(acc["n_images"])
    covariance = np.asarray(acc["covariance_sum"]) / n_images
    io_covariance = np.asarray(acc["io_covariance_sum"]) / n_images
    io_covariance_linearized = np.asarray(acc["io_covariance_linearized_sum"]) / n_images
    labels_variance = float(acc["labels_variance_sum"] / n_images)
    linearized_labels_variance = float(acc["linearized_labels_variance_sum"] / n_images)

    covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
    pseudo_inverse = np.linalg.pinv(covariance, hermitian=True)
    regression_vector = pseudo_inverse @ io_covariance

    denom = labels_variance
    reg_term = float(io_covariance @ pseudo_inverse @ io_covariance)
    regression_error = float(labels_variance - reg_term)
    normalized_regression_error = float(regression_error / denom) if denom != 0 else np.nan

    regression_error_gap = float(linearized_labels_variance - (io_covariance_linearized @ pseudo_inverse @ io_covariance_linearized))
    normalized_regression_error_gap = float(regression_error_gap / denom) if denom != 0 else np.nan

    regression_vector_norm = np.linalg.norm(regression_vector)
    global_direction = safe_unit_vector_np(regression_vector)
    global_overlap = global_direction @ covariance_eigenvectors

    variance_along_local_linear_rules = float(np.mean([
        float(local_results["variance_along_linear_rule"][category][feature])
        for category in categories_subset
    ]))

    mean_regression_direction = np.mean([
        safe_unit_vector_np(np.asarray(local_results["regression_vector"][category][feature]))
        for category in categories_subset
    ], axis=0)
    mean_regression_direction_norm = float(np.linalg.norm(mean_regression_direction))
    mean_regression_direction_unit_vector = safe_unit_vector_np(mean_regression_direction)

    lin_label_stds = np.array([
        np.sqrt(float(local_results["linearized_label_variance"][category][feature]))
        for category in categories_subset
    ], dtype=np.float64)
    linear_rule_stds = np.array([
        np.sqrt(float(local_results["variance_along_linear_rule"][category][feature]))
        for category in categories_subset
    ], dtype=np.float64)
    numerator = float(np.dot(lin_label_stds, linear_rule_stds) / len(categories_subset))
    denominator = np.sqrt(variance_along_local_linear_rules * linearized_labels_variance)
    variance_pearson = float(numerator / denominator) if denominator > 0 else np.nan

    projected_scales = []
    for category in categories_subset:
        local_w = np.asarray(local_results["regression_vector"][category][feature])
        local_norm = float(local_results["regression_vector_norm"][category][feature])
        if local_norm <= 0:
            projected_scales.append(np.nan)
            continue
        local_direction = local_w / local_norm
        overlap = float(local_direction @ mean_regression_direction_unit_vector)
        projected_scale = (1.0 / local_norm) * overlap
        projected_scales.append(projected_scale)

    projected_scales = np.array(projected_scales, dtype=np.float64)
    linear_rule_scale_mean = float(np.mean(projected_scales))
    linear_rule_scale_var = float(np.var(projected_scales))

    res_glob = make_empty_global_like(feature)
    res_glob["covariance"] = covariance
    res_glob["covariance_fluctuations"] = covariance.copy()
    res_glob["covariance_eigenvalues"] = covariance_eigenvalues
    res_glob["covariance_eigenvectors"] = covariance_eigenvectors
    res_glob["linear_rule_vs_PC_overlap"][feature] = global_overlap
    res_glob["io_covariance"][feature] = io_covariance
    res_glob["io_covariance_fluctuations"][feature] = io_covariance.copy()
    res_glob["io_covariance_linearized"][feature] = io_covariance_linearized
    res_glob["io_covariance_linearized_fluctuations"][feature] = io_covariance_linearized.copy()
    res_glob["regression_error"][feature] = regression_error
    res_glob["normalized_regression_error"][feature] = normalized_regression_error
    res_glob["regression_error_gap"][feature] = regression_error_gap
    res_glob["normalized_regression_error_gap"][feature] = normalized_regression_error_gap
    res_glob["regression_vector"][feature] = regression_vector
    res_glob["labels_variance"][feature] = labels_variance
    res_glob["labels_variance_fluctuations"][feature] = labels_variance
    res_glob["linearized_labels_variance"][feature] = linearized_labels_variance
    res_glob["linearized_labels_variance_fluctuations"][feature] = linearized_labels_variance
    res_glob["variance_along_local_linear_rules"][feature] = variance_along_local_linear_rules
    res_glob["mean_regression_direction"][feature] = mean_regression_direction
    res_glob["mean_regression_direction_norm"][feature] = mean_regression_direction_norm
    res_glob["mean_regression_direction_unit_vector"][feature] = mean_regression_direction_unit_vector
    res_glob["total_variance"] = float(np.trace(covariance))
    res_glob["variance_pearson"][feature] = variance_pearson
    res_glob["linear_rule_scale_mean"][feature] = linear_rule_scale_mean
    res_glob["linear_rule_scale_var"][feature] = linear_rule_scale_var
    return res_glob



def compute_local_projection_quantities(local_covariance_resampled: torch.Tensor, w_new: torch.Tensor):
    """Compute Sigma_1d and the orthogonal variances in the same way as the current theory script."""
    N = local_covariance_resampled.shape[0]
    w_direction = safe_unit_vector(w_new)

    Q = torch.empty((N, N), dtype=local_covariance_resampled.dtype, device=local_covariance_resampled.device)
    Q[:, 0] = w_direction

    if N > 1:
        rand_mat = torch.randn(N, N - 1, dtype=local_covariance_resampled.dtype, device=local_covariance_resampled.device)
        for i in range(N - 1):
            vec = rand_mat[:, i]
            vec = vec - torch.dot(vec, w_direction) * w_direction
            rand_mat[:, i] = vec
        Q_rest, _ = torch.linalg.qr(rand_mat)
        Q[:, 1:] = Q_rest

        C_rot = Q.t() @ local_covariance_resampled @ Q
        Sigma_1d = C_rot[0, 1:]
        C22 = C_rot[1:, 1:]
        eigenvalues, V = torch.linalg.eigh(C22)
        eigenvalues = eigenvalues.flip(0)
        V = V.flip(1)
        sigma_d_sq = eigenvalues
        sigma_1d_rotated = Sigma_1d @ V
    else:
        sigma_d_sq = torch.empty((0,), dtype=local_covariance_resampled.dtype, device=local_covariance_resampled.device)
        sigma_1d_rotated = torch.empty((0,), dtype=local_covariance_resampled.dtype, device=local_covariance_resampled.device)

    return w_direction, sigma_d_sq, sigma_1d_rotated



def build_resampling_statistics(original_theory_dict, categories, feature, device):
    """
    Estimate the across-category statistics of the local linear rules.

    These are the exact quantities the old resampling script used: the mean direction, its norm c,
    and the mean and variance of the local rule norm r_mu.
    """
    r_list = []
    u1_list = []
    for category in categories:
        w_mu = np.asarray(original_theory_dict["local"]["regression_vector"][category][feature])
        r_mu = np.linalg.norm(w_mu)
        if r_mu <= 1e-12:
            raise ValueError(f"Category {category} has an almost-zero local rule for feature {feature}. Cannot define u1.")
        r_list.append(r_mu)
        u1_list.append(w_mu / r_mu)

    r_list = np.asarray(r_list, dtype=np.float64)
    u1_list = np.asarray(u1_list, dtype=np.float64)

    mean_r = torch.tensor(np.mean(r_list), dtype=torch.float64, device=device)
    var_r = torch.tensor(np.var(r_list), dtype=torch.float64, device=device)

    cu = np.mean(u1_list, axis=0)
    c_value = float(np.linalg.norm(cu))
    if c_value <= 1e-12:
        u = np.zeros_like(cu)
        u[0] = 1.0
        c_value = 0.0
    else:
        u = cu / c_value

    return {
        "mean_r": mean_r,
        "var_r": var_r,
        "u": torch.tensor(u, dtype=torch.float64, device=device),
        "c": torch.tensor(c_value, dtype=torch.float64, device=device),
    }


# -------------------------------
# Load theory file, merge args, and initialize pipeline pieces
# -------------------------------
args_new = parser.parse_args()
with open(args_new.original_theory_file, "rb") as f:
    original_theory_dict = pickle.load(f)

saved_args = original_theory_dict["args"]
args = argparse.Namespace(**{**saved_args, **vars(args_new)})

# Keep backward compatibility with older theory files and older cv files.
for k in ["random_subsample_size", "random_subsample_seed"]:
    if not hasattr(args, k):
        setattr(args, k, None)

for k, v in {
    "local_dimensionality_reduction_style": "",
    "local_dimensionality_reduction_strength": 100,
}.items():
    if not hasattr(args, k):
        setattr(args, k, v)

for k, v in {
    "global_dimensionality_reduction_style": "",
    "global_dimensionality_reduction_strength": 100.0,
    "global_dimensionality_reduction_dimensions": 2048,
    "global_dimensionality_reduction_file": "./boh.pkl",
}.items():
    if not hasattr(args, k):
        setattr(args, k, v)

for k, v in {
    "manifold_subsample_size": None,
    "manifold_subsample_seed": 1,
    "manifold_subsampling_grouping_json": None,
}.items():
    if not hasattr(args, k):
        setattr(args, k, v)

if not hasattr(args, "n_subsampled_images"):
    args.n_subsampled_images = None
if not hasattr(args, "images_subsampling_seed"):
    args.images_subsampling_seed = 1
if not hasattr(args, "augment_dihedral"):
    args.augment_dihedral = False

# The current theory script may have stored the old cv file path under a different arg name.
if not hasattr(args, "cv_local_rule_file") and hasattr(args, "original_theory_file"):
    pass

print_args(args)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset_path = Path(args.dataset_folder)
if dataset_path.is_file() or dataset_path.suffix.lower() == ".h5":
    dataset_folder_name = dataset_path.parent.name
else:
    dataset_folder_name = dataset_path.name

# Load the model and transforms exactly like the current real-data theory script.
if os.path.basename(os.path.normpath(args.checkpoint)).lower() == "resnet50":
    model = LIB_model.ResNetBBoxModel(resnet_version="resnet50").to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
elif dataset_folder_name in [
    "representations_DicarloAsGenai_public_IT", "representations_DicarloAsGenai_public_V4",
]:
    model = None
    transform = None
    print("\nRepresentations dataset detected. Setting model to None\n")
else:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ckpt["model"].to(device)
    model.eval()
    transform = ckpt["transform"]

print(f"Dataset: {dataset_folder_name}", flush=True)

if dataset_folder_name in [
    "SDXL_dataset_test", "SDXL_dataset_train", "SDXL_dataset_validation",
    "DicarloAsGenai_public",
]:
    from LIB_dataloader import CategoryDatasetSDXL
    CategoryDataset = partial(
        CategoryDatasetSDXL,
        dataset_folder=args.dataset_folder,
        transform=transform,
        augment_dihedral=args.augment_dihedral,
    )
elif dataset_folder_name in [
    "representations_DicarloAsGenai_public_IT", "representations_DicarloAsGenai_public_V4",
]:
    from LIB_dataloader import CategoryDatasetRepresentation
    CategoryDataset = partial(
        CategoryDatasetRepresentation,
        dataset_folder=args.dataset_folder,
        transform=transform,
    )
else:
    raise ValueError(f"Dataset {dataset_folder_name} is not implemented in this script.")

# We intentionally run only one feature in this script.
feature = args.feature
feature_idx = ALL_BBOX_FEATURES.index(feature)
features = [feature]

# Use the categories in the input theory file as the authoritative category set.
# This guarantees that the local rules we need are available and that the category list matches the theory file.
categories = sorted(original_theory_dict["local"]["centroid"].keys())
if not categories:
    raise ValueError("The input theory file contains no categories.")

# Sanity-check that the chosen feature exists in the theory file.
example_category = categories[0]
if feature not in original_theory_dict["local"]["regression_vector"][example_category]:
    raise KeyError(f"Feature {feature} is not present in the input theory file.")

# Make sure those categories also exist on disk.
for category in categories:
    if feature not in original_theory_dict["local"]["regression_vector"][category]:
        raise KeyError(f"Category {category} is missing feature {feature} in the input theory file.")

resampling_stats = build_resampling_statistics(original_theory_dict, categories, feature, device)
cat_resampling_seeds = category_seeds(categories, args.manifolds_resampling_seed)

subset_definitions, subset_membership_map = build_nested_subsamples(
    categories=categories,
    requested_sizes=args.subsample_sizes,
    n_runs=args.n_independent_subsamplings,
    seed_offset=args.subsampling_seed_offset,
)

print(f"Number of categories in base run: {len(categories)}", flush=True)
print(f"Requested subset runs: {len(subset_definitions)}", flush=True)

# Resolve the output folder early so temporary files can live next to the final results.
# This follows the same checkpoint-relative convention used for the final pickle output.
checkpoint_dir = os.path.dirname(args.checkpoint) if hasattr(args, "checkpoint") else ""
if checkpoint_dir == "":
    checkpoint_dir = os.path.dirname(args.original_theory_file)
results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
os.makedirs(results_dir, exist_ok=True)

local_results = make_empty_local_results(categories, feature)
full_global_acc = None
subset_accumulators = None

fjlt_idx = None
fjlt_D = None
subsample_idx = None

if args.global_dimensionality_reduction_style != "":
    with open(args.global_dimensionality_reduction_file, "rb") as f:
        global_stats = pickle.load(f)
    global_eigenvalues = torch.tensor(global_stats["covariance_eigenvalues"], device=device, dtype=torch.float64)
    global_eigenvectors = torch.tensor(global_stats["covariance_eigenvectors"], device=device, dtype=torch.float64)
    global_projection_mean = torch.tensor(global_stats["mean"], device=device, dtype=torch.float64)
    if args.global_dimensionality_reduction_style == "dimensions":
        global_d = args.global_dimensionality_reduction_dimensions
        global_explained_variance = explained_variance_percent(global_eigenvalues, global_d)
        global_projector = global_eigenvectors[:, :global_d]
    elif args.global_dimensionality_reduction_style == "percentage":
        global_explained_variance = args.global_dimensionality_reduction_strength
        global_d = explained_variance_dim(global_eigenvalues, threshold=(args.global_dimensionality_reduction_strength / 100))
        global_projector = global_eigenvectors[:, :global_d]
    elif args.global_dimensionality_reduction_style == "neurons":
        neuron_var = einsum(global_eigenvectors ** 2, global_eigenvalues, "neuron eig, eig -> neuron")
        global_d = min(args.global_dimensionality_reduction_dimensions, neuron_var.numel())
        top_var, top_idx = torch.topk(neuron_var, k=global_d, largest=True, sorted=True)
        total_var = neuron_var.sum()
        global_explained_variance = (top_var.sum() / total_var * 100.0).item() if total_var > 0 else float("nan")
        N = neuron_var.numel()
        global_projector = torch.zeros((N, global_d), dtype=torch.float64, device=device)
        global_projector[top_idx, torch.arange(global_d, device=device)] = 1.0
    else:
        raise ValueError(f"Unsupported global dimensionality reduction style: {args.global_dimensionality_reduction_style}")
    print(
        f"Initializing global projector. Keeping first {global_d} dimensions, or {global_explained_variance}% of the variance",
        flush=True,
    )
else:
    global_d = None
    global_explained_variance = None

# Temporary directory for disk-backed subset accumulators.
# We keep these files next to the final results, inside a dedicated temp/ folder,
# so they do not end up in the system-wide temporary directory.
tmp_base = Path(results_dir) / "temp"
tmp_base.mkdir(parents=True, exist_ok=True)
temporary_directory_context = tempfile.TemporaryDirectory(prefix="subset_acc_", dir=str(tmp_base))

with temporary_directory_context as tmp_dir:
    tmp_dir = Path(tmp_dir)

    # ------------------------------------------------------------
    # Main category loop.
    # We intentionally follow the same overall pattern as the current real-data theory script:
    # compute one category at a time, update running global quantities immediately, and never keep
    # local covariance matrices from previous categories in memory.
    # ------------------------------------------------------------
    for category in categories:
        print(f"Processing category: {category}", flush=True)

        dataset = CategoryDataset(category=category)

        # Optional per-category image subsampling. This block is copied from the current theory script.
        if args.n_subsampled_images is not None:
            N_dataset = len(dataset)
            k = max(0, min(int(args.n_subsampled_images), N_dataset))
            if k < N_dataset:
                cat_key = f"{args.images_subsampling_seed}:{category}".encode("utf-8")
                cat_seed = int.from_bytes(hashlib.sha256(cat_key).digest()[:4], "big")
                rng = np.random.RandomState(cat_seed)
                subset_idx_images = rng.choice(np.arange(N_dataset), size=k, replace=False)
                subset_idx_images.sort()
                dataset = Subset(dataset, subset_idx_images.tolist())

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Running local sums. These are exactly the objects we need to center the manifold and labels.
        local_covariance_sum = None
        local_centroid_sum = None
        local_label_sum = None
        local_label_sq_sum = None
        local_io_covariance_sum = None
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

            if (fjlt_D is None) and (fjlt_idx is None) and (args.random_projection_size is not None):
                seed_everything(args.random_projection_seed)
                N_orig = reps.size(1)
                if is_a_power_of_2(N_orig):
                    N_power2 = N_orig
                else:
                    N_power2 = next_power_of_2(N_orig)
                D = (torch.randint(0, 2, (N_power2,), device=device) * 2 - 1).to(torch.float64)
                idx = torch.randperm(N_power2, device=device)[:args.random_projection_size]
                fjlt_D = D
                fjlt_idx = idx

            if args.random_projection_size is not None:
                reps = pad_to_power_of_2(reps)
                reps_d = reps * fjlt_D[None, :]
                reps_h = hadamard_transform(reps_d)
                reps = reps_h[:, fjlt_idx] * np.sqrt(N_power2 / args.random_projection_size)

            if args.global_dimensionality_reduction_style != "":
                reps = (reps - global_projection_mean) @ global_projector

            labels = gt_bboxes[:, feature_idx].to(dtype=torch.float64)

            cov_addition = einsum(reps, reps, "image neuron1, image neuron2 -> neuron1 neuron2")
            centroid_addition = torch.sum(reps, dim=0)
            label_sum_addition = torch.sum(labels)
            label_sq_sum_addition = torch.sum(labels ** 2)
            io_addition = einsum(reps, labels, "image neuron, image -> neuron")

            local_covariance_sum = cov_addition.clone() if local_covariance_sum is None else local_covariance_sum + cov_addition
            local_centroid_sum = centroid_addition.clone() if local_centroid_sum is None else local_centroid_sum + centroid_addition
            local_label_sum = label_sum_addition.clone() if local_label_sum is None else local_label_sum + label_sum_addition
            local_label_sq_sum = label_sq_sum_addition.clone() if local_label_sq_sum is None else local_label_sq_sum + label_sq_sum_addition
            local_io_covariance_sum = io_addition.clone() if local_io_covariance_sum is None else local_io_covariance_sum + io_addition

        if n_local_images <= 0:
            raise ValueError(f"Category {category} produced zero images.")

        # ------------------------------------------------------------
        # Step 1. Local centering.
        # This is the main conceptual difference relative to the old resampling script:
        # we keep only the fluctuation geometry and only the locally centered labels.
        # We never re-add local centroids or local label means later.
        # ------------------------------------------------------------
        local_centroid = local_centroid_sum / n_local_images
        local_label_mean = local_label_sum / n_local_images
        local_covariance = (local_covariance_sum / n_local_images) - torch.outer(local_centroid, local_centroid)
        local_label_variance = (local_label_sq_sum / n_local_images) - local_label_mean ** 2
        local_io_covariance = (local_io_covariance_sum / n_local_images) - (local_centroid * local_label_mean)

        # Local dimensionality reduction is applied at the same stage as in the real-data theory script.
        local_eigvals, local_eigvecs = torch.linalg.eigh(local_covariance)
        sort_idx = torch.argsort(local_eigvals, descending=True)
        local_eigvals = local_eigvals[sort_idx]
        local_eigvecs = local_eigvecs[:, sort_idx]

        style = args.local_dimensionality_reduction_style
        thresh = args.local_dimensionality_reduction_strength / 100
        if style != "":
            if style == "PR":
                d = participation_ratio(local_eigvals)
            elif style == "percentage":
                d = explained_variance_dim(local_eigvals, threshold=thresh)
            else:
                raise ValueError(f"Local dimensionality reduction style <{style}> is not implemented.")
            V = local_eigvecs[:, :d]
            local_eigvals[d:] = 0.0
            local_projector = V @ V.t()
            local_covariance = local_projector @ local_covariance @ local_projector
            local_io_covariance = local_projector @ local_io_covariance

        # ------------------------------------------------------------
        # Step 2. Resample the local rule orientation and norm, then rotate the centered manifold.
        # This reuses the core logic of the old resampling script, but now only for one feature.
        # ------------------------------------------------------------
        seed_everything(cat_resampling_seeds[category])
        original_w = torch.tensor(
            original_theory_dict["local"]["regression_vector"][category][feature],
            dtype=torch.float64,
            device=device,
        )
        original_w_direction = safe_unit_vector(original_w)

        c = resampling_stats["c"]
        u_common = resampling_stats["u"]
        mean_r = resampling_stats["mean_r"]
        var_r = resampling_stats["var_r"]
        N_neurons = local_covariance.shape[0]

        new_direction = c * u_common + torch.sqrt(torch.clamp(1 - c ** 2, min=0.0)) * torch.randn_like(u_common) / np.sqrt(N_neurons)
        new_direction = safe_unit_vector(new_direction)

        rotation_1 = rotation_matrix_from_a_to_b(original_w_direction, new_direction)
        rotation_orthogonal = random_rotation_around_axis(new_direction)
        rotation_total = rotation_orthogonal @ rotation_1

        local_covariance_resampled = rotation_total @ local_covariance @ rotation_total.T
        local_io_covariance_resampled = rotation_total @ local_io_covariance

        w_norm_new = torch.normal(
            mean=mean_r,
            std=torch.sqrt(torch.clamp(var_r, min=0.0)),
            size=(),
            device=device,
            dtype=torch.float64,
        )
        w_norm_new = torch.clamp(w_norm_new, min=1e-12)
        w_new = w_norm_new * new_direction

        # The eigenvalues are unchanged by rotation, but we still diagonalize the resampled covariance
        # because we need the overlaps with the principal components of the resampled manifold.
        resampled_eigvals, resampled_eigvecs = torch.linalg.eigh(local_covariance_resampled)
        sort_idx = torch.argsort(resampled_eigvals, descending=True)
        resampled_eigvals = resampled_eigvals[sort_idx]
        resampled_eigvecs = resampled_eigvecs[:, sort_idx]

        w_direction, sigma_d_sq, sigma_1d_rotated = compute_local_projection_quantities(local_covariance_resampled, w_new)
        overlaps = einsum(w_direction, resampled_eigvecs, "neuron, neuron eig -> eig")
        variance_along_linear_rule = w_direction @ local_covariance_resampled @ w_direction
        linearized_label_variance = w_new @ local_covariance_resampled @ w_new
        regression_error = (
            w_new @ local_covariance_resampled @ w_new
            + local_label_variance
            - 2 * (w_new @ local_io_covariance_resampled)
        )
        normalized_regression_error = regression_error / local_label_variance
        io_covariance_linearized = w_new @ local_covariance_resampled

        # Store local quantities immediately and discard the heavy local covariance afterwards.
        dim = local_covariance_resampled.shape[0]
        zero_centroid = np.zeros(dim, dtype=np.float64)
        local_results["n_images"][category] = int(n_local_images)
        local_results["covariance_eigenvalues"][category] = resampled_eigvals.detach().cpu().numpy()
        local_results["linear_rule_vs_PC_overlap"][category][feature] = overlaps.detach().cpu().numpy()
        local_results["centroid"][category] = zero_centroid
        local_results["centroid_norm"][category] = 0.0
        local_results["label_mean"][category][feature] = 0.0
        local_results["label_variance"][category][feature] = float(local_label_variance.detach().cpu().item())
        local_results["linearized_label_variance"][category][feature] = float(linearized_label_variance.detach().cpu().item())
        local_results["io_covariance"][category][feature] = local_io_covariance_resampled.detach().cpu().numpy()
        local_results["regression_error"][category][feature] = float(regression_error.detach().cpu().item())
        local_results["normalized_regression_error"][category][feature] = float(normalized_regression_error.detach().cpu().item())
        local_results["regression_vector"][category][feature] = w_new.detach().cpu().numpy()
        local_results["regression_vector_norm"][category][feature] = float(torch.linalg.vector_norm(w_new).detach().cpu().item())
        local_results["bias"][category][feature] = 0.0
        local_results["variance_along_linear_rule"][category][feature] = float(variance_along_linear_rule.detach().cpu().item())
        local_results["variances_orthogonal"][category][feature] = sigma_d_sq.detach().cpu().numpy()
        local_results["covariance_cross_terms"][category][feature] = sigma_1d_rotated.detach().cpu().numpy()
        local_results["total_variance"][category] = float(torch.trace(local_covariance_resampled).detach().cpu().item())

        # Convert the category contribution to NumPy once, then update the running accumulators.
        cov_update = local_covariance_resampled.detach().cpu().numpy() * n_local_images
        io_update = local_io_covariance_resampled.detach().cpu().numpy() * n_local_images
        io_lin_update = io_covariance_linearized.detach().cpu().numpy() * n_local_images
        lab_var_update = float(local_label_variance.detach().cpu().item()) * n_local_images
        lin_lab_var_update = float(linearized_label_variance.detach().cpu().item()) * n_local_images

        if full_global_acc is None:
            full_global_acc = make_in_memory_accumulator(dim)
            if subset_definitions:
                subset_accumulators = create_disk_backed_subset_accumulators(tmp_dir, subset_definitions, dim)
        update_in_memory_accumulator(
            full_global_acc,
            cov_update=cov_update,
            io_update=io_update,
            io_lin_update=io_lin_update,
            lab_var_update=lab_var_update,
            lin_lab_var_update=lin_lab_var_update,
            n_local_images=n_local_images,
        )

        if subset_accumulators is not None:
            for subset_key in subset_membership_map[category]:
                update_disk_backed_accumulator(
                    subset_accumulators[subset_key],
                    cov_update=cov_update,
                    io_update=io_update,
                    io_lin_update=io_lin_update,
                    lab_var_update=lab_var_update,
                    lin_lab_var_update=lin_lab_var_update,
                    n_local_images=n_local_images,
                )

        # A small cleanup helps keep the GPU memory footprint stable across long runs.
        del local_covariance_sum, local_centroid_sum, local_label_sum, local_label_sq_sum, local_io_covariance_sum
        del local_covariance, local_io_covariance, local_covariance_resampled, local_io_covariance_resampled
        del local_eigvals, local_eigvecs, resampled_eigvals, resampled_eigvecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Finalize the full global block and then add the local quantities that depend on the full common direction.
    # ------------------------------------------------------------
    if full_global_acc is None:
        raise RuntimeError("No categories were processed. Cannot build output file.")

    global_results = finalize_global_from_accumulator(
        acc=full_global_acc,
        local_results=local_results,
        categories_subset=categories,
        feature=feature,
    )

    full_common_direction = np.asarray(global_results["mean_regression_direction_unit_vector"][feature])
    for category in categories:
        local_w = np.asarray(local_results["regression_vector"][category][feature])
        local_norm = float(local_results["regression_vector_norm"][category][feature])
        if local_norm <= 0:
            local_scale = np.nan
            overlap = np.nan
            projected_scale = np.nan
        else:
            local_scale = 1.0 / local_norm
            local_direction = local_w / local_norm
            overlap = float(local_direction @ full_common_direction)
            projected_scale = local_scale * overlap
        local_results["linear_rule_scale"][category][feature] = local_scale
        local_results["local_vs_common_linear_rule_overlap"][category][feature] = overlap
        local_results["linear_rule_scale_projected"][category][feature] = projected_scale

    # ------------------------------------------------------------
    # Finalize every requested subset global block.
    # We do this after the main pass, but using the disk-backed accumulators built on the fly,
    # so we never needed to keep many covariance matrices in RAM during the run.
    # ------------------------------------------------------------
    subset_results = {
        "metadata": {
            "requested_sizes": sorted({int(s) for s in args.subsample_sizes}) if args.subsample_sizes else [],
            "n_independent_subsamplings": int(args.n_independent_subsamplings),
            "seed_offset": int(args.subsampling_seed_offset),
        },
        "runs": {},
    }

    if subset_accumulators is not None:
        for subset_key, subset_acc in subset_accumulators.items():
            subset_global = finalize_global_from_accumulator(
                acc=subset_acc,
                local_results=local_results,
                categories_subset=subset_definitions[subset_key]["categories"],
                feature=feature,
            )
            subset_results["runs"][subset_key] = {
                "size": subset_definitions[subset_key]["size"],
                "seed": subset_definitions[subset_key]["seed"],
                "categories": subset_definitions[subset_key]["categories"],
                "n_categories": len(subset_definitions[subset_key]["categories"]),
                "global": subset_global,
            }

    # Keep gamma present for compatibility with current readers.
    if not hasattr(args, "gamma") or args.gamma is None:
        args.gamma = 0

    args_dictionary = vars(args).copy()
    args_dictionary["selected_feature"] = feature
    args_dictionary["resampling_mode"] = "locally_centered_fluctuations_only"

    final_results = {
        "local": local_results,
        "global": global_results,
        "args": args_dictionary,
        "subsampled": subset_results,
    }

    final_results_numpy = convert_tensors_to_numpy(final_results)

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

    rnd_sub_string = (
        f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
        if args.random_subsample_size is not None else ""
    )

    if args.global_dimensionality_reduction_style != "":
        global_dim_red_string = f"_GlobalRed_{args.global_dimensionality_reduction_style}_Dim{global_d}_KV{global_explained_variance}"
    else:
        global_dim_red_string = ""

    subset_string = ""
    if args.subsample_sizes and args.n_independent_subsamplings > 0:
        subset_sizes_str = "-".join(str(s) for s in sorted({int(s) for s in args.subsample_sizes if int(s) > 0}))
        subset_string = f"_NestedManSub_{subset_sizes_str}_n{args.n_independent_subsamplings}"

    dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))
    results_file = os.path.join(
        results_dir,
        f"{args.job_id}_TheoryRegressionResampledCentered_{dataset_name}_{args.layer}_{feature}"
        + dim_red_string + rnd_proj_string + rnd_sub_string + global_dim_red_string
        + f"_ResampSeed{args.manifolds_resampling_seed}" + subset_string
        + f"_gamma{args.gamma}.pkl"
    )

    with open(results_file, 'wb') as file:
        pickle.dump(final_results_numpy, file)

    print(f"Saved results to: {results_file}", flush=True)
    print("Regression concluded. Exit", flush=True)
