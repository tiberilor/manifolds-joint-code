import os
import sys
import json
import math
import copy
import pickle
import random
import hashlib
import argparse
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from hadamard_transform import hadamard_transform, next_power_of_2, pad_to_power_of_2, is_a_power_of_2

# -----------------------------------------------------------------------------
# Local project imports
# -----------------------------------------------------------------------------
# NOTE:
# We intentionally keep the same import style / paths as the reference script,
# because the user explicitly asked to preserve the same overall structure.
sys.path.append('../../codebase_v5')
sys.path.append('../ANN_models/AvgPool_SDXLdataset')

import LIB_model


# -----------------------------------------------------------------------------
# Argument parsing helpers
# -----------------------------------------------------------------------------
def str2bool(value: Any) -> bool:
    """
    Robust boolean parser for argparse.

    This keeps the same CLI option names as the reference script while avoiding
    the usual pitfalls of using type=bool in argparse.
    """
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="0000",
                    help="A name/ID for this run (used for naming the results).")

# First model: keep the original option names where possible.
parser.add_argument("--layer", type=str, default="backbone",
                    help="Layer from which to extract the features for model 1.")
parser.add_argument("--checkpoint", type=str, default="./checkpoint.pth",
                    help="Path to the model-1 checkpoint file.")

# Second model: this is the main extension relative to the reference script.
parser.add_argument("--layer_2", type=str, default=None,
                    help="Layer from which to extract the features for model 2. "
                         "If omitted, uses --layer.")
parser.add_argument("--checkpoint_2", type=str, required=True,
                    help="Path to the model-2 checkpoint file.")

parser.add_argument("--dataset_folder", type=str, default="./",
                    help="Path to the dataset folder.")
parser.add_argument("--results_folder_name", type=str, default="./",
                    help="Name of the folder in the checkpoint-1 folder where we will store the results.")

# Shared post-processing options. These must be shared across the two models.
parser.add_argument("--random_projection_size", type=int, default=None,
                    help="The size of the shared random projection. Default None corresponds to no projection.")
parser.add_argument("--random_projection_seed", type=int, default=1,
                    help="Random seed used for the shared random projection.")
parser.add_argument("--random_subsample_size", type=int, default=None,
                    help="Number of neurons to keep via shared random subsampling. Default None = no subsampling.")
parser.add_argument("--random_subsample_seed", type=int, default=1,
                    help="Random seed used to pick the shared subsampled neuron indices.")

# Category subsampling / holdout.
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
                    help="How to use the held-out categories at eval time.")

# Optional per-category image subsampling.
parser.add_argument("--n_subsampled_images", type=int, default=None,
                    help="If set, the maximum number of images to use PER CATEGORY.")
parser.add_argument("--images_subsampling_seed", type=int, default=1,
                    help="Random seed for per-category image subsampling.")

# General execution options.
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images.")
parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")


# Optional 8-way dihedral augmentation. The reference script treats each block of
# 8 transforms as a single base-image group for image subsampling.
parser.add_argument("--augment_dihedral", action="store_true",
                    help="If set, augment each image with the 8 dihedral (D4) transforms.")

args = parser.parse_args()

# D4 augmentation implies 8 variants per base image.
AUGMENT_GROUP_SIZE = 8 if getattr(args, "augment_dihedral", False) else 1

# If the second layer is not specified, use the same layer name as the first one.
if args.layer_2 is None:
    args.layer_2 = args.layer


# -----------------------------------------------------------------------------
# Small utility functions
# -----------------------------------------------------------------------------
def print_args(parsed_args: argparse.Namespace) -> None:
    """Pretty-print all arguments for reproducibility."""
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(parsed_args).items()):
        print(f"{name:28s}: {value!r}")
    print("========================", flush=True)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for deterministic shared sampling/projection."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_tensors_to_numpy(obj: Any) -> Any:
    """Recursively convert torch tensors to numpy arrays before pickling results."""
    if isinstance(obj, dict):
        return {key: convert_tensors_to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_tensors_to_numpy(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(convert_tensors_to_numpy(value) for value in obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj


def sanitize_token(text: str) -> str:
    """Make strings safe to use inside filenames."""
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("-")
    return "".join(keep)


def clone_or_copy(value: Any) -> Any:
    """Clone tensors, deepcopy everything else."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def accumulate_cpu(old: Optional[Any], addition: Any) -> Any:
    """
    Accumulate on CPU.

    We intentionally keep the large sufficient statistics on CPU to reduce GPU
    memory pressure now that two models are evaluated per batch.
    """
    if isinstance(addition, torch.Tensor):
        addition = addition.detach().cpu()
    else:
        addition = copy.deepcopy(addition)

    if old is None:
        return clone_or_copy(addition)
    return old + addition


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model_bundle(
    checkpoint_path: str,
    layer_name: str,
    dataset_folder_name: str,
    device: torch.device,
    args_namespace: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Load one model bundle.

    Supported public cases:
    - special-case 'resnet50'
    - special-case precomputed representation datasets (model=None)
    - otherwise load model and transform from the checkpoint
    """
    bundle: Dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": os.path.basename(os.path.normpath(checkpoint_path)),
        "layer": layer_name,
        "model": None,
        "transform": None,
    }

    if os.path.basename(os.path.normpath(checkpoint_path)).lower() == "resnet50":
        model = LIB_model.ResNetBBoxModel(
            resnet_version="resnet50",
        ).to(device)
        model.eval()
        bundle["model"] = model
        bundle["transform"] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return bundle

    if dataset_folder_name in [
        "representations_DicarloAsGenai_public_IT",
        "representations_DicarloAsGenai_public_V4",
    ]:
        bundle["model"] = None
        bundle["transform"] = None
        return bundle

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].to(device)
    model.eval()
    bundle["model"] = model
    bundle["transform"] = ckpt["transform"]
    return bundle


# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------
def get_dataset_folder_name(dataset_folder: str) -> str:
    """Replicate the reference script logic for naming the dataset kind."""
    dataset_path = Path(dataset_folder)
    if dataset_path.is_file() or dataset_path.suffix.lower() == ".h5":
        return dataset_path.parent.name
    return dataset_path.name


def get_dataset_constructor(dataset_folder_name: str, transform: Any, parsed_args: argparse.Namespace):
    """Build the category-dataset constructor for the public datasets only."""
    print(f"Dataset: {dataset_folder_name}", flush=True)

    if dataset_folder_name in [
        "SDXL_dataset_test",
        "SDXL_dataset_train",
        "SDXL_dataset_validation",
        "DicarloAsGenai_public",
    ]:
        from LIB_dataloader import CategoryDatasetSDXL
        return partial(
            CategoryDatasetSDXL,
            dataset_folder=parsed_args.dataset_folder,
            transform=transform,
            augment_dihedral=parsed_args.augment_dihedral,
        )

    if dataset_folder_name in [
        "representations_DicarloAsGenai_public_IT",
        "representations_DicarloAsGenai_public_V4",
    ]:
        from LIB_dataloader import CategoryDatasetRepresentation
        return partial(
            CategoryDatasetRepresentation,
            dataset_folder=parsed_args.dataset_folder,
            transform=transform,
        )

    raise ValueError(f"Dataset {dataset_folder_name} is not in the implemented public datasets.")

def get_all_categories(dataset_folder: str) -> list:
    """Replicate the reference-script category discovery logic."""
    dataset_path = Path(dataset_folder)
    categories = [
        p.name
        for p in (dataset_path if dataset_path.is_dir() else dataset_path.parent).iterdir()
        if p.is_dir() and p.name.lower() != "crashes"
    ]
    integration_suffixes = ("_integration", "_integration_ref")
    categories = [
        c for c in categories
        if not any(c.endswith(sfx) for sfx in integration_suffixes)
    ]
    categories.sort()
    return categories


def apply_category_holdout_and_subsampling(all_categories: list, parsed_args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply held-out category selection and optional manifold subsampling.

    This follows the same category logic as the reference script.
    """
    excluded_categories = []
    categories = list(all_categories)

    if parsed_args.exclude_categories_Q is not None and parsed_args.exclude_categories_Q > 0:
        if parsed_args.exclude_categories_Q >= len(categories):
            raise ValueError(
                f"--exclude_categories_Q={parsed_args.exclude_categories_Q} must be < number of categories ({len(categories)})."
            )
        rng = random.Random(parsed_args.exclude_categories_seed)
        excluded_categories = rng.sample(categories, parsed_args.exclude_categories_Q)
        excluded_categories.sort()

    if parsed_args.exclude_categories_eval_mode == "only_excluded":
        if not excluded_categories:
            raise ValueError("exclude_categories_eval_mode='only_excluded' requires --exclude_categories_Q > 0")
        categories = excluded_categories
    elif parsed_args.exclude_categories_eval_mode == "exclude_excluded":
        if not excluded_categories:
            raise ValueError("exclude_categories_eval_mode='exclude_excluded' requires --exclude_categories_Q > 0")
        excluded_set = set(excluded_categories)
        categories = [c for c in categories if c not in excluded_set]

    if parsed_args.exclude_categories_eval_mode != "none":
        print(
            f"[Category holdout] mode={parsed_args.exclude_categories_eval_mode} | "
            f"Q={parsed_args.exclude_categories_Q} seed={parsed_args.exclude_categories_seed} | "
            f"excluded={excluded_categories}",
            flush=True,
        )

    if parsed_args.manifold_subsample_size is not None:
        seed_everything(parsed_args.manifold_subsample_seed)
        K = min(parsed_args.manifold_subsample_size, len(categories))

        if parsed_args.manifold_subsampling_grouping_json is not None:
            with open(parsed_args.manifold_subsampling_grouping_json, "r") as f:
                grouping = json.load(f)
            cat2macro = {row["category"]: row["macro"] for row in grouping}

            macros_to_cats: Dict[str, list] = {}
            for c in categories:
                macro = cat2macro.get(c, None)
                if macro is None:
                    continue
                macros_to_cats.setdefault(macro, []).append(c)

            if len(macros_to_cats) == 0:
                raise ValueError("No categories from the grouping JSON match the categories found on disk.")

            for macro in macros_to_cats:
                random.shuffle(macros_to_cats[macro])

            macros = sorted(macros_to_cats.keys())
            selected = []
            step = 0
            while len(selected) < K:
                target_macro = macros[step % len(macros)]
                if macros_to_cats[target_macro]:
                    selected.append(macros_to_cats[target_macro].pop(0))
                else:
                    non_empty = [m for m in macros if macros_to_cats[m]]
                    if not non_empty:
                        break
                    m_pick = random.choice(non_empty)
                    selected.append(macros_to_cats[m_pick].pop(0))
                step += 1
            categories = selected
        else:
            random.shuffle(categories)
            categories = categories[:K]

    return {
        "categories": categories,
        "excluded_categories": excluded_categories,
    }


def maybe_subsample_images_for_category(dataset, category: str, parsed_args: argparse.Namespace):
    """
    Apply the same per-category image subsampling logic as the reference script.

    Importantly, when dihedral augmentation is enabled, subsampling is done in
    base-image space and then expanded to the 8 transformed variants.
    """
    if parsed_args.n_subsampled_images is None:
        return dataset

    N = len(dataset)
    group_size = AUGMENT_GROUP_SIZE
    if group_size < 1:
        raise ValueError(f"Invalid AUGMENT_GROUP_SIZE={group_size}")
    if N % group_size != 0:
        raise ValueError(
            f"len(dataset)={N} not divisible by AUGMENT_GROUP_SIZE={group_size}. "
            "If augmentation is enabled, the dataset must return grouped blocks."
        )

    N_base = N // group_size
    k_base = max(0, min(int(parsed_args.n_subsampled_images), N_base))
    if k_base >= N_base:
        return dataset

    cat_key = f"{parsed_args.images_subsampling_seed}:{category}".encode("utf-8")
    cat_seed = int.from_bytes(hashlib.sha256(cat_key).digest()[:4], "big")
    rng = np.random.RandomState(cat_seed)

    subset_base = rng.choice(np.arange(N_base), size=k_base, replace=False)
    subset_base.sort()

    if group_size == 1:
        subset_idx = subset_base
    else:
        subset_idx = (subset_base[:, None] * group_size + np.arange(group_size)[None, :]).reshape(-1)

    return Subset(dataset, subset_idx.tolist())


# -----------------------------------------------------------------------------
# Representation extraction and shared post-processing
# -----------------------------------------------------------------------------
def extract_raw_representations(
    images: torch.Tensor,
    gt_category_idxs: torch.Tensor,
    gt_bboxes: torch.Tensor,
    bundle: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Extract one batch of raw (pre-shared-postprocessing) representations.
    """
    images = images.to(device)
    if bundle["layer"].lower() == "pixels":
        reps = images.flatten(1)
    elif bundle["model"] is not None:
        with torch.no_grad():
            reps = bundle["model"].forward_features(images, module_name=bundle["layer"])
    else:
        reps = images

    return reps.to(dtype=torch.float64)


class SharedPostprocessor:
    """
    Apply the shared post-processing that must be identical for the two models:
    - same neuron subsample
    - same random projection

    The state is created lazily from the first batch, exactly like in the
    reference script. The only difference is that it is shared across the two
    models, because this script compares them with CKA.
    """
    def __init__(self, device: torch.device, parsed_args: argparse.Namespace):
        self.device = device
        self.args = parsed_args
        self.raw_feature_dim: Optional[int] = None
        self.feature_dim_after_subsample: Optional[int] = None
        self.final_feature_dim: Optional[int] = None
        self.subsample_idx: Optional[torch.Tensor] = None
        self.fjlt_D: Optional[torch.Tensor] = None
        self.fjlt_idx: Optional[torch.Tensor] = None
        self.N_power2: Optional[int] = None

    def _initialize_feature_dim(self, reps: torch.Tensor, model_label: str) -> None:
        raw_dim = int(reps.size(1))
        if self.raw_feature_dim is None:
            self.raw_feature_dim = raw_dim
        elif raw_dim != self.raw_feature_dim:
            raise ValueError(
                f"The two models must have the same raw feature dimension before shared operations. "
                f"Got {self.raw_feature_dim} previously but {raw_dim} for {model_label}."
            )

    def _apply_shared_subsample(self, reps: torch.Tensor) -> torch.Tensor:
        if self.args.random_subsample_size is None:
            self.feature_dim_after_subsample = int(reps.size(1))
            return reps

        if self.subsample_idx is None:
            seed_everything(self.args.random_subsample_seed)
            N_orig = reps.size(1)
            if self.args.random_subsample_size > N_orig:
                raise ValueError(
                    f"--random_subsample_size ({self.args.random_subsample_size}) > neurons ({N_orig})."
                )
            self.subsample_idx = torch.randperm(N_orig, device=reps.device)[:self.args.random_subsample_size]
            self.feature_dim_after_subsample = int(self.args.random_subsample_size)

        return reps[:, self.subsample_idx]

    def _apply_shared_random_projection(self, reps: torch.Tensor) -> torch.Tensor:
        if self.args.random_projection_size is None:
            self.final_feature_dim = int(reps.size(1))
            return reps

        if self.fjlt_D is None or self.fjlt_idx is None:
            seed_everything(self.args.random_projection_seed)
            N_orig = reps.size(1)
            self.N_power2 = N_orig if is_a_power_of_2(N_orig) else next_power_of_2(N_orig)
            if self.args.random_projection_size > self.N_power2:
                raise ValueError(
                    f"--random_projection_size ({self.args.random_projection_size}) > padded dimension ({self.N_power2})."
                )
            self.fjlt_D = (torch.randint(0, 2, (self.N_power2,), device=reps.device) * 2 - 1).to(torch.float64)
            self.fjlt_idx = torch.randperm(self.N_power2, device=reps.device)[:self.args.random_projection_size]
            self.final_feature_dim = int(self.args.random_projection_size)

        reps = pad_to_power_of_2(reps)
        reps_d = reps * self.fjlt_D[None, :]
        reps_h = hadamard_transform(reps_d)
        reps = reps_h[:, self.fjlt_idx] * math.sqrt(self.N_power2 / self.args.random_projection_size)
        return reps

    def __call__(self, reps: torch.Tensor, model_label: str) -> torch.Tensor:
        self._initialize_feature_dim(reps, model_label=model_label)
        reps = self._apply_shared_subsample(reps)
        reps = self._apply_shared_random_projection(reps)
        if self.final_feature_dim is None:
            self.final_feature_dim = int(reps.size(1))
        return reps

    def metadata(self) -> Dict[str, Any]:
        return {
            "raw_feature_dim": self.raw_feature_dim,
            "feature_dim_after_subsample": self.feature_dim_after_subsample,
            "final_feature_dim": self.final_feature_dim,
            "subsample_idx": None if self.subsample_idx is None else self.subsample_idx.detach().cpu(),
            "fjlt_D": None if self.fjlt_D is None else self.fjlt_D.detach().cpu(),
            "fjlt_idx": None if self.fjlt_idx is None else self.fjlt_idx.detach().cpu(),
            "N_power2": self.N_power2,
        }


# -----------------------------------------------------------------------------
# CKA sufficient statistics
# -----------------------------------------------------------------------------
def initialize_stats() -> Dict[str, Any]:
    """
    Initialize the online sufficient statistics needed for both biased and
    debiased CKA.

    The definitions match the notation from the user's derivation.
    """
    return {
        "n": 0,
        "G_xy": None,
        "G_xx": None,
        "G_yy": None,
        "s_x": None,
        "s_y": None,
        "q_x": 0.0,
        "q_y": 0.0,
        "c_xy": None,
        "c_yx": None,
        "c_xx": None,
        "c_yy": None,
        "r_xy": 0.0,
        "r_xx": 0.0,
        "r_yy": 0.0,
    }


def update_cka_stats(stats: Dict[str, Any], x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> None:
    """
    Update the online sufficient statistics with one batch.

    Parameters
    ----------
    x_cpu : torch.Tensor, shape [B, p]
        Batch of processed representations for model 1.
    y_cpu : torch.Tensor, shape [B, p]
        Batch of processed representations for model 2.

    Notes
    -----
    The matrices / vectors accumulated here are exactly the ones appearing in
    the user's feature-space formulas:

        G_xy = Σ x^μ y^{μT}
        G_xx = Σ x^μ x^{μT}
        G_yy = Σ y^μ y^{μT}
        s_x  = Σ x^μ
        s_y  = Σ y^μ
        c_xy = Σ x^μ ||y^μ||^2
        ... etc.
    """
    if x_cpu.dtype != torch.float64:
        x_cpu = x_cpu.to(dtype=torch.float64)
    if y_cpu.dtype != torch.float64:
        y_cpu = y_cpu.to(dtype=torch.float64)

    batch_n = int(x_cpu.size(0))
    x_norm_sq = torch.sum(x_cpu * x_cpu, dim=1)
    y_norm_sq = torch.sum(y_cpu * y_cpu, dim=1)

    stats["n"] += batch_n
    stats["G_xy"] = accumulate_cpu(stats["G_xy"], x_cpu.T @ y_cpu)
    stats["G_xx"] = accumulate_cpu(stats["G_xx"], x_cpu.T @ x_cpu)
    stats["G_yy"] = accumulate_cpu(stats["G_yy"], y_cpu.T @ y_cpu)
    stats["s_x"] = accumulate_cpu(stats["s_x"], torch.sum(x_cpu, dim=0))
    stats["s_y"] = accumulate_cpu(stats["s_y"], torch.sum(y_cpu, dim=0))

    stats["q_x"] += float(torch.sum(x_norm_sq).item())
    stats["q_y"] += float(torch.sum(y_norm_sq).item())

    stats["c_xy"] = accumulate_cpu(stats["c_xy"], x_cpu.T @ y_norm_sq)
    stats["c_yx"] = accumulate_cpu(stats["c_yx"], y_cpu.T @ x_norm_sq)
    stats["c_xx"] = accumulate_cpu(stats["c_xx"], x_cpu.T @ x_norm_sq)
    stats["c_yy"] = accumulate_cpu(stats["c_yy"], y_cpu.T @ y_norm_sq)

    stats["r_xy"] += float(torch.dot(x_norm_sq, y_norm_sq).item())
    stats["r_xx"] += float(torch.dot(x_norm_sq, x_norm_sq).item())
    stats["r_yy"] += float(torch.dot(y_norm_sq, y_norm_sq).item())


# -----------------------------------------------------------------------------
# Final CKA computation from the accumulated statistics
# -----------------------------------------------------------------------------
def frobenius_sq(matrix: torch.Tensor) -> torch.Tensor:
    """Return Tr(M M^T) = ||M||_F^2."""
    return torch.sum(matrix * matrix)


def compute_biased_hsic_terms(stats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Compute the biased HSIC terms from the accumulated sufficient statistics.

    This implements the feature-space formulas:
        G_xy,c = G_xy - (1/n) s_x s_y^T
        HSIC_0 = Tr(G_xy,c G_xy,c^T) / (n-1)^2
    and similarly for XX / YY.
    """
    n = int(stats["n"])
    if n < 2:
        nan = torch.tensor(float("nan"), dtype=torch.float64)
        return {
            "hsic_xy": nan,
            "hsic_xx": nan,
            "hsic_yy": nan,
            "G_xy_c": None,
            "G_xx_c": None,
            "G_yy_c": None,
        }

    inv_n = 1.0 / float(n)
    G_xy_c = stats["G_xy"] - inv_n * torch.outer(stats["s_x"], stats["s_y"])
    G_xx_c = stats["G_xx"] - inv_n * torch.outer(stats["s_x"], stats["s_x"])
    G_yy_c = stats["G_yy"] - inv_n * torch.outer(stats["s_y"], stats["s_y"])

    denom = float((n - 1) ** 2)
    hsic_xy = frobenius_sq(G_xy_c) / denom
    hsic_xx = frobenius_sq(G_xx_c) / denom
    hsic_yy = frobenius_sq(G_yy_c) / denom

    return {
        "hsic_xy": hsic_xy,
        "hsic_xx": hsic_xx,
        "hsic_yy": hsic_yy,
        "G_xy_c": G_xy_c,
        "G_xx_c": G_xx_c,
        "G_yy_c": G_yy_c,
    }


def compute_unbiased_hsic_cross(stats: Dict[str, Any]) -> torch.Tensor:
    """
    Compute HSIC_1(K, L) from the user's online feature-space formula.
    """
    n = int(stats["n"])
    if n < 4:
        return torch.tensor(float("nan"), dtype=torch.float64)

    n_t = torch.tensor(float(n), dtype=torch.float64)
    q_x = torch.tensor(float(stats["q_x"]), dtype=torch.float64)
    q_y = torch.tensor(float(stats["q_y"]), dtype=torch.float64)
    r_xy = torch.tensor(float(stats["r_xy"]), dtype=torch.float64)

    term_1 = frobenius_sq(stats["G_xy"])

    sx_sq_minus_qx = torch.dot(stats["s_x"], stats["s_x"]) - q_x
    sy_sq_minus_qy = torch.dot(stats["s_y"], stats["s_y"]) - q_y
    term_2 = (sx_sq_minus_qx * sy_sq_minus_qy) / ((n_t - 1.0) * (n_t - 2.0))

    mixed = (
        torch.dot(stats["s_x"], stats["G_xy"] @ stats["s_y"])
        - torch.dot(stats["s_x"], stats["c_xy"])
        - torch.dot(stats["s_y"], stats["c_yx"])
    )
    term_3 = (2.0 / (n_t - 2.0)) * mixed

    term_4 = (n_t / (n_t - 2.0)) * r_xy

    return (term_1 + term_2 - term_3 - term_4) / (n_t * (n_t - 3.0))


def compute_unbiased_hsic_self(
    G_self: torch.Tensor,
    s_self: torch.Tensor,
    q_self: float,
    c_selfself: torch.Tensor,
    r_selfself: float,
    n: int,
) -> torch.Tensor:
    """
    Compute HSIC_1(K, K) or HSIC_1(L, L) from the user's formula.
    """
    if n < 4:
        return torch.tensor(float("nan"), dtype=torch.float64)

    n_t = torch.tensor(float(n), dtype=torch.float64)
    q_t = torch.tensor(float(q_self), dtype=torch.float64)
    r_t = torch.tensor(float(r_selfself), dtype=torch.float64)

    term_1 = frobenius_sq(G_self)
    term_2 = (torch.dot(s_self, s_self) - q_t) ** 2 / ((n_t - 1.0) * (n_t - 2.0))

    mixed = torch.dot(s_self, G_self @ s_self) - 2.0 * torch.dot(s_self, c_selfself)
    term_3 = (2.0 / (n_t - 2.0)) * mixed
    term_4 = (n_t / (n_t - 2.0)) * r_t

    return (term_1 + term_2 - term_3 - term_4) / (n_t * (n_t - 3.0))


def safe_cka_ratio(numerator: torch.Tensor, self_x: torch.Tensor, self_y: torch.Tensor) -> torch.Tensor:
    """
    Safely form CKA = numerator / sqrt(self_x self_y).

    We explicitly guard against invalid or slightly negative denominator terms,
    which can occur numerically for the debiased estimator on finite data.
    """
    if not torch.isfinite(numerator) or not torch.isfinite(self_x) or not torch.isfinite(self_y):
        return torch.tensor(float("nan"), dtype=torch.float64)
    if self_x <= 0 or self_y <= 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    return numerator / torch.sqrt(self_x * self_y)


def compute_all_cka_outputs(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all final scalar outputs from the accumulated sufficient statistics.

    Returned values include both biased and debiased HSIC / CKA, plus the
    centered matrices used by the biased estimator.
    """
    biased = compute_biased_hsic_terms(stats)
    unbiased_xy = compute_unbiased_hsic_cross(stats)
    unbiased_xx = compute_unbiased_hsic_self(
        G_self=stats["G_xx"],
        s_self=stats["s_x"],
        q_self=stats["q_x"],
        c_selfself=stats["c_xx"],
        r_selfself=stats["r_xx"],
        n=stats["n"],
    )
    unbiased_yy = compute_unbiased_hsic_self(
        G_self=stats["G_yy"],
        s_self=stats["s_y"],
        q_self=stats["q_y"],
        c_selfself=stats["c_yy"],
        r_selfself=stats["r_yy"],
        n=stats["n"],
    )

    return {
        "biased": {
            "hsic_xy": biased["hsic_xy"],
            "hsic_xx": biased["hsic_xx"],
            "hsic_yy": biased["hsic_yy"],
            "cka": safe_cka_ratio(biased["hsic_xy"], biased["hsic_xx"], biased["hsic_yy"]),
        },
        "debiased": {
            "hsic_xy": unbiased_xy,
            "hsic_xx": unbiased_xx,
            "hsic_yy": unbiased_yy,
            "cka": safe_cka_ratio(unbiased_xy, unbiased_xx, unbiased_yy),
        },
        "centered_feature_covariances": {
            "G_xy_c": biased["G_xy_c"],
            "G_xx_c": biased["G_xx_c"],
            "G_yy_c": biased["G_yy_c"],
        },
    }


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
print_args(args)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

dataset_folder_name = get_dataset_folder_name(args.dataset_folder)

# Load both models / bundles.
# IMPORTANT: by user request, we use ONLY the transform from model 1.
bundle_1 = load_model_bundle(args.checkpoint, args.layer, dataset_folder_name, device, args)
bundle_2 = load_model_bundle(args.checkpoint_2, args.layer_2, dataset_folder_name, device, args)

transform = bundle_1["transform"]

# Build the dataset constructor using the first model's transform, exactly as requested.
CategoryDataset = get_dataset_constructor(dataset_folder_name, transform, args)

# Retrieve and filter categories exactly as in the reference script.
all_categories = get_all_categories(args.dataset_folder)
category_info = apply_category_holdout_and_subsampling(all_categories, args)
categories = category_info["categories"]
excluded_categories = category_info["excluded_categories"]

print(f"Number of categories to process: {len(categories)}", flush=True)

shared_postprocessor = SharedPostprocessor(device=device, parsed_args=args)
stats = initialize_stats()

# -----------------------------------------------------------------------------
# Main category loop
# -----------------------------------------------------------------------------
# The user explicitly asked to keep the same processing style as the reference:
# iterate over categories, and then iterate over batches within each category,
# even though CKA itself only needs global accumulators.
for category in categories:
    print(f"PROCESSING CATEGORY: {category}", flush=True)

    dataset = CategoryDataset(category=category)
    dataset = maybe_subsample_images_for_category(dataset, category=category, parsed_args=args)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # Batch loop
    # -------------------------------------------------------------------------
    for batch in dataloader:
        images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch

        # Categories and bboxes are always needed downstream, because VAE mode uses
        # them to synthesize representations instead of forwarding images.
        gt_category_idxs = gt_category_idxs.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # Extract raw representations for the two models. We do this sequentially,
        # not in one giant combined forward pass, to keep peak GPU memory lower.
        reps_1 = extract_raw_representations(images, gt_category_idxs, gt_bboxes, bundle_1, device)
        reps_1 = shared_postprocessor(reps_1, model_label="model_1")
        reps_1_cpu = reps_1.detach().cpu()
        del reps_1

        reps_2 = extract_raw_representations(images, gt_category_idxs, gt_bboxes, bundle_2, device)
        reps_2 = shared_postprocessor(reps_2, model_label="model_2")
        reps_2_cpu = reps_2.detach().cpu()
        del reps_2

        # The online CKA update uses only the processed batch representations.
        update_cka_stats(stats, reps_1_cpu, reps_2_cpu)

        # Free batch-level CPU tensors promptly.
        del reps_1_cpu, reps_2_cpu

# -----------------------------------------------------------------------------
# Final outputs
# -----------------------------------------------------------------------------
cka_outputs = compute_all_cka_outputs(stats)

print("=== FINAL CKA RESULTS ===", flush=True)
print(f"Biased HSIC(X,Y):   {cka_outputs['biased']['hsic_xy']}", flush=True)
print(f"Biased HSIC(X,X):   {cka_outputs['biased']['hsic_xx']}", flush=True)
print(f"Biased HSIC(Y,Y):   {cka_outputs['biased']['hsic_yy']}", flush=True)
print(f"Biased CKA:         {cka_outputs['biased']['cka']}", flush=True)
print(f"Debiased HSIC(X,Y): {cka_outputs['debiased']['hsic_xy']}", flush=True)
print(f"Debiased HSIC(X,X): {cka_outputs['debiased']['hsic_xx']}", flush=True)
print(f"Debiased HSIC(Y,Y): {cka_outputs['debiased']['hsic_yy']}", flush=True)
print(f"Debiased CKA:       {cka_outputs['debiased']['cka']}", flush=True)

# Save the results under the first checkpoint's directory, matching the style of
# the reference script (single checkpoint -> checkpoint_dir; here we use model 1).
checkpoint_dir = os.path.dirname(args.checkpoint)
results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
os.makedirs(results_dir, exist_ok=True)

rnd_proj_string = (
    f"_RndProj_size{args.random_projection_size}_seed{args.random_projection_seed}"
    if args.random_projection_size is not None else ""
)
rnd_sub_string = (
    f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
    if args.random_subsample_size is not None else ""
)
manifolds_sub_string = (
    f"_ManSub_size{args.manifold_subsample_size}_seed{args.manifold_subsample_seed}"
    if args.manifold_subsample_size is not None else ""
)

dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))
ckpt1_name = sanitize_token(Path(args.checkpoint).stem)
ckpt2_name = sanitize_token(Path(args.checkpoint_2).stem)
layer1_name = sanitize_token(args.layer)
layer2_name = sanitize_token(args.layer_2)

core_name = (
    f"CKA_{dataset_name}"
    f"_M1-{ckpt1_name}_L1-{layer1_name}"
    f"_M2-{ckpt2_name}_L2-{layer2_name}"
    f"{rnd_proj_string}{rnd_sub_string}{manifolds_sub_string}"
)
results_file = os.path.join(results_dir, f"{args.job_id}_{core_name}.pkl")

results_to_save = {
    "args": vars(args),
    "dataset_folder_name": dataset_folder_name,
    "categories": categories,
    "excluded_categories": excluded_categories,
    "n_categories": len(categories),
    "model_1": {
        "checkpoint_path": args.checkpoint,
        "checkpoint_name": bundle_1["checkpoint_name"],
        "layer": args.layer,
        "transform_repr": repr(bundle_1["transform"]),
    },
    "model_2": {
        "checkpoint_path": args.checkpoint_2,
        "checkpoint_name": bundle_2["checkpoint_name"],
        "layer": args.layer_2,
        "transform_repr": repr(bundle_2["transform"]),
    },
    "shared_postprocessing": shared_postprocessor.metadata(),
    # Raw sufficient statistics collected online.
    "sufficient_statistics": stats,
    # Final scalar outputs.
    "results": cka_outputs,
}

results_to_save = convert_tensors_to_numpy(results_to_save)
with open(results_file, "wb") as f:
    pickle.dump(results_to_save, f)

print(f"Saved results to: {results_file}", flush=True)
