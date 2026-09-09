import argparse
import json
import math
import os
import pickle
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hadamard_transform import (
    hadamard_transform,
    is_a_power_of_2,
    next_power_of_2,
    pad_to_power_of_2,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_DIR = os.path.join(REPOSITORY_ROOT, "CNN_trainer")
sys.path.insert(0, REPOSITORY_ROOT)
sys.path.insert(0, MODEL_DIR)
import LIB_model


FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
SGD_DTYPE = torch.float32
SHRINK_APPLICATION_THRESHOLD = 1e-5


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="0000")
    parser.add_argument("--layer", type=str, default="backbone")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint.pth")
    parser.add_argument("--dataset_folder", type=str, default="./")
    parser.add_argument("--results_folder_name", type=str, default="./")

    parser.add_argument("--manifold_subsample_size", type=int, default=None)
    parser.add_argument("--manifold_subsample_seed", type=int, default=1)
    parser.add_argument("--manifold_subsampling_grouping_json", type=str, default=None)

    # Valid requested D values share one FJLT prefix. Only the first oversized
    # request is replaced by the unprojected native layer dimension.
    parser.add_argument("--random_projection_size", type=int, default=None)
    parser.add_argument("--random_projection_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--random_projection_seed", type=int, default=1)
    parser.add_argument("--random_subsample_size", type=int, default=None)
    parser.add_argument("--random_subsample_seed", type=int, default=1)

    parser.add_argument("--n_cv_splits", type=int, default=1)
    parser.add_argument("--n_l2_cv_splits", type=int, default=1)
    parser.add_argument("--l2_min_exp", type=float, default=-6)
    parser.add_argument("--l2_max_exp", type=float, default=2)
    parser.add_argument("--n_l2", type=int, default=9)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--storage_dtype",
        choices=["float64", "float32", "float16"],
        default="float32",
        help="Retained for command-line compatibility; SGD uses float32.",
    )
    parser.add_argument(
        "--regression_dtype",
        choices=["float64", "float32", "float16"],
        default="float32",
        help="Retained for command-line compatibility; SGD uses float32.",
    )

    parser.add_argument("--outer_split_method", choices=["kfold", "random"], default="kfold")
    parser.add_argument("--inner_split_method", choices=["kfold", "random"], default="kfold")
    parser.add_argument("--outer_test_ratio", type=float, default=0.2)
    parser.add_argument("--inner_test_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed_splits", type=int, default=43)

    # Retained so existing launchers fail explicitly rather than at parsing.
    parser.add_argument("--global_dimensionality_reduction_style", type=str, default="")
    parser.add_argument("--global_dimensionality_reduction_strength", type=float, default=100)
    parser.add_argument("--global_dimensionality_reduction_dimensions", type=int, default=2048)
    parser.add_argument("--global_dimensionality_reduction_file", type=str, default="./boh.pkl")

    parser.add_argument("--sgd_inner_epochs", type=int, default=3)
    parser.add_argument("--sgd_final_epochs", type=int, default=3)
    parser.add_argument("--sgd_learning_rate", type=float, default=1e-3)
    parser.add_argument("--sgd_min_learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--sgd_auto_learning_rate",
        action="store_true",
        help=(
            "Calibrate the representation scale and a stable weight learning "
            "rate separately for every readout dimension. In this mode, "
            "--sgd_learning_rate is an upper bound."
        ),
    )
    parser.add_argument(
        "--sgd_calibration_batches",
        type=int,
        default=8,
        help="Number of globally shuffled batches used for scale/LR calibration.",
    )
    parser.add_argument(
        "--sgd_lr_safety_factor",
        type=float,
        default=0.5,
        help="Upper bound on the preconditioned minibatch curvature.",
    )
    parser.add_argument(
        "--sgd_bias_lr_fraction",
        type=float,
        default=0.1,
        help="Fraction of the auto-LR stability budget reserved for the bias.",
    )
    parser.add_argument(
        "--sgd_lr_schedule",
        choices=["constant", "cosine"],
        default="cosine",
    )
    parser.add_argument(
        "--sgd_category_shuffle_seed",
        "--sgd_data_shuffle_seed",
        dest="sgd_category_shuffle_seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--no_sgd_shuffle_categories",
        "--no_sgd_shuffle",
        dest="no_sgd_shuffle_categories",
        action="store_true",
    )
    parser.add_argument("--progress_every_batches", type=int, default=250)
    parser.add_argument("--debug_max_categories", type=int, default=None)
    return parser


def print_args(args):
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:40s}: {value!r}")
    print("========================", flush=True)


def validate_args(args):
    if args.random_projection_sizes is not None and args.random_projection_size is not None:
        raise ValueError(
            "Use either --random_projection_sizes or --random_projection_size, not both."
        )
    if args.random_projection_sizes is None and args.random_projection_size is None:
        raise ValueError(
            "This SGD script requires --random_projection_sizes (or the legacy "
            "--random_projection_size)."
        )
    if args.sgd_inner_epochs < 1 or args.sgd_final_epochs < 1:
        raise ValueError("SGD inner and final epochs must both be at least 1.")
    if args.sgd_learning_rate <= 0 or args.sgd_min_learning_rate <= 0:
        raise ValueError("SGD learning rates must be positive.")
    if args.sgd_min_learning_rate > args.sgd_learning_rate:
        raise ValueError("--sgd_min_learning_rate cannot exceed --sgd_learning_rate.")
    if args.sgd_calibration_batches < 1:
        raise ValueError("--sgd_calibration_batches must be at least 1.")
    if not 0 < args.sgd_lr_safety_factor < 1:
        raise ValueError("--sgd_lr_safety_factor must be strictly between 0 and 1.")
    if not 0 < args.sgd_bias_lr_fraction < 1:
        raise ValueError("--sgd_bias_lr_fraction must be strictly between 0 and 1.")
    if args.n_cv_splits < 2 or args.n_l2_cv_splits < 2:
        raise ValueError("Nested CV requires at least two outer and inner splits.")
    if args.n_l2 < 1:
        raise ValueError("--n_l2 must be at least 1.")
    if args.progress_every_batches < 1:
        raise ValueError("--progress_every_batches must be at least 1.")
    if args.debug_max_categories is not None and args.debug_max_categories < 0:
        raise ValueError("--debug_max_categories cannot be negative.")
    if args.global_dimensionality_reduction_style:
        raise NotImplementedError(
            "Global dimensionality reduction is not combined with multi-D SGD. "
            "Use the FJLT projection sizes directly."
        )


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_splits(n_items, method="kfold", n_splits=5, test_ratio=0.2, seed=None):
    indices = list(range(n_items))
    rng = random.Random(seed)

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

    if method == "random":
        test_size = int(math.floor(n_items * test_ratio))
        folds = []
        for _ in range(n_splits):
            permutation = indices.copy()
            rng.shuffle(permutation)
            test_idx = torch.tensor(permutation[:test_size], dtype=torch.long)
            train_idx = torch.tensor(permutation[test_size:], dtype=torch.long)
            folds.append((train_idx, test_idx))
        return folds

    raise ValueError(f"Unknown split method: {method}")


def close_dataset(dataset):
    h5_file = getattr(dataset, "h5_file", None)
    if h5_file is not None:
        try:
            h5_file.close()
        except Exception:
            pass


def unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def dataset_folder_name(path):
    path = Path(path)
    if path.is_file() or path.suffix.lower() == ".h5":
        return path.parent.name
    return path.name


def load_model_and_transform(args, folder_name, device):
    if os.path.basename(os.path.normpath(args.checkpoint)).lower() == "resnet50":
        model = LIB_model.ResNetBBoxModel(resnet_version="resnet50").to(device)
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return model, transform

    representation_folders = {
        "representations_DicarloAsGenai_public_IT",
        "representations_DicarloAsGenai_public_V4",
    }
    if folder_name in representation_folders:
        print("Representations dataset detected. Setting model to None.", flush=True)
        return None, None

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = checkpoint["model"].to(device)
    model.eval()
    return model, checkpoint["transform"]


def build_category_dataset(args, folder_name, transform):
    if folder_name in {
        "SDXL_dataset_test",
        "SDXL_dataset_train",
        "SDXL_dataset_validation",
        "DicarloAsGenai_public",
    }:
        from LIB_dataloader import CategoryDatasetSDXL

        return partial(
            CategoryDatasetSDXL,
            dataset_folder=args.dataset_folder,
            transform=transform,
        )

    if folder_name in {
        "representations_DicarloAsGenai_public_IT",
        "representations_DicarloAsGenai_public_V4",
    }:
        from LIB_dataloader import CategoryDatasetRepresentation

        return partial(
            CategoryDatasetRepresentation,
            dataset_folder=args.dataset_folder,
            transform=transform,
        )

    raise ValueError(f"Dataset {folder_name!r} is not implemented.")

def discover_categories(args):
    path = Path(args.dataset_folder)
    root = path if path.is_dir() else path.parent
    categories = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and p.name.lower() != "crashes"
        and not p.name.endswith(("_integration", "_integration_ref"))
    )

    if args.manifold_subsample_size is not None:
        rng = random.Random(args.manifold_subsample_seed)
        keep = min(args.manifold_subsample_size, len(categories))
        if args.manifold_subsampling_grouping_json is None:
            rng.shuffle(categories)
            categories = categories[:keep]
        else:
            with open(args.manifold_subsampling_grouping_json, "r") as file:
                grouping = json.load(file)
            category_to_macro = {row["category"]: row["macro"] for row in grouping}
            pools = {}
            for category in categories:
                macro = category_to_macro.get(category)
                if macro is not None:
                    pools.setdefault(macro, []).append(category)
            if not pools:
                raise ValueError("No dataset categories match the grouping JSON.")
            for pool in pools.values():
                rng.shuffle(pool)
            macros = sorted(pools)
            selected = []
            step = 0
            while len(selected) < keep:
                macro = macros[step % len(macros)]
                if pools[macro]:
                    selected.append(pools[macro].pop())
                else:
                    nonempty = [name for name in macros if pools[name]]
                    if not nonempty:
                        break
                    selected.append(pools[rng.choice(nonempty)].pop())
                step += 1
            categories = selected

    if args.debug_max_categories is not None:
        categories = categories[: args.debug_max_categories]
    if not categories:
        raise RuntimeError("No categories were found.")
    return categories


def projection_sizes_from_args(args):
    if args.random_projection_sizes is not None:
        sizes = args.random_projection_sizes
    else:
        sizes = [args.random_projection_size]
    sizes = sorted(set(int(size) for size in sizes))
    if any(size <= 0 for size in sizes):
        raise ValueError("All random projection sizes must be positive.")
    return sizes


def learning_rate_for_epoch(args, epoch, n_epochs, base_learning_rate):
    if args.sgd_lr_schedule == "constant" or n_epochs == 1:
        return base_learning_rate
    fraction = epoch / (n_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    minimum = min(args.sgd_min_learning_rate, base_learning_rate)
    return minimum + (
        base_learning_rate - minimum
    ) * cosine


def category_order(categories, args, token):
    ordered = list(categories)
    if not args.no_sgd_shuffle_categories:
        rng = random.Random(args.sgd_category_shuffle_seed + int(token))
        rng.shuffle(ordered)
    return ordered


class MultiSizeFJLT:
    def __init__(self, model, layer, requested_projection_sizes, args, device):
        self.model = model
        self.layer = layer
        self.requested_projection_sizes = requested_projection_sizes
        self.projection_sizes = None
        self.random_projection_sizes = None
        self.dropped_projection_sizes = None
        self.replaced_projection_size = None
        self.max_projection_size = None
        self.full_representation_size = None
        self.unprojected_size = None
        self.args = args
        self.device = device
        self.signs = None
        self.indices = None
        self.subsample_indices = None
        self.original_dimension = None
        self.padded_dimension = None
        self.regression_input_scale = None
        self.native_center = None
        self.projected_center_prefix = None
        self.calibration_sample_count = None
        self.curvature_by_size = None
        self.base_weight_learning_rates = None
        self.base_bias_learning_rate = None

    def _extract(self, images):
        if self.layer.lower() == "pixels":
            return images.flatten(1)
        if self.model is not None:
            return self.model.forward_features(images, module_name=self.layer)
        return images.flatten(1)

    def _initialize(self, representations):
        if self.args.random_subsample_size is not None:
            if self.args.random_subsample_size > representations.shape[1]:
                raise ValueError(
                    f"random_subsample_size={self.args.random_subsample_size} exceeds "
                    f"the native dimension {representations.shape[1]}."
                )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.args.random_subsample_seed)
            self.subsample_indices = torch.randperm(
                representations.shape[1],
                device=self.device,
                generator=generator,
            )[: self.args.random_subsample_size]
            native_dimension = self.args.random_subsample_size
        else:
            native_dimension = representations.shape[1]

        self.original_dimension = int(native_dimension)
        self.full_representation_size = self.original_dimension
        # This provisional value is replaced by a measured RMS norm before
        # optimization when auto learning-rate calibration is active.
        self.regression_input_scale = math.sqrt(self.original_dimension)
        self.padded_dimension = int(
            native_dimension
            if is_a_power_of_2(native_dimension)
            else next_power_of_2(native_dimension)
        )

        first_exceeding_index = next(
            (
                index
                for index, size in enumerate(self.requested_projection_sizes)
                if size > self.full_representation_size
            ),
            None,
        )
        if first_exceeding_index is None:
            self.random_projection_sizes = list(self.requested_projection_sizes)
            self.dropped_projection_sizes = []
            self.projection_sizes = list(self.requested_projection_sizes)
        else:
            self.replaced_projection_size = self.requested_projection_sizes[
                first_exceeding_index
            ]
            preceding_sizes = self.requested_projection_sizes[:first_exceeding_index]
            # If native D itself was explicitly requested before the first
            # oversized D, emit one native endpoint and make it unprojected.
            self.random_projection_sizes = [
                size for size in preceding_sizes if size < self.full_representation_size
            ]
            self.unprojected_size = self.full_representation_size
            self.projection_sizes = self.random_projection_sizes + [
                self.unprojected_size
            ]
            self.dropped_projection_sizes = self.requested_projection_sizes[
                first_exceeding_index + 1:
            ]
        self.max_projection_size = (
            max(self.random_projection_sizes)
            if self.random_projection_sizes
            else None
        )

        if self.replaced_projection_size is not None:
            print(
                f"Requested D={self.replaced_projection_size} first exceeds native "
                f"D={self.full_representation_size}; replacing it with the "
                "unprojected native representation and truncating later values "
                f"{self.dropped_projection_sizes}.",
                flush=True,
            )

        if not self.random_projection_sizes:
            print(
                f"Initialized unprojected representation at native "
                f"D={self.original_dimension}.",
                flush=True,
            )
            return

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.args.random_projection_seed)
        self.signs = (
            torch.randint(
                0,
                2,
                (self.padded_dimension,),
                device=self.device,
                generator=generator,
            )
            * 2
            - 1
        ).to(SGD_DTYPE)
        self.indices = torch.randperm(
            self.padded_dimension,
            device=self.device,
            generator=generator,
        )[: self.max_projection_size]
        print(
            f"Initialized FJLT: native D={self.original_dimension}, padded D="
            f"{self.padded_dimension}, random projection sizes="
            f"{self.random_projection_sizes}"
            + (
                f"; oversized request replaced by unprojected D="
                f"{self.unprojected_size}"
                if self.unprojected_size is not None
                else "; no unprojected endpoint added"
            ),
            flush=True,
        )

    @torch.no_grad()
    def initialize_from_images(self, images):
        if self.projection_sizes is None:
            self._initialize(self._extract(images))

    @torch.no_grad()
    def native_representations(self, images):
        representations = self._extract(images).to(dtype=SGD_DTYPE)
        if self.projection_sizes is None:
            self._initialize(representations)
        if self.subsample_indices is not None:
            representations = representations.index_select(1, self.subsample_indices)
        if representations.shape[1] != self.original_dimension:
            raise RuntimeError(
                f"Representation dimension changed from {self.original_dimension} to "
                f"{representations.shape[1]}."
            )
        return representations

    @torch.no_grad()
    def project(self, images):
        representations = self.native_representations(images)
        full_representations = (
            representations if self.unprojected_size is not None else None
        )
        if not self.random_projection_sizes:
            return None, full_representations

        padded = pad_to_power_of_2(representations)
        transformed = hadamard_transform(padded * self.signs.unsqueeze(0))
        projected_prefix = transformed.index_select(1, self.indices)
        return projected_prefix, full_representations

    @torch.no_grad()
    def set_native_center(self, native_center):
        if native_center.shape != (self.original_dimension,):
            raise ValueError(
                f"Expected center shape ({self.original_dimension},), got "
                f"{tuple(native_center.shape)}."
            )
        self.native_center = native_center.to(
            device=self.device,
            dtype=SGD_DTYPE,
        )
        if self.random_projection_sizes:
            padded = pad_to_power_of_2(self.native_center.unsqueeze(0))
            transformed = hadamard_transform(padded * self.signs.unsqueeze(0))
            self.projected_center_prefix = transformed.index_select(
                1,
                self.indices,
            ).squeeze(0)

    def regression_inputs(self, projected_prefix, full_representations, size):
        if size == self.unprojected_size:
            representations = full_representations
            projection_scale = 1.0
            center = self.native_center
        else:
            representations = projected_prefix[:, :size]
            projection_scale = math.sqrt(self.padded_dimension / size)
            center = (
                None
                if self.projected_center_prefix is None
                else self.projected_center_prefix[:size]
            )
        if center is not None:
            representations = representations - center.unsqueeze(0)
        return representations * (projection_scale / self.regression_input_scale)


def initialize_projector(
    projector,
    data_source,
    device,
):
    images = data_source.probe_images().to(device, non_blocking=True)
    projector.initialize_from_images(images)


def build_fold_masks(batch_size, outer_split, args, device, include_inner=True):
    outer_splits = get_splits(
        batch_size,
        method=args.outer_split_method,
        n_splits=args.n_cv_splits,
        test_ratio=args.outer_test_ratio,
        seed=args.random_seed_splits,
    )
    outer_train_idx, outer_test_idx = outer_splits[outer_split]
    if outer_train_idx.numel() == 0 or outer_test_idx.numel() == 0:
        raise RuntimeError(
            f"Empty outer fold for batch size {batch_size}. Reduce n_cv_splits."
        )

    outer_train_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    outer_test_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    outer_train_mask[outer_train_idx.to(device)] = True
    outer_test_mask[outer_test_idx.to(device)] = True

    if not include_inner:
        return outer_train_mask, outer_test_mask, None, None

    inner_splits = get_splits(
        outer_train_idx.numel(),
        method=args.inner_split_method,
        n_splits=args.n_l2_cv_splits,
        test_ratio=args.inner_test_ratio,
        seed=args.random_seed_splits,
    )
    inner_train_mask = torch.zeros(
        batch_size,
        args.n_l2_cv_splits,
        dtype=torch.bool,
        device=device,
    )
    inner_validation_mask = torch.zeros_like(inner_train_mask)
    outer_train_idx_device = outer_train_idx.to(device)
    for inner_split, (train_local, validation_local) in enumerate(inner_splits):
        if train_local.numel() == 0 or validation_local.numel() == 0:
            raise RuntimeError(
                f"Empty inner fold for outer-training batch size "
                f"{outer_train_idx.numel()}. Reduce n_l2_cv_splits."
            )
        inner_train_mask[outer_train_idx_device[train_local.to(device)], inner_split] = True
        inner_validation_mask[
            outer_train_idx_device[validation_local.to(device)], inner_split
        ] = True
    return outer_train_mask, outer_test_mask, inner_train_mask, inner_validation_mask


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, (category, labels) = self.dataset[index]
        return index, image, category, labels


class BalancedCVAssignments:
    """Fixed fold membership matching the old category-local batch splits."""

    def __init__(self, category_indices, args):
        categories = np.asarray(category_indices, dtype=np.int64).reshape(-1)
        self.n_samples = len(categories)
        self.outer_test = torch.zeros(
            self.n_samples,
            args.n_cv_splits,
            dtype=torch.bool,
        )
        self.inner_validation = torch.zeros(
            self.n_samples,
            args.n_cv_splits,
            args.n_l2_cv_splits,
            dtype=torch.bool,
        )

        stable_order = np.argsort(categories, kind="stable")
        ordered_categories = categories[stable_order]
        boundaries = np.flatnonzero(np.diff(ordered_categories)) + 1
        category_groups = np.split(stable_order, boundaries)
        split_template_cache = {}

        def split_templates(chunk_size):
            if chunk_size not in split_template_cache:
                outer_splits = get_splits(
                    chunk_size,
                    method=args.outer_split_method,
                    n_splits=args.n_cv_splits,
                    test_ratio=args.outer_test_ratio,
                    seed=args.random_seed_splits,
                )
                inner_splits = []
                for outer_train, outer_test in outer_splits:
                    if outer_train.numel() == 0 or outer_test.numel() == 0:
                        raise RuntimeError(
                            f"Empty outer fold for category-local chunk size "
                            f"{chunk_size}. Reduce n_cv_splits."
                        )
                    splits = get_splits(
                        outer_train.numel(),
                        method=args.inner_split_method,
                        n_splits=args.n_l2_cv_splits,
                        test_ratio=args.inner_test_ratio,
                        seed=args.random_seed_splits,
                    )
                    if any(
                        train.numel() == 0 or validation.numel() == 0
                        for train, validation in splits
                    ):
                        raise RuntimeError(
                            f"Empty inner fold for category-local chunk size "
                            f"{chunk_size}. Reduce n_l2_cv_splits."
                        )
                    inner_splits.append(splits)
                split_template_cache[chunk_size] = outer_splits, inner_splits
            return split_template_cache[chunk_size]

        outer_ratios = []
        inner_ratios = []
        category_sizes = []
        for group in category_groups:
            category_sizes.append(len(group))
            for start in range(0, len(group), args.batch_size):
                chunk = torch.as_tensor(
                    group[start:start + args.batch_size],
                    dtype=torch.long,
                )
                outer_splits, inner_splits = split_templates(len(chunk))
                for outer_split, (outer_train, outer_test) in enumerate(outer_splits):
                    self.outer_test[chunk[outer_test], outer_split] = True
                    for inner_split, (_inner_train, inner_validation) in enumerate(
                        inner_splits[outer_split]
                    ):
                        validation_indices = chunk[
                            outer_train[inner_validation]
                        ]
                        self.inner_validation[
                            validation_indices,
                            outer_split,
                            inner_split,
                        ] = True

            group_tensor = torch.as_tensor(group, dtype=torch.long)
            category_outer_counts = self.outer_test[group_tensor].sum(dim=0)
            outer_ratios.extend(
                (category_outer_counts.to(torch.float64) / len(group)).tolist()
            )
            category_inner_counts = self.inner_validation[group_tensor].sum(dim=0)
            category_outer_train_counts = len(group) - category_outer_counts
            inner_ratio = category_inner_counts.to(torch.float64) / (
                category_outer_train_counts[:, None].to(torch.float64)
            )
            inner_ratios.extend(inner_ratio.reshape(-1).tolist())

        if torch.any(
            self.inner_validation
            & self.outer_test[:, :, None]
        ):
            raise RuntimeError("Inner-validation and outer-test assignments overlap.")
        if args.outer_split_method == "kfold":
            if not torch.all(self.outer_test.sum(dim=1) == 1):
                raise RuntimeError(
                    "K-fold outer assignments do not place every image in exactly "
                    "one outer-test fold."
                )
        if args.inner_split_method == "kfold":
            for outer_split in range(args.n_cv_splits):
                expected_count = (~self.outer_test[:, outer_split]).to(torch.int64)
                actual_count = self.inner_validation[:, outer_split].sum(dim=1)
                if not torch.equal(actual_count, expected_count):
                    raise RuntimeError(
                        "K-fold inner assignments do not place every outer-training "
                        "image in exactly one inner-validation fold."
                    )

        if len(set(category_sizes)) == 1:
            tolerance = 1e-12
            if max(outer_ratios) - min(outer_ratios) > tolerance:
                raise RuntimeError(
                    "Equal-sized categories received unequal outer-test ratios."
                )
            if max(inner_ratios) - min(inner_ratios) > tolerance:
                raise RuntimeError(
                    "Equal-sized categories received unequal inner-validation ratios."
                )

        print(
            f"Precomputed fixed category-balanced CV assignments for "
            f"{len(category_groups)} categories and {self.n_samples:,} images.",
            flush=True,
        )
        print(
            f"  Outer test ratios across category/fold pairs: "
            f"{min(outer_ratios):.6f} to {max(outer_ratios):.6f}",
            flush=True,
        )
        print(
            f"  Inner validation ratios within outer train sets: "
            f"{min(inner_ratios):.6f} to {max(inner_ratios):.6f}",
            flush=True,
        )

    def masks(self, batch_indices, outer_split, device, include_inner):
        batch_indices = batch_indices.to(dtype=torch.long, device="cpu")
        outer_test = self.outer_test[batch_indices, outer_split].to(device)
        outer_train = ~outer_test
        if not include_inner:
            return outer_train, outer_test, None, None

        inner_validation = self.inner_validation[
            batch_indices,
            outer_split,
        ].to(device)
        inner_train = outer_train[:, None] & ~inner_validation
        return outer_train, outer_test, inner_train, inner_validation


class GlobalSDXLDataSource:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.indexed_dataset = IndexedDataset(dataset)
        self.assignments = BalancedCVAssignments(
            dataset.category_indices.detach().cpu().numpy(),
            args,
        )

    def probe_images(self):
        image, _target = self.dataset[0]
        return image.unsqueeze(0)

    def normalization_stats(self):
        mean = np.asarray(self.dataset.bbox_mean).reshape(-1)
        std = np.asarray(self.dataset.bbox_std).reshape(-1)
        return (
            {feature: float(mean[index]) for index, feature in enumerate(FEATURES)},
            {feature: float(std[index]) for index, feature in enumerate(FEATURES)},
        )

    def iter_image_batches(self, args, device, max_batches, token):
        generator = torch.Generator()
        generator.manual_seed(args.sgd_category_shuffle_seed + int(token))
        dataloader = DataLoader(
            self.indexed_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for batch_number, batch in enumerate(dataloader, start=1):
            _batch_indices, images, _category_indices, _labels = batch
            yield images.to(device, non_blocking=True)
            if batch_number >= max_batches:
                break

    def iter_batches(
        self,
        projector,
        args,
        device,
        outer_split,
        token,
        include_inner,
        shuffle,
    ):
        do_shuffle = shuffle and not args.no_sgd_shuffle_categories
        generator = None
        if do_shuffle:
            generator = torch.Generator()
            generator.manual_seed(args.sgd_category_shuffle_seed + int(token))
        dataloader = DataLoader(
            self.indexed_dataset,
            batch_size=args.batch_size,
            shuffle=do_shuffle,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        total_batches = len(dataloader)
        for batch_number, batch in enumerate(dataloader, start=1):
            batch_indices, images, _category_indices, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=SGD_DTYPE, non_blocking=True)
            with torch.no_grad():
                projected_prefix, full_representations = projector.project(images)
            masks = self.assignments.masks(
                batch_indices,
                outer_split,
                device,
                include_inner,
            )
            if (
                batch_number == 1
                or batch_number % args.progress_every_batches == 0
                or batch_number == total_batches
            ):
                percentage = 100.0 * batch_number / total_batches
                print(
                    f"  batch {batch_number:,}/{total_batches:,} "
                    f"({percentage:.2f}%)",
                    flush=True,
                )
            yield projected_prefix, full_representations, labels, masks
            del images, labels, projected_prefix, full_representations

    def close(self):
        close_dataset(self.dataset)


class CategoryDataSource:
    """Compatibility path for the non-SDXL datasets supported by this script."""

    def __init__(self, categories, CategoryDataset, args):
        self.categories = categories
        self.CategoryDataset = CategoryDataset
        self.args = args

    def probe_images(self):
        dataset = self.CategoryDataset(category=self.categories[0])
        try:
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            return batch[0]
        finally:
            close_dataset(dataset)

    def normalization_stats(self):
        return get_normalization_stats(
            self.CategoryDataset,
            self.categories[0],
        )

    def iter_image_batches(self, args, device, max_batches, token):
        emitted = 0
        for category in category_order(self.categories, args, token):
            dataset = self.CategoryDataset(category=category)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            try:
                for batch in dataloader:
                    images = batch[0].to(device, non_blocking=True)
                    yield images
                    emitted += 1
                    if emitted >= max_batches:
                        return
            finally:
                close_dataset(dataset)

    def iter_batches(
        self,
        projector,
        args,
        device,
        outer_split,
        token,
        include_inner,
        shuffle,
    ):
        ordered_categories = category_order(self.categories, args, token)
        for category_number, category in enumerate(ordered_categories):
            if category_number % 25 == 0:
                print(
                    f"  category {category_number + 1}/{len(ordered_categories)}: "
                    f"{category}",
                    flush=True,
                )
            dataset = self.CategoryDataset(category=category)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            try:
                for batch in dataloader:
                    images, _category_indices, labels, _paths, _category_strings = batch
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, dtype=SGD_DTYPE, non_blocking=True)
                    with torch.no_grad():
                        projected_prefix, full_representations = projector.project(images)
                    masks = build_fold_masks(
                        labels.shape[0],
                        outer_split,
                        args,
                        device,
                        include_inner=include_inner,
                    )
                    yield projected_prefix, full_representations, labels, masks
                    del images, labels, projected_prefix, full_representations
            finally:
                close_dataset(dataset)

    def close(self):
        pass


def iter_projected_batches(
    data_source,
    projector,
    args,
    device,
    outer_split,
    token,
    include_inner,
    shuffle,
):
    yield from data_source.iter_batches(
        projector,
        args,
        device,
        outer_split,
        token,
        include_inner,
        shuffle,
    )


def training_fraction(args):
    outer_fraction = (
        (args.n_cv_splits - 1) / args.n_cv_splits
        if args.outer_split_method == "kfold"
        else 1.0 - args.outer_test_ratio
    )
    inner_fraction = (
        (args.n_l2_cv_splits - 1) / args.n_l2_cv_splits
        if args.inner_split_method == "kfold"
        else 1.0 - args.inner_test_ratio
    )
    return outer_fraction * inner_fraction


@torch.no_grad()
def calibrate_optimizer(data_source, projector, projection_sizes, args, device):
    if not args.sgd_auto_learning_rate:
        weight_rates = {
            size: args.sgd_learning_rate for size in projection_sizes
        }
        bias_rate = args.sgd_learning_rate
        projector.curvature_by_size = None
        projector.base_weight_learning_rates = weight_rates
        projector.base_bias_learning_rate = bias_rate
        return weight_rates, bias_rate

    print(
        f"Calibrating representation scale from "
        f"{args.sgd_calibration_batches} shuffled batches...",
        flush=True,
    )
    norm_sq_sum = 0.0
    representation_sum = None
    sample_count = 0
    for images in data_source.iter_image_batches(
        args,
        device,
        args.sgd_calibration_batches,
        token=970001,
    ):
        representations = projector.native_representations(images)
        norm_sq_sum += float(
            representations.square().sum(dtype=torch.float64).item()
        )
        batch_sum = representations.sum(dim=0, dtype=torch.float64)
        if representation_sum is None:
            representation_sum = batch_sum
        else:
            representation_sum.add_(batch_sum)
        sample_count += representations.shape[0]
        del images, representations
    if sample_count == 0 or not math.isfinite(norm_sq_sum) or norm_sq_sum <= 0:
        raise RuntimeError("Could not estimate a finite, positive representation scale.")

    native_center = representation_sum / sample_count
    centered_norm_sq_mean = (
        norm_sq_sum / sample_count
        - float(native_center.square().sum().item())
    )
    if not math.isfinite(centered_norm_sq_mean) or centered_norm_sq_mean <= 0:
        raise RuntimeError(
            "Could not estimate a finite, positive centered representation scale."
        )
    projector.set_native_center(native_center.to(dtype=SGD_DTYPE))
    projector.regression_input_scale = math.sqrt(centered_norm_sq_mean)
    projector.calibration_sample_count = sample_count
    del representation_sum, native_center
    print(
        f"Measured centered RMS representation norm from {sample_count:,} images: "
        f"{projector.regression_input_scale:.6g}",
        flush=True,
    )

    print(
        "Estimating the largest normalized minibatch curvature for each D...",
        flush=True,
    )
    curvature = {size: 0.0 for size in projection_sizes}
    curvature_batches = 0
    for images in data_source.iter_image_batches(
        args,
        device,
        args.sgd_calibration_batches,
        token=970002,
    ):
        projected, full_representations = projector.project(images)
        batch_size = images.shape[0]
        for size in projection_sizes:
            inputs = projector.regression_inputs(
                projected,
                full_representations,
                size,
            )
            gram = inputs @ inputs.T
            gram.div_(batch_size)
            largest = float(torch.linalg.eigvalsh(gram)[-1].item())
            if not math.isfinite(largest) or largest <= 0:
                raise RuntimeError(
                    f"Invalid curvature estimate {largest} for D={size}."
                )
            curvature[size] = max(curvature[size], largest)
        curvature_batches += 1
        del images, projected, full_representations

    # A covariance from an inner-training subset is bounded by the full-batch
    # covariance divided by the retained fraction. Applying this factor makes
    # the rates safe for the smallest masks used by nested CV.
    retained_fraction = training_fraction(args)
    curvature = {
        size: value / retained_fraction for size, value in curvature.items()
    }
    bias_rate = (
        args.sgd_lr_safety_factor * args.sgd_bias_lr_fraction
    )
    weight_budget = (
        args.sgd_lr_safety_factor * (1.0 - args.sgd_bias_lr_fraction)
    )
    weight_rates = {
        size: min(args.sgd_learning_rate, weight_budget / curvature[size])
        for size in projection_sizes
    }
    projector.curvature_by_size = curvature
    projector.base_weight_learning_rates = weight_rates
    projector.base_bias_learning_rate = bias_rate

    print(
        f"Auto-LR calibration used {curvature_batches} batches and inner-training "
        f"fraction {retained_fraction:.6g}.",
        flush=True,
    )
    for size in projection_sizes:
        print(
            f"  D={size}: conservative curvature={curvature[size]:.6g}, "
            f"weight lr={weight_rates[size]:.6g}, bias lr={bias_rate:.6g}",
            flush=True,
        )
    return weight_rates, bias_rate


class InnerReadouts(nn.Module):
    def __init__(self, projection_sizes, n_inner, n_l2, n_features, device):
        super().__init__()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        for size in projection_sizes:
            key = f"d{size}"
            self.weights[key] = nn.Parameter(
                torch.zeros(
                    size,
                    n_inner,
                    n_l2,
                    n_features,
                    device=device,
                    dtype=SGD_DTYPE,
                )
            )
            self.biases[key] = nn.Parameter(
                torch.zeros(
                    n_inner,
                    n_l2,
                    n_features,
                    device=device,
                    dtype=SGD_DTYPE,
                )
            )

    def predict(self, projected_prefix, full_representations, size, projector):
        key = f"d{size}"
        weight = self.weights[key]
        representations = projector.regression_inputs(
            projected_prefix,
            full_representations,
            size,
        )
        prediction = representations @ weight.reshape(size, -1)
        return prediction.reshape(
            representations.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        ) + self.biases[key].unsqueeze(0)


class FinalReadouts(nn.Module):
    def __init__(self, projection_sizes, n_features, device):
        super().__init__()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        for size in projection_sizes:
            key = f"d{size}"
            self.weights[key] = nn.Parameter(
                torch.zeros(size, n_features, device=device, dtype=SGD_DTYPE)
            )
            self.biases[key] = nn.Parameter(
                torch.zeros(n_features, device=device, dtype=SGD_DTYPE)
            )

    def predict(self, projected_prefix, full_representations, size, projector):
        key = f"d{size}"
        representations = projector.regression_inputs(
            projected_prefix,
            full_representations,
            size,
        )
        return representations @ self.weights[key] + self.biases[key]


def make_optimizers(
    model,
    projection_sizes,
    weight_learning_rates,
    bias_learning_rate,
):
    return {
        size: torch.optim.SGD(
            [
                {
                    "params": [model.weights[f"d{size}"]],
                    "lr": weight_learning_rates[size],
                },
                {
                    "params": [model.biases[f"d{size}"]],
                    "lr": bias_learning_rate,
                },
            ],
        )
        for size in projection_sizes
    }


@torch.no_grad()
def proximal_shrink_inner(
    model,
    size,
    l2_tensor,
    learning_rate,
    pending_log_shrink,
    force=False,
):
    pending_log_shrink.sub_(
        torch.log1p(
            2.0
            * learning_rate
            * l2_tensor
        )
    )
    apply = pending_log_shrink <= math.log1p(-SHRINK_APPLICATION_THRESHOLD)
    if force:
        apply = torch.ones_like(apply, dtype=torch.bool)
    factors = torch.where(
        apply,
        pending_log_shrink.exp(),
        torch.ones_like(pending_log_shrink),
    )
    model.weights[f"d{size}"].mul_(
        factors.to(dtype=SGD_DTYPE).view(1, 1, -1, 1)
    )
    pending_log_shrink.masked_fill_(apply, 0.0)


@torch.no_grad()
def proximal_shrink_final(
    model,
    size,
    selected_l2,
    learning_rate,
    pending_log_shrink,
    force=False,
):
    pending_log_shrink.sub_(
        torch.log1p(
            2.0
            * learning_rate
            * selected_l2
        )
    )
    apply = pending_log_shrink <= math.log1p(-SHRINK_APPLICATION_THRESHOLD)
    if force:
        apply = torch.ones_like(apply, dtype=torch.bool)
    factors = torch.where(
        apply,
        pending_log_shrink.exp(),
        torch.ones_like(pending_log_shrink),
    )
    model.weights[f"d{size}"].mul_(
        factors.to(dtype=SGD_DTYPE).view(1, -1)
    )
    pending_log_shrink.masked_fill_(apply, 0.0)


def empty_label_stats(n_groups, n_features, device, with_extrema=False):
    stats = {
        "count": torch.zeros(n_groups, dtype=torch.float64, device=device),
        "sum": torch.zeros(n_groups, n_features, dtype=torch.float64, device=device),
        "sum_sq": torch.zeros(n_groups, n_features, dtype=torch.float64, device=device),
    }
    if with_extrema:
        stats["minimum"] = torch.full(
            (n_groups, n_features), float("inf"), dtype=torch.float64, device=device
        )
        stats["maximum"] = torch.full(
            (n_groups, n_features), -float("inf"), dtype=torch.float64, device=device
        )
    return stats


@torch.no_grad()
def update_grouped_label_stats(stats, labels, masks):
    labels64 = labels.to(torch.float64)
    masks64 = masks.to(torch.float64)
    stats["count"].add_(masks64.sum(dim=0))
    stats["sum"].add_(torch.einsum("bg,bf->gf", masks64, labels64))
    stats["sum_sq"].add_(torch.einsum("bg,bf->gf", masks64, labels64.square()))
    if "minimum" in stats:
        for group in range(masks.shape[1]):
            selected = labels64[masks[:, group]]
            if selected.numel() > 0:
                stats["minimum"][group] = torch.minimum(
                    stats["minimum"][group], selected.amin(dim=0)
                )
                stats["maximum"][group] = torch.maximum(
                    stats["maximum"][group], selected.amax(dim=0)
                )


@torch.no_grad()
def update_global_label_stats(stats, labels):
    labels64 = labels.to(torch.float64)
    stats["count"].add_(labels64.shape[0])
    stats["sum"].add_(labels64.sum(dim=0, keepdim=True))
    stats["sum_sq"].add_(labels64.square().sum(dim=0, keepdim=True))
    stats["minimum"][0] = torch.minimum(stats["minimum"][0], labels64.amin(dim=0))
    stats["maximum"][0] = torch.maximum(stats["maximum"][0], labels64.amax(dim=0))


def stats_to_cpu(stats):
    return {name: value.detach().cpu().numpy() for name, value in stats.items()}


def train_inner_models(
    data_source,
    projector,
    projection_sizes,
    l2_strengths,
    args,
    device,
    outer_split,
    collect_global_stats,
    base_weight_learning_rates,
    base_bias_learning_rate,
):
    n_features = len(FEATURES)
    model = InnerReadouts(
        projection_sizes,
        args.n_l2_cv_splits,
        len(l2_strengths),
        n_features,
        device,
    )
    optimizers = make_optimizers(
        model,
        projection_sizes,
        base_weight_learning_rates,
        base_bias_learning_rate,
    )
    effective_l2_strengths = np.asarray(l2_strengths) / (
        projector.regression_input_scale ** 2
    )
    l2_tensor = torch.tensor(
        effective_l2_strengths,
        device=device,
        dtype=torch.float64,
    )
    pending_log_shrink = {
        size: torch.zeros(
            len(l2_strengths),
            device=device,
            dtype=torch.float64,
        )
        for size in projection_sizes
    }
    histories = {size: [] for size in projection_sizes}
    train_label_stats = empty_label_stats(
        args.n_l2_cv_splits, n_features, device
    )
    global_label_stats = (
        empty_label_stats(1, n_features, device, with_extrema=True)
        if collect_global_stats
        else None
    )

    for epoch in range(args.sgd_inner_epochs):
        weight_learning_rates = {
            size: learning_rate_for_epoch(
                args,
                epoch,
                args.sgd_inner_epochs,
                base_weight_learning_rates[size],
            )
            for size in projection_sizes
        }
        bias_learning_rate = learning_rate_for_epoch(
            args,
            epoch,
            args.sgd_inner_epochs,
            base_bias_learning_rate,
        )
        for size, optimizer in optimizers.items():
            optimizer.param_groups[0]["lr"] = weight_learning_rates[size]
            optimizer.param_groups[1]["lr"] = bias_learning_rate
        epoch_sse = {
            size: torch.zeros(
                args.n_l2_cv_splits,
                len(l2_strengths),
                n_features,
                dtype=torch.float64,
                device=device,
            )
            for size in projection_sizes
        }
        epoch_validation_sse = {
            size: torch.zeros(
                args.n_l2_cv_splits,
                len(l2_strengths),
                n_features,
                dtype=torch.float64,
                device=device,
            )
            for size in projection_sizes
        }
        epoch_count = torch.zeros(
            args.n_l2_cv_splits, dtype=torch.float64, device=device
        )
        epoch_validation_count = torch.zeros_like(epoch_count)

        print(
            f"Outer {outer_split + 1}/{args.n_cv_splits}, inner-training epoch "
            f"{epoch + 1}/{args.sgd_inner_epochs}, weight lr range="
            f"{min(weight_learning_rates.values()):.6g} to "
            f"{max(weight_learning_rates.values()):.6g}, "
            f"bias lr={bias_learning_rate:.6g}",
            flush=True,
        )
        token = outer_split * 100000 + epoch
        for projected, full_representations, labels, masks in iter_projected_batches(
            data_source,
            projector,
            args,
            device,
            outer_split,
            token,
            include_inner=True,
            shuffle=True,
        ):
            _outer_train, _outer_test, inner_train, inner_validation = masks
            counts = inner_train.sum(dim=0).to(torch.float64)
            validation_counts = inner_validation.sum(dim=0).to(torch.float64)
            if epoch == 0:
                update_grouped_label_stats(train_label_stats, labels, inner_train)
                if global_label_stats is not None:
                    update_global_label_stats(global_label_stats, labels)
            epoch_count.add_(counts)
            epoch_validation_count.add_(validation_counts)

            for size in projection_sizes:
                optimizer = optimizers[size]
                optimizer.zero_grad(set_to_none=True)
                prediction = model.predict(
                    projected,
                    full_representations,
                    size,
                    projector,
                )
                squared_error = (prediction - labels[:, None, None, :]).square()
                masked_sse = (
                    squared_error
                    * inner_train[:, :, None, None].to(squared_error.dtype)
                ).sum(dim=0)
                data_loss = (
                    masked_sse
                    / counts.to(masked_sse.dtype)[:, None, None]
                ).sum()
                if not torch.isfinite(data_loss):
                    raise FloatingPointError(
                        f"Non-finite inner loss for D={size}, epoch={epoch + 1}. "
                        "Reduce --sgd_learning_rate."
                    )
                data_loss.backward()
                optimizer.step()
                proximal_shrink_inner(
                    model,
                    size,
                    l2_tensor,
                    weight_learning_rates[size],
                    pending_log_shrink[size],
                )
                optimizer.zero_grad(set_to_none=True)
                epoch_sse[size].add_(masked_sse.detach().to(torch.float64))
                epoch_validation_sse[size].add_(
                    (
                        squared_error
                        * inner_validation[:, :, None, None].to(
                            squared_error.dtype
                        )
                    )
                    .sum(dim=0)
                    .detach()
                    .to(torch.float64)
                )

        for size in projection_sizes:
            proximal_shrink_inner(
                model,
                size,
                l2_tensor,
                0.0,
                pending_log_shrink[size],
                force=True,
            )
            mse = epoch_sse[size] / epoch_count[:, None, None]
            validation_mse = (
                epoch_validation_sse[size]
                / epoch_validation_count[:, None, None]
            )
            validation_mse_mean = validation_mse.mean(dim=0)
            online_best_indices = validation_mse_mean.argmin(dim=0)
            with torch.no_grad():
                weight = model.weights[f"d{size}"]
                weight_norm_sq = weight.square().sum(dim=0).to(torch.float64)
                objective = mse + (
                    torch.tensor(
                        effective_l2_strengths,
                        device=device,
                        dtype=torch.float64,
                    )
                    .view(1, -1, 1)
                    * weight_norm_sq
                )
            histories[size].append(
                {
                    "learning_rate": weight_learning_rates[size],
                    "weight_learning_rate": weight_learning_rates[size],
                    "bias_learning_rate": bias_learning_rate,
                    "training_mse": mse.cpu().numpy(),
                    "online_validation_mse": validation_mse.cpu().numpy(),
                    "online_best_l2_indices": online_best_indices.cpu().numpy(),
                    "online_best_l2": np.asarray(l2_strengths)[
                        online_best_indices.cpu().numpy()
                    ],
                    "training_objective": objective.cpu().numpy(),
                }
            )
            best_training_mse = np.asarray(
                [
                    float(mse[:, online_best_indices[feature], feature].mean())
                    for feature in range(n_features)
                ]
            )
            best_validation_mse = np.asarray(
                [
                    float(validation_mse_mean[online_best_indices[feature], feature])
                    for feature in range(n_features)
                ]
            )
            best_l2 = np.asarray(l2_strengths)[online_best_indices.cpu().numpy()]
            print(
                f"  D={size}: online train MSE="
                f"{dict(zip(FEATURES, best_training_mse))}; "
                f"online inner-val MSE="
                f"{dict(zip(FEATURES, best_validation_mse))}; "
                f"online best L2={dict(zip(FEATURES, best_l2))}",
                flush=True,
            )

    return (
        model,
        histories,
        stats_to_cpu(train_label_stats),
        stats_to_cpu(global_label_stats) if global_label_stats is not None else None,
    )


@torch.no_grad()
def evaluate_inner_models(
    model,
    data_source,
    projector,
    projection_sizes,
    l2_strengths,
    args,
    device,
    outer_split,
    inner_train_label_stats,
):
    n_features = len(FEATURES)
    sse = {
        size: torch.zeros(
            args.n_l2_cv_splits,
            len(l2_strengths),
            n_features,
            dtype=torch.float64,
            device=device,
        )
        for size in projection_sizes
    }
    validation_stats = empty_label_stats(
        args.n_l2_cv_splits, n_features, device
    )
    print(f"Outer {outer_split + 1}: evaluating inner validation folds", flush=True)
    token = outer_split * 100000 + 50000
    for projected, full_representations, labels, masks in iter_projected_batches(
        data_source,
        projector,
        args,
        device,
        outer_split,
        token,
        include_inner=True,
        shuffle=False,
    ):
        _outer_train, _outer_test, _inner_train, inner_validation = masks
        update_grouped_label_stats(validation_stats, labels, inner_validation)
        for size in projection_sizes:
            prediction = model.predict(
                projected,
                full_representations,
                size,
                projector,
            )
            squared_error = (prediction - labels[:, None, None, :]).square()
            sse[size].add_(
                (
                    squared_error
                    * inner_validation[:, :, None, None].to(squared_error.dtype)
                )
                .sum(dim=0)
                .to(torch.float64)
            )

    validation_stats_cpu = stats_to_cpu(validation_stats)
    validation_count = validation_stats_cpu["count"]
    validation_mean = validation_stats_cpu["sum"] / validation_count[:, None]
    validation_second_moment = (
        validation_stats_cpu["sum_sq"] / validation_count[:, None]
    )
    train_mean = (
        inner_train_label_stats["sum"]
        / inner_train_label_stats["count"][:, None]
    )
    validation_variance_about_train_mean = (
        validation_second_moment
        - 2.0 * train_mean * validation_mean
        + train_mean ** 2
    )

    output = {}
    for size in projection_sizes:
        mse_by_inner = sse[size].cpu().numpy() / validation_count[:, None, None]
        normalized_by_inner = (
            mse_by_inner / validation_variance_about_train_mean[:, None, :]
        )
        mean_mse = mse_by_inner.mean(axis=0)
        best_indices = np.argmin(mean_mse, axis=0)
        output[size] = {
            "mse_by_inner": mse_by_inner,
            "mse_mean": mean_mse,
            "mse_std": mse_by_inner.std(axis=0),
            "normalized_mse_mean": normalized_by_inner.mean(axis=0),
            "normalized_mse_std": normalized_by_inner.std(axis=0),
            "best_l2_indices": best_indices,
            "best_l2": np.asarray(l2_strengths)[best_indices],
        }
    return output


@torch.no_grad()
def initialize_final_from_inner(
    inner_model,
    inner_evaluation,
    projection_sizes,
    n_features,
    device,
):
    final_model = FinalReadouts(projection_sizes, n_features, device)
    for size in projection_sizes:
        inner_weight = inner_model.weights[f"d{size}"]
        inner_bias = inner_model.biases[f"d{size}"]
        for feature in range(n_features):
            l2_index = int(inner_evaluation[size]["best_l2_indices"][feature])
            final_model.weights[f"d{size}"][:, feature].copy_(
                inner_weight[:, :, l2_index, feature].mean(dim=1)
            )
            final_model.biases[f"d{size}"][feature].copy_(
                inner_bias[:, l2_index, feature].mean(dim=0)
            )
    return final_model


def train_final_models(
    model,
    inner_evaluation,
    data_source,
    projector,
    projection_sizes,
    args,
    device,
    outer_split,
    base_weight_learning_rates,
    base_bias_learning_rate,
):
    n_features = len(FEATURES)
    optimizers = make_optimizers(
        model,
        projection_sizes,
        base_weight_learning_rates,
        base_bias_learning_rate,
    )
    selected_l2 = {
        size: torch.tensor(
            inner_evaluation[size]["best_l2"]
            / (projector.regression_input_scale ** 2),
            dtype=torch.float64,
            device=device,
        )
        for size in projection_sizes
    }
    pending_log_shrink = {
        size: torch.zeros(
            n_features,
            device=device,
            dtype=torch.float64,
        )
        for size in projection_sizes
    }
    histories = {size: [] for size in projection_sizes}
    outer_train_label_stats = empty_label_stats(1, n_features, device)

    for epoch in range(args.sgd_final_epochs):
        weight_learning_rates = {
            size: learning_rate_for_epoch(
                args,
                epoch,
                args.sgd_final_epochs,
                base_weight_learning_rates[size],
            )
            for size in projection_sizes
        }
        bias_learning_rate = learning_rate_for_epoch(
            args,
            epoch,
            args.sgd_final_epochs,
            base_bias_learning_rate,
        )
        for size, optimizer in optimizers.items():
            optimizer.param_groups[0]["lr"] = weight_learning_rates[size]
            optimizer.param_groups[1]["lr"] = bias_learning_rate
        epoch_sse = {
            size: torch.zeros(n_features, dtype=torch.float64, device=device)
            for size in projection_sizes
        }
        epoch_count = torch.tensor(0.0, dtype=torch.float64, device=device)

        print(
            f"Outer {outer_split + 1}/{args.n_cv_splits}, final-training epoch "
            f"{epoch + 1}/{args.sgd_final_epochs}, weight lr range="
            f"{min(weight_learning_rates.values()):.6g} to "
            f"{max(weight_learning_rates.values()):.6g}, "
            f"bias lr={bias_learning_rate:.6g}",
            flush=True,
        )
        token = outer_split * 100000 + 60000 + epoch
        for projected, full_representations, labels, masks in iter_projected_batches(
            data_source,
            projector,
            args,
            device,
            outer_split,
            token,
            include_inner=False,
            shuffle=True,
        ):
            outer_train, _outer_test, _inner_train, _inner_validation = masks
            count = outer_train.sum().to(torch.float64)
            if epoch == 0:
                update_grouped_label_stats(
                    outer_train_label_stats, labels, outer_train[:, None]
                )
            epoch_count.add_(count)
            for size in projection_sizes:
                optimizer = optimizers[size]
                optimizer.zero_grad(set_to_none=True)
                prediction = model.predict(
                    projected,
                    full_representations,
                    size,
                    projector,
                )
                squared_error = (prediction - labels).square()
                feature_sse = squared_error[outer_train].sum(dim=0)
                data_loss = (feature_sse / count.to(feature_sse.dtype)).sum()
                if not torch.isfinite(data_loss):
                    raise FloatingPointError(
                        f"Non-finite final loss for D={size}, epoch={epoch + 1}. "
                        "Reduce --sgd_learning_rate."
                    )
                data_loss.backward()
                optimizer.step()
                proximal_shrink_final(
                    model,
                    size,
                    selected_l2[size],
                    weight_learning_rates[size],
                    pending_log_shrink[size],
                )
                optimizer.zero_grad(set_to_none=True)
                epoch_sse[size].add_(feature_sse.detach().to(torch.float64))

        for size in projection_sizes:
            proximal_shrink_final(
                model,
                size,
                selected_l2[size],
                0.0,
                pending_log_shrink[size],
                force=True,
            )
            mse = epoch_sse[size] / epoch_count
            with torch.no_grad():
                weight_norm_sq = (
                    model.weights[f"d{size}"].square().sum(dim=0).to(torch.float64)
                )
                objective = mse + selected_l2[size].to(torch.float64) * weight_norm_sq
            histories[size].append(
                {
                    "learning_rate": weight_learning_rates[size],
                    "weight_learning_rate": weight_learning_rates[size],
                    "bias_learning_rate": bias_learning_rate,
                    "training_mse": mse.cpu().numpy(),
                    "training_objective": objective.cpu().numpy(),
                }
            )
            print(
                f"  D={size}: online outer-train MSE="
                f"{dict(zip(FEATURES, mse.cpu().numpy()))}",
                flush=True,
            )
    return histories, stats_to_cpu(outer_train_label_stats)


@torch.no_grad()
def evaluate_final_models(
    model,
    data_source,
    projector,
    projection_sizes,
    args,
    device,
    outer_split,
    outer_train_label_stats,
):
    n_features = len(FEATURES)
    sse = {
        size: torch.zeros(n_features, dtype=torch.float64, device=device)
        for size in projection_sizes
    }
    test_label_stats = empty_label_stats(1, n_features, device, with_extrema=True)
    print(f"Outer {outer_split + 1}: evaluating outer test fold", flush=True)
    token = outer_split * 100000 + 90000
    for projected, full_representations, labels, masks in iter_projected_batches(
        data_source,
        projector,
        args,
        device,
        outer_split,
        token,
        include_inner=False,
        shuffle=False,
    ):
        _outer_train, outer_test, _inner_train, _inner_validation = masks
        update_grouped_label_stats(test_label_stats, labels, outer_test[:, None])
        for size in projection_sizes:
            prediction = model.predict(
                projected,
                full_representations,
                size,
                projector,
            )
            sse[size].add_(
                (prediction[outer_test] - labels[outer_test])
                .square()
                .sum(dim=0)
                .to(torch.float64)
            )

    test_stats = stats_to_cpu(test_label_stats)
    test_count = float(test_stats["count"][0])
    test_mean = test_stats["sum"][0] / test_count
    test_second_moment = test_stats["sum_sq"][0] / test_count
    train_mean = (
        outer_train_label_stats["sum"][0]
        / outer_train_label_stats["count"][0]
    )
    variance_about_train_mean = (
        test_second_moment - 2.0 * train_mean * test_mean + train_mean ** 2
    )
    output = {}
    for size in projection_sizes:
        mse = sse[size].cpu().numpy() / test_count
        output[size] = {
            "mse": mse,
            "normalized_mse": mse / variance_about_train_mean,
            "label_variance": variance_about_train_mean,
            "test_minimum": test_stats["minimum"][0],
            "test_maximum": test_stats["maximum"][0],
        }
    return output


def initialize_metrics(projection_sizes, args, n_features):
    return {
        size: {
            "inner_mse_mean": np.full(
                (args.n_cv_splits, args.n_l2, n_features), np.nan
            ),
            "inner_mse_std": np.full(
                (args.n_cv_splits, args.n_l2, n_features), np.nan
            ),
            "inner_normalized_mean": np.full(
                (args.n_cv_splits, args.n_l2, n_features), np.nan
            ),
            "inner_normalized_std": np.full(
                (args.n_cv_splits, args.n_l2, n_features), np.nan
            ),
            "best_l2": np.full((args.n_cv_splits, n_features), np.nan),
            "mse": np.full((args.n_cv_splits, n_features), np.nan),
            "normalized_mse": np.full((args.n_cv_splits, n_features), np.nan),
            "label_variance": np.full((args.n_cv_splits, n_features), np.nan),
            "label_minimum": np.full((args.n_cv_splits, n_features), np.nan),
            "label_maximum": np.full((args.n_cv_splits, n_features), np.nan),
            "inner_history": [None] * args.n_cv_splits,
            "final_history": [None] * args.n_cv_splits,
        }
        for size in projection_sizes
    }


def get_normalization_stats(CategoryDataset, category):
    dataset = CategoryDataset(category=category)
    try:
        base = unwrap_dataset(dataset)
        mean = getattr(base, "bbox_mean", None)
        std = getattr(base, "bbox_std", None)
        mean = None if mean is None else np.asarray(mean).reshape(-1)
        std = None if std is None else np.asarray(std).reshape(-1)
        mean_dict = {
            feature: None if mean is None else float(mean[index])
            for index, feature in enumerate(FEATURES)
        }
        std_dict = {
            feature: None if std is None else float(std[index])
            for index, feature in enumerate(FEATURES)
        }
        return mean_dict, std_dict
    finally:
        close_dataset(dataset)


def result_directory(args, projection_size, list_mode):
    checkpoint_dir = os.path.dirname(args.checkpoint)
    base = os.path.join(checkpoint_dir, args.results_folder_name)
    if not list_mode:
        return base
    return os.path.join(
        base,
        "projected_neurons",
        str(projection_size),
        str(args.random_projection_seed),
    )


def build_results(
    args,
    projection_size,
    metrics,
    l2_strengths,
    global_label_stats,
    normalization_mean,
    normalization_std,
    completed_outer_splits,
    projector,
):
    completed = list(range(completed_outer_splits))
    full_count = float(global_label_stats["count"][0])
    full_mean = global_label_stats["sum"][0] / full_count
    full_variance = global_label_stats["sum_sq"][0] / full_count - full_mean ** 2
    projection_style = (
        "unprojected_full_representation"
        if projection_size == projector.unprojected_size
        else "fjlt"
    )

    result_args = vars(args).copy()
    result_args["random_projection_size"] = projection_size
    result_args["requested_random_projection_sizes"] = list(
        projector.requested_projection_sizes
    )
    result_args["random_projection_sizes"] = list(
        projector.random_projection_sizes
    )
    result_args["readout_sizes"] = list(projector.projection_sizes)
    result_args["regression_solver"] = "proximal_sgd"
    results = {
        "args": result_args,
        "solver": "proximal_sgd",
        "representation_style": projection_style,
        "completed_outer_splits": completed,
        "inner_mse_test_avg": {},
        "inner_mse_test_std": {},
        "inner_normalized_mse_test_avg": {},
        "inner_normalized_mse_test_std": {},
        "best_l2_strength": {},
        "mse_test": {},
        "normalized_mse_test": {},
        "labels_variance": {},
        "labels_variance_full": {},
        "labels_max": {},
        "labels_min": {},
        "labels_max_full": {},
        "labels_min_full": {},
        "mse_test_avg": {},
        "mse_test_std": {},
        "normalized_mse_test_avg": {},
        "normalized_mse_test_std": {},
        "normalization_bbox_mean": normalization_mean,
        "normalization_bbox_std": normalization_std,
        "sgd_inner_training_history": {},
        "sgd_final_training_history": {},
        "native_representation_dimension": projector.original_dimension,
        "padded_representation_dimension": projector.padded_dimension,
        "regression_input_scale": projector.regression_input_scale,
        "regression_inputs_centered": projector.native_center is not None,
        "sgd_calibration_sample_count": projector.calibration_sample_count,
        "sgd_curvature_by_size": projector.curvature_by_size,
        "sgd_base_weight_learning_rates": projector.base_weight_learning_rates,
        "sgd_base_bias_learning_rate": projector.base_bias_learning_rate,
        "ridge_reparameterization": (
            "x_scaled=(x-calibration_mean)/measured_centered_RMS_norm, "
            "lambda_scaled=lambda/measured_RMS_norm^2"
            if args.sgd_auto_learning_rate
            else "x_scaled=x/sqrt(native_D), lambda_scaled=lambda/native_D"
        ),
        "replaced_oversized_projection": projector.replaced_projection_size,
    }

    for feature_index, feature in enumerate(FEATURES):
        results["inner_mse_test_avg"][feature] = {}
        results["inner_mse_test_std"][feature] = {}
        results["inner_normalized_mse_test_avg"][feature] = {}
        results["inner_normalized_mse_test_std"][feature] = {}
        results["best_l2_strength"][feature] = {}
        results["mse_test"][feature] = {}
        results["normalized_mse_test"][feature] = {}
        results["labels_variance"][feature] = {}
        results["labels_max"][feature] = {}
        results["labels_min"][feature] = {}
        results["labels_variance_full"][feature] = float(full_variance[feature_index])
        results["labels_max_full"][feature] = float(
            global_label_stats["maximum"][0, feature_index]
        )
        results["labels_min_full"][feature] = float(
            global_label_stats["minimum"][0, feature_index]
        )

        for split in completed:
            results["inner_mse_test_avg"][feature][split] = {
                l2: float(metrics["inner_mse_mean"][split, l2_index, feature_index])
                for l2_index, l2 in enumerate(l2_strengths)
            }
            results["inner_mse_test_std"][feature][split] = {
                l2: float(metrics["inner_mse_std"][split, l2_index, feature_index])
                for l2_index, l2 in enumerate(l2_strengths)
            }
            results["inner_normalized_mse_test_avg"][feature][split] = {
                l2: float(metrics["inner_normalized_mean"][split, l2_index, feature_index])
                for l2_index, l2 in enumerate(l2_strengths)
            }
            results["inner_normalized_mse_test_std"][feature][split] = {
                l2: float(metrics["inner_normalized_std"][split, l2_index, feature_index])
                for l2_index, l2 in enumerate(l2_strengths)
            }
            results["best_l2_strength"][feature][split] = float(
                metrics["best_l2"][split, feature_index]
            )
            results["mse_test"][feature][split] = float(
                metrics["mse"][split, feature_index]
            )
            results["normalized_mse_test"][feature][split] = float(
                metrics["normalized_mse"][split, feature_index]
            )
            results["labels_variance"][feature][split] = float(
                metrics["label_variance"][split, feature_index]
            )
            results["labels_max"][feature][split] = float(
                metrics["label_maximum"][split, feature_index]
            )
            results["labels_min"][feature][split] = float(
                metrics["label_minimum"][split, feature_index]
            )

        mse_values = metrics["mse"][completed, feature_index]
        normalized_values = metrics["normalized_mse"][completed, feature_index]
        results["mse_test_avg"][feature] = float(np.mean(mse_values))
        results["mse_test_std"][feature] = float(np.std(mse_values))
        results["normalized_mse_test_avg"][feature] = float(
            np.mean(normalized_values)
        )
        results["normalized_mse_test_std"][feature] = float(
            np.std(normalized_values)
        )
        results["sgd_inner_training_history"][feature] = {
            split: [
                {
                    "learning_rate": epoch["learning_rate"],
                    "weight_learning_rate": epoch["weight_learning_rate"],
                    "bias_learning_rate": epoch["bias_learning_rate"],
                    "training_mse": epoch["training_mse"][:, :, feature_index],
                    "online_validation_mse": epoch[
                        "online_validation_mse"
                    ][:, :, feature_index],
                    "online_best_l2_index": int(
                        epoch["online_best_l2_indices"][feature_index]
                    ),
                    "online_best_l2": float(
                        epoch["online_best_l2"][feature_index]
                    ),
                    "training_objective": epoch["training_objective"][:, :, feature_index],
                }
                for epoch in metrics["inner_history"][split]
            ]
            for split in completed
        }
        results["sgd_final_training_history"][feature] = {
            split: [
                {
                    "learning_rate": epoch["learning_rate"],
                    "weight_learning_rate": epoch["weight_learning_rate"],
                    "bias_learning_rate": epoch["bias_learning_rate"],
                    "training_mse": float(epoch["training_mse"][feature_index]),
                    "training_objective": float(
                        epoch["training_objective"][feature_index]
                    ),
                }
                for epoch in metrics["final_history"][split]
            ]
            for split in completed
        }
    return results


def save_results(
    args,
    projection_sizes,
    metrics_by_size,
    l2_strengths,
    global_label_stats,
    normalization_mean,
    normalization_std,
    completed_outer_splits,
    projector,
    list_mode,
):
    dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))
    random_subsample_tag = (
        f"_RndSub_size{args.random_subsample_size}_seed{args.random_subsample_seed}"
        if args.random_subsample_size is not None
        else ""
    )
    manifold_tag = (
        f"_ManSub_size{args.manifold_subsample_size}_seed{args.manifold_subsample_seed}"
        if args.manifold_subsample_size is not None
        else ""
    )
    saved_paths = []
    for size in projection_sizes:
        results = build_results(
            args,
            size,
            metrics_by_size[size],
            l2_strengths,
            global_label_stats,
            normalization_mean,
            normalization_std,
            completed_outer_splits,
            projector,
        )
        directory = result_directory(args, size, list_mode)
        os.makedirs(directory, exist_ok=True)
        representation_tag = (
            f"FullRep_size{size}"
            if size == projector.unprojected_size
            else f"RndProj_size{size}_seed{args.random_projection_seed}"
        )
        filename = (
            f"{args.job_id}_CVregressionSGD_{dataset_name}_{args.layer}_"
            f"nOutSplit{args.n_cv_splits}_nInSplit{args.n_l2_cv_splits}"
            f"{random_subsample_tag}_{representation_tag}"
            f"{manifold_tag}.pkl"
        )
        path = os.path.join(directory, filename)
        with open(path, "wb") as file:
            pickle.dump(results, file)
        saved_paths.append(path)
    return saved_paths


def print_memory_estimate(projector, args, n_features):
    projection_sizes = projector.projection_sizes
    sum_d = sum(projection_sizes)
    inner_parameter_bytes = (
        sum_d
        * args.n_l2_cv_splits
        * args.n_l2
        * n_features
        * torch.tensor([], dtype=SGD_DTYPE).element_size()
    )
    largest_gradient_bytes = (
        max(projection_sizes)
        * args.n_l2_cv_splits
        * args.n_l2
        * n_features
        * torch.tensor([], dtype=SGD_DTYPE).element_size()
    )
    element_size = torch.tensor([], dtype=SGD_DTYPE).element_size()
    projected_batch_bytes = 0
    if projector.max_projection_size is not None:
        projected_batch_bytes = (
            args.batch_size * projector.max_projection_size * element_size
        )
    full_batch_bytes = 0
    if projector.unprojected_size is not None:
        full_batch_bytes = (
            args.batch_size * projector.unprojected_size * element_size
        )
    gib = 1024 ** 3
    print("Approximate SGD memory components:")
    print(f"  sum(D): {sum_d:,}")
    print(f"  inner readout parameters: {inner_parameter_bytes / gib:.3f} GiB")
    print(f"  largest per-D gradient: {largest_gradient_bytes / gib:.3f} GiB")
    print(f"  largest FJLT prefix batch: {projected_batch_bytes / gib:.3f} GiB")
    if projector.unprojected_size is not None:
        print(f"  unprojected full batch: {full_batch_bytes / gib:.3f} GiB")
    print(
        "  Additional memory is used by raw activations, the padded FJLT, "
        "and CUDA workspaces."
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    print_args(args)
    seed_everything(args.random_seed_splits)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    folder_name = dataset_folder_name(args.dataset_folder)
    print(f"Dataset: {folder_name}", flush=True)
    model, transform = load_model_and_transform(args, folder_name, device)
    categories = discover_categories(args)
    global_sdxl_folders = {
        "SDXL_dataset_test",
        "SDXL_dataset_train",
        "SDXL_dataset_validation",
        "DicarloAsGenai_public",
    }
    if folder_name in global_sdxl_folders:
        from LIB_dataloader import DatasetSDXL

        all_categories = {
            path.name
            for path in Path(args.dataset_folder).iterdir()
            if path.is_dir() and path.name.lower() != "crashes"
        }
        selected_categories = set(categories)
        excluded_categories = sorted(all_categories - selected_categories)
        dataset = DatasetSDXL(
            dataset_folder=args.dataset_folder,
            transform=transform,
            exclude_categories=excluded_categories,
        )
        data_source = GlobalSDXLDataSource(dataset, args)
        args.sgd_data_loading_mode = "global_shuffle"
        args.cv_assignment_mode = "fixed_category_balanced_original_blocks"
        print(
            "Using one globally shuffled SDXL DataLoader with permanent, "
            "category-balanced CV assignments.",
            flush=True,
        )
    else:
        CategoryDataset = build_category_dataset(args, folder_name, transform)
        data_source = CategoryDataSource(categories, CategoryDataset, args)
        args.sgd_data_loading_mode = "category_compatibility"
        args.cv_assignment_mode = "batch_position"
        print(
            "Using the category-wise compatibility data source for this "
            "non-SDXL dataset.",
            flush=True,
        )
    requested_projection_sizes = projection_sizes_from_args(args)
    list_mode = args.random_projection_sizes is not None
    l2_strengths = np.logspace(
        args.l2_min_exp,
        args.l2_max_exp,
        num=args.n_l2,
        base=10.0,
    ).tolist()
    print(f"Categories: {len(categories)}", flush=True)

    normalization_mean, normalization_std = data_source.normalization_stats()
    projector = MultiSizeFJLT(
        model,
        args.layer,
        requested_projection_sizes,
        args,
        device,
    )
    initialize_projector(
        projector,
        data_source,
        device,
    )
    projection_sizes = projector.projection_sizes
    print(f"Requested projection sizes: {requested_projection_sizes}", flush=True)
    print(
        f"FJLT projection sizes trained jointly: "
        f"{projector.random_projection_sizes}",
        flush=True,
    )
    if projector.unprojected_size is not None:
        print(
            f"First oversized request D={projector.replaced_projection_size} "
            f"replaced by unprojected native D={projector.unprojected_size}",
            flush=True,
        )
    else:
        print(
            "No requested D exceeds the native dimension; using the requested "
            "list unchanged and adding no unprojected endpoint.",
            flush=True,
        )
    print(f"All readout sizes trained jointly: {projection_sizes}", flush=True)
    (
        base_weight_learning_rates,
        base_bias_learning_rate,
    ) = calibrate_optimizer(
        data_source,
        projector,
        projection_sizes,
        args,
        device,
    )
    print(
        f"Regression inputs divided by scale="
        f"{projector.regression_input_scale:.6g}; ridge strengths are rescaled "
        "internally to preserve the original objective.",
        flush=True,
    )
    print(f"L2 strengths trained jointly: {l2_strengths}", flush=True)
    print_memory_estimate(projector, args, len(FEATURES))

    metrics_by_size = initialize_metrics(projection_sizes, args, len(FEATURES))
    global_label_stats = None
    saved_paths = []

    for outer_split in range(args.n_cv_splits):
        print(
            f"\n========== OUTER SPLIT {outer_split + 1}/{args.n_cv_splits} ==========",
            flush=True,
        )
        (
            inner_model,
            inner_histories,
            inner_train_label_stats,
            newly_collected_global_stats,
        ) = train_inner_models(
            data_source,
            projector,
            projection_sizes,
            l2_strengths,
            args,
            device,
            outer_split,
            collect_global_stats=(outer_split == 0),
            base_weight_learning_rates=base_weight_learning_rates,
            base_bias_learning_rate=base_bias_learning_rate,
        )
        if newly_collected_global_stats is not None:
            global_label_stats = newly_collected_global_stats

        inner_evaluation = evaluate_inner_models(
            inner_model,
            data_source,
            projector,
            projection_sizes,
            l2_strengths,
            args,
            device,
            outer_split,
            inner_train_label_stats,
        )
        final_model = initialize_final_from_inner(
            inner_model,
            inner_evaluation,
            projection_sizes,
            len(FEATURES),
            device,
        )

        for size in projection_sizes:
            metrics = metrics_by_size[size]
            evaluation = inner_evaluation[size]
            metrics["inner_mse_mean"][outer_split] = evaluation["mse_mean"]
            metrics["inner_mse_std"][outer_split] = evaluation["mse_std"]
            metrics["inner_normalized_mean"][outer_split] = evaluation[
                "normalized_mse_mean"
            ]
            metrics["inner_normalized_std"][outer_split] = evaluation[
                "normalized_mse_std"
            ]
            metrics["best_l2"][outer_split] = evaluation["best_l2"]
            metrics["inner_history"][outer_split] = inner_histories[size]
            clean_validation_mse = evaluation["mse_mean"][
                evaluation["best_l2_indices"],
                np.arange(len(FEATURES)),
            ]
            print(
                f"D={size}: selected L2 by feature: "
                f"{dict(zip(FEATURES, evaluation['best_l2']))}; "
                f"clean inner-validation MSE: "
                f"{dict(zip(FEATURES, clean_validation_mse))}",
                flush=True,
            )

        del inner_model, inner_histories
        if device.type == "cuda":
            torch.cuda.empty_cache()

        final_histories, outer_train_label_stats = train_final_models(
            final_model,
            inner_evaluation,
            data_source,
            projector,
            projection_sizes,
            args,
            device,
            outer_split,
            base_weight_learning_rates,
            base_bias_learning_rate,
        )
        outer_evaluation = evaluate_final_models(
            final_model,
            data_source,
            projector,
            projection_sizes,
            args,
            device,
            outer_split,
            outer_train_label_stats,
        )
        for size in projection_sizes:
            metrics = metrics_by_size[size]
            metrics["mse"][outer_split] = outer_evaluation[size]["mse"]
            metrics["normalized_mse"][outer_split] = outer_evaluation[size][
                "normalized_mse"
            ]
            metrics["label_variance"][outer_split] = outer_evaluation[size][
                "label_variance"
            ]
            metrics["label_minimum"][outer_split] = outer_evaluation[size][
                "test_minimum"
            ]
            metrics["label_maximum"][outer_split] = outer_evaluation[size][
                "test_maximum"
            ]
            metrics["final_history"][outer_split] = final_histories[size]
            print(
                f"D={size}: outer NMSE by feature: "
                f"{dict(zip(FEATURES, outer_evaluation[size]['normalized_mse']))}",
                flush=True,
            )

        del final_model, final_histories, inner_evaluation, outer_evaluation
        if device.type == "cuda":
            torch.cuda.empty_cache()

        saved_paths = save_results(
            args,
            projection_sizes,
            metrics_by_size,
            l2_strengths,
            global_label_stats,
            normalization_mean,
            normalization_std,
            completed_outer_splits=outer_split + 1,
            projector=projector,
            list_mode=list_mode,
        )
        print(f"Checkpointed results after outer split {outer_split + 1}.", flush=True)

    print("\n### FINAL SUMMARY ###")
    for size in projection_sizes:
        print(f"D={size}")
        for feature_index, feature in enumerate(FEATURES):
            values = metrics_by_size[size]["mse"][:, feature_index]
            full_count = float(global_label_stats["count"][0])
            full_mean = global_label_stats["sum"][0, feature_index] / full_count
            full_variance = (
                global_label_stats["sum_sq"][0, feature_index] / full_count
                - full_mean ** 2
            )
            print(
                f"  {feature}: MSE/full variance = "
                f"{np.mean(values) / full_variance:.6g} +/- "
                f"{np.std(values) / full_variance:.6g}"
            )
    print("Saved:")
    for path in saved_paths:
        print(f"  {path}")
    data_source.close()
    print("SGD cross-validated regression concluded.", flush=True)


if __name__ == "__main__":
    main()

