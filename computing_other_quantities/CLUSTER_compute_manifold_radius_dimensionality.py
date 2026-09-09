#!/usr/bin/env python3

"""
Compute Cohen/Chung mean-field anchor radius and dimensionality for SDXL
category manifolds.

Each category is processed independently and written to its own HDF5 file.
Within each category file, compatible runs are grouped by a metadata hash. New
T seeds append anchor/T samples to the existing group; repeated T seeds are
skipped.

The expensive representation global mean is cached in the results directory.
If the matching global mean file is missing, the first process that acquires
the lock computes it by accumulating representations over the full category
list; other processes wait for the file to appear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.optimize import nnls
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = SCRIPT_DIR.parent
MODEL_DIR = REPOSITORY_DIR / "CNN_trainer"
sys.path.insert(0, str(REPOSITORY_DIR))
sys.path.insert(0, str(MODEL_DIR))

try:
    from hadamard_transform import (  # noqa: E402
        hadamard_transform,
        is_a_power_of_2,
        next_power_of_2,
        pad_to_power_of_2,
    )
except ModuleNotFoundError:
    def is_a_power_of_2(n: int) -> bool:
        n = int(n)
        return n > 0 and (n & (n - 1)) == 0

    def next_power_of_2(n: int) -> int:
        n = int(n)
        if n < 1:
            return 1
        return 1 << (n - 1).bit_length()

    def pad_to_power_of_2(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        n_power2 = n if is_a_power_of_2(n) else next_power_of_2(n)
        if n_power2 == n:
            return x
        pad = torch.zeros(*x.shape[:-1], n_power2 - n, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
        """Orthonormal Walsh-Hadamard transform along the last dimension."""
        n = x.shape[-1]
        if not is_a_power_of_2(n):
            raise ValueError(f"Hadamard input length must be a power of 2, got {n}.")
        batch_shape = x.shape[:-1]
        y = x
        h = 1
        while h < n:
            y = y.reshape(*batch_shape, -1, 2 * h)
            left = y[..., :h]
            right = y[..., h:]
            y = torch.cat([left + right, left - right], dim=-1)
            h *= 2
        return y.reshape(*batch_shape, n) / math.sqrt(n)

import LIB_model  # noqa: E402
from LIB_dataloader import CategoryDatasetSDXL  # noqa: E402


def optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    out = int(value)
    if out < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer or None.")
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str, default="0000")
parser.add_argument("--checkpoint", type=str, default="./checkpoint.pth")
parser.add_argument("--dataset_folder", type=str, required=True)
parser.add_argument("--layer", type=str, default="backbone")
parser.add_argument("--results_folder_name", type=str, default="./manifold_capacity_results")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--augment_dihedral", action="store_true")

parser.add_argument("--n_category_splits", type=int, default=1)
parser.add_argument("--category_split_index", type=int, default=1,
                    help="1-based category split index, matching Slurm array conventions.")

parser.add_argument("--n_images_per_category", type=optional_int, default=None)
parser.add_argument("--image_sampling_seed", type=int, default=1)

parser.add_argument("--random_projection_size", type=int, default=None)
parser.add_argument("--random_projection_seed", type=int, default=1)
parser.add_argument("--random_subsample_size", type=int, default=None)
parser.add_argument("--random_subsample_seed", type=int, default=1)

parser.add_argument("--kappa", type=float, default=0.0)
parser.add_argument("--t_seed", type=int, default=1)
parser.add_argument("--n_t_samples", type=int, default=1000)
parser.add_argument("--t_batch_size", type=int, default=128)

parser.add_argument("--svd_rank_rtol", type=float, default=1e-12)
parser.add_argument("--nnls_maxiter_factor", type=int, default=10)
parser.add_argument("--active_alpha_rtol", type=float, default=1e-10)
parser.add_argument("--feasibility_tol", type=float, default=1e-7)

parser.add_argument("--storage_dtype", type=str, choices=["float32", "float64"], default="float32")
parser.add_argument("--global_mean_wait_seconds", type=float, default=3 * 24 * 3600)
parser.add_argument("--global_mean_poll_seconds", type=float, default=30.0)
parser.add_argument("--estimate_chunk_rows", type=int, default=2048)
parser.add_argument("--representation_progress_every_batches", type=int, default=10,
                    help="Print representation extraction timing every this many batches.")

args = parser.parse_args()


def print_args() -> None:
    print("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print(f"{name:28s}: {value!r}")
    print("=========================", flush=True)


def canonical_json(payload: Dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_array_int(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.int64)
    return hashlib.sha1(values.tobytes()).hexdigest()


def decode_h5_strings(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not clean:
        clean = "category"
    return f"{clean}__{sha1_text(name)[:10]}.h5"


def stable_category_seed(base_seed: int, category: str) -> int:
    cat_int = int(sha1_text(category)[:8], 16)
    return (int(base_seed) + cat_int) % (2 ** 32)


def get_categories_from_labels(dataset_folder: str) -> List[str]:
    labels_path = Path(dataset_folder) / "labels.h5"
    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels.h5 at {labels_path}")
    with h5py.File(labels_path, "r") as h5:
        image_keys = decode_h5_strings(h5["image_keys"])
    categories = sorted({key.split("/")[0] for key in image_keys})
    return categories


def resolve_results_dir() -> Path:
    requested = Path(args.results_folder_name).expanduser()
    if requested.is_absolute():
        return requested
    checkpoint_dir = Path(args.checkpoint).expanduser().parent
    if str(checkpoint_dir) in {"", "."}:
        checkpoint_dir = Path.cwd()
    return (checkpoint_dir / requested).resolve()


def make_dataset_for_category(category: str, transform) -> CategoryDatasetSDXL:
    return CategoryDatasetSDXL(
        dataset_folder=args.dataset_folder,
        category=category,
        transform=transform,
        augment_dihedral=args.augment_dihedral,
    )


def selected_local_indices(category_dataset: CategoryDatasetSDXL, category: str) -> np.ndarray:
    n_items = len(category_dataset)
    if n_items < 1:
        raise RuntimeError(f"Category '{category}' has no images.")
    if args.n_images_per_category is None or args.n_images_per_category >= n_items:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(stable_category_seed(args.image_sampling_seed, category))
    idx = rng.choice(n_items, size=int(args.n_images_per_category), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def selected_image_keys(category_dataset: CategoryDatasetSDXL, local_indices: np.ndarray) -> List[str]:
    keys = []
    for local_idx in local_indices.tolist():
        if getattr(category_dataset, "augmentation_factor", 1) == 1:
            base_local_idx = int(local_idx)
        else:
            base_local_idx = int(local_idx) // int(category_dataset.augmentation_factor)
        real_idx = int(category_dataset.indices[base_local_idx])
        keys.append(str(category_dataset.image_keys[real_idx]))
    return keys


def split_categories(categories: List[str]) -> List[str]:
    if args.n_category_splits < 1:
        raise ValueError("--n_category_splits must be >= 1.")
    if not (1 <= args.category_split_index <= args.n_category_splits):
        raise ValueError(
            f"--category_split_index must be in 1..{args.n_category_splits}; "
            f"got {args.category_split_index}."
        )
    splits = np.array_split(np.asarray(categories, dtype=object), args.n_category_splits)
    return [str(x) for x in splits[args.category_split_index - 1].tolist()]


def load_model_and_transform(device: torch.device):
    if Path(args.checkpoint).name.lower() == "resnet50":
        model = LIB_model.ResNetBBoxModel(resnet_version="resnet50").to(device)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return model, transform

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ckpt["model"].to(device)
    model.eval()
    transform = ckpt["transform"]
    return model, transform


class ReductionState:
    def __init__(self):
        self.subsample_idx = None
        self.subsample_input_dim = None
        self.fjlt_D = None
        self.fjlt_idx = None
        self.fjlt_input_dim = None
        self.fjlt_padded_dim = None


def seed_torch(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def apply_reductions(reps: torch.Tensor, state: ReductionState, device: torch.device) -> torch.Tensor:
    reps = reps.to(dtype=torch.float64)

    if args.random_subsample_size is not None:
        if state.subsample_idx is None:
            seed_torch(args.random_subsample_seed)
            n_orig = reps.size(1)
            if args.random_subsample_size > n_orig:
                raise ValueError(
                    f"--random_subsample_size={args.random_subsample_size} exceeds "
                    f"representation dimension {n_orig}."
                )
            state.subsample_input_dim = int(n_orig)
            state.subsample_idx = torch.randperm(n_orig, device=device)[:args.random_subsample_size]
        reps = reps[:, state.subsample_idx]

    if args.random_projection_size is not None:
        if state.fjlt_D is None or state.fjlt_idx is None:
            seed_torch(args.random_projection_seed)
            n_orig = reps.size(1)
            n_power2 = n_orig if is_a_power_of_2(n_orig) else next_power_of_2(n_orig)
            if args.random_projection_size > n_power2:
                raise ValueError(
                    f"--random_projection_size={args.random_projection_size} exceeds "
                    f"padded representation dimension {n_power2}."
                )
            state.fjlt_input_dim = int(n_orig)
            state.fjlt_padded_dim = int(n_power2)
            state.fjlt_D = (torch.randint(0, 2, (n_power2,), device=device) * 2 - 1).to(torch.float64)
            state.fjlt_idx = torch.randperm(n_power2, device=device)[:args.random_projection_size]

        reps = pad_to_power_of_2(reps)
        reps_d = reps * state.fjlt_D[None, :]
        reps_h = hadamard_transform(reps_d)
        reps = reps_h[:, state.fjlt_idx] * math.sqrt(state.fjlt_padded_dim / args.random_projection_size)

    return reps


def iter_representation_batches(
    category: str,
    local_indices: np.ndarray,
    model,
    transform,
    state: ReductionState,
    device: torch.device,
    progress_prefix: str,
) -> Iterable[np.ndarray]:
    category_dataset = make_dataset_for_category(category, transform)
    subset = Subset(category_dataset, local_indices.tolist())
    effective_workers = min(max(0, args.num_workers), max(1, len(subset)))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=(device.type == "cuda"),
    )

    try:
        loader_iter = iter(loader)
        n_batches = len(loader)
        n_done = 0
        batch_idx = 0
        total_load_time = 0.0
        total_transfer_time = 0.0
        total_forward_reduction_time = 0.0
        total_cpu_copy_time = 0.0
        t_total0 = time.time()

        while True:
            t_load0 = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            load_time = time.time() - t_load0
            total_load_time += load_time
            batch_idx += 1

            images, _category_idx, _target, _img_relative_paths, _cat_strs = batch
            batch_size = int(images.shape[0])

            t_transfer0 = time.time()
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_time = time.time() - t_transfer0
            total_transfer_time += transfer_time

            t_forward0 = time.time()
            if args.layer.lower() == "pixels":
                reps = images.flatten(1)
            else:
                with torch.no_grad():
                    reps = model.forward_features(images, module_name=args.layer)
            reps = apply_reductions(reps, state, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_reduction_time = time.time() - t_forward0
            total_forward_reduction_time += forward_reduction_time

            t_cpu0 = time.time()
            reps_np = reps.detach().cpu().numpy().astype(np.float64, copy=False)
            cpu_copy_time = time.time() - t_cpu0
            total_cpu_copy_time += cpu_copy_time

            n_done += batch_size
            if (
                batch_idx % max(1, args.representation_progress_every_batches) == 0
                or batch_idx == n_batches
            ):
                elapsed = time.time() - t_total0
                print(
                    f"[repr timing] {progress_prefix} batch={batch_idx}/{n_batches} "
                    f"images={n_done}/{len(subset)} "
                    f"last_load={load_time:.3f}s "
                    f"last_transfer={transfer_time:.3f}s "
                    f"last_forward_reduce={forward_reduction_time:.3f}s "
                    f"last_cpu_copy={cpu_copy_time:.3f}s "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )

            yield reps_np

        elapsed = time.time() - t_total0
        print(
            f"[repr timing] {progress_prefix} completed "
            f"images={n_done} batches={batch_idx} "
            f"total_load={total_load_time:.2f}s "
            f"total_transfer={total_transfer_time:.2f}s "
            f"total_forward_reduce={total_forward_reduction_time:.2f}s "
            f"total_cpu_copy={total_cpu_copy_time:.2f}s "
            f"total_elapsed={elapsed:.2f}s",
            flush=True,
        )
    finally:
        try:
            category_dataset.h5_file.close()
        except Exception:
            pass


def extract_category_representations(
    category: str,
    local_indices: np.ndarray,
    model,
    transform,
    state: ReductionState,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    n_seen = 0
    t0 = time.time()
    for reps in iter_representation_batches(
        category,
        local_indices,
        model,
        transform,
        state,
        device,
        progress_prefix=f"category={category}",
    ):
        chunks.append(reps)
        n_seen += reps.shape[0]
    if not chunks:
        raise RuntimeError(f"No representations extracted for category '{category}'.")
    X = np.concatenate(chunks, axis=0)
    if X.shape[0] != len(local_indices):
        raise RuntimeError(
            f"Expected {len(local_indices)} representations for '{category}', got {X.shape[0]}."
        )
    elapsed = time.time() - t0
    print(f"[repr] category={category} X.shape={X.shape} elapsed={elapsed:.2f}s", flush=True)
    return X


def global_mean_payload(categories: List[str]) -> Dict:
    return {
        "analysis": "cohen_chung_global_mean",
        "script_version": 1,
        "dataset_folder": str(Path(args.dataset_folder).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "layer": args.layer,
        "augment_dihedral": bool(args.augment_dihedral),
        "n_images_per_category": args.n_images_per_category,
        "image_sampling_seed": int(args.image_sampling_seed),
        "random_subsample_size": args.random_subsample_size,
        "random_subsample_seed": int(args.random_subsample_seed),
        "random_projection_size": args.random_projection_size,
        "random_projection_seed": int(args.random_projection_seed),
        "categories_sha1": sha1_text("\n".join(categories)),
        "n_categories": len(categories),
    }


def acquire_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(f"host={socket.gethostname()}\npid={os.getpid()}\ntime={time.time()}\n")
    return True


def load_global_mean(global_mean_path: Path, expected_params_json: str) -> np.ndarray:
    with h5py.File(global_mean_path, "r") as h5:
        stored = h5.attrs.get("params_json", None)
        if stored != expected_params_json:
            raise RuntimeError(
                f"Global mean metadata mismatch in {global_mean_path}. "
                "This should not happen because the filename is metadata-hashed."
            )
        mean = h5["mean"][:].astype(np.float64, copy=False)
    print(f"[global mean] loaded {global_mean_path} with shape {mean.shape}", flush=True)
    return mean


def compute_global_mean(
    categories: List[str],
    model,
    transform,
    device: torch.device,
) -> Tuple[np.ndarray, int]:
    state = ReductionState()
    sum_vec = None
    n_total = 0
    for i, category in enumerate(categories, start=1):
        category_dataset = make_dataset_for_category(category, transform)
        try:
            local_indices = selected_local_indices(category_dataset, category)
        finally:
            try:
                category_dataset.h5_file.close()
            except Exception:
                pass
        print(
            f"[global mean] {i}/{len(categories)} category={category} "
            f"n_images={len(local_indices)}",
            flush=True,
        )
        category_t0 = time.time()
        for reps in iter_representation_batches(
            category,
            local_indices,
            model,
            transform,
            state,
            device,
            progress_prefix=f"global_mean category={category}",
        ):
            if sum_vec is None:
                sum_vec = np.zeros(reps.shape[1], dtype=np.float64)
            elif sum_vec.shape[0] != reps.shape[1]:
                raise RuntimeError("Representation dimension changed while computing global mean.")
            sum_vec += np.sum(reps, axis=0, dtype=np.float64)
            n_total += reps.shape[0]
        print(
            f"[global mean] finished category={category} "
            f"elapsed={time.time() - category_t0:.2f}s cumulative_images={n_total}",
            flush=True,
        )
    if sum_vec is None or n_total == 0:
        raise RuntimeError("No samples accumulated for global mean.")
    return sum_vec / float(n_total), n_total


def load_or_compute_global_mean(
    results_dir: Path,
    categories: List[str],
    model,
    transform,
    device: torch.device,
) -> Tuple[np.ndarray, Path, str]:
    payload = global_mean_payload(categories)
    params_json = canonical_json(payload)
    global_hash = sha1_text(params_json)
    global_mean_path = results_dir / f"global_mean_{global_hash}.h5"
    lock_path = results_dir / f"global_mean_{global_hash}.lock"

    if global_mean_path.exists():
        return load_global_mean(global_mean_path, params_json), global_mean_path, global_hash

    if acquire_lock(lock_path):
        print(f"[global mean] acquired lock {lock_path}", flush=True)
        try:
            mean, n_total = compute_global_mean(categories, model, transform, device)
            tmp_path = results_dir / f".global_mean_{global_hash}.{os.getpid()}.tmp"
            with h5py.File(tmp_path, "w") as h5:
                h5.attrs["params_json"] = params_json
                h5.attrs["hash"] = global_hash
                h5.attrs["n_total"] = int(n_total)
                h5.attrs["created_by_job_id"] = str(args.job_id)
                h5.create_dataset("mean", data=mean.astype(np.float64), dtype=np.float64)
            os.replace(tmp_path, global_mean_path)
            print(
                f"[global mean] wrote {global_mean_path} "
                f"shape={mean.shape} n_total={n_total}",
                flush=True,
            )
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        return load_global_mean(global_mean_path, params_json), global_mean_path, global_hash

    print(f"[global mean] waiting for existing lock {lock_path}", flush=True)
    t0 = time.time()
    while not global_mean_path.exists():
        elapsed = time.time() - t0
        if elapsed > args.global_mean_wait_seconds:
            raise TimeoutError(
                f"Timed out after {elapsed:.1f}s waiting for {global_mean_path}. "
                f"Lock file is {lock_path}."
            )
        time.sleep(args.global_mean_poll_seconds)
        print(f"[global mean] still waiting ({elapsed:.1f}s elapsed)...", flush=True)
    return load_global_mean(global_mean_path, params_json), global_mean_path, global_hash


def build_case_payload(
    category: str,
    selected_keys: List[str],
    global_mean_path: Path,
    global_mean_hash: str,
) -> Dict:
    return {
        "analysis": "cohen_chung_anchor_radius_dimension",
        "script_version": 1,
        "category": category,
        "dataset_folder": str(Path(args.dataset_folder).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "layer": args.layer,
        "augment_dihedral": bool(args.augment_dihedral),
        "n_images_per_category": args.n_images_per_category,
        "image_sampling_seed": int(args.image_sampling_seed),
        "selected_image_keys_sha1": sha1_text("\n".join(selected_keys)),
        "n_selected_images": len(selected_keys),
        "random_subsample_size": args.random_subsample_size,
        "random_subsample_seed": int(args.random_subsample_seed),
        "random_projection_size": args.random_projection_size,
        "random_projection_seed": int(args.random_projection_seed),
        "global_mean_path": str(global_mean_path),
        "global_mean_hash": global_mean_hash,
        "center_method": "global_mean_then_category_mean",
        "coordinate_method": "svd_empirical_variability_span",
        "svd_rank_rtol": float(args.svd_rank_rtol),
        "kappa": float(args.kappa),
    }


def build_cohen_coordinates(X: np.ndarray, global_mean: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError("X must have shape (n_images, n_features).")
    if X.shape[1] != global_mean.shape[0]:
        raise ValueError(
            f"Representation dim {X.shape[1]} does not match global mean dim {global_mean.shape[0]}."
        )

    X0 = X - global_mean[None, :]
    center = np.mean(X0, axis=0)
    center_norm = float(np.linalg.norm(center))
    if not np.isfinite(center_norm) or center_norm <= 0.0:
        raise RuntimeError("Category center norm is zero or non-finite.")

    Y = X0 - center[None, :]
    if X.shape[0] <= 1:
        singular_values = np.zeros((0,), dtype=np.float64)
        Z = np.zeros((X.shape[0], 0), dtype=np.float64)
        rank = 0
    else:
        U, singular_values, _Vt = np.linalg.svd(Y, full_matrices=False)
        if singular_values.size == 0:
            rank = 0
        else:
            tol = float(args.svd_rank_rtol) * float(singular_values[0])
            rank = int(np.count_nonzero(singular_values > tol))
        if rank == 0:
            Z = np.zeros((X.shape[0], 0), dtype=np.float64)
        else:
            Z = U[:, :rank] * singular_values[:rank][None, :]

    S = np.concatenate([Z / center_norm, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    variability_norms = np.linalg.norm(Y, axis=1)
    point_norms = np.linalg.norm(X0, axis=1)
    meta = {
        "representation_dim": int(X.shape[1]),
        "coordinate_rank": int(rank),
        "d_plus_one": int(S.shape[1]),
        "center_norm": center_norm,
        "mean_variability_norm": float(np.mean(variability_norms)),
        "rms_variability_norm": float(np.sqrt(np.mean(variability_norms ** 2))),
        "mean_global_centered_point_norm": float(np.mean(point_norms)),
        "rms_global_centered_point_norm": float(np.sqrt(np.mean(point_norms ** 2))),
        "max_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
        "min_retained_singular_value": float(singular_values[rank - 1]) if rank > 0 else 0.0,
    }
    return S, meta, center, singular_values


def category_file_path(results_dir: Path, category: str) -> Path:
    return results_dir / "categories" / sanitize_filename(category)


def ensure_case_group(
    category_path: Path,
    case_key: str,
    params_json: str,
    category: str,
    selected_keys: List[str],
    coordinate_meta: Dict[str, float],
    center: np.ndarray,
    singular_values: np.ndarray,
) -> h5py.Group:
    category_path.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(category_path, "a")
    if "file_format" not in h5.attrs:
        h5.attrs["file_format"] = "cohen_chung_category_anchor_samples_v1"
    if "category" not in h5.attrs:
        h5.attrs["category"] = category
    cases = h5.require_group("cases")

    if case_key in cases:
        grp = cases[case_key]
        if grp.attrs.get("params_json", None) != params_json:
            h5.close()
            raise RuntimeError(f"Metadata mismatch for existing case {case_key} in {category_path}.")
        expected_d = int(grp.attrs["d_plus_one"])
        if expected_d != int(coordinate_meta["d_plus_one"]):
            h5.close()
            raise RuntimeError(
                f"Existing d_plus_one={expected_d}, current d_plus_one={coordinate_meta['d_plus_one']}."
            )
        return grp

    grp = cases.create_group(case_key)
    grp.attrs["params_json"] = params_json
    grp.attrs["case_key"] = case_key
    grp.attrs["category"] = category
    grp.attrs["storage_dtype"] = args.storage_dtype
    grp.attrs["n_anchor_samples_total"] = 0
    for key, value in coordinate_meta.items():
        grp.attrs[key] = value

    d_plus_one = int(coordinate_meta["d_plus_one"])
    storage_dtype = np.float32 if args.storage_dtype == "float32" else np.float64
    chunk_rows = max(1, min(int(args.t_batch_size), 1024))

    grp.create_dataset("anchors", shape=(0, d_plus_one), maxshape=(None, d_plus_one),
                       dtype=storage_dtype, chunks=(chunk_rows, d_plus_one))
    grp.create_dataset("T", shape=(0, d_plus_one), maxshape=(None, d_plus_one),
                       dtype=storage_dtype, chunks=(chunk_rows, d_plus_one))

    for name in ("q_R_raw", "q_D_raw", "lambda", "max_violation"):
        grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64,
                           chunks=(max(1, min(int(args.n_t_samples), 100_000)),))
    grp.create_dataset("n_active", shape=(0,), maxshape=(None,), dtype=np.int32,
                       chunks=(max(1, min(int(args.n_t_samples), 100_000)),))
    grp.create_dataset("is_interior", shape=(0,), maxshape=(None,), dtype=np.uint8,
                       chunks=(max(1, min(int(args.n_t_samples), 100_000)),))

    for name in ("used_t_seeds", "run_n_samples", "run_start", "run_stop"):
        grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))

    str_dtype = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset("selected_image_keys", data=np.asarray(selected_keys, dtype=object), dtype=str_dtype)
    grp.create_dataset("category_center", data=center.astype(np.float64), dtype=np.float64)
    grp.create_dataset("singular_values", data=singular_values.astype(np.float64), dtype=np.float64)
    return grp


def close_group_file(grp: h5py.Group) -> None:
    h5 = grp.file
    h5.flush()
    h5.close()


def seed_already_used(category_path: Path, case_key: str, params_json: str, t_seed: int) -> bool:
    if not category_path.exists():
        return False
    with h5py.File(category_path, "r") as h5:
        cases = h5.get("cases", None)
        if cases is None or case_key not in cases:
            return False
        grp = cases[case_key]
        if grp.attrs.get("params_json", None) != params_json:
            raise RuntimeError(f"Metadata mismatch for existing case {case_key}.")
        used = grp["used_t_seeds"][:]
        return bool(np.any(used == int(t_seed)))


def append_1d(ds: h5py.Dataset, values: np.ndarray) -> Tuple[int, int]:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("append_1d expects a 1D array.")
    old = ds.shape[0]
    new = old + values.shape[0]
    ds.resize((new,))
    ds[old:new] = values
    return old, new


def append_2d(ds: h5py.Dataset, values: np.ndarray) -> Tuple[int, int]:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("append_2d expects a 2D array.")
    old = ds.shape[0]
    new = old + values.shape[0]
    ds.resize((new, ds.shape[1]))
    ds[old:new, :] = values
    return old, new


def solve_anchor_for_T(S: np.ndarray, A: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, float, float, int, float, float, bool]:
    d_var = S.shape[1] - 1
    margins = S @ T
    max_idx = int(np.argmax(margins))
    max_margin = float(margins[max_idx])

    if max_margin + args.kappa < 0.0:
        anchor = S[max_idx].copy()
        anchor_s = anchor[:d_var]
        norm_sq = float(anchor_s @ anchor_s)
        q_D = float((T[:d_var] @ anchor_s) ** 2 / norm_sq) if norm_sq > 0.0 else 0.0
        return anchor, norm_sq, q_D, 0, 0.0, max_margin + args.kappa, True

    target = T.copy()
    target[-1] += args.kappa
    maxiter = max(1, int(args.nnls_maxiter_factor * S.shape[0]))
    alpha, _residual = nnls(A, target, maxiter=maxiter)

    lambda_sum = float(np.sum(alpha))
    if not np.isfinite(lambda_sum) or lambda_sum <= 0.0:
        raise RuntimeError("NNLS returned non-positive or non-finite lambda.")

    active_tol = float(args.active_alpha_rtol) * max(1.0, lambda_sum)
    n_active = int(np.count_nonzero(alpha > active_tol))

    anchor = (A @ alpha) / lambda_sum
    anchor_s = anchor[:d_var]
    norm_sq = float(anchor_s @ anchor_s)
    q_D = float((T[:d_var] @ anchor_s) ** 2 / norm_sq) if norm_sq > 0.0 else 0.0

    V = T - A @ alpha
    max_violation = float(np.max(S @ V + args.kappa))
    return anchor, norm_sq, q_D, n_active, lambda_sum, max_violation, False


def sample_anchors(S: np.ndarray) -> Dict[str, np.ndarray]:
    if S.ndim != 2:
        raise ValueError("S must have shape (n_points, d_plus_one).")
    n_t = int(args.n_t_samples)
    d_plus_one = S.shape[1]
    rng = np.random.default_rng(args.t_seed)
    A = np.ascontiguousarray(S.T)

    anchors = np.empty((n_t, d_plus_one), dtype=np.float64)
    T_all = np.empty((n_t, d_plus_one), dtype=np.float64)
    q_R = np.empty(n_t, dtype=np.float64)
    q_D = np.empty(n_t, dtype=np.float64)
    n_active = np.empty(n_t, dtype=np.int32)
    lambda_arr = np.empty(n_t, dtype=np.float64)
    max_violation = np.empty(n_t, dtype=np.float64)
    is_interior = np.empty(n_t, dtype=np.uint8)

    done = 0
    n_bad_feasible = 0
    t0 = time.time()

    while done < n_t:
        bs = min(int(args.t_batch_size), n_t - done)
        t_batch0 = time.time()
        T_batch = rng.standard_normal((bs, d_plus_one))
        t_generate = time.time() - t_batch0
        t_solve0 = time.time()
        for j in range(bs):
            idx = done + j
            anchor, q_R[idx], q_D[idx], n_active[idx], lambda_arr[idx], max_violation[idx], interior = (
                solve_anchor_for_T(S=S, A=A, T=T_batch[j])
            )
            anchors[idx, :] = anchor
            T_all[idx, :] = T_batch[j]
            is_interior[idx] = 1 if interior else 0
            if (not interior) and max_violation[idx] > args.feasibility_tol:
                n_bad_feasible += 1
        solve_time = time.time() - t_solve0

        done += bs
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-12)
        print(
            f"[anchors] {done}/{n_t} T samples "
            f"({100.0 * done / n_t:6.2f}%), elapsed={elapsed:.1f}s, "
            f"last_generate_T={t_generate:.3f}s, "
            f"last_solve={solve_time:.3f}s, "
            f"rate={rate:.2f}/s",
            flush=True,
        )

    if n_bad_feasible:
        print(
            f"[warning] {n_bad_feasible} active anchor solves exceeded "
            f"feasibility_tol={args.feasibility_tol:g}.",
            flush=True,
        )

    return {
        "anchors": anchors,
        "T": T_all,
        "q_R_raw": q_R,
        "q_D_raw": q_D,
        "n_active": n_active,
        "lambda": lambda_arr,
        "max_violation": max_violation,
        "is_interior": is_interior,
    }


def append_samples_to_group(grp: h5py.Group, samples: Dict[str, np.ndarray]) -> None:
    old_total = int(grp["anchors"].shape[0])
    new_total = None
    for name, values in samples.items():
        if values.ndim == 2:
            _old, current_new = append_2d(grp[name], values)
        else:
            _old, current_new = append_1d(grp[name], values)
        if new_total is None:
            new_total = current_new
        elif current_new != new_total:
            raise RuntimeError("Append produced inconsistent dataset sizes.")

    append_1d(grp["used_t_seeds"], np.asarray([args.t_seed], dtype=np.int64))
    append_1d(grp["run_n_samples"], np.asarray([args.n_t_samples], dtype=np.int64))
    append_1d(grp["run_start"], np.asarray([old_total], dtype=np.int64))
    append_1d(grp["run_stop"], np.asarray([old_total + args.n_t_samples], dtype=np.int64))
    grp.attrs["n_anchor_samples_total"] = int(new_total)
    grp.file.flush()


def compute_estimates_from_group(grp: h5py.Group) -> Dict[str, float]:
    anchors_ds = grp["anchors"]
    T_ds = grp["T"]
    n = int(anchors_ds.shape[0])
    if n == 0:
        return {}

    d_var = int(anchors_ds.shape[1]) - 1
    chunk = max(1, int(args.estimate_chunk_rows))

    sum_anchor = np.zeros(d_var, dtype=np.float64)
    sum_q_R_raw = 0.0
    sum_q_D_raw = 0.0
    sum_n_active = 0.0
    sum_interior = 0.0

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        anchors = anchors_ds[start:stop, :d_var].astype(np.float64, copy=False)
        sum_anchor += np.sum(anchors, axis=0, dtype=np.float64)
        sum_q_R_raw += float(np.sum(grp["q_R_raw"][start:stop], dtype=np.float64))
        sum_q_D_raw += float(np.sum(grp["q_D_raw"][start:stop], dtype=np.float64))
        sum_n_active += float(np.sum(grp["n_active"][start:stop], dtype=np.float64))
        sum_interior += float(np.sum(grp["is_interior"][start:stop], dtype=np.float64))

    anchor_mean = sum_anchor / float(n)
    sum_q_R_centered = 0.0
    sum_q_D_centered = 0.0

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        anchors = anchors_ds[start:stop, :d_var].astype(np.float64, copy=False)
        T = T_ds[start:stop, :d_var].astype(np.float64, copy=False)
        centered = anchors - anchor_mean[None, :]
        norm_sq = np.sum(centered * centered, axis=1)
        dot = np.sum(T * centered, axis=1)
        q_D = np.zeros_like(norm_sq)
        valid = norm_sq > 0.0
        q_D[valid] = (dot[valid] * dot[valid]) / norm_sq[valid]
        sum_q_R_centered += float(np.sum(norm_sq, dtype=np.float64))
        sum_q_D_centered += float(np.sum(q_D, dtype=np.float64))

    estimates = {
        "n_anchor_samples_total": float(n),
        "radius_raw": math.sqrt(max(0.0, sum_q_R_raw / float(n))),
        "dimension_raw": sum_q_D_raw / float(n),
        "radius_centered": math.sqrt(max(0.0, sum_q_R_centered / float(n))),
        "dimension_centered": sum_q_D_centered / float(n),
        "anchor_mean_norm": float(np.linalg.norm(anchor_mean)),
        "mean_n_active": sum_n_active / float(n),
        "interior_fraction": sum_interior / float(n),
    }
    for key, value in estimates.items():
        grp.attrs[key] = value
    grp.file.flush()
    return estimates


def print_estimates(category: str, estimates: Dict[str, float]) -> None:
    if not estimates:
        print(f"[estimate] category={category}: no anchor samples stored yet.", flush=True)
        return
    print(
        f"[estimate] category={category} "
        f"n={int(estimates['n_anchor_samples_total'])} "
        f"R_raw={estimates['radius_raw']:.8g} "
        f"D_raw={estimates['dimension_raw']:.8g} "
        f"R_centered={estimates['radius_centered']:.8g} "
        f"D_centered={estimates['dimension_centered']:.8g} "
        f"anchor_mean_norm={estimates['anchor_mean_norm']:.8g} "
        f"mean_active={estimates['mean_n_active']:.8g} "
        f"interior_frac={estimates['interior_fraction']:.8g}",
        flush=True,
    )


def process_category(
    category: str,
    results_dir: Path,
    global_mean: np.ndarray,
    global_mean_path: Path,
    global_mean_hash: str,
    model,
    transform,
    device: torch.device,
) -> None:
    category_dataset = make_dataset_for_category(category, transform)
    try:
        local_indices = selected_local_indices(category_dataset, category)
        keys = selected_image_keys(category_dataset, local_indices)
    finally:
        try:
            category_dataset.h5_file.close()
        except Exception:
            pass
    payload = build_case_payload(category, keys, global_mean_path, global_mean_hash)
    params_json = canonical_json(payload)
    case_key = f"case_{sha1_text(params_json)}"
    category_path = category_file_path(results_dir, category)

    print("-" * 80, flush=True)
    print(
        f"[category] {category} | file={category_path} | "
        f"n_images={len(local_indices)} | case={case_key}",
        flush=True,
    )

    if seed_already_used(category_path, case_key, params_json, args.t_seed):
        print(
            f"[skip] t_seed={args.t_seed} already present for category={category}, case={case_key}.",
            flush=True,
        )
        with h5py.File(category_path, "a") as h5:
            estimates = compute_estimates_from_group(h5["cases"][case_key])
        print_estimates(category, estimates)
        return

    state = ReductionState()
    X = extract_category_representations(category, local_indices, model, transform, state, device)
    S, coordinate_meta, center, singular_values = build_cohen_coordinates(X, global_mean)
    print(
        f"[coords] category={category} D={S.shape[1] - 1} "
        f"center_norm={coordinate_meta['center_norm']:.8g} "
        f"rms_variability_norm={coordinate_meta['rms_variability_norm']:.8g}",
        flush=True,
    )

    grp = ensure_case_group(
        category_path=category_path,
        case_key=case_key,
        params_json=params_json,
        category=category,
        selected_keys=keys,
        coordinate_meta=coordinate_meta,
        center=center,
        singular_values=singular_values,
    )
    try:
        if np.any(grp["used_t_seeds"][:] == int(args.t_seed)):
            print(
                f"[skip] t_seed={args.t_seed} appeared while preparing category={category}.",
                flush=True,
            )
            estimates = compute_estimates_from_group(grp)
            print_estimates(category, estimates)
            return

        samples = sample_anchors(S)
        append_samples_to_group(grp, samples)
        estimates = compute_estimates_from_group(grp)
        print_estimates(category, estimates)
    finally:
        close_group_file(grp)


def main() -> None:
    print_args()
    if args.n_t_samples < 1:
        raise ValueError("--n_t_samples must be >= 1.")
    if args.t_batch_size < 1:
        raise ValueError("--t_batch_size must be >= 1.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[device] requested={args.device} using={device}", flush=True)

    results_dir = resolve_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "categories").mkdir(parents=True, exist_ok=True)
    print(f"[results] {results_dir}", flush=True)

    all_categories = get_categories_from_labels(args.dataset_folder)
    selected_categories = split_categories(all_categories)
    print(
        f"[categories] total={len(all_categories)} split={args.category_split_index}/"
        f"{args.n_category_splits} selected={len(selected_categories)}",
        flush=True,
    )
    print(f"[categories] selected list: {selected_categories}", flush=True)

    model, transform = load_model_and_transform(device)
    global_mean, global_mean_path, global_mean_hash = load_or_compute_global_mean(
        results_dir=results_dir,
        categories=all_categories,
        model=model,
        transform=transform,
        device=device,
    )

    t0 = time.time()
    for category in selected_categories:
        process_category(
            category=category,
            results_dir=results_dir,
            global_mean=global_mean,
            global_mean_path=global_mean_path,
            global_mean_hash=global_mean_hash,
            model=model,
            transform=transform,
            device=device,
        )

    elapsed = time.time() - t0
    print(f"[done] processed {len(selected_categories)} categories in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[warning] interrupted by user.", flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", flush=True)
        raise

