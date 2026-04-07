import os
import sys
import time
import pickle
import argparse
import traceback
import glob
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from brainscore_vision import load_benchmark
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


# ============================================================================
# USER-CONFIGURABLE SECTION
# Edit these paths / identifiers before running on the cluster.
# ============================================================================

# Add here every directory that must be on sys.path so that torch.load(...) can
# unpickle your full checkpoint objects. In particular, the directory containing
# LIB_model.py must be here.
EXTRA_SYS_PATHS = []


# Map each model_id to either:
#   - a .pth checkpoint path, or
#   - a directory path whose basename is "resnet50" for the special baseline.
#     Example: /.../some_run/resnet50/resnet50
MODEL_PATHS = {
    # examples:
    # "C": "path/checkpoint.pth",
    # "CR": "path/checkpoint.pth",
    # "R": "path/checkpoint.pth",
    # "CRloc": "path/checkpoint.pth",
    # "resnet50": "path/resnet50/resnet50",
}

# Candidate layers to score. These are the main ResNet block outputs plus the
# shared penultimate pooled representation.
CANDIDATE_LAYERS = [
    "backbone.layer1.0",
    "backbone.layer1.1",
    "backbone.layer1.2",
    "backbone.layer2.0",
    "backbone.layer2.1",
    "backbone.layer2.2",
    "backbone.layer2.3",
    "backbone.layer3.0",
    "backbone.layer3.1",
    "backbone.layer3.2",
    "backbone.layer3.3",
    "backbone.layer3.4",
    "backbone.layer3.5",
    "backbone.layer4.0",
    "backbone.layer4.1",
    "backbone.layer4.2",
    "backbone.avgpool",
]

# Public neural Brain-Score benchmarks.
BENCHMARKS = OrderedDict([
    ("V1", "FreemanZiemba2013public.V1-pls"),
    ("V2", "FreemanZiemba2013public.V2-pls"),
    ("V4", "MajajHong2015public.V4-pls"),
    ("IT", "MajajHong2015public.IT-pls"),
])

# Keep everything on CPU by default, which is the intended mode for this
# cluster script.
DEVICE = "cpu"

# Fallback transform if a checkpoint does not contain a saved transform.
DEFAULT_INPUT_RESOLUTION = (224, 224)
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]

# Brain-Score caches results based on the model identifier. Since we score every
# layer-region pair separately, the identifier must be unique for every pair.
# This script uses: <model_id>_<layer>_<region>

# Name of the pickle file written inside the chosen results folder.
RESULTS_FILENAME_TEMPLATE = "BrainScore_public_neural_{model_id}.pkl"


# ============================================================================
# Utility functions
# ============================================================================

def print_flush(msg: str) -> None:
    print(msg, flush=True)


def print_args(args) -> None:
    print_flush("=== SCRIPT ARGUMENTS ===")
    for name, value in sorted(vars(args).items()):
        print_flush(f"{name:20s}: {value!r}")
    print_flush("========================")


def append_sys_paths(paths):
    for p in paths:
        if p and p not in sys.path:
            sys.path.append(p)


def default_transform():
    return transforms.Compose([
        transforms.Resize(DEFAULT_INPUT_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_NORMALIZE_MEAN, std=DEFAULT_NORMALIZE_STD),
    ])


def infer_input_resolution(model):
    if hasattr(model, "input_resolution"):
        value = getattr(model, "input_resolution")
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return tuple(value)
    return DEFAULT_INPUT_RESOLUTION


def to_python_scalar(x):
    if x is None:
        return None

    # xarray / numpy style
    if hasattr(x, "values"):
        arr = np.asarray(x.values)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr.tolist()

    try:
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return arr.tolist()
    except Exception:
        pass

    try:
        return float(x)
    except Exception:
        return repr(x)


def score_to_dict(score):
    out = {
        "ceiled_score": to_python_scalar(score),
        "raw": None,
        "error": None,
        "ceiling": None,
        "model_identifier": None,
        "benchmark_identifier": None,
    }

    attrs = getattr(score, "attrs", {})
    if attrs:
        out["raw"] = to_python_scalar(attrs.get("raw", None))
        out["error"] = to_python_scalar(attrs.get("error", None))
        out["ceiling"] = to_python_scalar(attrs.get("ceiling", None))
        out["model_identifier"] = attrs.get("model_identifier", None)
        out["benchmark_identifier"] = attrs.get("benchmark_identifier", None)

    return out


def atomic_pickle_dump(obj, filepath: str) -> None:
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, filepath)


def load_existing_results(results_file: str):
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    return None


def cleanup_brainscore_cache_for_pair(model_identifier: str, region: str):
    """
    Delete only the Brain-Score cache files that are specific to one
    model-layer-region pair, while keeping shared benchmark/stimulus caches.

    Parameters
    ----------
    model_identifier : str
        The exact Brain-Score model identifier used for the current pair,
        e.g. "CR_backbone.layer4.1_IT".
    region : str
        Region label used in the script, e.g. "V1", "V2", "V4", "IT".

    Returns
    -------
    dict
        Summary of removed files / missing patterns / failures.
    """
    cache_root = Path.home() / ".result_caching"

    target_dirs = {
        "activations_core": cache_root / "brainscore_vision.model_helpers.activations.core.ActivationsExtractorHelper._from_paths_stored",
        "activations_pca": cache_root / "brainscore_vision.model_helpers.activations.pca.LayerPCA._pcas",
        "layer_scores": cache_root / "brainscore_vision.model_helpers.brain_transformation.neural.LayerScores._call",
        "layer_selection": cache_root / "brainscore_vision.model_helpers.brain_transformation.neural.LayerSelection._call",
    }

    patterns = [
        str(target_dirs["activations_core"] / f"identifier={model_identifier}_activations*.pkl"),
        str(target_dirs["activations_pca"] / f"identifier={model_identifier}_activations-pca_*.pkl"),
        str(target_dirs["layer_scores"] / f"model_identifier={model_identifier},*.pkl"),
        str(target_dirs["layer_selection"] / f"model_identifier={model_identifier}-pca_*,selection_identifier={region}.pkl"),
    ]

    removed = []
    missing_patterns = []
    failed = []
    total_bytes_removed = 0

    print_flush(f"[CACHE CLEANUP] Starting cleanup for model_identifier={model_identifier} | region={region}")

    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            missing_patterns.append(pattern)
            print_flush(f"[CACHE CLEANUP] No matches for pattern: {pattern}")
            continue

        for filepath in matches:
            try:
                size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                os.remove(filepath)
                removed.append(filepath)
                total_bytes_removed += size_bytes
                print_flush(f"[CACHE CLEANUP] Removed: {filepath} ({size_bytes} bytes)")
            except Exception as exc:
                failed.append({
                    "path": filepath,
                    "error": repr(exc),
                })
                print_flush(f"[CACHE CLEANUP] FAILED to remove: {filepath} | {repr(exc)}")

    summary = {
        "model_identifier": model_identifier,
        "region": region,
        "removed": removed,
        "missing_patterns": missing_patterns,
        "failed": failed,
        "n_removed": len(removed),
        "total_bytes_removed": total_bytes_removed,
    }

    print_flush(
        f"[CACHE CLEANUP] Finished cleanup | removed {summary['n_removed']} files | "
        f"freed {summary['total_bytes_removed']} bytes"
    )

    return summary


def update_summary_fields(results: dict) -> None:
    best_layer_by_region = {}
    best_score_by_region = {}

    for region in results["benchmarks"].keys():
        best_layer = None
        best_score = None

        region_scores = results["scores"].get(region, {})
        for layer, entry in region_scores.items():
            if entry.get("status") != "success":
                continue
            score_val = entry.get("ceiled_score", None)
            if score_val is None:
                continue
            if (best_score is None) or (score_val > best_score):
                best_score = score_val
                best_layer = layer

        best_layer_by_region[region] = best_layer
        best_score_by_region[region] = best_score

    results["best_layer_by_region"] = best_layer_by_region
    results["best_score_by_region"] = best_score_by_region

    completed_best_scores = [
        best_score_by_region[region]
        for region in results["benchmarks"].keys()
        if best_score_by_region[region] is not None
    ]

    results["n_regions_with_success"] = len(completed_best_scores)
    if len(completed_best_scores) == len(results["benchmarks"]):
        results["neural_mean"] = float(np.mean(completed_best_scores))
    else:
        results["neural_mean"] = None

    if len(completed_best_scores) > 0:
        results["neural_mean_available_regions"] = float(np.mean(completed_best_scores))
    else:
        results["neural_mean_available_regions"] = None


def load_model_and_transform(selected_model_path: str, device: str):
    """
    Load either:
      - a saved checkpoint containing ckpt['model'] and possibly ckpt['transform'], or
      - the special baseline case where the path basename is 'resnet50'.

    Returns
    -------
    model : torch.nn.Module
    image_transform : torchvision transform
    load_info : dict with metadata about how the model was loaded
    """
    normalized_path = os.path.normpath(selected_model_path)
    basename = os.path.basename(normalized_path).lower()

    # Important for torch.load(...) of full model objects.
    import LIB_model  # noqa: F401

    if basename == "resnet50":
        print_flush("Detected special resnet50 baseline path. Instantiating pretrained ResNet-50 backbone.")
        model = LIB_model.ResNetBBoxModel(
            resnet_version="resnet50",
        ).to(device)
        model.eval()
        image_transform = default_transform()
        load_info = {
            "load_mode": "resnet50_baseline",
            "checkpoint_path": selected_model_path,
            "used_checkpoint_transform": False,
        }
        return model, image_transform, load_info

    print_flush(f"Loading checkpoint from: {selected_model_path}")
    ckpt = torch.load(selected_model_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].to(device)
    model.eval()

    image_transform = ckpt.get("transform", None)
    used_checkpoint_transform = image_transform is not None
    if image_transform is None:
        image_transform = default_transform()

    load_info = {
        "load_mode": "checkpoint",
        "checkpoint_path": selected_model_path,
        "used_checkpoint_transform": used_checkpoint_transform,
    }
    return model, image_transform, load_info


# ============================================================================
# Brain-Score wrapper helpers
# ============================================================================

class SingleLayerBrainScoreModel(nn.Module):
    """
    Wrapper that extracts one internal layer via model.forward_features(...).

    The output is exposed as a single module called 'readout', so Brain-Score
    sees a one-layer candidate model.
    """
    def __init__(self, model, layer_name: str):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.readout = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            feats = self.model.forward_features(
                x,
                module_name=self.layer_name,
                flatten=True,
            )
            feats = feats.to(torch.float32)
            out = self.readout(feats)
        return out


def make_preprocessing_function(image_transform):
    """
    Brain-Score's PytorchWrapper preprocessing callable. It receives a list of
    image file paths and must return a batched tensor.
    """
    def custom_preprocessing(image_filepaths):
        batch = []
        for image_path in image_filepaths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                x = image_transform(img)
            batch.append(x)
        batch = torch.stack(batch, dim=0)
        return batch

    return custom_preprocessing


def build_brain_model(base_model, image_transform, chosen_layer: str,
                      model_identifier: str, image_size: int):
    single_layer_model = SingleLayerBrainScoreModel(
        model=base_model,
        layer_name=chosen_layer,
    ).to(DEVICE)
    single_layer_model.eval()

    activations_model = PytorchWrapper(
        identifier=model_identifier + "_activations",
        model=single_layer_model,
        preprocessing=make_preprocessing_function(image_transform),
    )
    activations_model.image_size = image_size

    brain_model = ModelCommitment(
        identifier=model_identifier,
        activations_model=activations_model,
        layers=["readout"],
    )
    return brain_model


# ============================================================================
# Main script
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute public neural Brain-Score benchmarks for one selected model across many layers."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Key used to select the model path from MODEL_PATHS inside this script.",
    )
    parser.add_argument(
        "--results_folder_name",
        type=str,
        required=True,
        help="Name of the folder created inside the checkpoint directory to store Brain-Score results.",
    )
    return parser.parse_args()


def initialize_results_dict(args, selected_model_path: str, results_dir: str, results_file: str):
    return {
        "args": vars(args).copy(),
        "script_name": os.path.basename(__file__),
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "selected_model_path": selected_model_path,
        "results_dir": results_dir,
        "results_file": results_file,
        "device": DEVICE,
        "candidate_layers": list(CANDIDATE_LAYERS),
        "benchmarks": dict(BENCHMARKS),
        "scores": {region: {} for region in BENCHMARKS.keys()},
        "best_layer_by_region": {region: None for region in BENCHMARKS.keys()},
        "best_score_by_region": {region: None for region in BENCHMARKS.keys()},
        "neural_mean": None,
        "neural_mean_available_regions": None,
        "n_regions_with_success": 0,
        "load_info": None,
        "completed_pairs": [],
        "failed_pairs": [],
    }


def main():
    args = parse_args()
    print_args(args)

    if args.model_id not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model_id '{args.model_id}'. Available keys are: {sorted(MODEL_PATHS.keys())}"
        )

    append_sys_paths(EXTRA_SYS_PATHS)
    print_flush(f"Added {len(EXTRA_SYS_PATHS)} user-configured paths to sys.path.")

    # Import now so that torch.load(...) can resolve the model class.
    print_flush("Importing LIB_model so checkpoint unpickling can resolve the model class...")
    import LIB_model  # noqa: F401

    selected_model_path = os.path.abspath(MODEL_PATHS[args.model_id])
    print_flush(f"Selected model_id: {args.model_id}")
    print_flush(f"Resolved model path: {selected_model_path}")

    # Store results in a folder created inside the checkpoint directory.
    checkpoint_dir = os.path.dirname(os.path.normpath(selected_model_path))
    results_dir = os.path.join(checkpoint_dir, args.results_folder_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir,
        RESULTS_FILENAME_TEMPLATE.format(model_id=args.model_id),
    )

    print_flush(f"Checkpoint directory: {checkpoint_dir}")
    print_flush(f"Results directory:    {results_dir}")
    print_flush(f"Results file:         {results_file}")

    existing_results = load_existing_results(results_file)
    if existing_results is not None:
        print_flush("Existing results file detected. Will resume and skip already-computed pairs.")
        results = existing_results
        # Update stored argument values before resuming.
        results["args"] = vars(args).copy()
        results["selected_model_path"] = selected_model_path
        results["results_dir"] = results_dir
        results["results_file"] = results_file
    else:
        results = initialize_results_dict(args, selected_model_path, results_dir, results_file)
        atomic_pickle_dump(results, results_file)
        print_flush("Initialized new results dictionary and saved the first partial file.")

    print_flush(f"Loading model on device: {DEVICE}")
    base_model, image_transform, load_info = load_model_and_transform(selected_model_path, DEVICE)
    results["load_info"] = load_info

    input_resolution = infer_input_resolution(base_model)
    image_size = int(input_resolution[0])

    print_flush(f"Loaded model type: {type(base_model)}")
    print_flush(f"Load mode: {load_info['load_mode']}")
    print_flush(f"Using transform from checkpoint: {load_info['used_checkpoint_transform']}")
    print_flush(f"Inferred input resolution: {input_resolution}")

    # Save immediately after loading info is known.
    update_summary_fields(results)
    atomic_pickle_dump(results, results_file)

    # Loop over public neural regions.
    for region, benchmark_identifier in BENCHMARKS.items():
        print_flush("\n" + "=" * 80)
        print_flush(f"Starting region: {region}")
        print_flush(f"Loading benchmark: {benchmark_identifier}")
        benchmark = load_benchmark(benchmark_identifier)
        print_flush(f"Benchmark loaded: {benchmark_identifier}")

        # Loop over candidate layers.
        for layer in CANDIDATE_LAYERS:
            if layer in results["scores"].get(region, {}):
                entry = results["scores"][region][layer]
                if entry.get("status") in ["success", "failed"]:
                    print_flush(
                        f"Skipping already-computed pair | region={region} | layer={layer} | status={entry.get('status')}"
                    )
                    continue

            model_identifier = f"{args.model_id}_{layer}_{region}"
            print_flush("-" * 80)
            print_flush(f"Scoring pair:")
            print_flush(f"  region          : {region}")
            print_flush(f"  benchmark       : {benchmark_identifier}")
            print_flush(f"  layer           : {layer}")
            print_flush(f"  model_identifier: {model_identifier}")

            pair_start_time = time.time()

            try:
                print_flush("Building Brain-Score model wrapper...")
                brain_model = build_brain_model(
                    base_model=base_model,
                    image_transform=image_transform,
                    chosen_layer=layer,
                    model_identifier=model_identifier,
                    image_size=image_size,
                )

                print_flush("Calling benchmark(model)...")
                score = benchmark(brain_model)
                score_dict = score_to_dict(score)
                elapsed = time.time() - pair_start_time

                entry = {
                    "status": "success",
                    "region": region,
                    "benchmark_identifier": benchmark_identifier,
                    "layer": layer,
                    "model_identifier": model_identifier,
                    "ceiled_score": score_dict["ceiled_score"],
                    "raw": score_dict["raw"],
                    "error": score_dict["error"],
                    "ceiling": score_dict["ceiling"],
                    "elapsed_seconds": elapsed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results["scores"][region][layer] = entry
                if [region, layer] not in results["completed_pairs"]:
                    results["completed_pairs"].append([region, layer])

                print_flush(f"Pair completed successfully in {elapsed:.2f} s")
                print_flush(f"  ceiled score : {entry['ceiled_score']}")
                print_flush(f"  raw          : {entry['raw']}")
                print_flush(f"  error        : {entry['error']}")
                print_flush(f"  ceiling      : {entry['ceiling']}")

            except Exception as exc:
                elapsed = time.time() - pair_start_time
                tb = traceback.format_exc()
                entry = {
                    "status": "failed",
                    "region": region,
                    "benchmark_identifier": benchmark_identifier,
                    "layer": layer,
                    "model_identifier": model_identifier,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": tb,
                    "elapsed_seconds": elapsed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results["scores"][region][layer] = entry
                if [region, layer] not in results["failed_pairs"]:
                    results["failed_pairs"].append([region, layer])

                print_flush(f"Pair FAILED after {elapsed:.2f} s")
                print_flush(f"  exception type   : {type(exc).__name__}")
                print_flush(f"  exception message: {str(exc)}")
                print_flush("  traceback follows:")
                print_flush(tb)

            # After every pair, recompute best layers and save partial results.
            update_summary_fields(results)
            atomic_pickle_dump(results, results_file)
            print_flush("Partial results saved before Brain-Score cache cleanup.")

            # Clean up the heavy Brain-Score cache files that are specific to the
            # current model-layer-region pair, now that our own partial results
            # have already been safely written to disk.
            cleanup_info = cleanup_brainscore_cache_for_pair(
                model_identifier=model_identifier,
                region=region,
            )
            results["scores"][region][layer]["cache_cleanup"] = cleanup_info

            # Save again so the cleanup report itself is also recorded.
            update_summary_fields(results)
            atomic_pickle_dump(results, results_file)
            print_flush("Partial results saved after Brain-Score cache cleanup.")
            print_flush(f"Current best layer for {region}: {results['best_layer_by_region'][region]}")
            print_flush(f"Current best score for {region}: {results['best_score_by_region'][region]}")
            print_flush(f"Current neural_mean_available_regions: {results['neural_mean_available_regions']}")
            print_flush(f"Current neural_mean (all 4 only): {results['neural_mean']}")

    results["timestamp_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    update_summary_fields(results)
    atomic_pickle_dump(results, results_file)

    print_flush("\n" + "=" * 80)
    print_flush("Brain-Score run completed.")
    print_flush(f"Best layer by region: {results['best_layer_by_region']}")
    print_flush(f"Best score by region: {results['best_score_by_region']}")
    print_flush(f"Neural mean available regions: {results['neural_mean_available_regions']}")
    print_flush(f"Neural mean (requires all 4): {results['neural_mean']}")
    print_flush(f"Final results file: {results_file}")


if __name__ == "__main__":
    main()
