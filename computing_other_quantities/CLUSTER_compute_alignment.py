import os
import argparse
import pickle
from einops import einsum
import torch
import numpy as np
import random
import h5py
from networkx.algorithms.centrality import katz_centrality_numpy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
from hadamard_transform import hadamard_transform, next_power_of_2, pad_to_power_of_2, is_a_power_of_2

# IMPORT DATALOADERS
# add path, import is done later
sys.path.append('../../codebase_v5')
# IMPORT MODELS
# -> AvgPool_SDXLdataset
sys.path.append('../ANN_models/AvgPool_SDXLdataset')
import LIB_model

# TODO: global dim reduction?

parser = argparse.ArgumentParser(description="Extract features and predictions for dataset images.")
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
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing images.")
parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
parser.add_argument("--local_dimensionality_reduction_style", type=str,
                    default="",
                    help="string identifier describing the dimensionality reduction type. Options are percentage or PR. "
                         "Default is no dimensionality reduction")
parser.add_argument("--local_dimensionality_reduction_strength", type=float, default=100,
                    help="Percentage of total covariance to explain (e.g., 90 for 90%). Only applicable to style percentage")

args = parser.parse_args()


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

# NEW - DATASET SELECTION. In other scripts, replace the old line determining dataset_folder_name with the block below
from pathlib import Path
dataset_path = Path(args.dataset_folder)
# If the user gives an .h5 file, use its parent folder name.
# Otherwise (directory case), use the directory name itself.
if dataset_path.is_file() or dataset_path.suffix.lower() == ".h5":
    dataset_folder_name = dataset_path.parent.name
else:
    dataset_folder_name = dataset_path.name
# END NEW

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
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # load on CPU
    model = ckpt["model"].to(device)
    model.eval()  # put into validation mode
    transform = ckpt["transform"]

# BEGIN NEW - DATASET SELECTION. In new scripts, replace the old dataset selection block with the block below
from functools import partial
import os

print(f"Dataset: {dataset_folder_name}", flush=True)

if dataset_folder_name in ["SDXL_dataset_test", "SDXL_dataset_train", "SDXL_dataset_validation",
                           "DicarloAsGenai_public"]:
    from LIB_dataloader import CategoryDatasetSDXL
    CategoryDataset = partial(
        CategoryDatasetSDXL,
        dataset_folder=args.dataset_folder,
        transform=transform,
    )
    features = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]

elif dataset_folder_name in ["representations_DicarloAsGenai_public_IT", "representations_DicarloAsGenai_public_V4"]:
    from LIB_dataloader import CategoryDatasetRepresentation
    CategoryDataset = partial(
        CategoryDatasetRepresentation,
        dataset_folder=args.dataset_folder,
        transform=transform,
    )
    features = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]

else:
    print(f"ERROR: dataset {dataset_folder_name} is not in the implemented public datasets", flush=True)
    exit(1)

# END NEW

# NEW - RenderedShapenet dataset addition. Replace with this below:
# retrieve categories (sorted for determinism)
all_categories = [
    p.name
    for p in (dataset_path if dataset_path.is_dir() else dataset_path.parent).iterdir()
    if p.is_dir() and p.name.lower() != "crashes"
]
# Remove integration variants (only keep base category names)
_INTEGRATION_SUFFIXES = ("_integration", "_integration_ref")
all_categories = [
    c for c in all_categories
    if not any(c.endswith(sfx) for sfx in _INTEGRATION_SUFFIXES)
]
all_categories.sort()  # deterministic order
# END NEW

categories = all_categories
n_categories = len(categories)

# define results dictionary.
results = {"covariance": {category: None for category in categories},
           "participation_ratio": {category: None for category in categories},
           "sqrt_trace_2": {category: None for category in categories},
           }

# define running quantities that are not defined in the global dictionary
fjlt_idx = None
fjlt_D = None

# loop over the categories: generate representations and do regression
for category in categories:
    print(f"Processing category: {category}", flush=True)

    # BEGIN NEW - DATASET SELECTION. In new scripts, replace the old dataset definition line with this line below
    dataset = CategoryDataset(category=category)  # <— only category here
    # END NEW

    # -> define the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # define running quantities
    local_covariance = None
    centroid = None

    # compute running quantities looping over batches
    n_local_images = 0
    for batch in dataloader:
        # DEBUG: START
        # batch_time_start = time.time()
        # DEBUG: END

        # retrieve elements of the batch
        images, gt_category_idxs, gt_bboxes, img_relative_paths, cat_strs = batch
        n_local_images += images.size()[0]
        images = images.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # Retrieve reps
        if model is not None:
            with torch.no_grad():
                reps = model.forward_features(images, module_name=args.layer)  # shape [batch size, # neurons]
        else:
            reps = images.to(device)
        # make them float64
        reps = reps.to(dtype=torch.float64)

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
            reps = pad_to_power_of_2(reps)  # NEW
            # (b) apply D
            reps_d = reps * fjlt_D[None, :]
            # (c) Hadamard (already orthonormal, no extra 1/√n needed)
            reps_h = hadamard_transform(reps_d)
            # DEBUG: check for stability
            # print(f"hadamard stable: {torch.allclose(hadamard_transform(reps_h),reps_d)}")
            # END DEBUG
            # (d) subsample + scale by √(n/k) with n = padded length
            reps = reps_h[:, fjlt_idx] * np.sqrt(N_power2 / args.random_projection_size)

            # print("Done performing the random projection", flush=True)
            # DEBUG:
            # print(f"projected reps shape: {reps.size()}", flush=True)
            # END DEBUG

        # <editor-fold desc="Update local quantities">
        # update local covariance
        local_covariance_addition = einsum(reps, reps, "image neuron1, image neuron2 -> neuron1 neuron2")
        local_covariance = local_covariance_addition.clone() \
            if local_covariance is None \
            else local_covariance + local_covariance_addition

        # update centroid
        centroid_addition = torch.sum(reps, dim=0)  # size [# neurons]
        centroid = centroid_addition.clone() \
            if centroid is None \
            else centroid + centroid_addition

    # divide local quantities by number of samples
    local_covariance /= n_local_images
    centroid /= n_local_images

    # center the covariance
    local_covariance -= torch.outer(centroid, centroid)

    # do local dimensionality reduction if required
    style = args.local_dimensionality_reduction_style
    thresh = args.local_dimensionality_reduction_strength / 100
    if style != "":

        eigenvalues, eigenvectors = torch.linalg.eigh(local_covariance)
        # Sort in descending order
        sorted_idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

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

    trace_2 = torch.trace(local_covariance @ local_covariance)
    participation_ratio = (torch.trace(local_covariance) ** 2) / trace_2
    print(f"participation ratio: {participation_ratio}")
    sqrt_trace_2 = torch.sqrt(trace_2)
    # store in dictionary
    results["covariance"][category] = local_covariance.cpu()
    results["participation_ratio"][category] = participation_ratio.cpu()
    results["sqrt_trace_2"][category] = sqrt_trace_2.cpu()

# === pair-wise alignments =====================================================
print("\nComputing pair-wise category alignments", flush=True)

# create containers -----------------------------------------------------------
pairwise_results = {
    "alignment": {},
    "alignment_random": {},
    "alignment_normalized": {},
}

# convenience views for the scalar stats --------------------------------------
sqrt_trace2 = {k: v.to(device) for k, v in results["sqrt_trace_2"].items()}
part_ratio = {k: v.to(device) for k, v in results["participation_ratio"].items()}

categories_sorted = sorted(results["covariance"])  # deterministic order
n_total = len(categories_sorted)

# main double loop ------------------------------------------------------------
for i, cat1 in enumerate(categories_sorted):
    print(f"Processing category {cat1} vs category...", flush=True)

    C1 = results["covariance"][cat1].to(device, non_blocking=True)
    M1 = sqrt_trace2[cat1]
    D1 = part_ratio[cat1]
    N = C1.shape[0]  # covariance dimension

    for cat2 in categories_sorted[i + 1:]:
        print(f"\t... {cat2}", flush=True)

        C2 = results["covariance"][cat2].to(device, non_blocking=True)
        M2 = sqrt_trace2[cat2]
        D2 = part_ratio[cat2]

        # fast O(N²) trace via element-wise dot (covariances are symmetric)
        tr_C1C2 = torch.einsum("ij,ij->", C1, C2)

        A = tr_C1C2 / (M1 * M2)  # raw alignment
        A_random = torch.sqrt(D1 * D2) / N  # random expectation
        A_norm = (A - A_random) / (1 - A_random)  # normalised

        key = f"{cat1}-{cat2}"
        pairwise_results["alignment"][key] = A.cpu().numpy()
        pairwise_results["alignment_random"][key] = A_random.cpu().numpy()
        pairwise_results["alignment_normalized"][key] = A_norm.cpu().numpy()

    # free GPU RAM for the next outer iteration
    del C1
    torch.cuda.empty_cache()

# === global mean ± std of pair-wise metrics ===================================
import numpy as np

print("\nRESULTS SUMMARY:")
for metric in ("alignment", "alignment_random", "alignment_normalized"):
    # each entry is a 0-d np.ndarray -> cast to float then to 1-D array
    vals = np.array([float(v) for v in pairwise_results[metric].values()],
                    dtype=np.float64)

    mean = vals.mean()
    std = vals.std()

    pairwise_results[f"{metric}_mean"] = mean
    pairwise_results[f"{metric}_std"] = std

    print(f"{metric}: {mean} +- {std}", flush=True)

# Now pairwise_results will contain
# {
#   "alignment":            { "catA-catB": …, … },
#   "alignment_random":     { … },
#   "alignment_normalized": { … },
#   "alignment_mean":            float,
#   "alignment_std":             float,
#   "alignment_random_mean":     float,
#   "alignment_random_std":      float,
#   "alignment_normalized_mean": float,
#   "alignment_normalized_std":  float,
# }

# STORE
pairwise_results["args"] = vars(args).copy()

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

dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))

results_file = os.path.join(results_dir, f"{args.job_id}_Alignment_{dataset_name}_{args.layer}" + dim_red_string + rnd_proj_string + ".pkl")

with open(results_file, 'wb') as file:
    pickle.dump(pairwise_results, file)

print("Script concluded. Exit", flush=True)

exit()

