import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


# We compute image statistics on the same spatial resolution used by the CNN
# before ImageNet normalization.
IMAGE_SIZE = 224

# Definition of local contrast: Gaussian-window local RMS luma contrast.
# sigma=3 px gives an approximately 7 px FWHM window at 224x224 resolution:
# local enough for edges/texture, but broad enough to suppress JPEG pixel noise.
LOCAL_CONTRAST_SIGMA_PX = 3.0
LOCAL_CONTRAST_EPS = 1e-6

FEATURE_NAMES = [
    "bbox_area",
    "mean_srgb_luma",
    "local_srgb_luma_contrast",
    "mean_hsv_saturation",
]

SRGB_LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from LIB_dataloader import DatasetSDXL  # noqa: E402


class DatasetSDXLForAdditionalLabels(Dataset):
    """
    Wrap DatasetSDXL so image loading follows the project dataloader convention,
    while exposing raw fractional bboxes from labels.h5.
    """

    def __init__(self, dataset_folder: str):
        image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.base = DatasetSDXL(
            dataset_folder=dataset_folder,
            transform=image_transform,
            bbox_means=[0.0, 0.0, 0.0, 0.0],
            bbox_stds=[1.0, 1.0, 1.0, 1.0],
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, (category_idx, _) = self.base[idx]
        raw_bbox = self.base.bboxes[idx]
        return int(idx), image, category_idx, raw_bbox


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute additional scalar image labels for an SDXL dataset: bbox area, "
            "mean sRGB luma, local sRGB-luma contrast, and mean HSV saturation."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset folder containing labels.h5 and category/images subfolders.",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default=None,
        help="Output HDF5 path. Default: <dataset_dir>/additional_labels.h5.",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default=None,
        help="Output stats text path. Default: <dataset_dir>/additional_labels_stats.txt.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional image limit. Omit for the full dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)


def compute_batch_features(images: torch.Tensor, raw_bboxes: torch.Tensor, kernel: torch.Tensor):
    images = images.to(dtype=torch.float32)
    raw_bboxes = raw_bboxes.to(dtype=torch.float32)

    # bbox area from raw fractional [center_x, center_y, width, height].
    bbox_area = raw_bboxes[:, 2] * raw_bboxes[:, 3]

    # sRGB luma, not linear-RGB luminance.
    weights = SRGB_LUMA_WEIGHTS.to(device=images.device).view(1, 3, 1, 1)
    luma = (images * weights).sum(dim=1, keepdim=True)
    mean_luma = luma.mean(dim=(1, 2, 3))

    # Local RMS luma contrast: mean_p std_G(Y')_p / (mean_G(Y')_p + eps).
    pad = kernel.shape[-1] // 2
    luma_pad = F.pad(luma, (pad, pad, pad, pad), mode="reflect")
    local_mean = F.conv2d(luma_pad, kernel)

    luma2_pad = F.pad(luma * luma, (pad, pad, pad, pad), mode="reflect")
    local_second_moment = F.conv2d(luma2_pad, kernel)

    local_var = (local_second_moment - local_mean * local_mean).clamp_min(0.0)
    local_std = torch.sqrt(local_var)
    local_contrast = (local_std / (local_mean + LOCAL_CONTRAST_EPS)).mean(dim=(1, 2, 3))

    # HSV saturation computed directly from sRGB channel values.
    rgb_max = images.max(dim=1).values
    rgb_min = images.min(dim=1).values
    chroma = rgb_max - rgb_min
    saturation = torch.where(
        rgb_max > 0.0,
        chroma / rgb_max.clamp_min(LOCAL_CONTRAST_EPS),
        torch.zeros_like(rgb_max),
    )
    mean_saturation = saturation.mean(dim=(1, 2))

    return torch.stack([bbox_area, mean_luma, local_contrast, mean_saturation], dim=1)


def resolve_output_paths(args):
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_h5 = args.output_h5 or os.path.join(dataset_dir, "additional_labels.h5")
    stats_file = args.stats_file or os.path.join(dataset_dir, "additional_labels_stats.txt")
    return dataset_dir, os.path.abspath(output_h5), os.path.abspath(stats_file)


def assert_can_write(path: str, overwrite: bool):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")


def save_h5(output_h5, dataset, selected_indices, additional_labels):
    image_keys_np = np.array([str(dataset.base.image_keys[i]) for i in selected_indices], dtype="S")
    feature_names_np = np.array(FEATURE_NAMES, dtype="S")
    category_indices_np = dataset.base.category_indices[selected_indices].cpu().numpy().astype(np.int32)

    with h5py.File(output_h5, "w") as hf:
        hf.create_dataset("image_keys", data=image_keys_np)
        hf.create_dataset("category_indices", data=category_indices_np)
        hf.create_dataset("feature_names", data=feature_names_np)
        hf.create_dataset("additional_labels", data=additional_labels, compression="gzip")

        for feature_idx, feature_name in enumerate(FEATURE_NAMES):
            hf.create_dataset(feature_name, data=additional_labels[:, feature_idx], compression="gzip")

        hf.attrs["image_size_px"] = IMAGE_SIZE
        hf.attrs["local_contrast_sigma_px"] = LOCAL_CONTRAST_SIGMA_PX
        hf.attrs["local_contrast_eps"] = LOCAL_CONTRAST_EPS
        hf.attrs["luma_definition"] = "Y_prime = 0.2126 R_prime + 0.7152 G_prime + 0.0722 B_prime on sRGB values"
        hf.attrs["local_contrast_definition"] = (
            "mean over pixels of Gaussian local std(Y_prime) / "
            "(Gaussian local mean(Y_prime) + eps)"
        )
        hf.attrs["saturation_definition"] = "HSV saturation from sRGB values: (max(R,G,B)-min(R,G,B))/max(R,G,B)"


def save_stats(stats_file, additional_labels):
    means = additional_labels.mean(axis=0)
    stds = additional_labels.std(axis=0)

    with open(stats_file, "w") as f:
        for i, feature_name in enumerate(FEATURE_NAMES):
            f.write(f"{feature_name}_mean: {means[i]}\n")
            f.write(f"{feature_name}_std: {stds[i]}\n")
            if i != len(FEATURE_NAMES) - 1:
                f.write("\n")


def main():
    args = parse_args()
    dataset_dir, output_h5, stats_file = resolve_output_paths(args)
    assert_can_write(output_h5, args.overwrite)
    assert_can_write(stats_file, args.overwrite)

    labels_h5 = os.path.join(dataset_dir, "labels.h5")
    if not os.path.isfile(labels_h5):
        raise FileNotFoundError(f"Could not find labels.h5 at {labels_h5}")

    requested_device = args.device
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Image size for image-derived labels: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"Local contrast Gaussian sigma: {LOCAL_CONTRAST_SIGMA_PX} px", flush=True)

    full_dataset = DatasetSDXLForAdditionalLabels(dataset_dir)
    selected_indices = np.arange(len(full_dataset), dtype=np.int64)
    dataset = full_dataset
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max_images must be positive when provided.")
        selected_indices = selected_indices[:min(args.max_images, len(full_dataset))]
        dataset = Subset(full_dataset, selected_indices.tolist())

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    n_images = len(dataset)
    additional_labels = np.empty((n_images, len(FEATURE_NAMES)), dtype=np.float32)
    kernel = gaussian_kernel_2d(LOCAL_CONTRAST_SIGMA_PX, device=device)

    start = time.time()
    n_seen = 0
    with torch.no_grad():
        for batch_idx, (indices, images, category_indices, raw_bboxes) in enumerate(loader, start=1):
            del indices, category_indices
            images = images.to(device, non_blocking=True)
            raw_bboxes = raw_bboxes.to(device, non_blocking=True)
            features = compute_batch_features(images, raw_bboxes, kernel)

            batch_size = images.shape[0]
            additional_labels[n_seen:n_seen + batch_size, :] = features.cpu().numpy().astype(np.float32)

            n_seen += batch_size
            if args.print_every > 0 and (batch_idx % args.print_every == 0 or n_seen == n_images):
                elapsed = time.time() - start
                print(f"Processed {n_seen}/{n_images} images in {elapsed:.1f} sec.", flush=True)

    save_h5(output_h5, full_dataset, selected_indices, additional_labels)
    save_stats(stats_file, additional_labels)

    print(f"Saved additional labels: {output_h5}", flush=True)
    print(f"Saved additional-label stats: {stats_file}", flush=True)


if __name__ == "__main__":
    main()

