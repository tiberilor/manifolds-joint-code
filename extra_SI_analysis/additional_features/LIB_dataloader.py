from __future__ import annotations
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance
import pickle


BBOX_FEATURE_NAMES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]

DEFAULT_BBOX_MEANS = [0.5015578269958496, 0.5011495351791382, 0.5563644170761108, 0.591561496257782]
DEFAULT_BBOX_STDS = [0.14438922703266144, 0.13441763818264008, 0.2118850201368332, 0.20007650554180145]

DEFAULT_TARGET_FEATURE_MEANS = {
    "bbox_center_x": DEFAULT_BBOX_MEANS[0],
    "bbox_center_y": DEFAULT_BBOX_MEANS[1],
    "bbox_x_length": DEFAULT_BBOX_MEANS[2],
    "bbox_y_length": DEFAULT_BBOX_MEANS[3],
    "bbox_area": 0.3387143909931183,
    "mean_srgb_luma": 0.48661008477211,
    "local_srgb_luma_contrast": 0.14033207297325134,
    "mean_hsv_saturation": 0.1964108794927597,
}

DEFAULT_TARGET_FEATURE_STDS = {
    "bbox_center_x": DEFAULT_BBOX_STDS[0],
    "bbox_center_y": DEFAULT_BBOX_STDS[1],
    "bbox_x_length": DEFAULT_BBOX_STDS[2],
    "bbox_y_length": DEFAULT_BBOX_STDS[3],
    "bbox_area": 0.19617179036140442,
    "mean_srgb_luma": 0.10358938574790955,
    "local_srgb_luma_contrast": 0.07069866359233856,
    "mean_hsv_saturation": 0.11258763074874878,
}

_BBOX_FEATURE_TO_COLUMN = {name: i for i, name in enumerate(BBOX_FEATURE_NAMES)}


def _as_feature_list(target_features):
    if target_features is None:
        return None
    if isinstance(target_features, str):
        features = [target_features]
    else:
        features = list(target_features)
    if not features:
        raise ValueError("target_features must contain at least one feature when it is not None.")
    return features


def _decode_h5_strings(values):
    return values[:].astype(str)


def _resolve_feature_stats(feature_names, provided_stats, default_stats, stat_name):
    if provided_stats is None:
        missing = [name for name in feature_names if name not in default_stats]
        if missing:
            raise ValueError(
                f"No default {stat_name} available for target feature(s): {missing}. "
                f"Pass {stat_name} explicitly for these features."
            )
        stats = [default_stats[name] for name in feature_names]
    elif isinstance(provided_stats, dict):
        missing = [name for name in feature_names if name not in provided_stats]
        if missing:
            raise ValueError(f"Missing {stat_name} for target feature(s): {missing}")
        stats = [provided_stats[name] for name in feature_names]
    else:
        stats = list(provided_stats)
        if len(stats) != len(feature_names):
            raise ValueError(
                f"{stat_name} must have length {len(feature_names)} for target_features={feature_names}, "
                f"got length {len(stats)}."
            )

    stats = np.asarray(stats, dtype=np.float32)
    if stat_name.endswith("stds") and np.any(stats == 0):
        raise ValueError(f"{stat_name} contains a zero entry for target_features={feature_names}.")
    return stats


def _target_default_stats_from_bbox_stats(bbox_stats, additional_default_stats):
    bbox_stats = np.asarray(bbox_stats, dtype=np.float32)
    if bbox_stats.shape[0] != len(BBOX_FEATURE_NAMES):
        raise ValueError(
            f"bbox stats must have length {len(BBOX_FEATURE_NAMES)}, got shape {bbox_stats.shape}."
        )
    stats = dict(additional_default_stats)
    for i, feature in enumerate(BBOX_FEATURE_NAMES):
        stats[feature] = float(bbox_stats[i])
    return stats


def _load_additional_feature_arrays(dataset_folder, image_keys, requested_features, additional_labels_filename):
    additional_features = [name for name in requested_features if name not in _BBOX_FEATURE_TO_COLUMN]
    if not additional_features:
        return {}

    additional_path = os.path.join(dataset_folder, additional_labels_filename)
    if not os.path.isfile(additional_path):
        raise FileNotFoundError(
            f"Requested additional target feature(s) {additional_features}, but {additional_path} does not exist."
        )

    out = {}
    with h5py.File(additional_path, "r") as h5:
        if "image_keys" not in h5:
            raise KeyError(f"{additional_path} does not contain required dataset 'image_keys'.")
        additional_image_keys = _decode_h5_strings(h5["image_keys"])
        if not np.array_equal(additional_image_keys, image_keys):
            raise ValueError(
                f"image_keys in {additional_path} do not match labels.h5. "
                "Refusing to align additional labels silently."
            )

        if "feature_names" in h5 and "additional_labels" in h5:
            feature_names = _decode_h5_strings(h5["feature_names"])
            labels_matrix = h5["additional_labels"]
            for feature in additional_features:
                matches = np.where(feature_names == feature)[0]
                if len(matches) == 1:
                    out[feature] = labels_matrix[:, int(matches[0])].astype(np.float32)

        for feature in additional_features:
            if feature not in out:
                if feature not in h5:
                    raise KeyError(
                        f"Feature '{feature}' was requested but was not found in {additional_path}."
                    )
                out[feature] = h5[feature][:].astype(np.float32)

    return out


def _build_target_array(dataset_folder, image_keys, bboxes_np, target_features, additional_labels_filename):
    additional_arrays = _load_additional_feature_arrays(
        dataset_folder,
        image_keys,
        target_features,
        additional_labels_filename,
    )

    columns = []
    for feature in target_features:
        if feature in _BBOX_FEATURE_TO_COLUMN:
            columns.append(bboxes_np[:, _BBOX_FEATURE_TO_COLUMN[feature]].astype(np.float32))
        else:
            columns.append(additional_arrays[feature])
    return np.stack(columns, axis=1).astype(np.float32)


def _replace_bbox_features_in_target(target_raw, target_features, bbox_raw):
    for i, feature in enumerate(target_features):
        if feature in _BBOX_FEATURE_TO_COLUMN:
            target_raw[i] = bbox_raw[_BBOX_FEATURE_TO_COLUMN[feature]]
    return target_raw


class DatasetSDXL(Dataset):
    """
    Returns:
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - `image` (torch.Tensor): Image tensor of shape `(C, H, W)`, dtype `torch.float32`.
            - Tuple containing:
                - `category` (torch.Tensor): Category index, shape `(1,)`, dtype `torch.long`.
                - `target` (torch.Tensor): By default, normalized bounding box coordinates
                  `(center_x, center_y, width, height)`, shape `(4,)`. If `target_features`
                  is provided, normalized targets in that requested feature order.
    """
    def __init__(self, dataset_folder, transform=None, exclude_categories=None,
                 bbox_means=DEFAULT_BBOX_MEANS,
                 bbox_stds=DEFAULT_BBOX_STDS,
                 target_features=None,
                 target_feature_means=None,
                 target_feature_stds=None,
                 additional_labels_filename="additional_labels.h5"):
        self.dataset_folder = dataset_folder
        self.h5_path = os.path.join(dataset_folder, "labels.h5")
        self.legacy_bbox_mode = target_features is None
        self.target_features = BBOX_FEATURE_NAMES.copy() if target_features is None else _as_feature_list(target_features)
        self.additional_labels_filename = additional_labels_filename

        # Open HDF5 file
        self.h5_file = h5py.File(self.h5_path, "r")

        # Read as numpy first so we can filter **before** converting to torch
        image_keys = _decode_h5_strings(self.h5_file["image_keys"])  # e.g. "guitar/img_00123"
        bboxes_np = self.h5_file["bboxes"][:]  # shape (N, 4), float
        cat_idx_np = self.h5_file["category_indices"][:]  # shape (N,), int

        raw_targets_np = None
        if not self.legacy_bbox_mode:
            raw_targets_np = _build_target_array(
                dataset_folder=self.dataset_folder,
                image_keys=image_keys,
                bboxes_np=bboxes_np,
                target_features=self.target_features,
                additional_labels_filename=self.additional_labels_filename,
            )

        # Optional category exclusion (case-insensitive match on the category prefix of image_keys)
        if exclude_categories:
            excl = {c.lower() for c in exclude_categories}
            keep = [i for i, key in enumerate(image_keys)
                    if key.split("/", 1)[0].lower() not in excl]
            if not keep:
                raise ValueError("All samples were filtered out by exclude_categories in DatasetSDXL.")
            image_keys = image_keys[keep]
            bboxes_np = bboxes_np[keep]
            cat_idx_np = cat_idx_np[keep]
            if raw_targets_np is not None:
                raw_targets_np = raw_targets_np[keep]

        # Store (convert to torch where needed)
        self.image_keys = image_keys
        self.bboxes = torch.tensor(bboxes_np, dtype=torch.float32)
        self.category_indices = torch.tensor(cat_idx_np, dtype=torch.long)
        self.raw_targets = (
            self.bboxes if self.legacy_bbox_mode
            else torch.tensor(raw_targets_np, dtype=torch.float32)
        )

        self.transform = transform

        # Keep bbox_mean/std as the legacy four-coordinate normalization stats.
        self.bbox_mean = torch.tensor(bbox_means, dtype=torch.float32)
        self.bbox_std = torch.tensor(bbox_stds, dtype=torch.float32)
        target_default_means = _target_default_stats_from_bbox_stats(
            bbox_means,
            DEFAULT_TARGET_FEATURE_MEANS,
        )
        target_default_stds = _target_default_stats_from_bbox_stats(
            bbox_stds,
            DEFAULT_TARGET_FEATURE_STDS,
        )
        if self.legacy_bbox_mode:
            self.target_mean = self.bbox_mean
            self.target_std = self.bbox_std
        else:
            self.target_mean = torch.tensor(
                _resolve_feature_stats(
                    self.target_features,
                    target_feature_means,
                    target_default_means,
                    "target_feature_means",
                ),
                dtype=torch.float32,
            )
            self.target_std = torch.tensor(
                _resolve_feature_stats(
                    self.target_features,
                    target_feature_stds,
                    target_default_stds,
                    "target_feature_stds",
                ),
                dtype=torch.float32,
            )

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Extract category and image name from HDF5 keys
        category, image_name = self.image_keys[idx].split("/")
        image_path = os.path.join(self.dataset_folder, category, "images", f"{image_name}.jpg")

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get and normalize the requested regression target.
        target = self.raw_targets[idx]
        target = (target - self.target_mean) / self.target_std

        category = self.category_indices[idx]

        return image, (category, target)

    def __del__(self):
        self.h5_file.close()  # Ensure HDF5 file is closed when the object is deleted


# USAGE EXAMPLE:
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images for CNN
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization
# ])
# dataset = DatasetSDXL(dataset_folder, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# for images, (categories, bboxes) in dataloader:
#     print(images.shape)       # Expected: (batch_size, C, H, W)
#     print(categories.shape)   # Expected: (batch_size,)  -> 1D tensor
#     print(bboxes.shape)       # Expected: (batch_size, 4)  -> 2D tensor


class CategoryDatasetSDXL(Dataset):
    """
    This dataset loads only the images for a given category. It reads from labels.h5
    to retrieve ground truth bounding boxes and category indices.
    It returns a tuple of:
        (image, category_idx, target, img_relative_path, category_str)
    so that images, category_idx, and targets can be batched automatically.
    """
    def __init__(self, dataset_folder, category, transform=None,
                 augment_dihedral=False,
                 bbox_means=DEFAULT_BBOX_MEANS,
                 bbox_stds=DEFAULT_BBOX_STDS,
                 target_features=None,
                 target_feature_means=None,
                 target_feature_stds=None,
                 additional_labels_filename="additional_labels.h5"):
        self.dataset_folder = dataset_folder
        self.category = category  # human-readable category string
        self.transform = transform
        self.legacy_bbox_mode = target_features is None
        self.target_features = BBOX_FEATURE_NAMES.copy() if target_features is None else _as_feature_list(target_features)
        self.additional_labels_filename = additional_labels_filename
        # Dihedral (D4) augmentation: 8 variants per base image (4 rotations x optional horizontal flip)
        self.augment_dihedral = bool(augment_dihedral)
        self.augmentation_factor = 8 if self.augment_dihedral else 1

        # Open the labels.h5 file (assumed to be in dataset_folder)
        h5_path = os.path.join(dataset_folder, "labels.h5")
        self.h5_file = h5py.File(h5_path, "r")
        self.image_keys = _decode_h5_strings(self.h5_file["image_keys"])
        self.bboxes = self.h5_file["bboxes"][:]  # shape (N, 4)
        self.category_indices = self.h5_file["category_indices"][:]
        self.raw_targets = (
            self.bboxes
            if self.legacy_bbox_mode
            else _build_target_array(
                dataset_folder=self.dataset_folder,
                image_keys=self.image_keys,
                bboxes_np=self.bboxes,
                target_features=self.target_features,
                additional_labels_filename=self.additional_labels_filename,
            )
        )

        # Filter indices to include only samples for the specified category.
        self.indices = [i for i, key in enumerate(self.image_keys) if key.startswith(f"{category}/")]

        # Keep bbox_mean/std as the legacy four-coordinate normalization stats.
        self.bbox_mean = np.array(bbox_means, dtype=np.float32)
        self.bbox_std = np.array(bbox_stds, dtype=np.float32)
        target_default_means = _target_default_stats_from_bbox_stats(
            self.bbox_mean,
            DEFAULT_TARGET_FEATURE_MEANS,
        )
        target_default_stds = _target_default_stats_from_bbox_stats(
            self.bbox_std,
            DEFAULT_TARGET_FEATURE_STDS,
        )
        if self.legacy_bbox_mode:
            self.target_mean = self.bbox_mean
            self.target_std = self.bbox_std
        else:
            self.target_mean = _resolve_feature_stats(
                self.target_features,
                target_feature_means,
                target_default_means,
                "target_feature_means",
            )
            self.target_std = _resolve_feature_stats(
                self.target_features,
                target_feature_stds,
                target_default_stds,
                "target_feature_stds",
            )

    # (k_rot, mirror) where k_rot is # of 90deg CCW rotations, mirror = horizontal flip AFTER rotation
    _D4 = [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        (2, False),
        (2, True),
        (3, False),
        (3, True),
    ]

    @staticmethod
    def _apply_d4_to_image(image: Image.Image, t_id: int) -> Image.Image:
        k_rot, mirror = CategoryDatasetSDXL._D4[int(t_id)]
        if k_rot == 1:
            image = image.transpose(Image.ROTATE_90)
        elif k_rot == 2:
            image = image.transpose(Image.ROTATE_180)
        elif k_rot == 3:
            image = image.transpose(Image.ROTATE_270)
        if mirror:
            image = ImageOps.mirror(image)
        return image

    @staticmethod
    def _apply_d4_to_bbox_frac(bbox_frac: np.ndarray, t_id: int) -> np.ndarray:
        """
        bbox_frac: raw fractional [cx, cy, w, h] in [0,1]
        Transform matches _apply_d4_to_image (rotate CCW, then optional horizontal mirror).
        """
        cx, cy, w, h = [float(x) for x in bbox_frac.tolist()]
        x0 = cx - w / 2.0
        x1 = cx + w / 2.0
        y0 = cy - h / 2.0
        y1 = cy + h / 2.0

        corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]

        k_rot, mirror = CategoryDatasetSDXL._D4[int(t_id)]

        def rot_ccw_90(x, y):
            # (x,y)->(y, 1-x) on unit square with origin top-left (x right, y down)
            return (y, 1.0 - x)

        def rot_ccw_k(x, y, k):
            k = int(k) % 4
            if k == 0:
                return (x, y)
            if k == 1:
                return rot_ccw_90(x, y)
            if k == 2:
                x2, y2 = rot_ccw_90(x, y)
                return rot_ccw_90(x2, y2)
            # k == 3
            x2, y2 = rot_ccw_90(x, y)
            x3, y3 = rot_ccw_90(x2, y2)
            return rot_ccw_90(x3, y3)

        new_pts = []
        for (x, y) in corners:
            xr, yr = rot_ccw_k(x, y, k_rot)
            if mirror:
                xr = 1.0 - xr
            new_pts.append((xr, yr))

        xs = [p[0] for p in new_pts]
        ys = [p[1] for p in new_pts]
        nx0, nx1 = min(xs), max(xs)
        ny0, ny1 = min(ys), max(ys)

        ncx = (nx0 + nx1) / 2.0
        ncy = (ny0 + ny1) / 2.0
        nw = (nx1 - nx0)
        nh = (ny1 - ny0)

        out = np.array([ncx, ncy, nw, nh], dtype=np.float32)
        out = np.clip(out, 0.0, 1.0)
        return out

    def __len__(self):
        return len(self.indices) * self.augmentation_factor

    def __getitem__(self, idx):
        idx = int(idx)
        if self.augmentation_factor == 1:
            base_local_idx = idx
            t_id = 0
        else:
            base_local_idx = idx // self.augmentation_factor
            t_id = idx % self.augmentation_factor

        # Map the base local index to the actual index in the h5 arrays.
        real_idx = self.indices[base_local_idx]
        key = self.image_keys[real_idx]  # e.g. "cat1/image001"
        parts = key.split("/")
        cat_str = parts[0]
        image_name = parts[1]

        # Build relative and absolute image paths.
        img_relative_path = os.path.join(cat_str, "images", f"{image_name}.jpg")
        img_path = os.path.join(self.dataset_folder, img_relative_path)

        # Load image.
        image = Image.open(img_path).convert("RGB")
        if self.augmentation_factor != 1:
            image = self._apply_d4_to_image(image, t_id)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        # Get the ground-truth bbox (raw fractional) and (optionally) transform it.
        bbox = np.asarray(self.bboxes[real_idx], dtype=np.float32)  # raw [cx, cy, w, h]
        if self.augmentation_factor != 1:
            bbox = self._apply_d4_to_bbox_frac(bbox, t_id)

        if self.legacy_bbox_mode:
            target = bbox
        else:
            target = np.asarray(self.raw_targets[real_idx], dtype=np.float32).copy()
            if self.augmentation_factor != 1:
                target = _replace_bbox_features_in_target(target, self.target_features, bbox)

        target_normalized = (target - self.target_mean) / self.target_std
        target_tensor = torch.tensor(target_normalized, dtype=torch.float32)

        # Get ground truth category index.
        category_idx = int(self.category_indices[real_idx])

        # Return a tuple so that collate_fn can batch tensors.
        return image_tensor, category_idx, target_tensor, img_relative_path, cat_str

    def __del__(self):
        # Close the h5 file when the dataset object is destroyed.
        self.h5_file.close()


class CategoryDatasetRepresentation(Dataset):
    """
    Loads pre-computed neural activations for a single category.

    Expects a folder structure like:
        root_folder/
            <category>/
                <category>.pkl

    The .pkl must contain a dict with keys:
        "values":          np.ndarray, shape (N_images, N_neurons)
        "presentation":    dict with lists:
            "img_relative_path" (list of str),
            "category_idx"      (list of int),
            "category"          (list of str),
            "bbox_center_x"     (list of float),
            "bbox_center_y"     (list of float),
            "bbox_x_length"     (list of float),
            "bbox_y_length"     (list of float)
    """

    def __init__(
        self,
        dataset_folder: str,
        category: str,
        transform=None,
        bbox_means=[0.5015578269958496, 0.5011495351791382, 0.5563644170761108, 0.591561496257782],
        bbox_stds =[0.14438922703266144,0.13441763818264008,0.2118850201368332,0.20007650554180145]
    ):
        self.dataset_folder = dataset_folder
        self.category    = category
        self.transform   = transform

        pkl_path = os.path.join(dataset_folder, category, f"{category}.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # raw activations: (N_images, N_neurons)
        self.values = torch.tensor(data["values"], dtype=torch.float32)

        pres = data["presentation"]
        self.img_relative_paths = pres["img_relative_path"]
        self.category_idxs      = pres["category_idx"]
        self.cat_strs           = pres["category"]

        # assemble raw bbox array of shape (N_images, 4)
        raw_bboxes = np.stack([
            pres["bbox_center_x"],
            pres["bbox_center_y"],
            pres["bbox_x_length"],
            pres["bbox_y_length"],
        ], axis=1).astype(np.float32)

        # normalise bboxes
        self.bbox_mean = np.array(bbox_means, dtype=np.float32)
        self.bbox_std = np.array(bbox_stds, dtype=np.float32)
        norm_bboxes = (raw_bboxes - self.bbox_mean[None, :]) / self.bbox_std[None, :]

        self.bboxes = torch.tensor(norm_bboxes, dtype=torch.float32)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        # “image” is actually the neural activation vector
        img_tensor = self.values[idx]               # shape [N_neurons]
        category_idx = int(self.category_idxs[idx])  # e.g. 0,1,2...
        bbox_tensor  = self.bboxes[idx]              # shape [4]
        img_rel_path = self.img_relative_paths[idx]  # e.g. "cat1/images/img_0001.jpg"
        cat_str      = self.cat_strs[idx]            # e.g. "cat1"

        # if you ever want to post-process the activations,
        # you can apply self.transform here:
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, category_idx, bbox_tensor, img_rel_path, cat_str

