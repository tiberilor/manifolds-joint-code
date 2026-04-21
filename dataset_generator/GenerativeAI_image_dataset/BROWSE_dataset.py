from pathlib import Path
import random
import pickle

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from PIL import Image, ImageDraw

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = "./dataset_sample"

# Main options
SHOW_BBOX_OVERLAY = True
# Category ordering
CATEGORY_ORDER_MODE = "alphabetical"   # "alphabetical" or "random"
CATEGORY_RANDOM_SEED = 1

# Other options
SORT_IMAGES = True
START_CATEGORY_INDEX = 0
START_PAGE_INDEX = 0
SHOW_FILENAMES = False
FILENAME_FONTSIZE = 8
FIGURE_FACE_COLOR = "black"
EMPTY_TILE_COLOR = "black"
TILE_SIZE = 224
N_ROWS = 4
N_COLS = 6
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
BBOX_COLOR = "white"
BBOX_LINE_WIDTH = 3

# Folder names inside each category folder
IMAGES_SUBFOLDER = "images"
BBOXES_SUBFOLDER = "bboxes"

def list_categories(dataset_root: Path):
    categories = [p for p in dataset_root.iterdir() if p.is_dir()]

    if CATEGORY_ORDER_MODE == "alphabetical":
        categories.sort(key=lambda p: p.name.lower())
    elif CATEGORY_ORDER_MODE == "random":
        rng = random.Random(CATEGORY_RANDOM_SEED)
        rng.shuffle(categories)
    else:
        raise ValueError(
            f"Unsupported CATEGORY_ORDER_MODE={CATEGORY_ORDER_MODE!r}. "
            f"Use 'alphabetical' or 'random'."
        )

    return categories


def list_images(category_dir: Path):
    images_dir = category_dir / IMAGES_SUBFOLDER
    if not images_dir.exists():
        return []
    image_paths = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if SORT_IMAGES:
        image_paths.sort(key=lambda p: p.name.lower())
    return image_paths


CANONICAL_BBOX_SIZE_PX = 512.0


def read_bbox_for_image(category_dir: Path, image_path: Path):
    bbox_path = category_dir / BBOXES_SUBFOLDER / f"{image_path.stem}.pkl"
    if not bbox_path.exists():
        return None

    try:
        with open(bbox_path, "rb") as f:
            bbox_dict = pickle.load(f)
    except Exception:
        return None

    required = ("bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max")
    if not all(k in bbox_dict for k in required):
        return None

    return [
        float(bbox_dict["bbox_x_min"]),
        float(bbox_dict["bbox_y_min"]),
        float(bbox_dict["bbox_x_max"]),
        float(bbox_dict["bbox_y_max"]),
    ]


def canonical_pixel_bbox_to_current_pixels(pixel_bbox_512, image_size):
    if pixel_bbox_512 is None:
        return None

    width, height = image_size
    x_min_rel = pixel_bbox_512[0] / CANONICAL_BBOX_SIZE_PX
    y_min_rel = pixel_bbox_512[1] / CANONICAL_BBOX_SIZE_PX
    x_max_rel = pixel_bbox_512[2] / CANONICAL_BBOX_SIZE_PX
    y_max_rel = pixel_bbox_512[3] / CANONICAL_BBOX_SIZE_PX

    x_min = x_min_rel * width
    y_min = y_min_rel * height
    x_max = x_max_rel * width
    y_max = y_max_rel * height
    return [x_min, y_min, x_max, y_max]


def overlay_bbox(image: Image.Image, pixel_bbox_512):
    if pixel_bbox_512 is None:
        return image

    pixel_bbox = canonical_pixel_bbox_to_current_pixels(pixel_bbox_512, image.size)
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle(pixel_bbox, outline=BBOX_COLOR, width=BBOX_LINE_WIDTH)
    return out


def load_tile(category_dir: Path, image_path: Path, tile_size: int):
    image = Image.open(image_path).convert("RGB")

    if SHOW_BBOX_OVERLAY:
        bbox = read_bbox_for_image(category_dir, image_path)
        image = overlay_bbox(image, bbox)

    if image.size != (tile_size, tile_size):
        image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

    return np.asarray(image)


class DatasetBrowser:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.categories = list_categories(dataset_root)
        if not self.categories:
            raise RuntimeError(f"No category folders found in {dataset_root}")

        self.category_index = max(0, min(START_CATEGORY_INDEX, len(self.categories) - 1))
        self.page_index = max(0, START_PAGE_INDEX)

        self.nrows = max(1, int(N_ROWS))
        self.ncols = max(1, int(N_COLS))
        self.page_size = self.nrows * self.ncols

        self.fig = plt.figure(facecolor=FIGURE_FACE_COLOR, figsize=(16, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self._build_buttons()
        self.render()

    def _build_buttons(self):
        self.ax_prev_cat = self.fig.add_axes([0.02, 0.93, 0.12, 0.045])
        self.ax_prev_page = self.fig.add_axes([0.16, 0.93, 0.12, 0.045])
        self.ax_next_page = self.fig.add_axes([0.72, 0.93, 0.12, 0.045])
        self.ax_next_cat = self.fig.add_axes([0.86, 0.93, 0.12, 0.045])

        self.btn_prev_cat = Button(self.ax_prev_cat, "Prev category")
        self.btn_prev_page = Button(self.ax_prev_page, "Prev page")
        self.btn_next_page = Button(self.ax_next_page, "Next page")
        self.btn_next_cat = Button(self.ax_next_cat, "Next category")

        self.btn_prev_cat.on_clicked(lambda event: self.prev_category())
        self.btn_prev_page.on_clicked(lambda event: self.prev_page())
        self.btn_next_page.on_clicked(lambda event: self.next_page())
        self.btn_next_cat.on_clicked(lambda event: self.next_category())

    def current_category_dir(self):
        return self.categories[self.category_index]

    def current_image_paths(self):
        return list_images(self.current_category_dir())

    def clamp_page_index(self, n_images):
        max_page_index = 0 if n_images == 0 else (n_images - 1) // self.page_size
        self.page_index = max(0, min(self.page_index, max_page_index))

    def next_page(self):
        n_images = len(self.current_image_paths())
        if n_images == 0:
            return
        max_page_index = (n_images - 1) // self.page_size
        if self.page_index < max_page_index:
            self.page_index += 1
            self.render()

    def prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.render()

    def next_category(self):
        if self.category_index < len(self.categories) - 1:
            self.category_index += 1
            self.page_index = 0
            self.render()

    def prev_category(self):
        if self.category_index > 0:
            self.category_index -= 1
            self.page_index = 0
            self.render()

    def on_key_press(self, event):
        key = event.key
        if key in ("right", "d", "l", "space", "pagedown"):
            self.next_page()
        elif key in ("left", "a", "h", "backspace", "pageup"):
            self.prev_page()
        elif key in ("down", "s", "j"):
            self.next_category()
        elif key in ("up", "w", "k"):
            self.prev_category()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def render(self):
        category_dir = self.current_category_dir()
        image_paths = self.current_image_paths()

        self.clamp_page_index(len(image_paths))

        start = self.page_index * self.page_size
        end = min(len(image_paths), start + self.page_size)
        page_paths = image_paths[start:end]

        # clear only image axes, keep button axes
        for ax in list(self.fig.axes):
            if ax not in [self.ax_prev_cat, self.ax_prev_page, self.ax_next_page, self.ax_next_cat]:
                ax.remove()

        title = (
            f"Category {self.category_index + 1}/{len(self.categories)}: {category_dir.name}    "
            f"Images {start + 1 if page_paths else 0}-{end}/{len(image_paths)}    "
            f"Grid {self.nrows}x{self.ncols} ({self.page_size} per page)    "
            f"Order: {CATEGORY_ORDER_MODE}"
        )
        self.fig.suptitle(title, fontsize=12, color="white", y=0.905)

        grid = self.fig.add_gridspec(
            self.nrows,
            self.ncols,
            left=0.01,
            right=0.99,
            bottom=0.03,
            top=0.88,
            wspace=0.02,
            hspace=0.08 if SHOW_FILENAMES else 0.02,
        )

        total_slots = self.nrows * self.ncols
        for slot_idx in range(total_slots):
            row = slot_idx // self.ncols
            col = slot_idx % self.ncols
            ax = self.fig.add_subplot(grid[row, col])
            ax.set_facecolor(EMPTY_TILE_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])

            if slot_idx < len(page_paths):
                image_path = page_paths[slot_idx]
                tile = load_tile(category_dir, image_path, TILE_SIZE)
                ax.imshow(tile)
                if SHOW_FILENAMES:
                    ax.set_title(image_path.name, fontsize=FILENAME_FONTSIZE, color="white", pad=2)
            else:
                ax.imshow(np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8))

        self.fig.canvas.draw_idle()


def main():
    dataset_root = Path(DATASET_ROOT)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    DatasetBrowser(dataset_root)
    plt.show()


if __name__ == "__main__":
    main()
