#!/usr/bin/env python3
# Plot alignment matrix with full plot control:
# - Arial everywhere (including colorbar ticks)
# - Diagonal forced to white (masked)
# - Manual color range (or auto from off-diagonals)
# - Optional axes/ticks/title (still defaults to plain matrix)
# - SVG output by default

import json, pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter

# =========================
# ===== USER SETTINGS =====
# =========================

# Ordered JSON and results PKL paths
JSON_PATH = "PATH/TO/categories_ordered.json"

RESULTS_PKL_PATH = "PATH/TO/alignment_results.pkl"

# Which metric to plot from the PKL
METRIC_KEY = "alignment_normalized"

# Value that conceptually belongs on the diagonal (we will mask it visually to white)
DIAGONAL_VALUE = 1.0

# If some JSON categories are missing in results, we drop them (to keep a square matrix in JSON order).
APPEND_RESULTS_EXTRAS_AT_END = False

# Missing pair fill (A-B not found)
MISSING_ENTRY_FILL = float("nan")

# =========================
# ===== PLOTTING CFG  =====
# =========================
PLOT_CFG = {
    # ---- Figure / saving ----
    "figsize": (1.2, 1.2),                 # (width, height) in inches
    "dpi": 300,
    "save_path": "alignment_matrix.svg",   # default: SVG
    "save_transparent": True,
    "bbox_inches": "tight",
    "tight_layout": False,
    "constrained_layout": False,
    "show_fig": True,

    # ---- Figure / axes facecolors ----
    "figure_facecolor": "white",
    "axes_facecolor": "white",

    # ---- Fonts: Arial everywhere ----
    "font_family": "Arial",
    # If you want to force-load Arial from a specific .ttf (useful on clusters),
    # set try_load_font_from_file=True and point to font_paths below.
    "try_load_font_from_file": False,
    "font_paths": [
        # Examples (edit to your machine if needed):
        # "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        # "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    ],
    "set_mathtext_to_font_family": True,   # keep mathtext (if any) consistent
    "text_usetex": False,
    "axes_unicode_minus": False,
    "axes_formatter_use_mathtext": False,

    # ---- Matrix rendering ----
    "cmap": "viridis",
    "interpolation": "nearest",
    "aspect": "equal",                    # "equal" or "auto"

    # ---- Masking / special colors ----
    "mask_diagonal": True,                # if True, diagonal is forced to diag_color
    "mask_invalid": True,                 # if True, NaNs are masked -> bad color
    "diag_color": "white",                # diagonal color (masked)
    "nan_color": "white",                 # how NaNs render (same mechanism as diag via set_bad)

    # ---- Color range ----
    # Set vmin/vmax to None to auto from off-diagonals; or set manually (e.g., 0.0 / 1.0)
    "vmin": 0.019797,                     # None -> auto from off-diagonals
    "vmax": 0.686438,                     # None -> auto from off-diagonals
    "auto_range_use_offdiag_only": True,  # keep original behavior (ignore diagonal + NaNs)
    "auto_range_fallback": (0.0, 1.0),    # used if degenerate

    # ---- Axes visibility (defaults preserve original "plain matrix") ----
    "axis_off": True,                     # original: ax.set_axis_off()
    "show_title": False,
    "title": "",
    "title_fontsize": 7,
    "title_pad": 2.0,

    # If axis_off=False, you can control ticks/labels:
    "show_ticks": False,
    "x_ticks": None,                      # e.g., [0, 10, 20]
    "y_ticks": None,
    "x_ticklabels": None,                 # e.g., ["a","b",...]
    "y_ticklabels": None,
    "xtick_labelsize": 7,
    "ytick_labelsize": 7,
    "xtick_rotation": 0.0,
    "ytick_rotation": 0.0,
    "tick_length": 2.0,
    "tick_width": 0.6,
    "tick_pad": 2.0,

    # Convenience: if you want category labels directly (can get crowded),
    # set these True (only used when axis_off=False).
    "use_category_labels_on_ticks": False,
    "category_ticklabel_stride": 1,       # show every k-th category label
    "category_ticklabel_max": None,       # optionally cap number of shown labels

    # ---- Colorbar ----
    "show_colorbar": True,
    "cbar_orientation": "vertical",       # "vertical" or "horizontal"
    "cbar_fraction": 0.046,
    "cbar_pad": 0.04,
    "cbar_ticks": None,                   # None = auto; or list of tick values
    "cbar_ticklabels": None,              # optional custom strings aligned with cbar_ticks
    "cbar_ticklabelsize": 7,
    "cbar_tick_length": 2.0,
    "cbar_tick_width": 0.6,
    "cbar_tick_pad": 2.0,
    "cbar_outline": True,
    "cbar_label": None,                   # e.g. "alignment"
    "cbar_label_fontsize": 7,
    "cbar_label_pad": 6.0,
    "cbar_label_rotation": 90,            # only meaningful for vertical bars
    "cbar_format": None,                  # e.g., "{x:.2f}" or None for default formatter
}

# =========================
# ====== HELPERS ==========
# =========================

def load_ordered_entries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("JSON must be a list of {category, macro, sub_macro}.")
    cat2group = {e["category"]: (e["macro"], e["sub_macro"]) for e in entries}
    return entries, cat2group

def load_metric_dict(pkl_path, metric_key):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if metric_key not in data:
        dict_like = [k for k, v in data.items() if isinstance(v, dict)]
        raise KeyError(f"Metric '{metric_key}' not found. Dict-like keys: {dict_like}")
    return {k: float(v) for k, v in data[metric_key].items()}

def categories_from_results(metric_dict):
    cats = set()
    for k in metric_dict.keys():
        a, b = k.split("-", 1)
        cats.add(a); cats.add(b)
    return cats

def build_order(entries, metric_dict, append_extras=False):
    res_cats = categories_from_results(metric_dict)
    json_cats = [e["category"] for e in entries]
    # Keep JSON order but only categories present in results
    cats_ordered = [c for c in json_cats if c in res_cats]
    if append_extras:
        extras = [c for c in sorted(res_cats) if c not in json_cats]
        cats_ordered.extend(extras)
    return cats_ordered

def build_square_matrix(categories, metric_dict, diag_value=1.0, missing_fill=np.nan):
    n = len(categories)
    M = np.full((n, n), missing_fill, dtype=np.float64)
    np.fill_diagonal(M, diag_value)

    def lookup(a, b):
        k1 = f"{a}-{b}"
        if k1 in metric_dict: return metric_dict[k1]
        k2 = f"{b}-{a}"
        if k2 in metric_dict: return metric_dict[k2]
        return None

    for i, a in enumerate(categories):
        for j in range(i + 1, n):
            b = categories[j]
            v = lookup(a, b)
            if v is None:
                v = missing_fill
            M[i, j] = v
            M[j, i] = v
    return M

# Arial activation + FontProperties (explicit assignment if desired)
def _enable_arial():
    # Core rcParams
    rcParams["text.usetex"] = bool(PLOT_CFG["text_usetex"])
    rcParams["axes.unicode_minus"] = bool(PLOT_CFG["axes_unicode_minus"])
    rcParams["axes.formatter.use_mathtext"] = bool(PLOT_CFG["axes_formatter_use_mathtext"])

    # Optionally load Arial from file(s)
    chosen = None
    if PLOT_CFG["try_load_font_from_file"]:
        for p in PLOT_CFG["font_paths"]:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                if chosen is None:
                    chosen = p

    if chosen is not None:
        fam = fm.FontProperties(fname=chosen).get_name()
        fp = fm.FontProperties(fname=chosen)
    else:
        fam = PLOT_CFG["font_family"]
        fp = fm.FontProperties(family=fam)

    rcParams.update({
        "font.family": fam,
        "font.sans-serif": [fam, "DejaVu Sans", "Liberation Sans", "sans-serif"],
    })

    if PLOT_CFG["set_mathtext_to_font_family"]:
        # Keep mathtext visually consistent with Arial (no TeX).
        rcParams.update({
            "mathtext.default": "regular",
            "mathtext.fontset": "custom",
            "mathtext.rm": fam,
            "mathtext.it": f"{fam}:italic",
            "mathtext.bf": f"{fam}:bold",
            "mathtext.sf": fam,
        })

    return fp

def _apply_ticks(ax, categories):
    # Only used if axis_off=False and show_ticks=True
    if not PLOT_CFG["show_ticks"]:
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Choose ticks
    if PLOT_CFG["use_category_labels_on_ticks"]:
        n = len(categories)
        stride = max(int(PLOT_CFG["category_ticklabel_stride"]), 1)

        idx = np.arange(0, n, stride)
        if PLOT_CFG["category_ticklabel_max"] is not None:
            idx = idx[:int(PLOT_CFG["category_ticklabel_max"])]

        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels([categories[i] for i in idx])
        ax.set_yticklabels([categories[i] for i in idx])
    else:
        if PLOT_CFG["x_ticks"] is not None:
            ax.xaxis.set_major_locator(FixedLocator(PLOT_CFG["x_ticks"]))
        if PLOT_CFG["y_ticks"] is not None:
            ax.yaxis.set_major_locator(FixedLocator(PLOT_CFG["y_ticks"]))
        if PLOT_CFG["x_ticklabels"] is not None:
            ax.xaxis.set_major_formatter(FixedFormatter([str(s) for s in PLOT_CFG["x_ticklabels"]]))
        if PLOT_CFG["y_ticklabels"] is not None:
            ax.yaxis.set_major_formatter(FixedFormatter([str(s) for s in PLOT_CFG["y_ticklabels"]]))

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=PLOT_CFG["xtick_labelsize"],
        length=PLOT_CFG["tick_length"],
        width=PLOT_CFG["tick_width"],
        pad=PLOT_CFG["tick_pad"],
    )
    # Separate y labelsize if needed
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(PLOT_CFG["ytick_labelsize"])

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(float(PLOT_CFG["xtick_rotation"]))
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(float(PLOT_CFG["ytick_rotation"]))

def _apply_cbar_format(cbar):
    if PLOT_CFG["cbar_format"] is None:
        return
    fmt = PLOT_CFG["cbar_format"]

    def _f(x, pos=None):
        try:
            return fmt.format(x=x)
        except Exception:
            return str(x)

    cbar.formatter = FuncFormatter(_f)
    cbar.update_ticks()

# =========================
# ========= MAIN ==========
# =========================
def main():
    # Fonts
    arial_fp = _enable_arial()

    # Load data
    entries, _ = load_ordered_entries(JSON_PATH)
    metric_dict = load_metric_dict(RESULTS_PKL_PATH, METRIC_KEY)
    cats_ordered = build_order(entries, metric_dict, append_extras=APPEND_RESULTS_EXTRAS_AT_END)
    if not cats_ordered:
        print("No overlapping categories between JSON and results; nothing to plot.", file=sys.stderr)
        return

    # Build matrix
    M = build_square_matrix(
        cats_ordered,
        metric_dict,
        diag_value=DIAGONAL_VALUE,
        missing_fill=MISSING_ENTRY_FILL
    )

    # Build the array we will plot
    A_plot = M.copy()

    # Mask diagonal (by setting it to NaN and masking invalids)
    if PLOT_CFG["mask_diagonal"]:
        np.fill_diagonal(A_plot, np.nan)

    # Compute auto range (optionally off-diagonals only)
    if PLOT_CFG["auto_range_use_offdiag_only"]:
        A_for_range = A_plot.copy()
        if not PLOT_CFG["mask_diagonal"]:
            np.fill_diagonal(A_for_range, np.nan)
    else:
        A_for_range = A_plot

    data_min = np.nanmin(A_for_range)
    data_max = np.nanmax(A_for_range)
    print(f"[matrix stats] min={data_min:.6g}, max={data_max:.6g}")

    # Determine vmin/vmax
    vmin = data_min if PLOT_CFG["vmin"] is None else float(PLOT_CFG["vmin"])
    vmax = data_max if PLOT_CFG["vmax"] is None else float(PLOT_CFG["vmax"])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = map(float, PLOT_CFG["auto_range_fallback"])

    # Prepare masked array
    if PLOT_CFG["mask_invalid"]:
        A = np.ma.masked_invalid(A_plot)
    else:
        A = np.ma.array(A_plot, mask=np.zeros_like(A_plot, dtype=bool))

    cmap = plt.get_cmap(PLOT_CFG["cmap"]).copy()
    # One "bad" color handles both NaNs and masked diagonal
    # (if you want different diag vs NaN colors, you’d need a more complex mapping)
    cmap.set_bad(PLOT_CFG["diag_color"] if PLOT_CFG["mask_diagonal"] else PLOT_CFG["nan_color"])

    # Plot
    fig, ax = plt.subplots(
        figsize=PLOT_CFG["figsize"],
        dpi=PLOT_CFG["dpi"],
        constrained_layout=PLOT_CFG["constrained_layout"],
    )
    fig.patch.set_facecolor(PLOT_CFG["figure_facecolor"])
    ax.set_facecolor(PLOT_CFG["axes_facecolor"])

    im = ax.imshow(
        A,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation=PLOT_CFG["interpolation"],
        aspect=PLOT_CFG["aspect"]
    )

    # Axes / ticks / title control
    if PLOT_CFG["axis_off"]:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
        _apply_ticks(ax, cats_ordered)

    if PLOT_CFG["show_title"]:
        ax.set_title(PLOT_CFG["title"], fontsize=PLOT_CFG["title_fontsize"], pad=PLOT_CFG["title_pad"])

    # Colorbar
    if PLOT_CFG["show_colorbar"]:
        cbar = fig.colorbar(
            im, ax=ax,
            fraction=PLOT_CFG["cbar_fraction"],
            pad=PLOT_CFG["cbar_pad"],
            orientation=PLOT_CFG["cbar_orientation"],
        )

        if PLOT_CFG["cbar_ticks"] is not None:
            cbar.set_ticks(PLOT_CFG["cbar_ticks"])
            if PLOT_CFG["cbar_ticklabels"] is not None:
                cbar.set_ticklabels([str(s) for s in PLOT_CFG["cbar_ticklabels"]])

        _apply_cbar_format(cbar)

        cbar.ax.tick_params(
            labelsize=PLOT_CFG["cbar_ticklabelsize"],
            length=PLOT_CFG["cbar_tick_length"],
            width=PLOT_CFG["cbar_tick_width"],
            pad=PLOT_CFG["cbar_tick_pad"],
        )

        # Explicitly force Arial on colorbar tick labels
        for lbl in (cbar.ax.get_yticklabels() + cbar.ax.get_xticklabels()):
            lbl.set_fontproperties(arial_fp)

        # Explicitly force Arial on colorbar tick labels
        for lbl in (cbar.ax.get_yticklabels() + cbar.ax.get_xticklabels()):
            lbl.set_fontproperties(arial_fp)
            lbl.set_fontsize(PLOT_CFG["cbar_ticklabelsize"])  # <-- add this

        if PLOT_CFG["cbar_label"] is not None:
            cbar.set_label(
                PLOT_CFG["cbar_label"],
                fontsize=PLOT_CFG["cbar_label_fontsize"],
                labelpad=PLOT_CFG["cbar_label_pad"],
                rotation=PLOT_CFG["cbar_label_rotation"],
                fontproperties=arial_fp,
            )

        if not PLOT_CFG["cbar_outline"]:
            cbar.outline.set_visible(False)

    # Layout & Save
    if PLOT_CFG["tight_layout"]:
        fig.tight_layout()

    if PLOT_CFG["save_path"]:
        outdir = os.path.dirname(PLOT_CFG["save_path"])
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(
            PLOT_CFG["save_path"],
            bbox_inches=PLOT_CFG["bbox_inches"],
            transparent=PLOT_CFG["save_transparent"],
        )
        print(f"[OK] Saved: {PLOT_CFG['save_path']}")

    if PLOT_CFG["show_fig"]:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
