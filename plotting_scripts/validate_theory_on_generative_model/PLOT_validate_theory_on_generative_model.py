import gc
import hashlib
import json
import pickle
import warnings
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, font_manager as fm
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, LogLocator
from scipy.optimize import root_scalar


# ============================================================
# ======================== USER CONFIG ========================
# ============================================================

# Ordered list of theory-regression result files produced by the resampled-
# manifold theory script.
#
# IMPORTANT:
# The script processes these files sequentially.
#
# The first file plays a special role:
#   - its internal subset runs provide the points from small P up to the full
#     single-file category count;
#   - then the second file is loaded and merged into the running accumulators to
#     create the 2*P full-category point;
#   - then the third file is loaded and merged to create the 3*P point; etc.
THEORY_RESULTS_FILES = [
    "PATH/TO/theory_results_seed1.pkl",
    "PATH/TO/theory_results_seed2.pkl",
    "PATH/TO/theory_results_seed3.pkl",
    "PATH/TO/theory_results_seed4.pkl",
    "PATH/TO/theory_results_seed5.pkl",
    "PATH/TO/theory_results_seed6.pkl",
    "PATH/TO/theory_results_seed7.pkl",
]

# Bounding-box feature to plot. If None, the script infers it from the first
# file and checks compatibility as additional files are loaded.
FEATURE = None
# FEATURE = "bbox_center_x"
# FEATURE = "bbox_center_y"
# FEATURE = "bbox_x_length"
# FEATURE = "bbox_y_length"

# Which subset sizes P to plot from the FIRST file only.
# If None, the script uses all subset sizes present in the first file.
P_LIST = None

# Which independent within-file subsampling realizations to use for the subset
# points taken from the FIRST file.
# - None  -> use all subset seeds found in the first file
# - [1]   -> use only seed 1
# - [1,2] -> use seeds 1 and 2 and show error bars across them
SELECTED_SUBSAMPLING_SEEDS = [1]

# Whether to include the full-category point of the first file.
INCLUDE_SINGLE_FILE_FULL_POINT = True

# Whether to include the cumulative merged full-category points obtained from
# the ordered prefixes of the file list.
#
# Example:
#   files = [f1, f2, f3, f4]
#   k = 2 -> merge f1 + f2
#   k = 3 -> merge f1 + f2 + f3
#   k = 4 -> merge f1 + f2 + f3 + f4
INCLUDE_MERGED_FULL_POINTS = True

# Which cumulative prefix lengths k to include.
# - None    -> use k = 2, 3, ..., n_files
# - [2, 4]  -> compute only the points from files[0:2] and files[0:4]
MERGED_FULL_FILE_COUNTS = None

# Error-bar mode across realizations contributing to one plotted point.
#
# For this streaming prefix-based script, error bars can appear for the within-
# file subset points because the first file may contain several subset seeds.
# The cumulative merged full-category points are single realizations, so their
# error bars are zero.
REALIZATION_ERRORBAR_MODE = "std"   # "std", "sem", "none"

# ------------------------------------------------------------
# Cache control.
# ------------------------------------------------------------
ENABLE_CACHE = True
FORCE_RECOMPUTE = False

# If None, the cache directory is created next to the first input pkl file.
CACHE_DIR = None
CACHE_PREFIX = "resampled_multi_curve_cache"

# Output figure path. If None, the figure is saved next to the first input pkl.
OUTPUT_FIGURE = None
OUTPUT_FORMAT = "svg"

# Used only for titles and filenames. The script is otherwise model-agnostic.
MODEL_NAME = "C"

# ============================================================
# ===================== PLOT CONFIG (STYLE) ===================
# ============================================================
# Plot configuration for the resampled-manifold normalized-gap plot.
PLOT_CFG = {
    # ---- Global behavior ----
    "show_figures": True,
    "close_after_save": False,
    "raise_on_nonpositive_log": False,

    # ---- Fonts ----
    "font": {
        "family": "Arial",
        "default_size": 7,
        "usetex": False,
        "axes_unicode_minus": False,
        "try_load_from_file": False,
        "font_dir": "/usr/share/fonts",
        "font_candidates": ["Arial.ttf", "ArialMT.ttf", "arial.ttf"],
        "font_bold_candidates": ["Arial Bold.ttf", "Arial-Bold.ttf", "ArialMT-Bold.ttf"],
        "font_italic_candidates": ["Arial Italic.ttf", "Arial-Italic.ttf", "ArialMT-Italic.ttf"],
        "set_mathtext_to_family": True,
    },

    # ---- Figure creation ----
    "figure": {
        "figsize": (4.2, 2.5),
        "dpi": 300,
        "constrained_layout": False,
        "tight_layout": True,
        "tight_layout_rect": None,
    },

    # ---- Saving ----
    "save": {
        "enabled": True,
        "path_template": "{model_name}_{metric_key}_{feature}.{ext}",
        "formats": ["svg"],
        "transparent": True,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
    },

    # ---- Suptitle ----
    "suptitle": {
        "show": False,
        "text_template": "{model_name} — {metric_key} — {feature}",
        "fontsize": 7,
        "y": 0.98,
    },

    # ---- X axis ----
    "xaxis": {
        "label": "P",
        "label_fontsize": 7,
        "scale": "log",
        "xlim": (7, 2500),
        "ticks": None,
        "ticklabels": None,
        "num_ticks": None,
        "tick_rotation": 0,
        "tick_ha": "center",
        "tick_fontsize": 7,
        "tick_pad": 2.0,
        "log_locator": {
            "base": 10.0,
            "subs": (1.0,),
        },
    },

    # ---- Y axis ----
    "yaxis": {
        "normalized_gap": {
            "title": "local-global gap",
            "ylabel": r"$\Delta E/\hat{\sigma}_{y}^2$",
            "yscale": "log",
            "ylim": None,
            "yticks": None,
            "y_num_ticks": None,
            "symlog_linthresh_mode": "fraction_of_median_abs",
            "symlog_linthresh_fixed": 1e-3,
            "symlog_linthresh_fraction": 0.10,
        },
        "_shared": {
            "title_fontsize": 7,
            "label_fontsize": 7,
            "tick_fontsize": 7,
            "tick_pad": 2.0,
        },
    },

    # ---- Curves ----
    "curves": {
        "measured": {"label": "emp", "color": "tab:blue", "marker": "o", "linestyle": "-"},
        "theory_finite": {"label": "theo (fin. P)", "color": "tab:green", "marker": "x", "linestyle": "--"},
        "theory_infinite": {
            "label": "theo (inf. P)",
            "color": "tab:red",
            "marker": "",
            "linestyle": "--",
        },
    },
    "curve_defaults": {
        "connect_points": True,
        "alpha": 0.95,
        "linewidth": 0.9,
        "marker": "o",
        "markersize": 3.0,
        "elinewidth": 0.6,
        "capsize": 1.6,
        "capthick": 0.6,
        "ecolor": None,
        "zorder_marker": 2.0,
        "zorder_error": 3.5,
        "hide_if_all_zero_err": True,
    },

    # ---- Grid & spines ----
    "grid": {
        "show": True,
        "which": "both",
        "axis": "both",
        "alpha": 0.5,
        "linestyle": ":",
        "linewidth": 0.5,
    },
    "spines": {
        "top": False,
        "right": False,
        "left": True,
        "bottom": True,
    },

    # ---- Legend ----
    "legend": {
        "show": True,
        "loc": "best",
        "fontsize": 7,
        "frameon": False,
        "ncol": 1,
        "handlelength": 1.2,
        "handletextpad": 0.4,
        "labelspacing": 0.2,
        "borderaxespad": 0.2,
        "columnspacing": 0.8,
        "markerscale": 1.0,
    },
}



# ============================================================
# ======================== PLOT HELPERS =======================
# ============================================================
def _add_font_if_exists(path: str) -> bool:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return True
    return False


def apply_plot_style(cfg: dict) -> str:
    """
    Apply the matplotlib configuration.
    """
    fcfg = cfg["font"]
    rcParams["text.usetex"] = bool(fcfg["usetex"])
    rcParams["axes.unicode_minus"] = bool(fcfg["axes_unicode_minus"])

    family_name = fcfg["family"]
    if fcfg.get("try_load_from_file", False):
        font_dir = fcfg.get("font_dir", "")
        regular_path = None
        for fn in fcfg.get("font_candidates", []):
            p = os.path.join(font_dir, fn)
            if _add_font_if_exists(p):
                regular_path = p
                break
        for fn in fcfg.get("font_bold_candidates", []) + fcfg.get("font_italic_candidates", []):
            _add_font_if_exists(os.path.join(font_dir, fn))
        if regular_path is not None:
            family_name = fm.FontProperties(fname=regular_path).get_name()

    base = int(fcfg["default_size"])
    rcParams.update({
        "font.family": family_name,
        "font.sans-serif": [family_name, "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base,
    })

    if fcfg.get("set_mathtext_to_family", True):
        rcParams.update({
            "mathtext.default": "regular",
            "mathtext.fontset": "custom",
            "mathtext.rm": family_name,
            "mathtext.it": f"{family_name}:italic",
            "mathtext.bf": f"{family_name}:bold",
            "mathtext.sf": family_name,
        })

    return family_name


def _merge_config(base_cfg: dict, override: dict | None):
    """
    Shallow-then-nested merge for plotting configuration dictionaries.
    """
    if override is None:
        return base_cfg

    cfg = dict(base_cfg)
    cfg.update(override)

    for k in ["font", "figure", "save", "suptitle", "xaxis", "grid", "spines",
              "legend", "curves", "curve_defaults", "yaxis"]:
        if k in override and isinstance(override[k], dict):
            tmp = dict(base_cfg.get(k, {}))
            tmp.update(override[k])
            cfg[k] = tmp

    if "yaxis" in override:
        yb = dict(base_cfg.get("yaxis", {}))
        for mk, md in override["yaxis"].items():
            if isinstance(md, dict) and mk in yb and isinstance(yb[mk], dict):
                t2 = dict(yb[mk])
                t2.update(md)
                yb[mk] = t2
            else:
                yb[mk] = md
        cfg["yaxis"] = yb

    return cfg


def _maybe_check_log_safe(cfg, y, yscale_name: str):
    if yscale_name != "log":
        return

    y = np.asarray(y, float)
    if np.any(~np.isfinite(y)):
        return

    if np.any(y <= 0):
        msg = "[plot] yscale='log' requested but some values are <= 0."
        if cfg.get("raise_on_nonpositive_log", False):
            raise ValueError(msg)
        print("WARNING:", msg, "Falling back to symlog.", flush=True)


def _apply_symlog(ax, ycfg, vals):
    absvals = np.abs(vals[np.isfinite(vals)])
    mode = ycfg.get("symlog_linthresh_mode", "fixed")

    if mode == "fixed":
        linthresh = float(ycfg.get("symlog_linthresh_fixed", 1e-3))
    elif mode == "fraction_of_median_abs":
        fixed = float(ycfg.get("symlog_linthresh_fixed", 1e-3))
        frac = float(ycfg.get("symlog_linthresh_fraction", 0.1))
        linthresh = fixed if absvals.size == 0 else max(frac * float(np.median(absvals)), fixed)
    else:
        raise ValueError(f"Unknown symlog_linthresh_mode: {mode}")

    ax.set_yscale("symlog", linthresh=linthresh)


def _set_y_scale_with_fallback(ax, cfg, ycfg, yvals):
    yscale = ycfg.get("yscale", "linear")
    yvals = np.asarray(yvals, float)
    yvals = yvals[np.isfinite(yvals)]

    if yscale == "linear":
        ax.set_yscale("linear")
        return

    if yscale == "symlog":
        _apply_symlog(ax, ycfg, yvals)
        return

    if yscale == "log":
        if yvals.size and np.any(yvals <= 0):
            _maybe_check_log_safe(cfg, yvals, "log")
            _apply_symlog(ax, ycfg, yvals)
        else:
            ax.set_yscale("log")
        return

    raise ValueError(f"Unknown yscale: {yscale}")


def _maybe_hide_zero_err(yerr, hide_if_all_zero: bool):
    if yerr is None:
        return None
    yerr = np.asarray(yerr, float)
    if hide_if_all_zero and np.all(yerr == 0):
        return None
    return yerr

# ============================================================
# ===================== THEORY HELPER CODE ====================
# ============================================================


def _as_feature_value(field, feature):
    """
    Some stored quantities are plain arrays/scalars, others are dicts keyed by
    feature. This helper makes the readout uniform.
    """
    if isinstance(field, dict):
        return field[feature]
    return field



def infer_feature(data_dict):
    """
    Infer the selected bbox feature from one loaded theory file.
    """
    args = data_dict.get("args", {})
    if args.get("selected_feature") is not None:
        return args["selected_feature"]

    labels_var = data_dict["global"].get("linearized_labels_variance", None)
    if isinstance(labels_var, dict):
        keys = list(labels_var.keys())
        if len(keys) == 1:
            return keys[0]

    raise ValueError("Could not infer the feature automatically. Please set FEATURE explicitly.")



def self_consistent_equation(oparam, var_d_mu, var_1_mu, sigma_1d_mu, c, n_neurons, n_categories):
    """
    Same self-consistent equation used in the original plotting utility.
    """
    alpha = n_categories / n_neurons

    u_d_mu = var_d_mu + oparam
    u_1_mu = var_1_mu + (oparam / (1.0 - c ** 2))
    s_d_mu = (sigma_1d_mu ** 2) / u_d_mu

    u_d_mu_inv_sum = np.sum(1.0 / u_d_mu, axis=1)
    s_d_mu_sum = np.sum(s_d_mu, axis=1)
    u_1_mu_corrected = u_1_mu - s_d_mu_sum

    term_d = np.mean(u_d_mu_inv_sum) / n_neurons
    term_1 = np.mean(1.0 / u_1_mu_corrected) / (n_neurons * (1.0 - c ** 2))
    term_correction = np.mean(np.sum((sigma_1d_mu / u_d_mu) ** 2, axis=1) / u_1_mu_corrected) / n_neurons

    return oparam * (term_1 + term_d + term_correction) - 1.0 + 1.0 / (n_neurons * alpha)



def solve_self_consistent_equation(var_d_mu, var_1_mu, sigma_1d_mu, c, n_neurons, n_categories,
                                   lower_bound=1e-6, upper_bound=1e8):
    """
    Solve the order-parameter equation entering the finite-P theory.
    """

    def f(oparam):
        return self_consistent_equation(oparam, var_d_mu, var_1_mu, sigma_1d_mu, c, n_neurons, n_categories)

    result = root_scalar(f, bracket=(lower_bound, upper_bound), method="brentq")
    if not result.converged:
        raise RuntimeError("Root finding for the finite-P theory did not converge.")
    return float(result.root)



def _get_fluctuation_global_fields(global_results, feature):
    """
    Read the centered fluctuation blocks from one global theory block.
    """
    if "covariance_fluctuations" in global_results:
        covariance = np.asarray(_as_feature_value(global_results["covariance_fluctuations"], feature), dtype=np.float64)
    else:
        covariance = np.asarray(_as_feature_value(global_results["covariance"], feature), dtype=np.float64)

    if "io_covariance_linearized_fluctuations" in global_results:
        io_covariance = np.asarray(
            _as_feature_value(global_results["io_covariance_linearized_fluctuations"], feature), dtype=np.float64
        )
    else:
        io_covariance = np.asarray(_as_feature_value(global_results["io_covariance_linearized"], feature), dtype=np.float64)

    if "linearized_labels_variance_fluctuations" in global_results:
        linearized_labels_variance = float(
            _as_feature_value(global_results["linearized_labels_variance_fluctuations"], feature)
        )
    else:
        linearized_labels_variance = float(_as_feature_value(global_results["linearized_labels_variance"], feature))

    return covariance, io_covariance, linearized_labels_variance



def compute_empirical_gap_from_global_block(global_results, feature):
    """
    Compute the fluctuation regression-error gap directly from one global block:

        gap = var_y_lin - d^T C^+ d

    using the centered fluctuation covariance, the centered fluctuation
    input-output covariance for the linearized labels, and the corresponding
    linearized-label variance.

    The normalized quantity plotted on the y-axis is gap / var_y_lin.
    """
    covariance, io_covariance, normalization = _get_fluctuation_global_fields(global_results, feature)

    if normalization <= 0:
        raise ValueError(
            f"The linearized fluctuation-label variance for feature '{feature}' is non-positive: {normalization}"
        )

    covariance_pinv = np.linalg.pinv(covariance)
    regression_explained = float(io_covariance @ covariance_pinv @ io_covariance)
    raw_gap = float(normalization - regression_explained)
    normalized_gap = raw_gap / normalization

    return {
        "normalization": normalization,
        "empirical_raw": raw_gap,
        "empirical": normalized_gap,
    }



def initialize_running_local_accumulator():
    """
    Create the lightweight local accumulator used for theory.

    We keep only the category-wise local quantities actually needed by the
    finite-P and infinite-category theory. This is much lighter than keeping the
    full theory files in memory, and avoids storing the large global covariance
    matrices from multiple files simultaneously.
    """
    return {
        "r_mu": [],
        "u1_mu": [],
        "var1_mu": [],
        "vartot_mu": [],
        "var_d_mu": [],
        "sigma_1d_mu": [],
    }



def append_full_file_local_quantities(local_accumulator, local_results, categories, feature):
    """
    Append the per-category local quantities from one full theory file into the
    running local accumulator.
    """
    for category in categories:
        w_mu = np.asarray(local_results["regression_vector"][category][feature], dtype=np.float64)
        r_mu = float(np.linalg.norm(w_mu))
        if r_mu <= 0:
            raise ValueError(f"Category '{category}' has zero local regression-vector norm.")

        u1_mu = w_mu / r_mu
        local_accumulator["r_mu"].append(r_mu)
        local_accumulator["u1_mu"].append(u1_mu)
        local_accumulator["var1_mu"].append(float(local_results["variance_along_linear_rule"][category][feature]))
        local_accumulator["vartot_mu"].append(float(_as_feature_value(local_results["total_variance"][category], feature)))
        local_accumulator["var_d_mu"].append(
            np.asarray(local_results["variances_orthogonal"][category][feature], dtype=np.float64)
        )
        local_accumulator["sigma_1d_mu"].append(
            np.asarray(local_results["covariance_cross_terms"][category][feature], dtype=np.float64)
        )



def extract_local_arrays_for_subset(local_results, categories_subset, feature):
    """
    Build the local theory arrays for one subset of categories.

    This is used only for the small-P points coming from the subset runs stored
    inside the FIRST file.
    """
    local_accumulator = initialize_running_local_accumulator()
    append_full_file_local_quantities(local_accumulator, local_results, categories_subset, feature)
    return finalize_local_accumulator(local_accumulator)



def finalize_local_accumulator(local_accumulator):
    """
    Convert the running local accumulator from Python lists to NumPy arrays.
    """
    if len(local_accumulator["r_mu"]) == 0:
        raise ValueError("The local accumulator is empty.")

    return {
        "r_mu": np.asarray(local_accumulator["r_mu"], dtype=np.float64),
        "u1_mu": np.asarray(local_accumulator["u1_mu"], dtype=np.float64),
        "var1_mu": np.asarray(local_accumulator["var1_mu"], dtype=np.float64),
        "vartot_mu": np.asarray(local_accumulator["vartot_mu"], dtype=np.float64),
        "var_d_mu": np.asarray(local_accumulator["var_d_mu"], dtype=np.float64),
        "sigma_1d_mu": np.asarray(local_accumulator["sigma_1d_mu"], dtype=np.float64),
    }



def compute_theory_quantities_from_local_arrays(local_arrays, normalization, feature):
    """
    Compute the finite-P and infinite-category theory from the local arrays
    currently stored in memory.
    """
    del feature  # kept in the signature for readability and future edits

    if normalization <= 0:
        raise ValueError("The normalization is non-positive.")

    r_mu = np.asarray(local_arrays["r_mu"], dtype=np.float64)
    u1_mu = np.asarray(local_arrays["u1_mu"], dtype=np.float64)
    var1_mu = np.asarray(local_arrays["var1_mu"], dtype=np.float64)
    vartot_mu = np.asarray(local_arrays["vartot_mu"], dtype=np.float64)
    var_d_mu = np.asarray(local_arrays["var_d_mu"], dtype=np.float64)
    sigma_1d_mu = np.asarray(local_arrays["sigma_1d_mu"], dtype=np.float64)

    mean_r = float(np.mean(r_mu))
    var_r = float(np.var(r_mu))
    mean_var1 = float(np.mean(var1_mu))
    mean_vartot = float(np.mean(vartot_mu))

    mean_direction = np.mean(u1_mu, axis=0)
    c = float(np.linalg.norm(mean_direction))
    if c <= 0:
        raise ValueError("The mean local regression direction has zero norm, so c is zero.")

    n_categories = int(r_mu.shape[0])
    n_neurons = int(u1_mu.shape[1])

    scale_contribution = var_r / (mean_r ** 2)
    orientation_contribution = (mean_vartot / mean_var1) * ((1.0 - c ** 2) / (c ** 2)) / n_neurons
    theory_infinite_normalized = 1.0 - 1.0 / ((1.0 + scale_contribution) * (1.0 + orientation_contribution))

    try:
        oparam = solve_self_consistent_equation(var_d_mu, var1_mu, sigma_1d_mu, c, n_neurons, n_categories)

        u_d_mu = var_d_mu + oparam
        u_1_mu = var1_mu + (oparam / (1.0 - c ** 2))
        s_d_mu = (sigma_1d_mu ** 2) / u_d_mu

        u_d_mu_inv_sum = np.sum(1.0 / u_d_mu, axis=1)
        s_d_mu_sum = np.sum(s_d_mu, axis=1)
        u_1_mu_corrected = u_1_mu - s_d_mu_sum
        var1_mu_corrected = var1_mu - s_d_mu_sum
        last_factor_mu = (var1_mu_corrected ** 2) / u_1_mu_corrected

        mean_r_var1 = float(np.mean(r_mu * var1_mu))
        mean_r2_var1 = float(np.mean((r_mu ** 2) * var1_mu))

        a_coeff = (
            + mean_var1 * (c ** 2)
            + oparam / n_categories
            - ((n_neurons - 1) * (c ** 2) * oparam) / n_neurons
            - (c ** 2) * np.mean(s_d_mu_sum)
            + (c ** 2) * (oparam ** 2) * np.mean(u_d_mu_inv_sum) / n_neurons
            - (c ** 2) * np.mean(last_factor_mu)
        )

        b_coeff = (
            - 2.0 * c * mean_r_var1
            + 2.0 * c * np.mean(s_d_mu_sum * r_mu)
            + 2.0 * c * np.mean(last_factor_mu * r_mu)
        )

        c_coeff = (
            + mean_r2_var1
            - np.mean(s_d_mu_sum * (r_mu ** 2))
            - np.mean(last_factor_mu * (r_mu ** 2))
        )

        raw_theory_finite = float(c_coeff - (b_coeff ** 2) / (4.0 * a_coeff))
        theory_finite_normalized = raw_theory_finite / normalization
    except Exception as exc:
        warnings.warn(
            f"Finite-P theory failed for feature '{feature}' and a collection of {n_categories} categories. "
            f"Returning NaN. Original error: {exc}",
            stacklevel=1,
        )
        raw_theory_finite = np.nan
        theory_finite_normalized = np.nan

    return {
        "theory_finite_raw": raw_theory_finite,
        "theory_finite": theory_finite_normalized,
        "theory_infinite": theory_infinite_normalized,
    }



def compute_gap_quantities_from_local_arrays_and_global(local_arrays, global_results, feature):
    """
    Compute the empirical gap plus the finite/infinite theory from:
      - one global fluctuation block,
      - one local theory-array collection describing the same manifolds.
    """
    empirical_stats = compute_empirical_gap_from_global_block(global_results, feature)
    theory_stats = compute_theory_quantities_from_local_arrays(
        local_arrays=local_arrays,
        normalization=empirical_stats["normalization"],
        feature=feature,
    )

    return {
        "normalization": empirical_stats["normalization"],
        "empirical_raw": empirical_stats["empirical_raw"],
        "empirical": empirical_stats["empirical"],
        "theory_finite_raw": theory_stats["theory_finite_raw"],
        "theory_finite": theory_stats["theory_finite"],
        "theory_infinite": theory_stats["theory_infinite"],
    }


# ============================================================
# ===================== MERGING UTILITIES =====================
# ============================================================


def total_image_count_for_categories(local_results, categories):
    """
    Total number of examples behind one chosen set of categories.
    """
    return int(sum(int(local_results["n_images"][category]) for category in categories))



def initialize_running_global_sums():
    """
    Create the running sums needed to merge full-file global fluctuation blocks
    across the ordered prefix of files.
    """
    return {
        "total_weight": 0.0,
        "covariance_sum": None,
        "io_covariance_sum": None,
        "linearized_var_sum": 0.0,
    }



def update_running_global_sums(running_sums, global_results, feature, weight):
    """
    Update the cumulative weighted sums with one additional full-file global
    block.
    """
    covariance, io_covariance, linearized_var = _get_fluctuation_global_fields(global_results, feature)
    covariance = np.asarray(covariance, dtype=np.float64)
    io_covariance = np.asarray(io_covariance, dtype=np.float64)
    linearized_var = float(linearized_var)
    weight = float(weight)

    if running_sums["covariance_sum"] is None:
        running_sums["covariance_sum"] = weight * covariance
        running_sums["io_covariance_sum"] = weight * io_covariance
    else:
        running_sums["covariance_sum"] += weight * covariance
        running_sums["io_covariance_sum"] += weight * io_covariance

    running_sums["linearized_var_sum"] += weight * linearized_var
    running_sums["total_weight"] += weight



def build_global_block_from_running_sums(running_sums, feature):
    """
    Convert the running weighted sums into one merged global block.
    """
    total_weight = float(running_sums["total_weight"])
    if total_weight <= 0:
        raise ValueError("The total weight in the running global sums is non-positive.")

    return {
        "covariance_fluctuations": {feature: running_sums["covariance_sum"] / total_weight},
        "io_covariance_linearized_fluctuations": {feature: running_sums["io_covariance_sum"] / total_weight},
        "linearized_labels_variance_fluctuations": {feature: running_sums["linearized_var_sum"] / total_weight},
    }



def collect_subset_runs_from_loaded_data(data_dict):
    """
    Extract the subset-run records from one loaded theory file.
    """
    subset_runs = data_dict.get("subsampled", {}).get("runs", {})
    runs = []
    for _, run_data in subset_runs.items():
        runs.append(
            {
                "size": int(run_data["size"]),
                "seed": int(run_data["seed"]),
                "categories": list(run_data["categories"]),
                "global": run_data["global"],
            }
        )
    return runs


# ============================================================
# ============================ CACHE ==========================
# ============================================================


def make_cache_config_signature(feature):
    """
    Build a stable dictionary containing exactly the settings that affect the
    computed curve points.

    Plot cosmetics are intentionally left out so that the same cached points can
    be replotted with different visual settings.
    """
    resolved_paths = [str(Path(path).expanduser().resolve()) for path in THEORY_RESULTS_FILES]
    mtimes = []
    sizes = []
    for path in resolved_paths:
        stat = Path(path).stat()
        mtimes.append(int(stat.st_mtime_ns))
        sizes.append(int(stat.st_size))

    return {
        "feature": feature,
        "theory_results_files": resolved_paths,
        "theory_results_files_mtime_ns": mtimes,
        "theory_results_files_size": sizes,
        "p_list": None if P_LIST is None else [int(value) for value in P_LIST],
        "selected_subsampling_seeds": None if SELECTED_SUBSAMPLING_SEEDS is None else [int(value) for value in SELECTED_SUBSAMPLING_SEEDS],
        "include_single_file_full_point": bool(INCLUDE_SINGLE_FILE_FULL_POINT),
        "include_merged_full_points": bool(INCLUDE_MERGED_FULL_POINTS),
        "merged_full_file_counts": None if MERGED_FULL_FILE_COUNTS is None else [int(value) for value in MERGED_FULL_FILE_COUNTS],
        "realization_errorbar_mode": REALIZATION_ERRORBAR_MODE,
    }



def get_cache_dir():
    """
    Resolve the directory used to store cached curve points.
    """
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR).expanduser().resolve()
    else:
        cache_dir = Path(THEORY_RESULTS_FILES[0]).expanduser().resolve().parent / "cache_resampled_multi_plot"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir



def get_cache_path(feature):
    """
    Use a hash of the compatibility-defining inputs so that one cache file
    corresponds to one unique curve-computation setup.
    """
    signature = make_cache_config_signature(feature)
    signature_json = json.dumps(signature, sort_keys=True)
    signature_hash = hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:16]
    cache_path = get_cache_dir() / f"{CACHE_PREFIX}_{feature}_{signature_hash}.pkl"
    return cache_path, signature



def try_load_curve_data_from_cache(feature):
    """
    Try to load previously computed curve points from disk.
    """
    if not ENABLE_CACHE or FORCE_RECOMPUTE:
        return None, None

    cache_path, signature = get_cache_path(feature)
    if not cache_path.exists():
        print(f"Cache miss: no cache file found at {cache_path}", flush=True)
        return None, cache_path

    print(f"Checking cache file: {cache_path}", flush=True)
    with open(cache_path, "rb") as file_handle:
        cached_payload = pickle.load(file_handle)

    if cached_payload.get("signature", None) != signature:
        print("Cache miss: cache signature does not match the current inputs.", flush=True)
        return None, cache_path

    print(f"Cache hit: loading precomputed curve points from {cache_path}", flush=True)
    return cached_payload["curve_data"], cache_path



def save_curve_data_to_cache(curve_data, feature, cache_path=None):
    """
    Save the computed curve points to disk for future reuse.
    """
    if not ENABLE_CACHE:
        return None

    if cache_path is None:
        cache_path, signature = get_cache_path(feature)
    else:
        _, signature = get_cache_path(feature)

    payload = {
        "signature": signature,
        "curve_data": curve_data,
    }

    with open(cache_path, "wb") as file_handle:
        pickle.dump(payload, file_handle)

    print(f"Saved curve-point cache to: {cache_path}", flush=True)
    return cache_path


# ============================================================
# ==================== PLOTTING UTILITIES =====================
# ============================================================


def aggregate_across_realizations(values, mode):
    """
    Aggregate one list of realization values into a mean and an error bar.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan

    mean_value = float(np.mean(values))
    if values.size == 1 or mode == "none":
        return mean_value, 0.0
    if mode == "std":
        return mean_value, float(np.std(values, ddof=1))
    if mode == "sem":
        return mean_value, float(np.std(values, ddof=1) / np.sqrt(values.size))
    raise ValueError(f"Unknown REALIZATION_ERRORBAR_MODE: {mode}")



def make_default_output_path(theory_files, feature, output_format, plot_cfg=None):
    if plot_cfg is None:
        plot_cfg = PLOT_CFG
    metric_key = "normalized_gap"
    save_cfg = plot_cfg["save"]
    filename = save_cfg["path_template"].format(
        model_name=MODEL_NAME,
        feature=feature,
        metric_key=metric_key,
        ext=output_format,
    )
    return Path(theory_files[0]).expanduser().resolve().parent / filename



def plot_curve_bundle(curve_data, output_figure=None, output_format="svg", config_override=None):
    """
    Plot the empirical curve, the finite-P theory curve, and the horizontal
    infinite-categories line computed from the cumulative merge of all files.
    """
    cfg = _merge_config(PLOT_CFG, config_override)
    apply_plot_style(cfg)

    metric_key = "normalized_gap"
    ycfg = cfg["yaxis"][metric_key]
    shared = cfg["yaxis"].get("_shared", {})
    xcfg = cfg["xaxis"]
    cdef = cfg["curve_defaults"]

    feature = curve_data["feature"]
    results_per_p = curve_data["results_per_p"]
    infinite_line_value = curve_data["infinite_line_value"]

    p_sorted = sorted(results_per_p.keys())
    empirical_mean = np.array([results_per_p[p]["empirical_mean"] for p in p_sorted], dtype=np.float64)
    empirical_err = np.array([results_per_p[p]["empirical_err"] for p in p_sorted], dtype=np.float64)
    theory_finite_mean = np.array([results_per_p[p]["theory_finite_mean"] for p in p_sorted], dtype=np.float64)
    theory_finite_err = np.array([results_per_p[p]["theory_finite_err"] for p in p_sorted], dtype=np.float64)
    theory_infinite_curve = np.full(len(p_sorted), float(infinite_line_value), dtype=np.float64)
    theory_infinite_err = np.zeros(len(p_sorted), dtype=np.float64)

    empirical_err = _maybe_hide_zero_err(empirical_err, bool(cdef.get("hide_if_all_zero_err", False)))
    theory_finite_err = _maybe_hide_zero_err(theory_finite_err, bool(cdef.get("hide_if_all_zero_err", True)))
    theory_infinite_err = _maybe_hide_zero_err(theory_infinite_err, bool(cdef.get("hide_if_all_zero_err", True)))

    fig, ax = plt.subplots(
        figsize=cfg["figure"]["figsize"],
        dpi=cfg["figure"]["dpi"],
        constrained_layout=cfg["figure"].get("constrained_layout", False),
    )

    if cfg["suptitle"].get("show", False):
        suptitle = cfg["suptitle"]["text_template"].format(
            model_name=MODEL_NAME,
            metric_key=metric_key,
            feature=feature,
        )
        fig.suptitle(suptitle, fontsize=cfg["suptitle"]["fontsize"], y=cfg["suptitle"].get("y", 0.98))

    ax.set_title(ycfg["title"], fontsize=shared.get("title_fontsize", cfg["font"]["default_size"]))
    ax.set_xlabel(xcfg["label"], fontsize=xcfg.get("label_fontsize", cfg["font"]["default_size"]))
    ax.set_ylabel(ycfg["ylabel"], fontsize=shared.get("label_fontsize", cfg["font"]["default_size"]))

    ax.set_xscale(xcfg.get("scale", "linear"))
    if xcfg.get("scale", "linear") == "log" and xcfg.get("log_locator", None):
        locator_cfg = xcfg["log_locator"]
        ax.xaxis.set_major_locator(LogLocator(base=locator_cfg.get("base", 10.0), subs=locator_cfg.get("subs", (1.0,))))

    all_values = np.concatenate([empirical_mean, theory_finite_mean, theory_infinite_curve])
    _set_y_scale_with_fallback(ax, cfg, ycfg, all_values)

    if xcfg.get("xlim", None) is not None:
        ax.set_xlim(*xcfg["xlim"])
    if ycfg.get("ylim", None) is not None:
        ax.set_ylim(*ycfg["ylim"])

    if xcfg.get("ticks", None) is not None:
        ax.xaxis.set_major_locator(FixedLocator(xcfg["ticks"]))
        if xcfg.get("ticklabels", None) is not None:
            ax.xaxis.set_major_formatter(FixedFormatter([str(tick) for tick in xcfg["ticklabels"]]))
    elif xcfg.get("num_ticks", None) is not None:
        ax.xaxis.set_major_locator(MaxNLocator(xcfg["num_ticks"]))

    ax.tick_params(axis="x", labelsize=xcfg.get("tick_fontsize", cfg["font"]["default_size"]), pad=xcfg.get("tick_pad", 2.0))
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_rotation(xcfg.get("tick_rotation", 0))
        ticklabel.set_ha(xcfg.get("tick_ha", "center"))

    ax.tick_params(axis="y", labelsize=shared.get("tick_fontsize", cfg["font"]["default_size"]), pad=shared.get("tick_pad", 2.0))
    if ycfg.get("yticks", None) is not None:
        ax.yaxis.set_major_locator(FixedLocator(ycfg["yticks"]))
        ax.yaxis.set_major_formatter(FixedFormatter([str(tick) for tick in ycfg["yticks"]]))
    elif ycfg.get("y_num_ticks", None) is not None:
        ax.yaxis.set_major_locator(MaxNLocator(ycfg["y_num_ticks"]))

    if cfg["grid"].get("show", True):
        ax.grid(
            True,
            which=cfg["grid"].get("which", "major"),
            axis=cfg["grid"].get("axis", "both"),
            alpha=cfg["grid"].get("alpha", 0.5),
            linestyle=cfg["grid"].get("linestyle", ":"),
            linewidth=cfg["grid"].get("linewidth", 0.5),
        )
    else:
        ax.grid(False)

    for spine_name, visible in cfg["spines"].items():
        if spine_name in ax.spines:
            ax.spines[spine_name].set_visible(bool(visible))

    def _resolve_style(curve_key):
        style = dict(cdef)
        style.update(cfg["curves"].get(curve_key, {}))
        if not style.get("connect_points", True):
            style["linestyle"] = "None"
        return style

    def _plot_curve(curve_key, x_values, y_values, yerr_values):
        style = _resolve_style(curve_key)
        color = style.get("color", None)
        ecolor = style.get("ecolor", None) or color
        ax.errorbar(
            x_values, y_values, yerr=yerr_values,
            fmt=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 0.9),
            markersize=style.get("markersize", 3.0),
            alpha=style.get("alpha", 0.95),
            color=color,
            ecolor=ecolor if yerr_values is not None else None,
            elinewidth=style.get("elinewidth", 0.6) if yerr_values is not None else 0,
            capsize=style.get("capsize", 1.6) if yerr_values is not None else 0,
            capthick=style.get("capthick", 0.6) if yerr_values is not None else 0,
            zorder=style.get("zorder_marker", 2.0),
            label=style.get("label", curve_key),
        )

    _plot_curve("measured", p_sorted, empirical_mean, empirical_err)
    _plot_curve("theory_finite", p_sorted, theory_finite_mean, theory_finite_err)
    _plot_curve("theory_infinite", p_sorted, theory_infinite_curve, theory_infinite_err)

    legend_cfg = cfg["legend"]
    if legend_cfg.get("show", True):
        ax.legend(
            loc=legend_cfg.get("loc", "best"),
            ncol=legend_cfg.get("ncol", 1),
            fontsize=legend_cfg.get("fontsize", cfg["font"]["default_size"]),
            frameon=legend_cfg.get("frameon", False),
            handlelength=legend_cfg.get("handlelength", 1.2),
            handletextpad=legend_cfg.get("handletextpad", 0.4),
            labelspacing=legend_cfg.get("labelspacing", 0.2),
            borderaxespad=legend_cfg.get("borderaxespad", 0.2),
            columnspacing=legend_cfg.get("columnspacing", 0.8),
            markerscale=legend_cfg.get("markerscale", 1.0),
        )

    if cfg["figure"].get("tight_layout", False):
        rect = cfg["figure"].get("tight_layout_rect", None)
        fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()

    if output_figure is None:
        output_figure = make_default_output_path(THEORY_RESULTS_FILES, feature, output_format, plot_cfg=cfg)
    output_figure = Path(output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    save_cfg = cfg["save"]
    if save_cfg.get("enabled", True):
        fig.savefig(
            output_figure,
            transparent=save_cfg.get("transparent", True),
            bbox_inches=save_cfg.get("bbox_inches", "tight"),
            pad_inches=save_cfg.get("pad_inches", 0.02),
        )
        print(f"Saved figure to: {output_figure}", flush=True)

    if cfg.get("close_after_save", False):
        plt.close(fig)
    if cfg.get("show_figures", True):
        plt.show()


# ============================================================
# ==================== CURVE CONSTRUCTION =====================
# ============================================================


def validate_loaded_file(data_dict, feature, reference_gamma, reference_dim, path_str):
    """
    Check that one newly loaded file is compatible with the already established
    feature / dimensionality conventions.
    """
    current_feature = infer_feature(data_dict)
    if FEATURE is None and current_feature != feature:
        raise ValueError(
            f"The file {path_str} implies feature '{current_feature}', but the first file implies '{feature}'."
        )

    current_gamma = data_dict.get("args", {}).get("gamma", None)
    if current_gamma != reference_gamma:
        warnings.warn(
            f"The file {path_str} has gamma={current_gamma}, whereas the first file has gamma={reference_gamma}.",
            stacklevel=1,
        )

    local_results = data_dict["local"]
    categories = list(local_results["n_images"].keys())
    if len(categories) == 0:
        raise ValueError(f"The file {path_str} contains no local categories.")

    current_dim = len(np.asarray(local_results["regression_vector"][categories[0]][feature], dtype=np.float64))
    if current_dim != reference_dim:
        raise ValueError(
            f"The file {path_str} has neuron dimension {current_dim}, but the first file has {reference_dim}."
        )



def build_curve_data_streaming(theory_files, first_data, feature):
    """
    Build the full curve while processing at most ONE theory file at a time.

    Workflow:
    1) use the first file for the within-file subset points and for the single-
       file full-category point;
    2) keep only the lightweight local theory quantities plus the cumulative
       global sums in memory;
    3) load the second file, update the accumulators, compute the 2*P point,
       then discard the file;
    4) repeat for the third file, etc.
    """
    results_per_p = {}

    # --------------------------------------------------------
    # First file: subset points and the single-file full point.
    # --------------------------------------------------------
    first_local = first_data["local"]
    first_global = first_data["global"]
    first_categories = list(first_local["n_images"].keys())
    first_n_full_categories = len(first_categories)
    first_n_full_images = total_image_count_for_categories(first_local, first_categories)

    subset_runs = collect_subset_runs_from_loaded_data(first_data)
    available_sizes = sorted({run["size"] for run in subset_runs})
    available_seeds = sorted({run["seed"] for run in subset_runs})

    if SELECTED_SUBSAMPLING_SEEDS is None:
        selected_seeds = available_seeds
    else:
        selected_seeds = [int(seed) for seed in SELECTED_SUBSAMPLING_SEEDS]

    if P_LIST is None:
        p_list = available_sizes.copy()
    else:
        p_list = [int(p) for p in P_LIST]
        missing_sizes = [p for p in p_list if p not in available_sizes]
        if missing_sizes:
            warnings.warn(
                f"These requested subset sizes are not present in the first file and will be ignored: {missing_sizes}",
                stacklevel=1,
            )
        p_list = [p for p in p_list if p in available_sizes]

    for p in p_list:
        print(f"=== Computing within-file subset stage from the first file: P={p} ===", flush=True)
        runs_here = [run for run in subset_runs if run["size"] == p and run["seed"] in selected_seeds]
        if len(runs_here) == 0:
            warnings.warn(
                f"No subset runs found for P={p} after applying the selected seeds {selected_seeds}. Skipping this point.",
                stacklevel=1,
            )
            continue

        realization_stats = []
        realization_labels = []
        for run in sorted(runs_here, key=lambda item: item["seed"]):
            print(f"Processing within-file subset point: P={p}, seed={run['seed']}", flush=True)
            local_arrays_subset = extract_local_arrays_for_subset(first_local, run["categories"], feature)
            stats = compute_gap_quantities_from_local_arrays_and_global(
                local_arrays=local_arrays_subset,
                global_results=run["global"],
                feature=feature,
            )
            realization_stats.append(stats)
            realization_labels.append(f"seed{run['seed']}")

        empirical_values = [stats["empirical"] for stats in realization_stats]
        theory_finite_values = [stats["theory_finite"] for stats in realization_stats]
        empirical_mean, empirical_err = aggregate_across_realizations(empirical_values, REALIZATION_ERRORBAR_MODE)
        theory_finite_mean, theory_finite_err = aggregate_across_realizations(theory_finite_values, REALIZATION_ERRORBAR_MODE)

        results_per_p[p] = {
            "empirical_mean": empirical_mean,
            "empirical_err": empirical_err,
            "theory_finite_mean": theory_finite_mean,
            "theory_finite_err": theory_finite_err,
            "n_realizations": len(realization_stats),
            "realizations": realization_labels,
        }

    # --------------------------------------------------------
    # Initialize the cumulative accumulators with the first file.
    # --------------------------------------------------------
    print(
        f"=== Initializing cumulative full-category accumulators from file 1/{len(theory_files)}: "
        f"categories={first_n_full_categories} ===",
        flush=True,
    )
    running_local = initialize_running_local_accumulator()
    append_full_file_local_quantities(running_local, first_local, first_categories, feature)

    running_global_sums = initialize_running_global_sums()
    update_running_global_sums(running_global_sums, first_global, feature, weight=first_n_full_images)

    cumulative_category_count = first_n_full_categories
    cumulative_file_count = 1

    if INCLUDE_SINGLE_FILE_FULL_POINT:
        print(
            f"=== Computing cumulative full-category stage: k=1 file, total categories={cumulative_category_count} ===",
            flush=True,
        )
        current_local_arrays = finalize_local_accumulator(running_local)
        current_global_block = build_global_block_from_running_sums(running_global_sums, feature)
        stats = compute_gap_quantities_from_local_arrays_and_global(
            local_arrays=current_local_arrays,
            global_results=current_global_block,
            feature=feature,
        )
        results_per_p[cumulative_category_count] = {
            "empirical_mean": stats["empirical"],
            "empirical_err": 0.0,
            "theory_finite_mean": stats["theory_finite"],
            "theory_finite_err": 0.0,
            "n_realizations": 1,
            "realizations": ["prefix_k1"],
        }

    # Explicitly release the first loaded file after its useful content has been
    # transferred into the lightweight accumulators.
    del first_data
    gc.collect()

    # --------------------------------------------------------
    # Remaining files: one at a time, always updating the cumulative prefix.
    # --------------------------------------------------------
    if INCLUDE_MERGED_FULL_POINTS and len(theory_files) >= 2:
        if MERGED_FULL_FILE_COUNTS is None:
            merge_counts = list(range(2, len(theory_files) + 1))
        else:
            merge_counts = sorted({int(k) for k in MERGED_FULL_FILE_COUNTS if 2 <= int(k) <= len(theory_files)})
    else:
        merge_counts = []

    reference_gamma = None
    reference_dim = len(np.asarray(first_local["regression_vector"][first_categories[0]][feature], dtype=np.float64))
    # Recover gamma from the first file only after we have already extracted what we need.
    # It is a small scalar, so keeping it here is fine.
    # We cannot query first_data anymore because it was intentionally released.
    # Therefore, store reference_gamma directly from the original first global context.
    # The caller will pass it separately if needed.
    # We instead infer it from the file content before releasing the first file in main().
    # The value is injected by assigning to the function attribute below.
    reference_gamma = build_curve_data_streaming.reference_gamma

    for file_index in range(1, len(theory_files)):
        theory_path = Path(theory_files[file_index]).expanduser().resolve()
        next_k = file_index + 1
        print(f"Loading cumulative-prefix file [{next_k}/{len(theory_files)}]: {theory_path}", flush=True)
        with open(theory_path, "rb") as file_handle:
            data_dict = pickle.load(file_handle)

        validate_loaded_file(
            data_dict=data_dict,
            feature=feature,
            reference_gamma=reference_gamma,
            reference_dim=reference_dim,
            path_str=str(theory_path),
        )

        local_results = data_dict["local"]
        global_results = data_dict["global"]
        categories = list(local_results["n_images"].keys())
        n_categories = len(categories)
        n_images = total_image_count_for_categories(local_results, categories)

        print(
            f"Updating cumulative prefix with file {next_k}/{len(theory_files)}: "
            f"added_categories={n_categories}, cumulative_before={cumulative_category_count}",
            flush=True,
        )
        append_full_file_local_quantities(running_local, local_results, categories, feature)
        update_running_global_sums(running_global_sums, global_results, feature, weight=n_images)

        cumulative_category_count += n_categories
        cumulative_file_count = next_k

        if next_k in merge_counts:
            print(
                f"=== Computing cumulative full-category stage: k={next_k} files, "
                f"total categories={cumulative_category_count} ===",
                flush=True,
            )
            current_local_arrays = finalize_local_accumulator(running_local)
            current_global_block = build_global_block_from_running_sums(running_global_sums, feature)
            stats = compute_gap_quantities_from_local_arrays_and_global(
                local_arrays=current_local_arrays,
                global_results=current_global_block,
                feature=feature,
            )
            results_per_p[cumulative_category_count] = {
                "empirical_mean": stats["empirical"],
                "empirical_err": 0.0,
                "theory_finite_mean": stats["theory_finite"],
                "theory_finite_err": 0.0,
                "n_realizations": 1,
                "realizations": [f"prefix_k{next_k}"],
            }

        del data_dict, local_results, global_results
        gc.collect()

    # --------------------------------------------------------
    # Infinite-theory horizontal line: use the final cumulative prefix that
    # contains ALL provided files.
    # --------------------------------------------------------
    print(
        f"=== Computing infinite-categories horizontal line from the cumulative merge of all {cumulative_file_count} files ===",
        flush=True,
    )
    all_local_arrays = finalize_local_accumulator(running_local)
    all_global_block = build_global_block_from_running_sums(running_global_sums, feature)
    full_merged_stats = compute_gap_quantities_from_local_arrays_and_global(
        local_arrays=all_local_arrays,
        global_results=all_global_block,
        feature=feature,
    )

    return {
        "feature": feature,
        "results_per_p": results_per_p,
        "infinite_line_value": float(full_merged_stats["theory_infinite"]),
        "infinite_line_total_categories": cumulative_category_count,
        "available_sizes": available_sizes,
        "available_seeds": available_seeds,
        "selected_seeds": selected_seeds,
        "n_files": len(theory_files),
    }


# ============================================================
# ============================== MAIN =========================
# ============================================================


def main():
    if len(THEORY_RESULTS_FILES) == 0:
        raise ValueError("THEORY_RESULTS_FILES is empty. Please provide at least one .pkl file.")

    resolved_files = [str(Path(path).expanduser().resolve()) for path in THEORY_RESULTS_FILES]
    print(f"Preparing streaming multi-file plot from {len(resolved_files)} theory file(s).", flush=True)

    # We load only the FIRST file here. This is enough both to infer the feature
    # and, if needed, to start the streaming computation. We do not load the
    # remaining files unless the cache misses.
    first_path = Path(resolved_files[0])
    print(f"Loading first theory file: {first_path}", flush=True)
    with open(first_path, "rb") as file_handle:
        first_data = pickle.load(file_handle)

    feature = FEATURE if FEATURE is not None else infer_feature(first_data)
    print(f"Using feature: {feature}", flush=True)

    cache_curve_data, cache_path = try_load_curve_data_from_cache(feature)
    if cache_curve_data is not None:
        curve_data = cache_curve_data
        del first_data
        gc.collect()
    else:
        first_local = first_data["local"]
        first_categories = list(first_local["n_images"].keys())
        if len(first_categories) == 0:
            raise ValueError(f"The file {first_path} contains no local categories.")

        # Store small compatibility references on the function object so that
        # the streaming builder can release the first large file dictionary early.
        build_curve_data_streaming.reference_gamma = first_data.get("args", {}).get("gamma", None)

        curve_data = build_curve_data_streaming(resolved_files, first_data, feature)
        save_curve_data_to_cache(curve_data, feature, cache_path=cache_path)

    print(
        f"Available subset sizes in the first file: {curve_data['available_sizes']} | "
        f"Available within-file seeds in the first file: {curve_data['available_seeds']} | "
        f"Using seeds: {curve_data['selected_seeds']} | "
        f"Number of input files: {curve_data['n_files']}",
        flush=True,
    )

    for p in sorted(curve_data["results_per_p"].keys()):
        stats = curve_data["results_per_p"][p]
        print(
            f"P={p}: empirical={stats['empirical_mean']:.6g} +/- {stats['empirical_err']:.6g}, "
            f"theory_finite={stats['theory_finite_mean']:.6g} +/- {stats['theory_finite_err']:.6g}, "
            f"n_realizations={stats['n_realizations']}",
            flush=True,
        )

    print(
        f"Infinite-categories horizontal line (computed from all provided files together, "
        f"total categories={curve_data['infinite_line_total_categories']}): "
        f"{curve_data['infinite_line_value']:.6g}",
        flush=True,
    )

    plot_curve_bundle(curve_data, output_figure=OUTPUT_FIGURE, output_format=OUTPUT_FORMAT)


if __name__ == "__main__":
    main()
