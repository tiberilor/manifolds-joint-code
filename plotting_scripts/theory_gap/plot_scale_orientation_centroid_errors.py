import pickle
import numpy as np
import LIB_plot_utility as myplt
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

import json
import hashlib
from datetime import datetime
import sys
import importlib
import os


# ============================================================
# =========== NUMPY PICKLE BACKCOMPAT SHIM (EARLY) ============
# ============================================================
# Needed if any loaded .pkl references numpy._core.* modules (older numpy pickles).
def _alias(old, new):
    try:
        sys.modules[old] = importlib.import_module(new)
    except Exception:
        pass

_alias("numpy._core", "numpy.core")
for _name in ("numeric", "multiarray", "_multiarray_umath", "shape_base", "overrides"):
    _alias(f"numpy._core.{_name}", f"numpy.core.{_name}")


# ============================================================
# ======================= COMMON PATHS =======================
# ============================================================
# Shared across ALL plots and the shared compute pass. Edit once.

COMMON_PATHS = {
    "CR": {
        "global_L2_DimReduced_path": "PATH/TO/CR/global_results.pkl",
        "global_L2_DimReduced_fluct_only_path": "PATH/TO/CR/global_fluctuations_only_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/CR/local_results",
        "theory_regression_folder": "PATH/TO/CR/theory_results",
    },
    "C": {
        "global_L2_DimReduced_path": "PATH/TO/C/global_results.pkl",
        "global_L2_DimReduced_fluct_only_path": "PATH/TO/C/global_fluctuations_only_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/C/local_results",
        "theory_regression_folder": "PATH/TO/C/theory_results",
    },
}


# ============================================================
# ===================== COMMON COMPUTE CFG ===================
# ============================================================
# Shared across ALL plots. This is the ONLY compute-config.
# Expensive computation runs once using this config.

COMMON_COMPUTE_CFG = {
    # which models to compute
    "MODEL_ORDER_COMPUTE": ["CR", "C"],

    # which features to compute
    "FEATURES": ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"],

    # Keep this on "outer" to reproduce the paper results
    "best_gamma_theory_method": "outer",  # "inner" | "outer"

    # std aggregation choice
    "std_of_avg_split_and_file": True,

    # shared compute caching (disk + in-memory)
    "CACHE_CFG": {
        "use_cache": True,
        "overwrite_cache": False,
        "verbose": True,

        # written next to this script (__file__ directory)
        "shared_compute_cache_filename": "plot_cache__three_errors_master.pkl",

        # prints extra info if cache is missed
        "debug_cache": True,
    },
}


# ============================================================
# ======================= PLOT CONFIGS =======================
# ============================================================
# IMPORTANT:
# - NO ARGPARSE.
# - Paths and compute-config are shared above.
# - EACH plot below repeats ALL plotting controls independently.
# - Each plot can reorder/subset models for display via MODEL_ORDER_TO_PLOT.
# - Each plot can choose its own tick labels via FEATURE_TICK_LABELS (cosmetic only).


# ==============================
# ========= SCALE PLOT =========
# ==============================

SCALE_PLOT_CFG = {
    # which models to display (subset/reorder OK)
    "MODEL_ORDER_TO_PLOT": ["CR", "C"],

    # cosmetic tick labels (must align with COMMON_COMPUTE_CFG["FEATURES"])
    "FEATURE_TICK_LABELS": [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"],

    "BAR_KEYS_TO_PLOT": ["scale"],
    "BAR_ORDER": ["scale", "orient", "centroid"],

    "SAVE_CFG": {
        "save": True,
        "filename": "scale_error__relative.svg",
        "transparent": True,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "dpi": 300,
    },

    "FONT_CFG": {"family": "Arial", "use_global_rc": True},
    "STYLE_CFG": {"hide_top_right_spines": True, "spine_color": "#333333", "spine_linewidth": 1.0},

    "TEXT_CFG": {
        "title": r"scale error $E_s$",
        "x_label": None,
        "y_label": r"$E_s$ (normalized)",
        "title_size": 7,
        "x_label_size": 7,
        "y_label_size": 7,
        "x_tick_size": 5,
        "y_tick_size": 5,
        "x_tick_rotation": 0,
    },

    "FIG_CFG": {"figsize": (1.6, 1.45), "dpi": 150, "tight_layout": True, "constrained_layout": False},

    "AXES_CFG": {
        "y_scale": "log",  # "log" | "linear" | "symlog"
        "y_lim": (4e-4, 20),
        "x_lim": None,
        "y_ticks": (1e-3, 1e-2, 1e-1, 1, 10),
        "y_ticklabels": None,
        "y_log_base": 10,
        "symlog_linthresh": 1e-4,
        "symlog_linscale": 1.0,
        "symlog_base": 10,
        "symlog_subs": None,
        "show_grid": True,
        "grid_axis": "y",
        "grid_alpha": 1.0,
        "grid_linestyle": ":",
        "grid_linewidth": 0.8,
    },

    "GROUP_CFG": {"bar_width": 0.1, "inner_gap": 0.00, "model_gap": 0.20, "feature_gap": 0.05},

    "BAR_CFG": {
        "scale_color": "tab:olive",
        "orient_color": "tab:cyan",
        "centroid_color": "tab:pink",
        "alpha": 0.95,
        "edgecolor": None,
        "linewidth": 0.5,
        "show_error": True,
        "error_cap_size": 1.5,
        "error_linewidth": 0.8,
        "error_color": "#333333",
    },

    "LEGEND_CFG": {
        "show": False,
        "labels": {
            "scale": r"$E_s\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "orient": r"$E_o\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "centroid": r"$E_c\,/\,E_{\mathrm{loc}}$",
        },
        "loc": "upper left",
        "fontsize": 7,
        "frameon": False,
        "ncols": 1,
    },
}


# ==============================
# ======= ORIENTATION PLOT =====
# ==============================

ORIENT_PLOT_CFG = {
    "MODEL_ORDER_TO_PLOT": ["CR", "C"],
    "FEATURE_TICK_LABELS": [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"],

    "BAR_KEYS_TO_PLOT": ["orient"],
    "BAR_ORDER": ["scale", "orient", "centroid"],

    "SAVE_CFG": {
        "save": True,
        "filename": "orientation_error__relative.svg",
        "transparent": True,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "dpi": 300,
    },

    "FONT_CFG": {"family": "Arial", "use_global_rc": True},
    "STYLE_CFG": {"hide_top_right_spines": True, "spine_color": "#333333", "spine_linewidth": 1.0},

    "TEXT_CFG": {
        "title": r"orientation error $E_o$",
        "x_label": None,
        "y_label": r"$E_o$ (normalized)",
        "title_size": 7,
        "x_label_size": 7,
        "y_label_size": 7,
        "x_tick_size": 5,
        "y_tick_size": 5,
        "x_tick_rotation": 0,
    },

    "FIG_CFG": {"figsize": (1.6, 1.45), "dpi": 150, "tight_layout": True, "constrained_layout": False},

    "AXES_CFG": {
        "y_scale": "log",
        "y_lim": (4e-4, 20),
        "x_lim": None,
        "y_ticks": (1e-3, 1e-2, 1e-1, 1, 10),
        "y_ticklabels": None,
        "y_log_base": 10,
        "symlog_linthresh": 1e-4,
        "symlog_linscale": 1.0,
        "symlog_base": 10,
        "symlog_subs": None,
        "show_grid": True,
        "grid_axis": "y",
        "grid_alpha": 1.0,
        "grid_linestyle": ":",
        "grid_linewidth": 0.8,
    },

    "GROUP_CFG": {"bar_width": 0.1, "inner_gap": 0.00, "model_gap": 0.20, "feature_gap": 0.05},

    "BAR_CFG": {
        "scale_color": "tab:olive",
        "orient_color": "tab:cyan",
        "centroid_color": "tab:purple",
        "alpha": 0.95,
        "edgecolor": None,
        "linewidth": 0.5,
        "show_error": True,
        "error_cap_size": 1.5,
        "error_linewidth": 0.8,
        "error_color": "#333333",
    },

    "LEGEND_CFG": {
        "show": False,
        "labels": {
            "scale": r"$E_s\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "orient": r"$E_o\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "centroid": r"$E_c\,/\,E_{\mathrm{loc}}$",
        },
        "loc": "upper left",
        "fontsize": 7,
        "frameon": False,
        "ncols": 1,
    },
}


# ==============================
# ========= CENTROID PLOT =======
# ==============================

CENTROID_PLOT_CFG = {
    "MODEL_ORDER_TO_PLOT": ["CR", "C"],
    "FEATURE_TICK_LABELS": [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"],

    "BAR_KEYS_TO_PLOT": ["centroid"],
    "BAR_ORDER": ["scale", "orient", "centroid"],

    "SAVE_CFG": {
        "save": True,
        "filename": "centroid_error__relative.svg",
        "transparent": True,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "dpi": 300,
    },

    "FONT_CFG": {"family": "Arial", "use_global_rc": True},
    "STYLE_CFG": {"hide_top_right_spines": True, "spine_color": "#333333", "spine_linewidth": 1.0},

    "TEXT_CFG": {
        "title": r"centroid error $E_c$",
        "x_label": None,
        "y_label": r"$E_c$ (normalized)",
        "title_size": 7,
        "x_label_size": 7,
        "y_label_size": 7,
        "x_tick_size": 5,
        "y_tick_size": 5,
        "x_tick_rotation": 0,
    },

    "FIG_CFG": {"figsize": (1.6, 1.45), "dpi": 150, "tight_layout": True, "constrained_layout": False},

    "AXES_CFG": {
        "y_scale": "log",
        "y_lim": (4e-4, 20),
        "x_lim": None,
        "y_ticks": (1e-3, 1e-2, 1e-1, 1, 10),
        "y_ticklabels": None,
        "y_log_base": 10,
        "symlog_linthresh": 1e-4,
        "symlog_linscale": 1.0,
        "symlog_base": 10,
        "symlog_subs": None,
        "show_grid": True,
        "grid_axis": "y",
        "grid_alpha": 1.0,
        "grid_linestyle": ":",
        "grid_linewidth": 0.8,
    },

    "GROUP_CFG": {"bar_width": 0.1, "inner_gap": 0.00, "model_gap": 0.20, "feature_gap": 0.05},

    "BAR_CFG": {
        "scale_color": "tab:olive",
        "orient_color": "tab:cyan",
        "centroid_color": "tab:pink",
        "alpha": 0.95,
        "edgecolor": None,
        "linewidth": 0.5,
        "show_error": True,
        "error_cap_size": 1.5,
        "error_linewidth": 0.8,
        "error_color": "#333333",
    },

    "LEGEND_CFG": {
        "show": False,
        "labels": {
            "scale": r"$E_s\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "orient": r"$E_o\,\frac{\hat{\sigma}^2}{\sigma^2}\,/\,E_{\mathrm{loc}}$",
            "centroid": r"$E_c\,/\,E_{\mathrm{loc}}$",
        },
        "loc": "upper left",
        "fontsize": 7,
        "frameon": False,
        "ncols": 1,
    },
}


# ============================================================
# ===================== SMALL HELPERS ========================
# ============================================================

def _script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _safe_div(a, b):
    if b is None:
        return np.nan
    b = float(b)
    if b == 0:
        return np.nan
    return float(a) / b


def tex_log_formatter(x, pos=None, base=10):
    if not np.isfinite(x) or x <= 0:
        return ""
    k = int(round(np.log(x) / np.log(base)))
    return r"$10^{%+d}$" % k


def tex_symlog_formatter(x, pos=None, base=10, linthresh=1e-4):
    if not np.isfinite(x):
        return ""
    if x == 0:
        return r"$0$"
    ax = abs(x)
    sign = "-" if x < 0 else ""
    if ax < linthresh:
        s = f"{x:.0e}"
        return rf"${s}$"
    k = int(round(np.log(ax) / np.log(base)))
    if sign:
        return r"$-10^{%+d}$" % k
    return r"$10^{%+d}$" % k


# ============================================================
# ================ ERRORBAR PROPAGATION ======================
# ============================================================

def _ratio_and_std(num, num_std, den, den_std):
    num = float(num)
    den = float(den)
    num_std = float(num_std)
    den_std = float(den_std)

    r = _safe_div(num, den)
    if not np.isfinite(r) or den == 0:
        return r, np.nan

    var = (num_std / den) ** 2 + (num * den_std / (den ** 2)) ** 2
    return r, float(np.sqrt(max(var, 0.0)))


def _product_ratio_and_std(numer_terms, denom_terms):
    r = 1.0
    rel_var = 0.0

    for a, a_std in numer_terms:
        a = float(a)
        a_std = float(a_std)
        r *= a
        rel_var = rel_var + (a_std / a) ** 2 if a != 0 else np.inf

    for b, b_std in denom_terms:
        b = float(b)
        b_std = float(b_std)
        r = _safe_div(r, b)
        rel_var = rel_var + (b_std / b) ** 2 if b != 0 else np.inf

    if not np.isfinite(r) or not np.isfinite(rel_var):
        return r, np.nan
    return r, float(abs(r) * np.sqrt(max(rel_var, 0.0)))


# ============================================================
# ===================== SHARED COMPUTE CACHE =================
# ============================================================

_SHARED_RESULTS_MEMO = {}  # in-memory memo


def _canonicalize_paths_dict(paths_dict):
    out = {}
    for k in sorted(paths_dict.keys()):
        out[k] = {kk: str(vv) for kk, vv in sorted(paths_dict[k].items())}
    return out


def _shared_compute_cache_path() -> Path:
    fn = COMMON_COMPUTE_CFG["CACHE_CFG"].get("shared_compute_cache_filename", "plot_cache__three_errors_master.pkl")
    return _script_dir() / fn


def _compute_key() -> str:
    payload = {
        "COMMON_PATHS_ordered": _canonicalize_paths_dict(COMMON_PATHS),
        "MODEL_ORDER_COMPUTE": list(COMMON_COMPUTE_CFG["MODEL_ORDER_COMPUTE"]),
        "FEATURES": list(COMMON_COMPUTE_CFG["FEATURES"]),
        "best_gamma_theory_method": str(COMMON_COMPUTE_CFG["best_gamma_theory_method"]),
        "std_of_avg_split_and_file": bool(COMMON_COMPUTE_CFG["std_of_avg_split_and_file"]),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_shared_cache_file() -> dict:
    p = _shared_compute_cache_path()
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj if isinstance(obj, dict) else {}


def _save_shared_cache_file(obj: dict):
    p = _shared_compute_cache_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


# ============================================================
# ==================== STYLE HELPERS =========================
# ============================================================

def _apply_global_style(plot_cfg: dict):
    font_family = plot_cfg["FONT_CFG"].get("family", None)
    if font_family:
        plt.rcParams["font.family"] = font_family
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _apply_spines(ax, plot_cfg: dict):
    sc = plot_cfg["STYLE_CFG"].get("spine_color", "#333333")
    lw = plot_cfg["STYLE_CFG"].get("spine_linewidth", 1.0)
    for spine in ax.spines.values():
        spine.set_color(sc)
        spine.set_linewidth(lw)
    if plot_cfg["STYLE_CFG"].get("hide_top_right_spines", True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ============================================================
# ================== LOADERS / RETRIEVERS ====================
# ============================================================

def _load_global_mse_any_path(path_str: str, feature: str, std_of_avg_split_and_file: bool):
    path = Path(path_str).expanduser()

    if path.is_file() and path.suffix.lower() == ".pkl":
        mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(str(path), feature)
        return float(mse_avg), float(mse_avg_std), float(labels_variance), None

    if path.is_dir():
        mse_avg_per_file = []
        mse_avg_std_per_file = []
        mse_per_outer_split_per_file = []
        labels_var_per_file = []

        pkl_files = sorted(path.glob("*.pkl"))
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl files found under: {path}")

        for pkl_file in pkl_files:
            mse_avg_i, mse_avg_std_i, labels_variance_i, mse_per_outer_split = myplt.retrieve_global_mse(
                pkl_file, feature, return_mse_per_outer_split=True
            )
            mse_avg_per_file.append(float(mse_avg_i))
            mse_avg_std_per_file.append(float(mse_avg_std_i))
            mse_per_outer_split_per_file.append(np.asarray(mse_per_outer_split, dtype=float))
            labels_var_per_file.append(float(labels_variance_i))

        mse_avg = float(np.mean(mse_avg_per_file))

        mse_avg_std_avg = float(np.mean(mse_avg_std_per_file))
        std_across_files = float(np.std(mse_avg_per_file))
        mse_avg_std = float(np.sqrt((mse_avg_std_avg**2) + (std_across_files**2)))

        mse_per_outer_split_per_file = np.stack(mse_per_outer_split_per_file, axis=0)
        mse_std_of_avg_split_and_file = float(np.std(mse_per_outer_split_per_file) / np.sqrt(mse_per_outer_split_per_file.size))
        mse_avg_std_final = mse_std_of_avg_split_and_file if std_of_avg_split_and_file else mse_avg_std

        labels_variance = float(np.mean(labels_var_per_file))
        if np.std(labels_var_per_file) > 1e-9:
            warnings.warn(f"labels_variance varies across files under {path}. Using mean={labels_variance}.", RuntimeWarning)

        return mse_avg, mse_avg_std_final, labels_variance, None

    raise FileNotFoundError(f"Global path does not exist or isn’t a .pkl file/folder: {path}")


def _load_local_mse_and_best_gamma(local_folder: str, feature: str,
                                  cv_global_mse_avg: float, cv_global_mse_std: float, labels_variance: float):
    best_gamma_theory_method = COMMON_COMPUTE_CFG["best_gamma_theory_method"]
    std_of_avg_split_and_file = COMMON_COMPUTE_CFG["std_of_avg_split_and_file"]
    path_local = Path(local_folder).expanduser()

    if path_local.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_local.iterdir()):
        AvgLocalMse_avg, AvgLocalMse_avg_std, best_gamma_per_outer_split = myplt.retrieve_avg_local_mse_gamma_regularization_fast(
            local_folder, feature, return_best_gammas_per_outer_split=True
        )
        if best_gamma_theory_method == "inner":
            best_gamma_theory = float(np.mean(best_gamma_per_outer_split))
        elif best_gamma_theory_method == "outer":
            gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                local_folder, feature, cv_global_mse_avg, cv_global_mse_std, labels_variance
            )
            best_gamma_theory = float(gamma_cv_list[int(np.argmax(cv_nmse_gap_fixed_gamma_avg))])
        else:
            raise ValueError(f"Invalid best_gamma_theory_method: {best_gamma_theory_method}")
        return float(AvgLocalMse_avg), float(AvgLocalMse_avg_std), best_gamma_theory, False

    if path_local.is_dir() and any(d.is_dir() for d in path_local.iterdir()):
        AvgLocalMse_avg_per_file = []
        AvgLocalMse_avg_std_per_file = []
        LocalMse_per_outer_split_per_file = []
        best_gamma_theory = {}

        run_dirs = sorted([d for d in path_local.iterdir() if d.is_dir()])
        for run_dir in run_dirs:
            seed_str = run_dir.name
            (AvgLocalMse_avg_i, AvgLocalMse_avg_std_i,
             best_gamma_per_outer_split, LocalMse_per_outer_split) = myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                run_dir,
                feature,
                return_mse_per_outer_split=True,
                return_best_gammas_per_outer_split=True,
            )

            AvgLocalMse_avg_per_file.append(float(AvgLocalMse_avg_i))
            AvgLocalMse_avg_std_per_file.append(float(AvgLocalMse_avg_std_i))
            LocalMse_per_outer_split_per_file.append(np.asarray(LocalMse_per_outer_split, dtype=float))

            if best_gamma_theory_method == "inner":
                best_gamma_theory[seed_str] = float(np.mean(best_gamma_per_outer_split))
            elif best_gamma_theory_method == "outer":
                gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                    run_dir, feature, cv_global_mse_avg, cv_global_mse_std, labels_variance
                )
                best_gamma_theory[seed_str] = float(gamma_cv_list[int(np.argmax(cv_nmse_gap_fixed_gamma_avg))])
            else:
                raise ValueError(f"Invalid best_gamma_theory_method: {best_gamma_theory_method}")

        AvgLocalMse_avg = float(np.mean(AvgLocalMse_avg_per_file))

        AvgLocalMse_avg_std_avg = float(np.mean(AvgLocalMse_avg_std_per_file))
        std_across_files = float(np.std(AvgLocalMse_avg_per_file))
        AvgLocalMse_avg_std = float(np.sqrt((AvgLocalMse_avg_std_avg**2) + (std_across_files**2)))

        LocalMse_per_outer_split_per_file = np.stack(LocalMse_per_outer_split_per_file, axis=0)
        LocalMse_std_of_avg_split_and_file = float(np.std(LocalMse_per_outer_split_per_file) / np.sqrt(LocalMse_per_outer_split_per_file.size))

        cv_local_mse_std = LocalMse_std_of_avg_split_and_file if std_of_avg_split_and_file else AvgLocalMse_avg_std
        return AvgLocalMse_avg, cv_local_mse_std, best_gamma_theory, True

    raise FileNotFoundError(f"Local folder does not exist or isn’t a .pkl file/folder: {path_local}")


def _load_theory_Es_Eo_hatvar(theory_folder: str, feature: str, best_gamma_theory):
    std_of_avg_split_and_file = COMMON_COMPUTE_CFG["std_of_avg_split_and_file"]
    path_theory = Path(theory_folder).expanduser()

    def _read_one(folder, gamma):
        res = myplt.retrieve_theory_gap_closest_gamma(folder, feature, gamma, return_detailed_results=True)
        detailed_results = res[-1]
        return (
            float(detailed_results["mse_gap_scale_contribution"]),
            float(detailed_results["mse_gap_orientation_contribution"]),
            float(detailed_results["hat_var_y_1"]),
        )

    if path_theory.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_theory.iterdir()):
        Es, Eo, hatv = _read_one(theory_folder, float(best_gamma_theory))
        return Es, 0.0, Eo, 0.0, hatv, 0.0

    if path_theory.is_dir() and any(d.is_dir() for d in path_theory.iterdir()):
        Es_list, Eo_list, hatv_list = [], [], []
        run_dirs = sorted([d for d in path_theory.iterdir() if d.is_dir()])
        for run_dir in run_dirs:
            seed_str = run_dir.name
            gamma = float(best_gamma_theory[seed_str])
            Es, Eo, hatv = _read_one(run_dir, gamma)
            Es_list.append(Es)
            Eo_list.append(Eo)
            hatv_list.append(hatv)

        Es_avg = float(np.mean(Es_list))
        Eo_avg = float(np.mean(Eo_list))
        hatv_avg = float(np.mean(hatv_list))

        norm = np.sqrt(len(Es_list)) if std_of_avg_split_and_file else 1.0
        Es_std = float(np.std(Es_list) / norm)
        Eo_std = float(np.std(Eo_list) / norm)
        hatv_std = float(np.std(hatv_list) / norm)
        return Es_avg, Es_std, Eo_avg, Eo_std, hatv_avg, hatv_std

    raise FileNotFoundError(f"Theory folder does not exist or isn’t a .pkl file/folder: {path_theory}")


# ============================================================
# ===================== CORE COMPUTATION =====================
# ============================================================

def compute_three_errors_for_model_and_feature(model_cfg: dict, feature: str):
    stdflag = COMMON_COMPUTE_CFG["std_of_avg_split_and_file"]

    global_path = model_cfg["global_L2_DimReduced_path"]
    global_fluct_only_path = model_cfg["global_L2_DimReduced_fluct_only_path"]
    local_folder = model_cfg["local_Gamma_DimReduced_folder"]
    theory_folder = model_cfg["theory_regression_folder"]

    g_mse_avg, g_mse_std, labels_variance, _ = _load_global_mse_any_path(global_path, feature, stdflag)
    gf_mse_avg, gf_mse_std, _gf_var, _ = _load_global_mse_any_path(global_fluct_only_path, feature, stdflag)

    g_nmse_avg = _safe_div(g_mse_avg, labels_variance)
    g_nmse_std = _safe_div(g_mse_std, labels_variance)
    gf_nmse_avg = _safe_div(gf_mse_avg, labels_variance)
    gf_nmse_std = _safe_div(gf_mse_std, labels_variance)

    centroid_nmse_avg = float(g_nmse_avg - gf_nmse_avg)
    centroid_nmse_std = float(np.sqrt(g_nmse_std**2 + gf_nmse_std**2))

    local_mse_avg, local_mse_std, best_gamma_theory, _is_multi = _load_local_mse_and_best_gamma(
        local_folder, feature,
        cv_global_mse_avg=g_mse_avg,
        cv_global_mse_std=g_mse_std,
        labels_variance=labels_variance,
    )
    E_loc = _safe_div(local_mse_avg, labels_variance)
    E_loc_std = _safe_div(local_mse_std, labels_variance)

    Es, Es_std, Eo, Eo_std, hat_var_y_1, hat_var_y_1_std = _load_theory_Es_Eo_hatvar(
        theory_folder, feature, best_gamma_theory
    )

    pref = _safe_div(hat_var_y_1, labels_variance)
    pref_std = _safe_div(hat_var_y_1_std, labels_variance)

    scale_rel, scale_rel_std = _product_ratio_and_std(
        numer_terms=[(Es, Es_std), (pref, pref_std)],
        denom_terms=[(E_loc, E_loc_std)],
    )

    orient_rel, orient_rel_std = _product_ratio_and_std(
        numer_terms=[(Eo, Eo_std), (pref, pref_std)],
        denom_terms=[(E_loc, E_loc_std)],
    )

    centroid_rel, centroid_rel_std = _ratio_and_std(
        centroid_nmse_avg, centroid_nmse_std,
        E_loc, E_loc_std,
    )

    return scale_rel, scale_rel_std, orient_rel, orient_rel_std, centroid_rel, centroid_rel_std


def _compute_master_arrays():
    model_ids = list(COMMON_COMPUTE_CFG["MODEL_ORDER_COMPUTE"])
    features = list(COMMON_COMPUTE_CFG["FEATURES"])
    n_models = len(model_ids)
    n_features = len(features)

    scale_means = np.zeros((n_features, n_models), dtype=float)
    scale_stds = np.zeros((n_features, n_models), dtype=float)
    orient_means = np.zeros((n_features, n_models), dtype=float)
    orient_stds = np.zeros((n_features, n_models), dtype=float)
    centroid_means = np.zeros((n_features, n_models), dtype=float)
    centroid_stds = np.zeros((n_features, n_models), dtype=float)

    for m_idx, model_name in enumerate(model_ids):
        model_cfg = COMMON_PATHS[model_name]
        print(f"\nModel: {model_name}")
        for f_idx, feature in enumerate(features):
            print(f"  Feature: {feature}")
            (s_mu, s_sd, o_mu, o_sd, c_mu, c_sd) = compute_three_errors_for_model_and_feature(model_cfg, feature)
            scale_means[f_idx, m_idx] = s_mu
            scale_stds[f_idx, m_idx] = s_sd
            orient_means[f_idx, m_idx] = o_mu
            orient_stds[f_idx, m_idx] = o_sd
            centroid_means[f_idx, m_idx] = c_mu
            centroid_stds[f_idx, m_idx] = c_sd

    return {
        "model_ids": model_ids,
        "FEATURES": features,
        "scale_means": scale_means,
        "scale_stds": scale_stds,
        "orient_means": orient_means,
        "orient_stds": orient_stds,
        "centroid_means": centroid_means,
        "centroid_stds": centroid_stds,
    }


def get_master_arrays():
    ck = _compute_key()
    cache_cfg = COMMON_COMPUTE_CFG["CACHE_CFG"]
    verbose = cache_cfg.get("verbose", True)
    debug = cache_cfg.get("debug_cache", True)
    cache_path = _shared_compute_cache_path()

    if ck in _SHARED_RESULTS_MEMO and not cache_cfg.get("overwrite_cache", False):
        if verbose:
            print("[MEMO] Using in-memory computed results.")
        return _SHARED_RESULTS_MEMO[ck]

    if cache_cfg.get("use_cache", True) and not cache_cfg.get("overwrite_cache", False):
        if verbose:
            print(f"[CACHE] Looking for compute cache at: {cache_path.resolve()}")
        if cache_path.exists():
            if debug:
                st = cache_path.stat()
                print(f"[CACHE] Cache file exists (size={st.st_size} bytes, mtime={datetime.fromtimestamp(st.st_mtime)}).")
            try:
                cache_obj = _load_shared_cache_file()
                if debug:
                    print(f"[CACHE] Loaded cache file with {len(cache_obj)} key(s). Current key prefix: {ck[:12]}")
                entry = cache_obj.get(ck, None)
                if isinstance(entry, dict):
                    out = {
                        "model_ids": entry["model_ids"],
                        "FEATURES": entry["FEATURES"],
                        "scale_means": np.asarray(entry["scale_means"], dtype=float),
                        "scale_stds": np.asarray(entry["scale_stds"], dtype=float),
                        "orient_means": np.asarray(entry["orient_means"], dtype=float),
                        "orient_stds": np.asarray(entry["orient_stds"], dtype=float),
                        "centroid_means": np.asarray(entry["centroid_means"], dtype=float),
                        "centroid_stds": np.asarray(entry["centroid_stds"], dtype=float),
                    }
                    _SHARED_RESULTS_MEMO[ck] = out
                    if verbose:
                        print("[CACHE] Hit. Using cached computed arrays.")
                    return out
                if verbose:
                    print("[CACHE] Miss (key not found). Will recompute.")
            except Exception as e:
                print(f"[CACHE] Failed to read cache file ({type(e).__name__}): {e}. Will recompute.")
        else:
            if verbose:
                print("[CACHE] No cache file found. Will compute and create it.")

    out = _compute_master_arrays()
    _SHARED_RESULTS_MEMO[ck] = out

    if cache_cfg.get("use_cache", True):
        try:
            cache_obj = {}
            if cache_path.exists() and not cache_cfg.get("overwrite_cache", False):
                try:
                    cache_obj = _load_shared_cache_file()
                except Exception:
                    cache_obj = {}
            if not isinstance(cache_obj, dict):
                cache_obj = {}

            cache_obj[ck] = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "COMMON_PATHS_ordered": _canonicalize_paths_dict(COMMON_PATHS),
                "COMMON_COMPUTE_CFG": {
                    "MODEL_ORDER_COMPUTE": list(COMMON_COMPUTE_CFG["MODEL_ORDER_COMPUTE"]),
                    "FEATURES": list(COMMON_COMPUTE_CFG["FEATURES"]),
                    "best_gamma_theory_method": str(COMMON_COMPUTE_CFG["best_gamma_theory_method"]),
                    "std_of_avg_split_and_file": bool(COMMON_COMPUTE_CFG["std_of_avg_split_and_file"]),
                },
                "model_ids": out["model_ids"],
                "FEATURES": out["FEATURES"],
                "scale_means": np.asarray(out["scale_means"], dtype=float),
                "scale_stds": np.asarray(out["scale_stds"], dtype=float),
                "orient_means": np.asarray(out["orient_means"], dtype=float),
                "orient_stds": np.asarray(out["orient_stds"], dtype=float),
                "centroid_means": np.asarray(out["centroid_means"], dtype=float),
                "centroid_stds": np.asarray(out["centroid_stds"], dtype=float),
            }
            _save_shared_cache_file(cache_obj)
            if verbose:
                print(f"[CACHE] Saved compute cache key prefix {ck[:12]} to: {cache_path.resolve()}")
        except Exception as e:
            print(f"[CACHE] Failed to save cache ({type(e).__name__}): {e}")

    return out


# ============================================================
# ======================== PLOTTING ==========================
# ============================================================

def _get_kind_arrays(arrays: dict, kind: str):
    if kind == "scale":
        return arrays["scale_means"], arrays["scale_stds"]
    if kind == "orient":
        return arrays["orient_means"], arrays["orient_stds"]
    if kind == "centroid":
        return arrays["centroid_means"], arrays["centroid_stds"]
    raise ValueError(f"Unknown kind: {kind}")


def _kind_to_color(plot_cfg: dict, kind: str):
    bar_cfg = plot_cfg["BAR_CFG"]
    if kind == "scale":
        return bar_cfg["scale_color"]
    if kind == "orient":
        return bar_cfg["orient_color"]
    if kind == "centroid":
        return bar_cfg["centroid_color"]
    return "tab:blue"


def _subset_and_reorder_arrays(arrays: dict, model_order_to_plot):
    compute_order = arrays["model_ids"]
    idx = [compute_order.index(m) for m in model_order_to_plot]
    out = dict(arrays)
    out["model_ids"] = list(model_order_to_plot)
    for k in ("scale_means", "scale_stds", "orient_means", "orient_stds", "centroid_means", "centroid_stds"):
        out[k] = out[k][:, idx]
    return out


def plot_from_arrays(plot_cfg: dict, arrays_full: dict):
    allowed = {"scale", "orient", "centroid"}
    bo = list(plot_cfg["BAR_ORDER"])
    if set(bo) != allowed or len(bo) != 3:
        raise ValueError(f"BAR_ORDER must be a permutation of {sorted(allowed)}. Got: {bo}")

    bar_keys = list(plot_cfg["BAR_KEYS_TO_PLOT"])
    for k in bar_keys:
        if k not in allowed:
            raise ValueError(f"BAR_KEYS_TO_PLOT contains invalid kind: {k}")

    kinds = [k for k in bo if k in bar_keys]
    if len(kinds) == 0:
        raise ValueError("No bars selected: BAR_KEYS_TO_PLOT is empty after filtering BAR_ORDER.")

    arrays = _subset_and_reorder_arrays(arrays_full, plot_cfg["MODEL_ORDER_TO_PLOT"])

    model_ids = arrays["model_ids"]
    feature_tick_labels = list(plot_cfg["FEATURE_TICK_LABELS"])
    n_models = len(model_ids)
    n_features = len(feature_tick_labels)

    fig_cfg = plot_cfg["FIG_CFG"]
    fig = plt.figure(figsize=fig_cfg["figsize"], dpi=fig_cfg.get("dpi", 150))
    ax = fig.add_subplot(1, 1, 1)

    group_cfg = plot_cfg["GROUP_CFG"]
    bar_width = group_cfg["bar_width"]
    inner_gap = group_cfg["inner_gap"]
    model_gap = group_cfg["model_gap"]
    feature_gap = group_cfg["feature_gap"]

    nB = len(kinds)
    group_width = nB * bar_width + (nB - 1) * inner_gap
    model_width = n_features * group_width + (n_features - 1) * feature_gap
    model_centers = np.arange(n_models, dtype=float) * (model_width + model_gap)

    x_by_kind = {k: [] for k in kinds}
    y_by_kind = {k: [] for k in kinds}
    yerr_by_kind = {k: [] for k in kinds}
    tick_pos, tick_labels = [], []

    for m_idx, model_name in enumerate(model_ids):
        mc = model_centers[m_idx]
        left_edge = mc - model_width / 2.0

        for f_idx, feat_label in enumerate(feature_tick_labels):
            base = left_edge + f_idx * (group_width + feature_gap)

            x_centers = [base + j * (bar_width + inner_gap) for j in range(nB)]
            tick_center = base + (bar_width + inner_gap) * (nB - 1) / 2.0 if nB > 1 else base
            tick_pos.append(tick_center)
            tick_labels.append(f"{feat_label}\n{model_name}")

            for j, kind in enumerate(kinds):
                means, stds = _get_kind_arrays(arrays, kind)
                x_by_kind[kind].append(x_centers[j])
                y_by_kind[kind].append(float(means[f_idx, m_idx]))
                yerr_by_kind[kind].append(float(stds[f_idx, m_idx]))

    bar_cfg = plot_cfg["BAR_CFG"]
    for kind in kinds:
        color = _kind_to_color(plot_cfg, kind)
        ax.bar(
            x_by_kind[kind],
            y_by_kind[kind],
            width=bar_width,
            color=color,
            alpha=bar_cfg.get("alpha", 0.95),
            edgecolor=bar_cfg.get("edgecolor", None),
            linewidth=bar_cfg.get("linewidth", 0.5),
            yerr=yerr_by_kind[kind] if bar_cfg.get("show_error", True) else None,
            capsize=bar_cfg.get("error_cap_size", 3),
            ecolor=bar_cfg.get("error_color", "#333333"),
            error_kw={"elinewidth": bar_cfg.get("error_linewidth", 0.8)},
        )

    text_cfg = plot_cfg["TEXT_CFG"]
    if text_cfg.get("title", None):
        ax.set_title(text_cfg["title"], fontsize=text_cfg.get("title_size", 7))
    if text_cfg.get("x_label", None):
        ax.set_xlabel(text_cfg["x_label"], fontsize=text_cfg.get("x_label_size", 7))
    if text_cfg.get("y_label", None):
        ax.set_ylabel(text_cfg["y_label"], fontsize=text_cfg.get("y_label_size", 7))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        tick_labels,
        fontsize=text_cfg.get("x_tick_size", 7),
        rotation=text_cfg.get("x_tick_rotation", 0),
    )
    ax.tick_params(axis="y", labelsize=text_cfg.get("y_tick_size", 5))

    axes_cfg = plot_cfg["AXES_CFG"]
    y_scale = axes_cfg.get("y_scale", "log")
    if y_scale == "log":
        ax.set_yscale("log", base=axes_cfg.get("y_log_base", 10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: tex_log_formatter(x, pos, base=axes_cfg.get("y_log_base", 10))))
    elif y_scale == "symlog":
        linthresh = axes_cfg.get("symlog_linthresh", 1e-4)
        ax.set_yscale(
            "symlog",
            linthresh=linthresh,
            linscale=axes_cfg.get("symlog_linscale", 1.0),
            base=axes_cfg.get("symlog_base", 10),
            subs=axes_cfg.get("symlog_subs", None),
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: tex_symlog_formatter(
            x, pos, base=axes_cfg.get("symlog_base", 10), linthresh=linthresh)))
    else:
        ax.set_yscale("linear")

    if axes_cfg.get("y_lim", None) is not None:
        ax.set_ylim(axes_cfg["y_lim"])
    if axes_cfg.get("x_lim", None) is not None:
        ax.set_xlim(axes_cfg["x_lim"])

    if axes_cfg.get("y_ticks", None) is not None:
        ax.set_yticks(list(axes_cfg["y_ticks"]))
        if axes_cfg.get("y_ticklabels", None) is not None:
            ax.set_yticklabels(list(axes_cfg["y_ticklabels"]))

    if axes_cfg.get("show_grid", False):
        ax.grid(
            True,
            axis=axes_cfg.get("grid_axis", "y"),
            alpha=axes_cfg.get("grid_alpha", 0.4),
            linestyle=axes_cfg.get("grid_linestyle", ":"),
            linewidth=axes_cfg.get("grid_linewidth", 0.8),
        )

    _apply_spines(ax, plot_cfg)

    leg_cfg = plot_cfg["LEGEND_CFG"]
    if leg_cfg.get("show", False):
        handles = [mpatches.Patch(color=_kind_to_color(plot_cfg, k), label=leg_cfg["labels"].get(k, k)) for k in kinds]
        ax.legend(
            handles=handles,
            loc=leg_cfg.get("loc", "upper left"),
            fontsize=leg_cfg.get("fontsize", 7),
            frameon=leg_cfg.get("frameon", False),
            ncols=leg_cfg.get("ncols", 1),
        )

    if fig_cfg.get("tight_layout", True):
        fig.tight_layout()

    save_cfg = plot_cfg["SAVE_CFG"]
    if save_cfg.get("save", False):
        out_path = _script_dir() / save_cfg["filename"]
        fig.savefig(
            out_path,
            dpi=save_cfg.get("dpi", 300),
            transparent=save_cfg.get("transparent", True),
            bbox_inches=save_cfg.get("bbox_inches", "tight"),
            pad_inches=save_cfg.get("pad_inches", 0.02),
        )
        print(f"[SAVE] Wrote: {out_path.resolve()}")

    return fig, ax


def run_one_plot(plot_cfg: dict, arrays: dict):
    if plot_cfg["FONT_CFG"].get("use_global_rc", True):
        _apply_global_style(plot_cfg)
    plot_from_arrays(plot_cfg, arrays)


def main():
    arrays = get_master_arrays()

    print("\n================== SCALE PLOT ==================")
    run_one_plot(SCALE_PLOT_CFG, arrays)

    print("\n=============== ORIENTATION PLOT ===============")
    run_one_plot(ORIENT_PLOT_CFG, arrays)

    print("\n================ CENTROID PLOT ================")
    run_one_plot(CENTROID_PLOT_CFG, arrays)

    plt.show()


if __name__ == "__main__":
    main()
