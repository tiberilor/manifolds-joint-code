import pickle
import numpy as np
import LIB_plot_utility as myplt
import os
from pathlib import Path
import warnings
import json
import hashlib
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, AutoMinorLocator

# ============================================================
# ===================== HELPER FUNCTIONS =====================
# ============================================================

def find_seeded_pkl(size_dir: str | Path, seed: int) -> str:
    size_dir = Path(size_dir)
    matches = list(size_dir.glob(f"*seed{seed}.pkl"))  # use .rglob(...) if nested
    if not matches:
        raise FileNotFoundError(f"No file matching *seed{seed}.pkl in {size_dir}")
    if len(matches) > 1:
        # choose the most recent if multiple
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def tex_log_formatter(x, pos=None, base=10):
    """Format positive log ticks as 10^{k} using mathtext."""
    if not np.isfinite(x) or x <= 0:
        return ""
    k = int(round(np.log(x) / np.log(base)))
    return r"$10^{%+d}$" % k  # e.g., $10^{-3}$


def _apply_global_style():
    """Set global matplotlib style (Arial, white background, mathtext on)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False  # use ASCII '-' (U+002D)


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b if abs(b) > eps else (np.sign(b) * eps if b != 0 else eps)))


# ------------------------------------------------------------
# NumPy 2.x → 1.x pickle compatibility shim (from original)
# ------------------------------------------------------------
import sys, importlib
def _alias(old, new):
    try:
        sys.modules[old] = importlib.import_module(new)
    except Exception:
        pass

_alias("numpy._core", "numpy.core")
for _name in ("numeric", "multiarray", "_multiarray_umath", "shape_base", "overrides"):
    _alias(f"numpy._core.{_name}", f"numpy.core.{_name}")


# ============================================================
# ======================== CACHE CONFIG =======================
# ============================================================

SCRIPT_VERSION = "2026-03-06.v_centroid_plus_cache"  # bump to invalidate cache when computations change

CACHE_CFG = {
    "enabled": True,
    "cache_filename": "plot_theory_vs_emp__cache.pkl",  # stored next to this script
    "verbose": True,
}


def _canonicalize_paths_dict(d: dict) -> dict:
    """Convert all paths to normalized absolute strings for stable hashing."""
    out = {}
    for model, cfg in d.items():
        out[model] = {}
        for k, v in cfg.items():
            if v is None:
                out[model][k] = None
            else:
                out[model][k] = os.path.abspath(os.path.expanduser(str(v)))
    return out


def _sig_hash_from_paths(paths_dict: dict) -> str:
    sig = {
        "script_version": SCRIPT_VERSION,
        "paths": _canonicalize_paths_dict(paths_dict),
        "normalize_gap_by_local_error": bool(NORMALIZE_GAP_BY_LOCAL_ERROR),
    }
    blob = json.dumps(sig, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_file_path() -> Path:
    return (Path(__file__).resolve().parent / CACHE_CFG["cache_filename"]).expanduser()


# ============================================================
# ======================== DATA CONFIG =======================
# ============================================================

paths = {
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

# All 4 bbox features
FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
FEATURE_TICK_LABELS = [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"]

# Keep on "outer" for reproducing paper results.
BEST_GAMMA_THEORY_METHOD = "outer"

# If True, pool SEM across ALL outer-split values from ALL files (when directories)
STD_OF_AVG_SPLIT_AND_FILE = True


# If True, normalize the gap (empirical and theory) by the local error E_loc, i.e. plot ΔE/E_loc
NORMALIZE_GAP_BY_LOCAL_ERROR = True


# ============================================================
# ======================== PLOT CONFIG =======================
# ============================================================

SAVE_CFG = {
    "save": True,
    "filename": "NMSE_gap_theory_vs_experiment_all_features.svg",
    "transparent": True,
    "bbox_inches": "tight",
    "pad_inches": 0.02,
    "dpi": 300,
}

FONT_CFG = {"family": "Arial", "use_global_rc": True}

STYLE_CFG = {
    "hide_top_right_spines": True,
    "spine_color": "#333333",
    "spine_linewidth": 1.0,
}

TEXT_CFG = {
    "title": r"local-global gap (theory vs empirical)",
    "x_label": None,
    "y_label": r"$\Delta E \; / \; E_{\mathrm{loc}}$",
    "title_size": 7,
    "x_label_size": 7,
    "y_label_size": 7,
    "x_tick_size": 7,
    "y_tick_size": 5,
    "x_tick_rotation": 0,
    "figsize": (2.5, 1.45),
    "dpi": 150,
    "tight_layout": True,
    "constrained_layout": False,
}

AXES_CFG = {
    "y_scale": "log",
    "y_lim": (4e-4, 20),
    "x_lim": None,
    "y_ticks": (1e-3, 1e-2, 1e-1, 1, 10),
    "y_ticklabels": None,
    "y_log_base": 10,
    "y_num_ticks": 7,
    "y_minor_num_ticks": None,
    "show_grid": True,
    "grid_axis": "y",
    "grid_alpha": 0.25,
    "grid_linestyle": "-",
    "grid_linewidth": 0.8,
}

GROUP_CFG = {
    "bar_width": 0.18,
    "between_model_gap": 0.06,
    "between_feature_gap": 0.60,
}

BAR_CFG = {
    "show_error": True,
    "cv_color": "tab:blue",
    "theory_color": "tab:red",
    "bar_edgecolor": None,
    "bar_linewidth": 0.4,
    "bar_alpha": 0.95,
    "error_cap_size": 2,
    "error_linewidth": 0.6,
    "error_color": "#333333",
}

LEGEND_CFG = {
    "show": True,
    "labels": {"cv": r"emp", "theory": r"theo"},
    "loc": "upper left",
    "fontsize": 7,
    "frameon": False,
    "ncols": 1,
}


# ============================================================
# ======================== DATA LOADING ======================
# ============================================================

def _load_global_mse_any_path(path_str: str, feature: str, std_of_avg_split_and_file: bool):
    """
    Load global MSE stats from either:
      - a single .pkl file
      - a directory of .pkl files

    Returns: (mse_avg, mse_avg_std, labels_variance)
    """
    path = Path(path_str).expanduser()
    multi_file = False

    if path.is_file() and path.suffix.lower() == ".pkl":
        mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(str(path), feature)
        return float(mse_avg), float(mse_avg_std), float(labels_variance)

    if path.is_dir():
        multi_file = True
        mse_avg_per_file = []
        mse_avg_std_per_file = []
        mse_per_outer_split_per_file = []
        labels_var_per_file = []

        pkl_files = sorted(path.glob("*.pkl"))
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl files found under: {path}")

        for i, pkl_file in enumerate(pkl_files):
            if not pkl_file.is_file():
                continue
            mse_avg_i, mse_avg_std_i, labels_variance_i, mse_per_outer_split = myplt.retrieve_global_mse(
                pkl_file, feature, return_mse_per_outer_split=True
            )
            mse_avg_per_file.append(float(mse_avg_i))
            mse_avg_std_per_file.append(float(mse_avg_std_i))
            mse_per_outer_split_per_file.append(np.array(mse_per_outer_split, dtype=float))
            labels_var_per_file.append(float(labels_variance_i))

        mse_avg = float(np.mean(mse_avg_per_file))
        mse_avg_std_avg = float(np.mean(mse_avg_std_per_file))
        std_across_files = float(np.std(mse_avg_per_file))
        mse_avg_std = float(np.sqrt((mse_avg_std_avg**2) + (std_across_files**2)))

        mse_per_outer_split_per_file = np.stack(mse_per_outer_split_per_file, axis=0)
        mse_std_of_avg_split_and_file = float(
            np.std(mse_per_outer_split_per_file) / np.sqrt(mse_per_outer_split_per_file.size)
        )

        if std_of_avg_split_and_file and multi_file:
            mse_avg_std_final = mse_std_of_avg_split_and_file
        else:
            mse_avg_std_final = mse_avg_std

        labels_variance = float(np.mean(labels_var_per_file))
        if np.std(labels_var_per_file) > 1e-9:
            warnings.warn(
                f"labels_variance varies across files under {path}. Using mean={labels_variance}.",
                RuntimeWarning,
            )

        return mse_avg, mse_avg_std_final, labels_variance

    raise FileNotFoundError(f"Global path does not exist or isn’t a .pkl file/folder: {path}")


def compute_gaps_for_model_and_feature(
    global_path_str: str,
    global_fluct_only_path_str: str,
    local_folder_str: str,
    theory_folder_str: str,
    feature: str,
    std_of_avg_split_and_file: bool,
    best_gamma_theory_method: str,
):
    """
    Compute:
      - empirical CV nMSE gap (global - local) mean and std
      - theory finite nMSE gap mean and std

    The theory term is computed for locally centered manifolds (fluctuations only).
    The plotted theory gap adds the centroid-error contribution:
        centroid_error_nmse = global_nmse - global_fluct_only_nmse
        theory_nmse_gap_total = theory_nmse_gap_fluct + centroid_error_nmse
    """

    print(f"  feature: {feature}")

    # ----------------------- GLOBAL (CV) -----------------------
    print("\tProcessing global (centroids+fluct)...")
    cv_global_mse_avg, cv_global_mse_avg_std, labels_variance = _load_global_mse_any_path(
        global_path_str, feature, std_of_avg_split_and_file
    )

    print("\tProcessing global (fluct-only)...")
    cv_global_fluct_mse_avg, cv_global_fluct_mse_avg_std, labels_variance_fluct = _load_global_mse_any_path(
        global_fluct_only_path_str, feature, std_of_avg_split_and_file
    )
    # variance should match; use the centroids global one as reference
    print(f"labels_variance: {labels_variance}, labels_variance_fluct: {labels_variance_fluct}")

    if abs(labels_variance_fluct - labels_variance) > 1e-8 * max(1.0, labels_variance):
        warnings.warn(
            f"labels_variance mismatch between global and fluct-only files: "
            f"{labels_variance} vs {labels_variance_fluct}. Using {labels_variance}.",
            RuntimeWarning,
        )

    # centroid error contribution in nMSE units
    global_nmse = _safe_div(cv_global_mse_avg, labels_variance)
    global_nmse_std = _safe_div(cv_global_mse_avg_std, labels_variance)
    global_fluct_nmse = _safe_div(cv_global_fluct_mse_avg, labels_variance)
    global_fluct_nmse_std = _safe_div(cv_global_fluct_mse_avg_std, labels_variance)

    centroid_error_nmse = global_nmse - global_fluct_nmse
    centroid_error_nmse_std = float(np.sqrt(global_nmse_std**2 + global_fluct_nmse_std**2))

    # ----------------------- LOCAL (CV) -----------------------
    print("\tProcessing local...")
    path_local = Path(local_folder_str).expanduser()
    multi_file_local = False

    if path_local.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_local.iterdir()):
        print("\t\tProcessing a single local folder")
        (AvgLocalMse_avg,
         AvgLocalMse_avg_std,
         best_gamma_per_outer_split) = myplt.retrieve_avg_local_mse_gamma_regularization_fast(
            local_folder_str, feature, return_best_gammas_per_outer_split=True
        )

        if best_gamma_theory_method == "inner":
            best_gamma_theory = np.mean(best_gamma_per_outer_split)
        elif best_gamma_theory_method == "outer":
            gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                local_folder_str,
                feature,
                cv_global_mse_avg,
                cv_global_mse_avg_std,
                labels_variance,
            )
            best_gamma_theory_idx = np.argmax(cv_nmse_gap_fixed_gamma_avg)
            best_gamma_theory = gamma_cv_list[best_gamma_theory_idx]
        else:
            raise ValueError(
                f"Invalid BEST_GAMMA_THEORY_METHOD={best_gamma_theory_method!r}; use 'inner' or 'outer'."
            )

    elif path_local.is_dir() and any(d.is_dir() for d in path_local.iterdir()):
        print("\t\tProcessing multiple local folders")
        multi_file_local = True

        AvgLocalMse_avg_per_file = []
        AvgLocalMse_avg_std_per_file = []
        LocalMse_per_outer_split_per_file = []
        best_gamma_theory = {}  # seed -> gamma

        run_directories = sorted([d for d in path_local.iterdir() if d.is_dir()])
        for i, run_dir in enumerate(run_directories):
            print(f"\t\t  local folder {i+1}/{len(run_directories)}")
            seed_str = run_dir.name

            (AvgLocalMse_avg_i,
             AvgLocalMse_avg_std_i,
             best_gamma_per_outer_split,
             LocalMse_per_outer_split) = myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                run_dir,
                feature,
                return_mse_per_outer_split=True,
                return_best_gammas_per_outer_split=True,
            )

            AvgLocalMse_avg_per_file.append(AvgLocalMse_avg_i)
            AvgLocalMse_avg_std_per_file.append(AvgLocalMse_avg_std_i)
            LocalMse_per_outer_split_per_file.append(LocalMse_per_outer_split)

            if best_gamma_theory_method == "inner":
                best_gamma_theory[seed_str] = np.mean(best_gamma_per_outer_split)
            elif best_gamma_theory_method == "outer":
                gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                    run_dir,
                    feature,
                    cv_global_mse_avg,
                    cv_global_mse_avg_std,
                    labels_variance,
                )
                best_gamma_theory_idx = np.argmax(cv_nmse_gap_fixed_gamma_avg)
                best_gamma_theory[seed_str] = gamma_cv_list[best_gamma_theory_idx]
            else:
                raise ValueError(
                    f"Invalid BEST_GAMMA_THEORY_METHOD={best_gamma_theory_method!r}; use 'inner' or 'outer'."
                )

        AvgLocalMse_avg = np.mean(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std_avg = np.mean(AvgLocalMse_avg_std_per_file)
        std_across_files = np.std(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std = np.sqrt((AvgLocalMse_avg_std_avg**2) + (std_across_files**2))

        LocalMse_per_outer_split_per_file = np.stack(LocalMse_per_outer_split_per_file, axis=0)
        LocalMse_std_of_avg_split_and_file = np.std(LocalMse_per_outer_split_per_file) / np.sqrt(
            LocalMse_per_outer_split_per_file.size
        )

    else:
        raise FileNotFoundError(f"Local folder does not exist or isn’t a .pkl file/folder: {path_local}")

    cv_local_mse_avg = AvgLocalMse_avg
    if std_of_avg_split_and_file and multi_file_local:
        cv_local_mse_avg_std = LocalMse_std_of_avg_split_and_file
    else:
        cv_local_mse_avg_std = AvgLocalMse_avg_std

    # CV gap (MSE -> nMSE)
    cv_mse_gap_avg = cv_global_mse_avg - cv_local_mse_avg
    cv_mse_gap_std = np.sqrt(cv_global_mse_avg_std**2 + cv_local_mse_avg_std**2)
    cv_nmse_gap_avg = cv_mse_gap_avg / labels_variance
    cv_nmse_gap_std = cv_mse_gap_std / labels_variance

    # ----------------------- THEORY (finite P; fluctuations only) -----------------------
    print("\tProcessing theory...")
    path_theory = Path(theory_folder_str).expanduser()
    multi_file_theory = False

    if path_theory.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_theory.iterdir()):
        print("\t\tProcessing a single theory folder")

        (_,  # empirical_mse_gap
         _,  # empirical_mse_gap_fluctuations
         _,  # mse_gap_centroids
         _,  # mse_gap_fluctuations_theory_infinite
         _,  # mse_gap_scale_contribution_expanded
         _,  # mse_gap_orientation_contribution_expanded
         mse_gap_fluctuations_theory_finite) = myplt.retrieve_theory_gap_closest_gamma(
            theory_folder_str,
            feature,
            best_gamma_theory,
        )

        theory_nmse_gap_fluct = mse_gap_fluctuations_theory_finite / labels_variance
        theory_nmse_gap_fluct_std = 0.0

    elif path_theory.is_dir() and any(d.is_dir() for d in path_theory.iterdir()):
        print("\t\tProcessing multiple theory folders")
        multi_file_theory = True

        theory_nmse_gap_fluct_per_file = []

        run_directories = sorted([d for d in path_theory.iterdir() if d.is_dir()])
        for i, run_dir in enumerate(run_directories):
            print(f"\t\t  theory folder {i+1}/{len(run_directories)}")
            seed_str = run_dir.name

            (_, _, _, _, _, _, mse_gap_fluctuations_theory_finite) = myplt.retrieve_theory_gap_closest_gamma(
                run_dir,
                feature,
                best_gamma_theory[seed_str],
            )
            theory_nmse_gap_fluct_per_file.append(mse_gap_fluctuations_theory_finite / labels_variance)

        theory_nmse_gap_fluct = float(np.mean(theory_nmse_gap_fluct_per_file))
        if std_of_avg_split_and_file:
            norm = np.sqrt(len(theory_nmse_gap_fluct_per_file))
        else:
            norm = 1.0
        theory_nmse_gap_fluct_std = float(np.std(theory_nmse_gap_fluct_per_file) / norm)

    else:
        raise FileNotFoundError(f"Theory folder does not exist or isn’t a .pkl file/folder: {path_theory}")

    # ----------------------- ADD centroid-error contribution -----------------------
    theory_nmse_gap_total = float(theory_nmse_gap_fluct + centroid_error_nmse)
    theory_nmse_gap_total_std = float(np.sqrt(theory_nmse_gap_fluct_std**2 + centroid_error_nmse_std**2))

    # ----------------------- OPTIONAL: normalize by local error -----------------------
    if NORMALIZE_GAP_BY_LOCAL_ERROR:
        # local error in nMSE units
        cv_local_nmse_avg = float(_safe_div(cv_local_mse_avg, labels_variance))
        cv_local_nmse_avg_std = float(_safe_div(cv_local_mse_avg_std, labels_variance))

        if not (np.isfinite(cv_local_nmse_avg) and abs(cv_local_nmse_avg) > 1e-12):
            warnings.warn(
                f'Cannot normalize by E_loc because E_loc is non-finite or ~0 (E_loc={cv_local_nmse_avg}). Returning NaNs.',
                RuntimeWarning,
            )
            return float('nan'), float('nan'), float('nan'), float('nan')

        # Empirical: (global - local)/local = global/local - 1
        cv_rel_gap_avg = float(_safe_div(global_nmse, cv_local_nmse_avg) - 1.0)
        cv_rel_gap_std = float(np.sqrt(
            (global_nmse_std / cv_local_nmse_avg) ** 2
            + ((global_nmse * cv_local_nmse_avg_std) / (cv_local_nmse_avg ** 2)) ** 2
        ))

        # Theory: ΔE_theory / E_loc
        th_rel_gap_avg = float(_safe_div(theory_nmse_gap_total, cv_local_nmse_avg))
        th_rel_gap_std = float(np.sqrt(
            (theory_nmse_gap_total_std / cv_local_nmse_avg) ** 2
            + ((theory_nmse_gap_total * cv_local_nmse_avg_std) / (cv_local_nmse_avg ** 2)) ** 2
        ))

        return cv_rel_gap_avg, cv_rel_gap_std, th_rel_gap_avg, th_rel_gap_std

    return cv_nmse_gap_avg, cv_nmse_gap_std, theory_nmse_gap_total, theory_nmse_gap_total_std


# ============================================================
# ========================= MAIN PLOT ========================
# ============================================================

def main():
    # CLI overrides for caching
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_recompute_cache", action="store_true",
                        help="If set, recompute and overwrite cache even if paths are unchanged.")
    parser.add_argument("--no_cache", action="store_true",
                        help="If set, disable cache read/write.")
    args = parser.parse_args()

    if FONT_CFG.get("use_global_rc", True):
        _apply_global_style()

    # ------------------------ Cache load ------------------------
    cache_enabled = bool(CACHE_CFG.get("enabled", True)) and (not args.no_cache)
    cache_force = bool(args.force_recompute_cache)

    cache_path = _cache_file_path()
    sig_hash = _sig_hash_from_paths(paths)

    if cache_enabled and (not cache_force) and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("sig_hash", None) == sig_hash:
                if CACHE_CFG.get("verbose", True):
                    print(f"[cache] Using cached results from: {cache_path}", flush=True)
                cv_gap_mean = cached["cv_gap_mean"]
                cv_gap_std = cached["cv_gap_std"]
                th_gap_mean = cached["th_gap_mean"]
                th_gap_std = cached["th_gap_std"]
                model_ids = cached["model_ids"]
                n_models = cached["n_models"]
                n_features = cached["n_features"]
                # go straight to plotting
                _plot_from_arrays(cv_gap_mean, cv_gap_std, th_gap_mean, th_gap_std, model_ids, n_models, n_features)
                return
            else:
                if CACHE_CFG.get("verbose", True):
                    print("[cache] Cache signature mismatch; recomputing.", flush=True)
        except Exception as e:
            if CACHE_CFG.get("verbose", True):
                print(f"[cache] Failed to load cache; recomputing. ({e})", flush=True)

    # ------------------------ Compute ------------------------
    model_ids = list(paths.keys())
    n_models = len(model_ids)
    n_features = len(FEATURES)

    # Arrays: [n_features, n_models]
    cv_gap_mean = np.zeros((n_features, n_models), dtype=float)
    cv_gap_std = np.zeros((n_features, n_models), dtype=float)
    th_gap_mean = np.zeros((n_features, n_models), dtype=float)
    th_gap_std = np.zeros((n_features, n_models), dtype=float)

    for m_idx, (model_name, model_cfg) in enumerate(paths.items()):
        print(f"\nModel: {model_name}")
        g_path = model_cfg["global_L2_DimReduced_path"]

        gf_path = (
            model_cfg.get("global_L2_DimReduced_fluct_only_path", None)
            or model_cfg.get("global_L2_DimReduced_fluctuations_path", None)
            or model_cfg.get("global_L2_DimReduced_fluctuations_only_path", None)
        )
        if gf_path is None:
            raise KeyError(
                f"Model {model_name} is missing the fluctuations-only global path. "
                f"Add 'global_L2_DimReduced_fluct_only_path' (or 'global_L2_DimReduced_fluctuations_path') to paths[{model_name!r}]."
            )

        l_folder = model_cfg["local_Gamma_DimReduced_folder"]
        t_folder = model_cfg["theory_regression_folder"]

        for f_idx, feat in enumerate(FEATURES):
            (cv_mean, cv_std, th_mean, th_std) = compute_gaps_for_model_and_feature(
                g_path,
                gf_path,
                l_folder,
                t_folder,
                feat,
                STD_OF_AVG_SPLIT_AND_FILE,
                BEST_GAMMA_THEORY_METHOD,
            )
            cv_gap_mean[f_idx, m_idx] = cv_mean
            cv_gap_std[f_idx, m_idx] = cv_std
            th_gap_mean[f_idx, m_idx] = th_mean
            th_gap_std[f_idx, m_idx] = th_std

    # ------------------------ Cache write ------------------------
    if cache_enabled:
        payload = dict(
            sig_hash=sig_hash,
            cv_gap_mean=cv_gap_mean,
            cv_gap_std=cv_gap_std,
            th_gap_mean=th_gap_mean,
            th_gap_std=th_gap_std,
            model_ids=model_ids,
            n_models=n_models,
            n_features=n_features,
        )
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(payload, f)
            if CACHE_CFG.get("verbose", True):
                print(f"[cache] Wrote cache: {cache_path}", flush=True)
        except Exception as e:
            if CACHE_CFG.get("verbose", True):
                print(f"[cache] Failed to write cache: {e}", flush=True)

    _plot_from_arrays(cv_gap_mean, cv_gap_std, th_gap_mean, th_gap_std, model_ids, n_models, n_features)


def _plot_from_arrays(cv_gap_mean, cv_gap_std, th_gap_mean, th_gap_std, model_ids, n_models, n_features):
    # Layout
    bar_width = GROUP_CFG["bar_width"]
    between_model_gap = GROUP_CFG["between_model_gap"]      # inner gap (between features within a model)
    between_feature_gap = GROUP_CFG["between_feature_gap"]  # outer gap (between models)

    nM = n_models
    nF = n_features

    pair_width = 2.0 * bar_width
    inner_gap = between_model_gap
    outer_gap = between_feature_gap

    model_width = nF * pair_width + (nF - 1) * inner_gap
    model_centers = np.arange(nM, dtype=float) * (model_width + outer_gap)

    x_cv = np.zeros((nF, nM), dtype=float)
    x_th = np.zeros((nF, nM), dtype=float)

    for m_idx in range(nM):
        mc = model_centers[m_idx]
        left_edge = mc - model_width / 2.0
        for f_idx in range(nF):
            offset_in_model = f_idx * (pair_width + inner_gap)
            base = left_edge + offset_in_model
            x_cv[f_idx, m_idx] = base + bar_width / 2.0
            x_th[f_idx, m_idx] = base + 3.0 * bar_width / 2.0

    pair_centers = 0.5 * (x_cv + x_th)

    x_cv_flat = x_cv.ravel()
    x_th_flat = x_th.ravel()
    cv_flat = cv_gap_mean.ravel()
    cv_std_flat = cv_gap_std.ravel()
    th_flat = th_gap_mean.ravel()
    th_std_flat = th_gap_std.ravel()

    tick_positions = pair_centers.ravel()
    tick_labels = []
    for f_idx, feat_label in enumerate(FEATURE_TICK_LABELS):
        for m_idx, mid in enumerate(model_ids):
            tick_labels.append(f"{feat_label}\n{mid}")

    fig = plt.figure(
        figsize=TEXT_CFG["figsize"],
        dpi=TEXT_CFG["dpi"],
        constrained_layout=TEXT_CFG["constrained_layout"],
    )
    ax = fig.add_subplot(111)

    if STYLE_CFG.get("hide_top_right_spines", True):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(STYLE_CFG.get("spine_color", "#333333"))
        ax.spines[spine].set_linewidth(STYLE_CFG.get("spine_linewidth", 1.0))

    if AXES_CFG["show_grid"]:
        ax.grid(
            True,
            axis=AXES_CFG["grid_axis"],
            alpha=AXES_CFG["grid_alpha"],
            linestyle=AXES_CFG["grid_linestyle"],
            linewidth=AXES_CFG["grid_linewidth"],
        )

    ax.set_yscale(AXES_CFG.get("y_scale", "linear"))

    err_kw = dict(
        capsize=BAR_CFG["error_cap_size"],
        elinewidth=BAR_CFG["error_linewidth"],
        ecolor=BAR_CFG["error_color"],
    )

    def _yerr_or_none(std_arr):
        if std_arr is None:
            return None
        std_arr = np.asarray(std_arr, dtype=float)
        if np.all(np.isclose(std_arr, 0.0, rtol=0, atol=1e-15)):
            return None
        return np.ma.masked_where(np.isclose(std_arr, 0.0, rtol=0, atol=1e-15), std_arr)

    cv_yerr = _yerr_or_none(cv_std_flat) if BAR_CFG["show_error"] else None
    th_yerr = _yerr_or_none(th_std_flat) if BAR_CFG["show_error"] else None

    bar_kw = dict(
        width=bar_width,
        edgecolor=BAR_CFG["bar_edgecolor"],
        linewidth=BAR_CFG["bar_linewidth"],
        alpha=BAR_CFG["bar_alpha"],
    )

    cv_bars = ax.bar(
        x_cv_flat,
        cv_flat,
        yerr=cv_yerr,
        color=BAR_CFG["cv_color"],
        error_kw=err_kw if cv_yerr is not None else None,
        label=LEGEND_CFG["labels"]["cv"],
        **bar_kw,
    )

    th_bars = ax.bar(
        x_th_flat,
        th_flat,
        yerr=th_yerr,
        color=BAR_CFG["theory_color"],
        error_kw=err_kw if th_yerr is not None else None,
        label=LEGEND_CFG["labels"]["theory"],
        **bar_kw,
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        fontsize=TEXT_CFG["x_tick_size"],
        rotation=TEXT_CFG["x_tick_rotation"],
    )

    ax.tick_params(axis="y", labelsize=TEXT_CFG["y_tick_size"])

    if AXES_CFG["y_ticks"] is not None:
        ax.set_yticks(AXES_CFG["y_ticks"])
        if AXES_CFG["y_ticklabels"] is not None:
            ax.set_yticklabels(AXES_CFG["y_ticklabels"])
    else:
        if AXES_CFG.get("y_scale", "linear") == "log":
            ax.yaxis.set_major_locator(
                LogLocator(base=AXES_CFG.get("y_log_base", 10),
                           numticks=AXES_CFG.get("y_num_ticks", 7))
            )
        else:
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=AXES_CFG.get("y_num_ticks", 7))
            )

    if AXES_CFG.get("y_scale", "linear") == "log":
        ax.yaxis.set_major_formatter(FuncFormatter(tex_log_formatter))

    if AXES_CFG["y_lim"] is not None:
        ax.set_ylim(*AXES_CFG["y_lim"])

    if AXES_CFG["x_lim"] is not None:
        ax.set_xlim(*AXES_CFG["x_lim"])

    title = TEXT_CFG.get("title")
    if title and str(title).strip():
        ax.set_title(title, fontsize=TEXT_CFG["title_size"])

    xlab = TEXT_CFG.get("x_label")
    if xlab and str(xlab).strip():
        ax.set_xlabel(xlab, fontsize=TEXT_CFG["x_label_size"])
    else:
        ax.set_xlabel(None)

    ylab = TEXT_CFG.get("y_label")
    if ylab and str(ylab).strip():
        ax.set_ylabel(ylab, fontsize=TEXT_CFG["y_label_size"])
    else:
        ax.set_ylabel(None)

    if LEGEND_CFG["show"]:
        ax.legend(
            handles=[cv_bars[0], th_bars[0]],
            labels=[LEGEND_CFG["labels"]["cv"], LEGEND_CFG["labels"]["theory"]],
            loc=LEGEND_CFG["loc"],
            fontsize=LEGEND_CFG["fontsize"],
            frameon=LEGEND_CFG["frameon"],
            ncols=LEGEND_CFG["ncols"],
        )

    if TEXT_CFG["tight_layout"]:
        fig.tight_layout()

    if SAVE_CFG["save"]:
        out_path = Path.cwd() / SAVE_CFG["filename"]
        fig.savefig(
            out_path,
            dpi=SAVE_CFG["dpi"],
            bbox_inches=SAVE_CFG["bbox_inches"],
            pad_inches=SAVE_CFG["pad_inches"],
            transparent=SAVE_CFG["transparent"],
        )
        print(f"Saved figure to: {out_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()