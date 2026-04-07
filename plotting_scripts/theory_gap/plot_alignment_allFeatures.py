import pickle
import numpy as np
import LIB_plot_utility as myplt
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from matplotlib.ticker import FuncFormatter
import json
import hashlib

# ============================================================
# ===================== SMALL HELPERS ========================
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
    """Tick formatter for log scale: returns $10^{k}$ style labels with ASCII '-'."""
    if not np.isfinite(x) or x <= 0:
        return ""
    k = int(round(np.log(x) / np.log(base)))
    return r"$10^{%+d}$" % k


def _apply_global_style():
    """Set global matplotlib style (Arial, white background, mathtext, ASCII minus)."""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["axes.unicode_minus"] = False


# ============================================================
# ======================== DATA CONFIG =======================
# ============================================================

paths = {
    "CR": {
        "global_L2_DimReduced_path": "PATH/TO/CR/global_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/CR/local_results",
        "theory_regression_folder": "PATH/TO/CR/theory_results",
    },
    "C": {
        "global_L2_DimReduced_path": "PATH/TO/C/global_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/C/local_results",
        "theory_regression_folder": "PATH/TO/C/theory_results",
    },
}

# ==== Features: PLOT ALL 4 ====
FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
FEATURE_TICK_LABELS = [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"]

best_gamma_theory_method = "outer"  # "inner"
std_of_avg_split_and_file = True

# ============================================================
# ======================== PLOT CONFIG =======================
# ============================================================

# Bump to invalidate cache if you change *computation* (not just plot styling)
SCRIPT_VERSION = "2026-03-07.v1"

# Cache computed quantities (per paths/features) to avoid recomputation.
# Pattern mirrors the caching used in your consolidated plotting script.
CACHE_CFG = {
    "enabled": True,
    "force_recompute": False,
    "recursive_folder_scan": True,
    "cache_filename": "plot_alignment_allFeatures__cache.pkl",
    "verbose": True,
}

SAVE_CFG = {
    "save": True,
    "filename": "alignment_allFeatures.svg",
    "transparent": True,
    "bbox_inches": "tight",
    "pad_inches": 0.02,
    "dpi": 300,
}

FONT_CFG = {
    "family": "Arial",
    "use_global_rc": True,
}

STYLE_CFG = {
    "hide_top_right_spines": True,
    "spine_color": "#333333",
    "spine_linewidth": 1.0,
}

TEXT_CFG = {
    "title": r"alignment $a$",
    "x_label": None,          # None or "" to suppress
    "y_label": r"$a$",

    "title_size": 7,
    "x_label_size": 7,
    "y_label_size": 7,

    "x_tick_size": 5,
    "y_tick_size": 5,
    "x_tick_rotation": 0,
}

FIG_CFG = {
    "figsize": (1.6, 1.45),
    "dpi": 150,
    "tight_layout": True,
    "constrained_layout": False,
}

AXES_CFG = {
    # Set to "log" or "linear"
    "y_scale": "linear",
    "y_lim": (0.0, 1.0),
    "x_lim": None,

    "y_ticks": None,
    "y_ticklabels": None,
    "y_log_base": 10,

    "show_grid": True,
    "grid_axis": "y",
    "grid_alpha": 0.4,
    "grid_linestyle": ":",
    "grid_linewidth": 0.8,
}

GROUP_CFG = {
    "bar_width": 0.1,            # width of each bar
    "intra_group_gap": 0.05,     # gap between models within feature
    "inter_group_gap": 0.20,     # gap between features
}

BAR_CFG = {
    "color": "tab:green",
    "alpha": 0.95,
    "edgecolor": "#333333",
    "linewidth": 0.5,

    "show_error": True,
    "error_cap_size": 3,
    "error_linewidth": 0.8,
    "error_color": "#333333",
}

LEGEND_CFG = {
    "show": False,
    "label": r"$a$",
    "loc": "upper right",
    "fontsize": 7,
    "frameon": False,
    "ncols": 1,
}

# Alignment definition option:
#   - a_definition = 'c'  -> a = c
#   - a_definition = 'c2' -> a = c^2
ALIGNMENT_CFG = {
    'a_definition': 'c2',
}


# ============================================================
# =========== NUMPY 2.x → 1.x PICKLE COMPAT SHIM ============
# ============================================================

import sys, importlib
def alias(old, new):
    try:
        sys.modules[old] = importlib.import_module(new)
    except Exception:
        pass

alias('numpy._core', 'numpy.core')
for name in ('numeric', 'multiarray', '_multiarray_umath', 'shape_base', 'overrides'):
    alias(f'numpy._core.{name}', f'numpy.core.{name}')


# ============================================================
# === CORE: ALIGNMENT a PER MODEL/FEATURE ===================
# ============================================================

def compute_alignment_for_model_and_feature(model_cfg: dict, feature: str):
    """
    For a given model and feature, compute:
      - align_avg  = a (defined from detailed_results["c"] via ALIGNMENT_CFG["a_definition"])
      - align_std
    using the same global/local/theory logic as the original script, per-feature.
    """
    global_L2_path = model_cfg["global_L2_DimReduced_path"]
    local_folder   = model_cfg["local_Gamma_DimReduced_folder"]
    theory_folder  = model_cfg["theory_regression_folder"]

    # ---------- GLOBAL CV (for consistency) ----------
    print(f"    GLOBAL (feature={feature})")
    path = Path(global_L2_path).expanduser()
    multi_file_global = False

    if path.is_file() and path.suffix.lower() == ".pkl":
        print("\t\tSingle .pkl global")
        mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(global_L2_path, feature)
    elif path.is_dir():
        print("\t\tMultiple .pkl global")
        multi_file_global = True
        mse_avg_per_file = []
        mse_avg_std_per_file = []
        mse_per_outer_split_per_file = []
        pkl_files = sorted(path.glob("*.pkl"))
        if not pkl_files:
            subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
            collected = []
            for d in subdirs:
                pkls = sorted(d.glob("*.pkl"))
                if len(pkls) == 1:
                    collected.append(pkls[0])
                elif len(pkls) > 1:
                    warnings.warn(f"Multiple .pkl files in {d}; using most recent.")
                    pkls.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    collected.append(pkls[0])
            pkl_files = collected

        for i, pkl_file in enumerate(pkl_files):
            print(f"\t\t  global file {i+1}/{len(pkl_files)}")
            if pkl_file.is_file():
                mse_avg_i, mse_avg_std_i, labels_variance, mse_per_outer_split = \
                    myplt.retrieve_global_mse(pkl_file, feature, return_mse_per_outer_split=True)
                mse_avg_per_file.append(mse_avg_i)
                mse_avg_std_per_file.append(mse_avg_std_i)
                mse_per_outer_split_per_file.append(mse_per_outer_split)

        mse_avg = np.mean(mse_avg_per_file)
        mse_avg_std_avg = np.mean(mse_avg_std_per_file)
        std_across_files = np.std(mse_avg_per_file)
        mse_avg_std = np.sqrt((mse_avg_std_avg ** 2) + (std_across_files ** 2))

        mse_per_outer_split_per_file = np.stack(mse_per_outer_split_per_file, axis=0)
        mse_std_of_avg_split_and_file = np.std(mse_per_outer_split_per_file) / np.sqrt(
            mse_per_outer_split_per_file.size
        )
    else:
        raise FileNotFoundError(f"Global path does not exist or isn’t a .pkl file/folder: {path}")

    if std_of_avg_split_and_file and multi_file_global:
        cv_global_mse_avg_std = mse_std_of_avg_split_and_file
    else:
        cv_global_mse_avg_std = mse_avg_std  # kept for completeness

    # ---------- LOCAL CV + BEST GAMMA FOR THEORY ----------
    print("\t    LOCAL")
    path_local = Path(local_folder).expanduser()
    multi_file_local = False

    if path_local.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_local.iterdir()):
        print("\t\tSingle local folder")
        AvgLocalMse_avg, AvgLocalMse_avg_std, best_gamma_per_outer_split = \
            myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                local_folder, feature, return_best_gammas_per_outer_split=True
            )
        if best_gamma_theory_method == "inner":
            best_gamma_theory = float(np.mean(best_gamma_per_outer_split))
        elif best_gamma_theory_method == "outer":
            gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, cv_nmse_gap_fixed_gamma_std = \
                myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                    local_folder, feature, 0.0, 0.0, 1.0
                )
            best_idx = int(np.argmax(cv_nmse_gap_fixed_gamma_avg))
            best_gamma_theory = float(gamma_cv_list[best_idx])
        else:
            raise ValueError(f"Invalid best_gamma_theory_method: {best_gamma_theory_method}")

    elif path_local.is_dir() and any(d.is_dir() for d in path_local.iterdir()):
        print("\t\tMultiple local folders")
        multi_file_local = True

        AvgLocalMse_avg_per_file = []
        AvgLocalMse_avg_std_per_file = []
        LocalMse_per_outer_split_per_file = []
        best_gamma_theory = {}

        run_dirs = sorted([d for d in path_local.iterdir() if d.is_dir()])
        for i, run_dir in enumerate(run_dirs):
            print(f"\t\t  local folder {i+1}/{len(run_dirs)}")
            seed_str = run_dir.name
            (AvgLocalMse_avg_i, AvgLocalMse_avg_std_i,
             best_gamma_per_outer_split, LocalMse_per_outer_split) = \
                myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                    run_dir, feature,
                    return_mse_per_outer_split=True,
                    return_best_gammas_per_outer_split=True,
                )

            AvgLocalMse_avg_per_file.append(AvgLocalMse_avg_i)
            AvgLocalMse_avg_std_per_file.append(AvgLocalMse_avg_std_i)
            LocalMse_per_outer_split_per_file.append(LocalMse_per_outer_split)

            if best_gamma_theory_method == "inner":
                best_gamma_theory[seed_str] = float(np.mean(best_gamma_per_outer_split))
            elif best_gamma_theory_method == "outer":
                gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, cv_nmse_gap_fixed_gamma_std = \
                    myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                        run_dir, feature, 0.0, 0.0, 1.0
                    )
                best_idx = int(np.argmax(cv_nmse_gap_fixed_gamma_avg))
                best_gamma_theory[seed_str] = float(gamma_cv_list[best_idx])
            else:
                raise ValueError(f"Invalid best_gamma_theory_method: {best_gamma_theory_method}")

        AvgLocalMse_avg = np.mean(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std_avg = np.mean(AvgLocalMse_avg_std_per_file)
        std_across_files = np.std(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std = np.sqrt((AvgLocalMse_avg_std_avg ** 2) + (std_across_files ** 2))

        LocalMse_per_outer_split_per_file = np.stack(LocalMse_per_outer_split_per_file, axis=0)
        LocalMse_std_of_avg_split_and_file = np.std(LocalMse_per_outer_split_per_file) / np.sqrt(
            LocalMse_per_outer_split_per_file.size
        )
    else:
        raise FileNotFoundError(f"Local folder does not exist or isn’t a .pkl file/folder: {path_local}")

    # ---------- THEORY → alignment a ----------
    a_definition = str(ALIGNMENT_CFG.get("a_definition", "c"))
    print(f"\t    THEORY (a_definition={a_definition})")
    path_theory = Path(theory_folder).expanduser()

    if path_theory.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_theory.iterdir()):
        print("\t\tSingle theory folder")
        (_emp_gap, _emp_fluct, _centroids, _inf, _scale_exp,
         _orient_exp, _finite, detailed_results) = myplt.retrieve_theory_gap_closest_gamma(
            theory_folder, feature, best_gamma_theory, return_detailed_results=True
        )
        c_val = float(detailed_results["c"])
        if a_definition == "c":
            align_avg = c_val
        elif a_definition == "c2":
            align_avg = c_val ** 2
        else:
            raise ValueError(f"Invalid ALIGNMENT_CFG[a_definition]={a_definition!r}; expected 'c' or 'c2'.")
        align_std = 0.0

    elif path_theory.is_dir() and any(d.is_dir() for d in path_theory.iterdir()):
        print("\t\tMultiple theory folders")
        a_per_file = []
        run_dirs = sorted([d for d in path_theory.iterdir() if d.is_dir()])
        for i, run_dir in enumerate(run_dirs):
            print(f"\t\t  theory folder {i+1}/{len(run_dirs)}")
            seed_str = run_dir.name
            (_emp_gap, _emp_fluct, _centroids, _inf, _scale_exp,
             _orient_exp, _finite, detailed_results) = myplt.retrieve_theory_gap_closest_gamma(
                run_dir, feature, best_gamma_theory[seed_str], return_detailed_results=True
            )
            c_val = float(detailed_results["c"])
            if a_definition == "c":
                a_per_file.append(c_val)
            elif a_definition == "c2":
                a_per_file.append(c_val ** 2)
            else:
                raise ValueError(f"Invalid ALIGNMENT_CFG[a_definition]={a_definition!r}; expected 'c' or 'c2'.")

        align_avg = float(np.mean(a_per_file))
        if std_of_avg_split_and_file:
            align_std = float(np.std(a_per_file) / np.sqrt(len(a_per_file)))
        else:
            align_std = float(np.std(a_per_file))
    else:
        raise FileNotFoundError(f"Theory folder does not exist or isn’t a .pkl file/folder: {path_theory}")

    return align_avg, align_std


# ============================================================
# ============================ CACHE =========================
# ============================================================


def _max_mtime_under_path(path: str | Path, recursive: bool) -> float:
    p = Path(path).expanduser()
    if not p.exists():
        return -1.0
    try:
        if p.is_file():
            return p.stat().st_mtime
        if not p.is_dir():
            return p.stat().st_mtime
        if recursive:
            mt = p.stat().st_mtime
            for sub in p.rglob("*"):
                try:
                    mt = max(mt, sub.stat().st_mtime)
                except Exception:
                    pass
            return mt
        else:
            mt = p.stat().st_mtime
            for sub in p.iterdir():
                try:
                    mt = max(mt, sub.stat().st_mtime)
                except Exception:
                    pass
            return mt
    except Exception:
        return -1.0


def _json_hash(d) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _cache_signature() -> dict:
    sig = dict(
        script_version=SCRIPT_VERSION,
        quantity="alignment_a",
        paths=paths,
        features=FEATURES,
        best_gamma_theory_method=best_gamma_theory_method,
        std_of_avg_split_and_file=std_of_avg_split_and_file,
        a_definition=str(ALIGNMENT_CFG.get('a_definition','c')),
    )

    # Invalidate cache if underlying result files change
    mpaths = {}
    recursive = bool(CACHE_CFG.get("recursive_folder_scan", True))
    for model_name, model_cfg in paths.items():
        for k, p in model_cfg.items():
            if p is None:
                continue
            mpaths[f"{model_name}:{k}"] = _max_mtime_under_path(p, recursive=recursive)
    sig["paths_mtime"] = mpaths
    sig["sig_hash"] = _json_hash(sig)
    return sig


# ============================================================
# =========================== MAIN ===========================
# ============================================================

def main():
    if FONT_CFG.get("use_global_rc", True):
        _apply_global_style()

    model_ids = list(paths.keys())
    n_models = len(model_ids)
    n_features = len(FEATURES)

    # ---- Load from cache or compute ----
    cache_enabled = bool(CACHE_CFG.get("enabled", True))
    force = bool(CACHE_CFG.get("force_recompute", False))
    cache_file = Path(CACHE_CFG.get("cache_filename", "plot_alignment_allFeatures__cache.pkl")).expanduser()
    sig = _cache_signature()

    align_means = None
    align_stds = None
    if cache_enabled and (not force) and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get("signature", {}).get("sig_hash") == sig["sig_hash"]:
                if CACHE_CFG.get("verbose", True):
                    print(f"[cache] Using cached results from: {cache_file}", flush=True)
                align_means = cached["align_means"]
                align_stds = cached["align_stds"]
            else:
                if CACHE_CFG.get("verbose", True):
                    print("[cache] Cache signature mismatch; recomputing.", flush=True)
        except Exception:
            if CACHE_CFG.get("verbose", True):
                print("[cache] Failed to load cache; recomputing.", flush=True)

    if align_means is None or align_stds is None:
        align_means = np.zeros((n_features, n_models), dtype=float)
        align_stds = np.zeros((n_features, n_models), dtype=float)

        for m_idx, (model_name, model_cfg) in enumerate(paths.items()):
            print(f"\nModel: {model_name}")
            for f_idx, feature in enumerate(FEATURES):
                print(f"  Feature: {feature}")
                avg_val, std_val = compute_alignment_for_model_and_feature(model_cfg, feature)
                align_means[f_idx, m_idx] = avg_val
                align_stds[f_idx, m_idx] = std_val

        if cache_enabled:
            payload = dict(signature=sig, align_means=align_means, align_stds=align_stds)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(payload, f)
                if CACHE_CFG.get("verbose", True):
                    print(f"[cache] Wrote cache: {cache_file}", flush=True)
            except Exception as e:
                if CACHE_CFG.get("verbose", True):
                    print(f"[cache] Failed to write cache: {e}", flush=True)

    # =======================================================
    # ========== X positions with inverted grouping =========
    # =======================================================
    # Now: outer grouping = MODEL (CR, then C),
    #      inner grouping = FEATURES within each model.
    bar_width   = GROUP_CFG["bar_width"]
    feature_gap = GROUP_CFG["intra_group_gap"]   # small gap between features within a model
    model_gap   = GROUP_CFG["inter_group_gap"]   # larger gap between models

    nF, nM = n_features, n_models

    # Width occupied by one model block (all features)
    model_width = nF * bar_width + (nF - 1) * feature_gap

    # Centers of each model block
    model_centers = np.arange(nM, dtype=float) * (model_width + model_gap)

    x_flat     = []
    vals_flat  = []
    std_flat   = []
    tick_labels = []

    for m_idx, mid in enumerate(model_ids):
        center = model_centers[m_idx]
        left_edge = center - model_width / 2.0

        for f_idx, feat_label in enumerate(FEATURE_TICK_LABELS):
            # bar position for this feature inside this model
            x = left_edge + f_idx * (bar_width + feature_gap)

            x_flat.append(x)
            vals_flat.append(align_means[f_idx, m_idx])
            std_flat.append(align_stds[f_idx, m_idx])

            # Feature name on top, model label on bottom
            tick_labels.append(f"{feat_label}\n{mid}")

    x_flat    = np.asarray(x_flat, dtype=float)
    vals_flat = np.asarray(vals_flat, dtype=float)
    std_flat  = np.asarray(std_flat, dtype=float)

    # ----- Figure & axes -----
    fig = plt.figure(
        figsize=FIG_CFG["figsize"],
        dpi=FIG_CFG["dpi"],
        constrained_layout=FIG_CFG["constrained_layout"],
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

    # Y scale (supports "log" or "linear")
    y_scale = AXES_CFG.get("y_scale", "log")
    if y_scale == "log":
        ax.set_yscale("log", base=AXES_CFG.get("y_log_base", 10))
    else:
        ax.set_yscale(y_scale)

    def _yerr_or_none(std_arr):
        if not BAR_CFG["show_error"]:
            return None
        std_arr = np.asarray(std_arr, dtype=float)
        if np.all(np.isclose(std_arr, 0.0, rtol=0, atol=1e-15)):
            return None
        return std_arr

    yerr = _yerr_or_none(std_flat)
    err_kw = dict(
        capsize=BAR_CFG["error_cap_size"],
        lw=BAR_CFG["error_linewidth"],
        ecolor=BAR_CFG["error_color"],
    ) if yerr is not None else None

    bars = ax.bar(
        x_flat,
        vals_flat,
        width=bar_width,
        color=BAR_CFG["color"],
        alpha=BAR_CFG["alpha"],
        edgecolor=BAR_CFG["edgecolor"],
        linewidth=BAR_CFG["linewidth"],
        yerr=yerr,
        error_kw=err_kw,
        label=LEGEND_CFG["label"],
    )

    # X ticks & labels: "feature\nmodel"
    ax.set_xticks(x_flat)
    ax.set_xticklabels(
        tick_labels,
        fontsize=TEXT_CFG["x_tick_size"],
        rotation=TEXT_CFG["x_tick_rotation"],
    )

    # Y limits
    ax.tick_params(axis="y", labelsize=TEXT_CFG["y_tick_size"])

    if AXES_CFG["y_lim"] is not None:
        ax.set_ylim(*AXES_CFG["y_lim"])
    else:
        positive_vals = vals_flat[vals_flat > 0]
        if positive_vals.size == 0:
            ymin, ymax = 1e-8, 1.0
        else:
            ymin = positive_vals.min()
            ymax = positive_vals.max()
            if ymin <= 0:
                ymin = ymax * 1e-4
        ax.set_ylim(ymin, ymax * 1.1)

    if AXES_CFG["x_lim"] is not None:
        ax.set_xlim(*AXES_CFG["x_lim"])

    # Y ticks & formatter
    if AXES_CFG["y_ticks"] is not None:
        ax.set_yticks(AXES_CFG["y_ticks"])
        if AXES_CFG["y_ticklabels"] is not None:
            ax.set_yticklabels(AXES_CFG["y_ticklabels"])
        else:
            if y_scale == "log":
                ax.yaxis.set_major_formatter(FuncFormatter(tex_log_formatter))
    else:
        if y_scale == "log":
            ax.yaxis.set_major_formatter(FuncFormatter(tex_log_formatter))

    # Labels & title
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
            handles=[bars[0]],
            labels=[LEGEND_CFG["label"]],
            loc=LEGEND_CFG["loc"],
            fontsize=LEGEND_CFG["fontsize"],
            frameon=LEGEND_CFG["frameon"],
            ncols=LEGEND_CFG["ncols"],
        )

    if FIG_CFG["tight_layout"]:
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
