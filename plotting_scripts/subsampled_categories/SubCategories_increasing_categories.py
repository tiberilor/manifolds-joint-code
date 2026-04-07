import warnings
from pathlib import Path
import os
import sys
import importlib
import json
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager as fm
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, LogLocator, FuncFormatter

import LIB_plot_utility as myplt


# ============================================================
# ======================== USER CONFIG ========================
# ============================================================

# -------------------------
# Feature
# -------------------------
# feature = "bbox_center_x"
# feature = "bbox_center_y"
# feature = "bbox_x_length"
feature = "bbox_y_length"

model_name = "C"   # used only for titles/filenames; script is otherwise model-agnostic

# -------------------------
# Sampling
# -------------------------
P_list = [8, 16, 32, 64, 128, 265]
seeds = [1, 2, 3, 4]

full_P = 265
full_seed = None  # keep None unless your full run is keyed by an integer seed

# -------------------------
# Paths
# -------------------------
global_subsampled_base_tpl = "PATH/TO/global_subsampled/{P}/{seed}"
local_subsampled_folder_tpl = "PATH/TO/local_subsampled/{P}/{seed}"
theory_subsampled_folder_tpl = "PATH/TO/theory_subsampled/{P}/{seed}"

global_full_path = "PATH/TO/global_full/global_results.pkl"
local_full_folder = "PATH/TO/local_full"
theory_full_folder = "PATH/TO/theory_full"


# -------------------------
# Computation choices
# -------------------------
best_gamma_theory_method = "outer"     # "inner" or "outer"
std_of_avg_split_and_file = True
min_local_mse_for_ratio = 1e-12


# -------------------------
# Aggregation / errorbars (across subsampling seeds)
# -------------------------
seed_errorbar_mode_rel = "seed_std"   # "seed_std", "seed_sem", "internal_only", "seed_std_plus_internal", "none"


# -------------------------
# Cache (reference-style, but per-feature)
# -------------------------
CACHE_CFG = {
    "enabled": True,
    "force_recompute": False,
    "cache_file": "./relative_gap_cache.pkl",  # one file storing entries per feature
    "verbose": True,
}


# ============================================================
# ===================== PLOT CONFIG (STYLE) ===================
# ============================================================
# Arial everywhere; all tweakable fontsize defaults set to 7.
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
        "figsize": (2.2, 1.5),   # (width, height) inches
        "dpi": 300,
        "constrained_layout": False,
        "tight_layout": True,
        "tight_layout_rect": None,
    },

    # ---- Saving ----
    "save": {
        "enabled": True,  # toggle here
        # Available fields: {model_name}, {feature}, {metric_key}, {ext}
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
        "scale": "log",          # "linear" or "log"
        "xlim": (7, 290),            # (xmin, xmax) or None
        "ticks": None,           # explicit list or None
        "ticklabels": None,      # list same length as ticks, or None
        "num_ticks": None,       # if set -> MaxNLocator
        "tick_rotation": 0,
        "tick_ha": "center",
        "tick_fontsize": 7,
        "tick_pad": 2.0,
        # Log tick control (only if scale=="log")
        "log_locator": {
            "base": 10.0,
            "subs": (1.0,),       # major ticks at 10^k
        },
    },

    # ---- Y axis (single metric: rel_gap) ----
    "yaxis": {
        "rel_gap": {
            "title": "Relative local-global gap",
            "ylabel": r"$\Delta E/E_{\mathrm{loc}}$",
            "yscale": "log",      # "linear", "log", "symlog"
            "ylim": None,         # (ymin,ymax) or None
            "yticks": None,       # explicit ticks list or None
            "y_num_ticks": None,  # if set -> MaxNLocator
            # symlog params (only used if yscale=="symlog" OR if log requested but <=0 values and fallback triggers)
            "symlog_linthresh_mode": "fraction_of_median_abs",  # "fixed" or "fraction_of_median_abs"
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
        "theory_finite": {"label": "theo (fin. P)", "color": "tab:green", "marker": "s", "linestyle": "-"},
        "theory_infinite": {"label": "theo (inf. P)", "color": "tab:red", "marker": "D", "linestyle": "-"},
    },
    "curve_defaults": {
        "connect_points": True,
        "alpha": 0.95,
        "linewidth": 0.9,
        "marker": "o",
        "markersize": 3.0,
        # errorbars
        "elinewidth": 0.6,
        "capsize": 1.6,
        "capthick": 0.6,
        "ecolor": None,         # None -> use curve color
        "zorder_marker": 2.0,
        "zorder_error": 3.5,
        "hide_if_all_zero_err": True,
    },

    # ---- Grid & spines ----
    "grid": {
        "show": True,
        "which": "both",       # "major", "minor", "both"
        "axis": "both",        # "x", "y", "both"
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
# ========== NumPy 2.x → 1.x pickle compatibility shim =========
# ============================================================
def _alias(old, new):
    try:
        sys.modules[old] = importlib.import_module(new)
    except Exception:
        pass

_alias("numpy._core", "numpy.core")
for _name in ("numeric", "multiarray", "_multiarray_umath", "shape_base", "overrides"):
    _alias(f"numpy._core.{_name}", f"numpy.core.{_name}")


# ============================================================
# ======================== CACHE HELPERS =======================
# ============================================================
def _try_load_cache(cache_file: Path):
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception as e:
        print(f"[cache] WARNING: could not load cache file '{cache_file}': {e}")
        return None


def _save_cache(cache_file: Path, obj: dict):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[cache] ERROR: could not save cache file '{cache_file}': {e}")
        raise


def _list_all_pkls_under(path: Path):
    if not path.exists():
        return []
    if path.is_file() and path.suffix.lower() == ".pkl":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.pkl"))
    return []


def _signature_for_inputs(
    *,
    feature: str,
    P_list,
    seeds,
    full_P,
    full_seed,
    best_gamma_theory_method,
    std_of_avg_split_and_file,
    min_local_mse_for_ratio,
    # paths
    global_subsampled_base_tpl,
    local_subsampled_folder_tpl,
    theory_subsampled_folder_tpl,
    global_full_path,
    local_full_folder,
    theory_full_folder,
):
    # Collect file mtimes/sizes for all relevant .pkl files.
    # (This is conservative; it ensures cache invalidates if any input changes.)
    items = []

    def add_path(p: str, tag: str):
        pp = Path(p).expanduser()
        for f in _list_all_pkls_under(pp):
            try:
                st = f.stat()
                items.append((f"{tag}::{str(f)}", int(st.st_mtime_ns), int(st.st_size)))
            except FileNotFoundError:
                items.append((f"{tag}::{str(f)}", -1, -1))
        # also include raw configured path string
        items.append((f"CFG::{tag}::{p}", 0, 0))

    # Full paths
    add_path(global_full_path, "global_full")
    add_path(local_full_folder, "local_full")
    add_path(theory_full_folder, "theory_full")

    # Subsampled templates: enumerate actual (P,seed) used
    for P in P_list:
        if P == full_P:
            continue
        for sd in seeds:
            add_path(global_subsampled_base_tpl.format(P=P, seed=sd), f"global_sub_P{P}_seed{sd}")
            add_path(local_subsampled_folder_tpl.format(P=P, seed=sd), f"local_sub_P{P}_seed{sd}")
            add_path(theory_subsampled_folder_tpl.format(P=P, seed=sd), f"theory_sub_P{P}_seed{sd}")

    items = sorted(items, key=lambda x: x[0])

    payload = {
        "feature": feature,
        "P_list": list(P_list),
        "seeds": list(seeds),
        "full_P": int(full_P),
        "full_seed": full_seed,
        "best_gamma_theory_method": str(best_gamma_theory_method),
        "std_of_avg_split_and_file": bool(std_of_avg_split_and_file),
        "min_local_mse_for_ratio": float(min_local_mse_for_ratio),
        "items": items,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"payload": payload, "sha256": digest}


# ============================================================
# ======================== PLOT HELPERS =======================
# ============================================================
def _add_font_if_exists(path: str) -> bool:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return True
    return False


def apply_plot_style(cfg: dict) -> str:
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
        if regular_path:
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
    if override is None:
        return base_cfg
    cfg = dict(base_cfg)
    cfg.update(override)
    # merge nested dicts
    for k in ["font", "figure", "save", "suptitle", "xaxis", "grid", "spines", "legend", "curves", "curve_defaults", "yaxis"]:
        if k in override and isinstance(override[k], dict):
            tmp = dict(base_cfg.get(k, {}))
            tmp.update(override[k])
            cfg[k] = tmp
    # merge yaxis metric sub-dicts
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
        print("WARNING:", msg, "Falling back to symlog.")


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
# ===================== COMPUTATION HELPERS ===================
# ============================================================
def find_seeded_pkl(size_dir: str | Path, seed: int) -> str:
    size_dir = Path(size_dir)
    matches = list(size_dir.glob(f"*seed{seed}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No file matching *seed{seed}.pkl in {size_dir}")
    if len(matches) > 1:
        warnings.warn(f"Multiple matches for seed {seed} in {size_dir}; using first: {matches[0]}")
    return str(matches[0])


def rel_gap_measured_and_std(G, sG, L, sL, minL=1e-12):
    L_eff = float(L)
    if abs(L_eff) < minL:
        L_eff = (np.sign(L_eff) * minL) if (L_eff != 0) else minL
    R = (float(G) - L_eff) / L_eff
    sR = np.sqrt((float(sG) / L_eff) ** 2 + ((float(G) * float(sL)) / (L_eff ** 2)) ** 2)
    return float(R), float(sR)


def rel_gap_theory_and_std(T, sT, L, sL, minL=1e-12):
    L_eff = float(L)
    if abs(L_eff) < minL:
        L_eff = (np.sign(L_eff) * minL) if (L_eff != 0) else minL
    R = float(T) / L_eff
    sR = np.sqrt((float(sT) / L_eff) ** 2 + ((float(T) * float(sL)) / (L_eff ** 2)) ** 2)
    return float(R), float(sR)


def aggregate_over_seeds(values, internal_stds, mode: str):
    v = np.array(values, dtype=float)
    s_int = np.array(internal_stds, dtype=float) if internal_stds is not None else None

    mean = float(np.mean(v)) if v.size else np.nan

    if mode == "none":
        return mean, None

    if v.size <= 1:
        seed_std = 0.0
        seed_sem = 0.0
    else:
        seed_std = float(np.std(v, ddof=1))
        seed_sem = float(seed_std / np.sqrt(v.size))

    if mode == "seed_std":
        return mean, seed_std
    if mode == "seed_sem":
        return mean, seed_sem

    if s_int is None or s_int.size == 0:
        return mean, None

    internal_rms = float(np.sqrt(np.mean(s_int ** 2)))

    if mode == "internal_only":
        return mean, internal_rms
    if mode == "seed_std_plus_internal":
        return mean, float(np.sqrt(seed_std ** 2 + internal_rms ** 2))

    raise ValueError(f"Unknown mode: {mode}")


def _seed_sort_key(x):
    if x is None:
        return (1, 0)
    try:
        return (0, int(x))
    except Exception:
        return (0, 0)


def compute_all_results_for_feature():
    # Build run list
    runs = []
    for P in P_list:
        if P == full_P:
            runs.append(dict(
                P=P,
                seed=full_seed,
                global_L2_DimReduced_path=global_full_path,
                local_Gamma_DimReduced_folder=local_full_folder,
                theory_regression_folder=theory_full_folder,
            ))
        else:
            for sd in seeds:
                base_global = global_subsampled_base_tpl.format(P=P, seed=sd)
                runs.append(dict(
                    P=P,
                    seed=sd,
                    global_L2_DimReduced_path=find_seeded_pkl(base_global, sd),
                    local_Gamma_DimReduced_folder=local_subsampled_folder_tpl.format(P=P, seed=sd),
                    theory_regression_folder=theory_subsampled_folder_tpl.format(P=P, seed=sd),
                ))

    # Compute per (P,seed)
    results = {}  # results[P][seed] = {...}

    for i, run in enumerate(runs):
        P = run["P"]
        sd = run["seed"]
        sd_print = "full" if sd is None else str(sd)

        results.setdefault(P, {})
        results[P].setdefault(sd, {})

        print(f"[{i+1}/{len(runs)}] model={model_name}  P={P}  seed={sd_print}")

        global_path = run["global_L2_DimReduced_path"]
        local_folder = run["local_Gamma_DimReduced_folder"]
        theory_folder = run["theory_regression_folder"]

        # ---- Global
        path = Path(global_path).expanduser()
        multi_file_global = False

        if path.is_file() and path.suffix.lower() == ".pkl":
            mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(global_path, feature)
            mse_std_of_avg_split_and_file = None

        elif path.is_dir():
            multi_file_global = True
            mse_avg_per_file, mse_avg_std_per_file, mse_per_outer_split_per_file = [], [], []

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

            for pkl_file in pkl_files:
                mse_avg_i, mse_avg_std_i, labels_variance, mse_per_outer_split = myplt.retrieve_global_mse(
                    pkl_file, feature, return_mse_per_outer_split=True
                )
                mse_avg_per_file.append(mse_avg_i)
                mse_avg_std_per_file.append(mse_avg_std_i)
                mse_per_outer_split_per_file.append(mse_per_outer_split)

            mse_avg = float(np.mean(mse_avg_per_file))
            mse_avg_std_avg = float(np.mean(mse_avg_std_per_file))
            std_across_files = float(np.std(mse_avg_per_file))
            mse_avg_std = float(np.sqrt((mse_avg_std_avg ** 2) + (std_across_files ** 2)))

            mse_per_outer_split_per_file = np.stack(mse_per_outer_split_per_file, axis=0)
            mse_std_of_avg_split_and_file = float(
                np.std(mse_per_outer_split_per_file) / np.sqrt(mse_per_outer_split_per_file.size)
            )
        else:
            raise FileNotFoundError(f"Global path does not exist or isn’t a .pkl file/folder: {path}")

        cv_global_mse_avg = float(mse_avg)
        cv_global_mse_avg_std = float(mse_std_of_avg_split_and_file) if (std_of_avg_split_and_file and multi_file_global) else float(mse_avg_std)

        # ---- Local
        path = Path(local_folder).expanduser()
        multi_file_local = False
        local_mse_by_subfolder = {}
        local_mse_std_by_subfolder = {}

        if path.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path.iterdir()):
            AvgLocalMse_avg, AvgLocalMse_avg_std, best_gamma_per_outer_split = (
                myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                    local_folder,
                    feature,
                    return_best_gammas_per_outer_split=True
                )
            )

            if best_gamma_theory_method == "inner":
                best_gamma_theory = float(np.mean(best_gamma_per_outer_split))
            elif best_gamma_theory_method == "outer":
                gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = (
                    myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                        local_folder,
                        feature,
                        cv_global_mse_avg,
                        cv_global_mse_avg_std,
                        labels_variance
                    )
                )
                best_gamma_theory = float(gamma_cv_list[int(np.argmax(cv_nmse_gap_fixed_gamma_avg))])
            else:
                raise ValueError(f"best_gamma_theory_method={best_gamma_theory_method} not valid")

            LocalMse_std_of_avg_split_and_file = None

        elif path.is_dir() and any(d.is_dir() for d in path.iterdir()):
            multi_file_local = True
            AvgLocalMse_avg_per_file, AvgLocalMse_avg_std_per_file, LocalMse_per_outer_split_per_file = [], [], []
            best_gamma_theory = {}

            run_directories = sorted([d for d in path.iterdir() if d.is_dir()])
            for run_dir in run_directories:
                run_name = run_dir.name
                AvgLocalMse_avg_i, AvgLocalMse_avg_std_i, best_gamma_per_outer_split, LocalMse_per_outer_split = (
                    myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                        run_dir,
                        feature,
                        return_mse_per_outer_split=True,
                        return_best_gammas_per_outer_split=True
                    )
                )

                AvgLocalMse_avg_per_file.append(AvgLocalMse_avg_i)
                AvgLocalMse_avg_std_per_file.append(AvgLocalMse_avg_std_i)
                LocalMse_per_outer_split_per_file.append(LocalMse_per_outer_split)

                local_mse_by_subfolder[run_name] = float(AvgLocalMse_avg_i)
                local_mse_std_by_subfolder[run_name] = float(AvgLocalMse_avg_std_i)

                if best_gamma_theory_method == "inner":
                    best_gamma_theory[run_name] = float(np.mean(best_gamma_per_outer_split))
                elif best_gamma_theory_method == "outer":
                    gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, _ = (
                        myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
                            run_dir,
                            feature,
                            cv_global_mse_avg,
                            cv_global_mse_avg_std,
                            labels_variance
                        )
                    )
                    best_gamma_theory[run_name] = float(gamma_cv_list[int(np.argmax(cv_nmse_gap_fixed_gamma_avg))])
                else:
                    raise ValueError(f"best_gamma_theory_method={best_gamma_theory_method} not valid")

            AvgLocalMse_avg = float(np.mean(AvgLocalMse_avg_per_file))
            AvgLocalMse_avg_std_avg = float(np.mean(AvgLocalMse_avg_std_per_file))
            std_across_files = float(np.std(AvgLocalMse_avg_per_file))
            AvgLocalMse_avg_std = float(np.sqrt((AvgLocalMse_avg_std_avg ** 2) + (std_across_files ** 2)))

            LocalMse_per_outer_split_per_file = np.stack(LocalMse_per_outer_split_per_file, axis=0)
            LocalMse_std_of_avg_split_and_file = float(
                np.std(LocalMse_per_outer_split_per_file) / np.sqrt(LocalMse_per_outer_split_per_file.size)
            )

        else:
            raise FileNotFoundError(f"Local path does not exist or isn’t a folder: {path}")

        cv_local_mse_avg = float(AvgLocalMse_avg)
        cv_local_mse_avg_std = float(LocalMse_std_of_avg_split_and_file) if (std_of_avg_split_and_file and multi_file_local) else float(AvgLocalMse_avg_std)

        # ---- Measured relative gap
        measured_rel_avg, measured_rel_std = rel_gap_measured_and_std(
            cv_global_mse_avg, cv_global_mse_avg_std,
            cv_local_mse_avg, cv_local_mse_avg_std,
            minL=min_local_mse_for_ratio
        )

        # ---- Theory
        tpath = Path(theory_folder).expanduser()
        if tpath.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in tpath.iterdir()):
            (_, _, _, mse_gap_fluctuations_theory_infinite,
             _, _, mse_gap_fluctuations_theory_finite) = myplt.retrieve_theory_gap_closest_gamma(
                theory_folder, feature, best_gamma_theory
            )

            theory_finite_rel_avg, theory_finite_rel_std = rel_gap_theory_and_std(
                T=float(mse_gap_fluctuations_theory_finite), sT=0.0,
                L=cv_local_mse_avg, sL=cv_local_mse_avg_std,
                minL=min_local_mse_for_ratio
            )
            theory_infinite_rel_avg, theory_infinite_rel_std = rel_gap_theory_and_std(
                T=float(mse_gap_fluctuations_theory_infinite), sT=0.0,
                L=cv_local_mse_avg, sL=cv_local_mse_avg_std,
                minL=min_local_mse_for_ratio
            )

        elif tpath.is_dir() and any(d.is_dir() for d in tpath.iterdir()):
            rel_finite_per_file = []
            rel_infinite_per_file = []

            run_directories = sorted([d for d in tpath.iterdir() if d.is_dir()])
            for run_dir in run_directories:
                run_name = run_dir.name
                bg = best_gamma_theory[run_name] if isinstance(best_gamma_theory, dict) else best_gamma_theory

                (_, _, _, mse_gap_fluctuations_theory_infinite,
                 _, _, mse_gap_fluctuations_theory_finite) = myplt.retrieve_theory_gap_closest_gamma(
                    run_dir, feature, bg
                )

                L0 = local_mse_by_subfolder.get(run_name, cv_local_mse_avg)
                if abs(L0) < min_local_mse_for_ratio:
                    L0 = min_local_mse_for_ratio

                rel_finite_per_file.append(float(mse_gap_fluctuations_theory_finite) / float(L0))
                rel_infinite_per_file.append(float(mse_gap_fluctuations_theory_infinite) / float(L0))

            theory_finite_rel_avg = float(np.mean(rel_finite_per_file))
            theory_infinite_rel_avg = float(np.mean(rel_infinite_per_file))

            std_norm = float(np.sqrt(len(rel_finite_per_file))) if std_of_avg_split_and_file else 1.0
            theory_finite_rel_std_runs = float(np.std(rel_finite_per_file) / std_norm) if len(rel_finite_per_file) > 1 else 0.0
            theory_infinite_rel_std_runs = float(np.std(rel_infinite_per_file) / std_norm) if len(rel_infinite_per_file) > 1 else 0.0

            # conservative combo with local uncertainty:
            theory_finite_rel_std = float(np.sqrt(
                theory_finite_rel_std_runs**2 +
                (abs(theory_finite_rel_avg) * (cv_local_mse_avg_std / max(cv_local_mse_avg, min_local_mse_for_ratio)))**2
            ))
            theory_infinite_rel_std = float(np.sqrt(
                theory_infinite_rel_std_runs**2 +
                (abs(theory_infinite_rel_avg) * (cv_local_mse_avg_std / max(cv_local_mse_avg, min_local_mse_for_ratio)))**2
            ))
        else:
            raise FileNotFoundError(f"Theory path does not exist or isn’t a folder: {tpath}")

        results[P][sd] = {
            "measured_rel": (float(measured_rel_avg), float(measured_rel_std)),
            "theory_finite_rel": (float(theory_finite_rel_avg), float(theory_finite_rel_std)),
            "theory_infinite_rel": (float(theory_infinite_rel_avg), float(theory_infinite_rel_std)),
        }

    return results


def collect_and_aggregate(results, key: str, mode: str):
    P_sorted = sorted(P_list)
    means, errs, all_vals = [], [], []

    for P in P_sorted:
        seeds_here = sorted(results.get(P, {}).keys(), key=_seed_sort_key)
        vals, internal = [], []
        for sd in seeds_here:
            v, s = results[P][sd][key]
            vals.append(v)
            internal.append(s)
        all_vals.extend(vals)

        m, e = aggregate_over_seeds(vals, internal, mode)
        means.append(m)
        errs.append(e)

    yerr = None if all(e is None for e in errs) else np.array([0.0 if e is None else float(e) for e in errs], float)
    return np.array(P_sorted, float), np.array(means, float), yerr, np.array(all_vals, float)


# ============================================================
# ========================= PLOTTING ==========================
# ============================================================
def plot_relative_gap_configurable(*, results, config_override=None):
    cfg = _merge_config(PLOT_CFG, config_override)
    apply_plot_style(cfg)

    metric_key = "rel_gap"
    ycfg = cfg["yaxis"][metric_key]
    shared = cfg["yaxis"].get("_shared", {})
    xcfg = cfg["xaxis"]
    cdef = cfg["curve_defaults"]

    P_sorted, m_meas, yerr_meas, vals_meas = collect_and_aggregate(results, "measured_rel", seed_errorbar_mode_rel)
    P_sorted, m_fin,  yerr_fin,  vals_fin  = collect_and_aggregate(results, "theory_finite_rel", seed_errorbar_mode_rel)
    P_sorted, m_inf,  yerr_inf,  vals_inf  = collect_and_aggregate(results, "theory_infinite_rel", seed_errorbar_mode_rel)

    # optional: hide theory errorbars if all zeros
    yerr_meas = _maybe_hide_zero_err(yerr_meas, bool(cdef.get("hide_if_all_zero_err", False)))
    yerr_fin  = _maybe_hide_zero_err(yerr_fin,  bool(cdef.get("hide_if_all_zero_err", True)))
    yerr_inf  = _maybe_hide_zero_err(yerr_inf,  bool(cdef.get("hide_if_all_zero_err", True)))

    fig, ax = plt.subplots(
        figsize=cfg["figure"]["figsize"],
        dpi=cfg["figure"]["dpi"],
        constrained_layout=cfg["figure"].get("constrained_layout", False),
    )

    # Suptitle
    if cfg["suptitle"]["show"]:
        st = cfg["suptitle"]["text_template"].format(model_name=model_name, metric_key=metric_key, feature=feature)
        fig.suptitle(st, fontsize=cfg["suptitle"]["fontsize"], y=cfg["suptitle"].get("y", 0.98))

    # Titles/labels
    ax.set_title(ycfg["title"], fontsize=shared.get("title_fontsize", cfg["font"]["default_size"]))
    ax.set_xlabel(xcfg["label"], fontsize=xcfg.get("label_fontsize", cfg["font"]["default_size"]))
    ax.set_ylabel(ycfg["ylabel"], fontsize=shared.get("label_fontsize", cfg["font"]["default_size"]))

    # X scale
    ax.set_xscale(xcfg.get("scale", "linear"))
    if xcfg.get("scale", "linear") == "log" and xcfg.get("log_locator", None):
        ll = xcfg["log_locator"]
        ax.xaxis.set_major_locator(LogLocator(base=ll.get("base", 10.0), subs=ll.get("subs", (1.0,))))

    # Y scale with fallback if log and <=0
    all_rel_vals = np.concatenate([vals_meas, vals_fin, vals_inf])
    _set_y_scale_with_fallback(ax, cfg, ycfg, all_rel_vals)

    # Limits
    if xcfg.get("xlim", None) is not None:
        ax.set_xlim(*xcfg["xlim"])
    if ycfg.get("ylim", None) is not None:
        ax.set_ylim(*ycfg["ylim"])

    # Ticks (x)
    if xcfg.get("ticks", None) is not None:
        ax.xaxis.set_major_locator(FixedLocator(xcfg["ticks"]))
        if xcfg.get("ticklabels", None) is not None:
            ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in xcfg["ticklabels"]]))
    elif xcfg.get("num_ticks", None) is not None:
        ax.xaxis.set_major_locator(MaxNLocator(xcfg["num_ticks"]))

    ax.tick_params(
        axis="x",
        labelsize=xcfg.get("tick_fontsize", cfg["font"]["default_size"]),
        pad=xcfg.get("tick_pad", 2.0),
    )
    # If user provided rotation/ha, apply to existing ticklabels
    for tl in ax.get_xticklabels():
        tl.set_rotation(xcfg.get("tick_rotation", 0))
        tl.set_ha(xcfg.get("tick_ha", "center"))

    # Ticks (y)
    ax.tick_params(
        axis="y",
        labelsize=shared.get("tick_fontsize", cfg["font"]["default_size"]),
        pad=shared.get("tick_pad", 2.0),
    )
    if ycfg.get("yticks", None) is not None:
        ax.yaxis.set_major_locator(FixedLocator(ycfg["yticks"]))
        ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ycfg["yticks"]]))
    elif ycfg.get("y_num_ticks", None) is not None:
        ax.yaxis.set_major_locator(MaxNLocator(ycfg["y_num_ticks"]))

    # Grid & spines
    if cfg["grid"]["show"]:
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

    for spine, visible in cfg["spines"].items():
        if spine in ax.spines:
            ax.spines[spine].set_visible(bool(visible))

    def _resolve_style(curve_key: str):
        style = dict(cdef)
        style.update(cfg["curves"].get(curve_key, {}))
        if not style.get("connect_points", True):
            style["linestyle"] = "None"
        return style

    def _plot_curve(curve_key, x, y, yerr):
        st = _resolve_style(curve_key)
        color = st.get("color", None)
        ecolor = st.get("ecolor", None) or color

        cont = ax.errorbar(
            x, y, yerr=yerr,
            fmt=st.get("marker", "o"),
            linestyle=st.get("linestyle", "-"),
            linewidth=st.get("linewidth", 0.9),
            markersize=st.get("markersize", 3.0),
            alpha=st.get("alpha", 0.95),
            color=color,
            ecolor=ecolor if yerr is not None else None,
            elinewidth=st.get("elinewidth", 0.6) if yerr is not None else 0,
            capsize=st.get("capsize", 1.6) if yerr is not None else 0,
            capthick=st.get("capthick", 0.6) if yerr is not None else 0,
            zorder=st.get("zorder_marker", 2.0),
            label=st.get("label", curve_key),
        )
        return cont

    _plot_curve("measured", P_sorted, m_meas, yerr_meas)
    _plot_curve("theory_finite", P_sorted, m_fin, yerr_fin)
    _plot_curve("theory_infinite", P_sorted, m_inf, yerr_inf)

    # Legend
    lcfg = cfg["legend"]
    if lcfg.get("show", True):
        ax.legend(
            loc=lcfg.get("loc", "best"),
            ncol=lcfg.get("ncol", 1),
            fontsize=lcfg.get("fontsize", cfg["font"]["default_size"]),
            frameon=lcfg.get("frameon", False),
            handlelength=lcfg.get("handlelength", 1.2),
            handletextpad=lcfg.get("handletextpad", 0.4),
            labelspacing=lcfg.get("labelspacing", 0.2),
            borderaxespad=lcfg.get("borderaxespad", 0.2),
            columnspacing=lcfg.get("columnspacing", 0.8),
            markerscale=lcfg.get("markerscale", 1.0),
        )

    # Layout
    if cfg["figure"].get("tight_layout", False):
        rect = cfg["figure"].get("tight_layout_rect", None)
        fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()

    # Save
    if cfg["save"]["enabled"]:
        for ext in cfg["save"]["formats"]:
            out_path = cfg["save"]["path_template"].format(
                model_name=model_name, feature=feature, metric_key=metric_key, ext=ext
            )
            fig.savefig(
                out_path,
                transparent=cfg["save"].get("transparent", True),
                bbox_inches=cfg["save"].get("bbox_inches", "tight"),
                pad_inches=cfg["save"].get("pad_inches", 0.02),
            )
            print(f"[save] {out_path}")

    if cfg.get("close_after_save", False):
        plt.close(fig)
    if cfg["show_figures"]:
        plt.show()


# ============================================================
# ===================== CACHE + RUN (MAIN) ====================
# ============================================================
cache_file = Path(CACHE_CFG["cache_file"]).expanduser()
cached_obj = _try_load_cache(cache_file) if CACHE_CFG["enabled"] else None
if cached_obj is None:
    cached_obj = {"version": 1, "entries": {}}  # entries: feature -> {"signature":..., "data":...}

# current signature for THIS feature
sig = _signature_for_inputs(
    feature=feature,
    P_list=P_list,
    seeds=seeds,
    full_P=full_P,
    full_seed=full_seed,
    best_gamma_theory_method=best_gamma_theory_method,
    std_of_avg_split_and_file=std_of_avg_split_and_file,
    min_local_mse_for_ratio=min_local_mse_for_ratio,
    global_subsampled_base_tpl=global_subsampled_base_tpl,
    local_subsampled_folder_tpl=local_subsampled_folder_tpl,
    theory_subsampled_folder_tpl=theory_subsampled_folder_tpl,
    global_full_path=global_full_path,
    local_full_folder=local_full_folder,
    theory_full_folder=theory_full_folder,
)

entries = cached_obj.get("entries", {})
entry = entries.get(feature, None)

use_cache = False
if CACHE_CFG["enabled"] and (not CACHE_CFG["force_recompute"]) and (entry is not None):
    if entry.get("signature", {}).get("sha256", None) == sig["sha256"]:
        use_cache = True

if use_cache:
    if CACHE_CFG["verbose"]:
        print(f"[cache] loading stored results for feature='{feature}' from: {cache_file}")
        print(f"[cache] signature match: {sig['sha256'][:12]}")
    results = entry["data"]
else:
    if CACHE_CFG["enabled"]:
        if CACHE_CFG["verbose"]:
            if CACHE_CFG["force_recompute"]:
                print(f"[cache] FORCE_RECOMPUTE=True -> recomputing feature='{feature}'")
            elif entry is None:
                print(f"[cache] no stored results for feature='{feature}' -> computing and caching")
            else:
                old = entry.get("signature", {}).get("sha256", "")[:12]
                new = sig.get("sha256", "")[:12]
                print(f"[cache] stored results exist for feature='{feature}' but signature changed -> recomputing")
                print(f"[cache] old sig: {old}, new sig: {new}")

    results = compute_all_results_for_feature()

    if CACHE_CFG["enabled"]:
        entries[feature] = {"signature": sig, "data": results}
        cached_obj["entries"] = entries
        _save_cache(cache_file, cached_obj)
        if CACHE_CFG["verbose"]:
            print(f"[cache] saved feature='{feature}' results to: {cache_file}")

# Plot
plot_relative_gap_configurable(results=results, config_override=None)

