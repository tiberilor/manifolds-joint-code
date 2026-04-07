import pickle
import numpy as np
import LIB_plot_utility as myplt
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


# BEGIN DATA CONFIG (unchanged)

models = {
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

feature = "bbox_center_x"
# feature = "bbox_center_y"
# feature = "bbox_x_length"
# feature = "bbox_y_length"
best_gamma_theory_method = "outer"  # "inner"
X = 0.9  # fraction of variance to explain
covariance_rescaling = "none"          # options: "none", "mean_sq_norm", "sq_mean_norm"
# "mean_sq_norm"  -> divide covariance eigenvalues by  ⟨||c||^2⟩
# "sq_mean_norm"  -> divide covariance eigenvalues by (⟨||c||⟩)^2

# END DATA CONFIG (unchanged)


def _n_for_variance(eigs: np.ndarray, frac: float) -> int:
    """Smallest n so that the first n eigenvalues explain `frac` of total variance."""
    if eigs.size == 0:
        return 0
    c = np.cumsum(eigs)
    tot = c[-1]
    if tot <= 0:
        return eigs.size
    n = int(np.searchsorted(c / tot, frac, side="left") + 1)
    return min(max(n, 1), eigs.size)


# containers for per-model statistics required for plotting
model_names = list(models.keys())
per_model_stats = {}  # name -> dict of arrays and scalars

# ------------------------------------------------------------------------------------
# Collect raw eigenvalues and per-category totals first for both models,
# then normalize each model's eigenvalues using the same denominator:
# the average total variance between the two models for each category.
# ------------------------------------------------------------------------------------
_tmp = {}  # temporary store per model before we compute shared-denominator normalization


# <editor-fold desc="PROCESS EACH MODEL (kept same; we **defer** eig normalization)">
for model_name in model_names:
    print(f"\n=== Model: {model_name} ===")
    model = models[model_name]

    print(f"model: {model}")
    global_L2_DimReduced_path = model["global_L2_DimReduced_path"]
    local_Gamma_DimReduced_folder = model["local_Gamma_DimReduced_folder"]
    theory_regression_folder = model["theory_regression_folder"]

    print("\tProcessing global...")
    path = Path(global_L2_DimReduced_path).expanduser()
    mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(global_L2_DimReduced_path, feature)
    cv_global_mse_avg = mse_avg
    cv_global_mse_avg_std = mse_avg_std

    print("\tProcessing local...")
    path = Path(local_Gamma_DimReduced_folder).expanduser()
    gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, cv_nmse_gap_fixed_gamma_std = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(
        local_Gamma_DimReduced_folder, feature, 0.0, 0.0, 1.0
    )
    best_gamma_theory_idx = np.argmax(cv_nmse_gap_fixed_gamma_avg)
    best_gamma_theory = gamma_cv_list[best_gamma_theory_idx]

    local_mses_per_outer_split_per_category, categories_local_mses, _ = myplt.retrieve_local_mses_per_outer_split_closest_gamma(
        local_Gamma_DimReduced_folder, best_gamma_theory, feature
    )
    local_mses_per_category = np.mean(local_mses_per_outer_split_per_category, axis=0)
    print(f"shape of local_mses_per_category {np.shape(local_mses_per_category)}")

    print("\tProcessing theory...")
    (empirical_mse_gap, empirical_mse_gap_fluctuations, mse_gap_centroids, mse_gap_fluctuations_theory_infinite,
     mse_gap_scale_contribution_expanded, mse_gap_orientation_contribution_expanded,
     mse_gap_fluctuations_theory_finite, detailed_results) = myplt.retrieve_theory_gap_closest_gamma(
        theory_regression_folder, feature, best_gamma_theory, return_detailed_results=True
    )

    overlaps_per_category = detailed_results["overlaps_per_category"]**2

    # --- OPTIONAL: rescale covariance eigenvalues using centroid norms ---
    centroids = np.asarray(detailed_results["centroids"])
    centroid_norms = np.linalg.norm(centroids, axis=1)

    mean_centroid_norm_sq = float(np.mean(centroid_norms**2))  # ⟨||c||^2⟩
    mean_centroid_norm = float(np.mean(centroid_norms))        # ⟨||c||⟩

    if covariance_rescaling == "mean_sq_norm":
        if mean_centroid_norm_sq > 0.0:
            eigen_scale = 1.0 / mean_centroid_norm_sq
        else:
            eigen_scale = 1.0
    elif covariance_rescaling == "sq_mean_norm":
        if mean_centroid_norm > 0.0:
            eigen_scale = 1.0 / (mean_centroid_norm**2)
        else:
            eigen_scale = 1.0
    else:
        eigen_scale = 1.0

    print(
        f"{model_name}: ⟨||c||⟩ = {mean_centroid_norm:.4f}, "
        f"⟨||c||^2⟩ = {mean_centroid_norm_sq:.4f}, "
        f"eigen_scale = {eigen_scale:.4e}"
    )

    eigenvalues_raw = np.asarray(detailed_results["eigenvalues_per_category"]) * eigen_scale
    categories = np.asarray(detailed_results["categories"])
    row_totals = eigenvalues_raw.sum(axis=1)

    PR_per_category = detailed_results["PR_per_category"]

    mse_dict = dict(zip(categories_local_mses, local_mses_per_category))
    local_mses_per_category = np.array([mse_dict[c] for c in categories])

    n_cat = overlaps_per_category.shape[0]
    ov_mean  = overlaps_per_category.mean(axis=0)
    ov_sem   = overlaps_per_category.std(axis=0, ddof=0) / np.sqrt(n_cat)

    ov_cumsum_per_cat = np.cumsum(overlaps_per_category, axis=1)
    ov_cumsum_mean    = ov_cumsum_per_cat.mean(axis=0)
    ov_cumsum_sem     = ov_cumsum_per_cat.std(axis=0, ddof=0) / np.sqrt(n_cat)

    _tmp[model_name] = dict(
        categories=categories,
        eigenvalues_raw=eigenvalues_raw,
        row_totals=row_totals,
        PR_per_category=PR_per_category,
        ov_mean=ov_mean,
        ov_sem=ov_sem,
        ov_cumsum_mean=ov_cumsum_mean,
        ov_cumsum_sem=ov_cumsum_sem,
        n_cat=n_cat,
    )
# </editor-fold>


# ------------------------------------------------------------------------------------
# Build a shared denominator per category = average total variance across models.
# ------------------------------------------------------------------------------------
all_cats = set()
for d in _tmp.values():
    all_cats.update(map(str, d["categories"]))

avg_total_by_cat = {}
for cat in all_cats:
    vals = []
    for d in _tmp.values():
        idx = np.where(d["categories"] == cat)[0]
        if idx.size == 1:
            vals.append(float(d["row_totals"][idx[0]]))
    if len(vals) == 0:
        avg_total_by_cat[cat] = 0.0
    else:
        avg_total_by_cat[cat] = float(np.mean(vals))

per_model_stats = {}
for model_name in model_names:
    d = _tmp[model_name]
    cats = d["categories"]
    eig_raw = d["eigenvalues_raw"]

    denom = np.array([avg_total_by_cat[str(c)] for c in cats], dtype=eig_raw.dtype)

    eig_norm = np.divide(
        eig_raw, denom[:, None],
        out=np.zeros_like(eig_raw),
        where=denom[:, None] != 0
    )

    eig_mean = eig_norm.mean(axis=0)
    eig_sem  = eig_norm.std(axis=0, ddof=0) / np.sqrt(d["n_cat"])

    n_global = _n_for_variance(eig_mean, X)

    pr_mean = float(np.mean(d["PR_per_category"]))

    per_model_stats[model_name] = dict(
        eig_mean=eig_mean,
        eig_sem=eig_sem,
        ov_mean=d["ov_mean"],
        ov_sem=d["ov_sem"],
        ov_cumsum_mean=d["ov_cumsum_mean"],
        ov_cumsum_sem=d["ov_cumsum_sem"],
        n_global=n_global,
        pr_mean=pr_mean,
    )


# =========================
# PLOTTING (CONFIGURABLE): ONLY Overlaps
# =========================

from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter, LogLocator, FuncFormatter
import textwrap

def tex_log_formatter(x, pos=None, base=10):
    # Optional log tick formatter (mathtext). Use only if you enable it below.
    if not np.isfinite(x) or x <= 0:
        return ""
    k = int(round(np.log(x) / np.log(base)))
    return r"$10^{%+d}$" % k

PLOT_CFG = {
    # ---- Figure & export ----
    "figsize": (1.7, 0.5),   # (width, height) inches
    "dpi": 300,
    "save_path": "overlaps_multi_model.svg",
    "save_transparent": True,
    "bbox_inches": "tight",
    "tight_layout": False,
    "constrained_layout": False,

    # ---- Fonts (Arial everywhere; no TeX) ----
    "font_family": "Arial",
    "try_load_font_from_file": False,    # if True, will try to add font files below
    "font_dir": "/usr/share/fonts",      # only used if try_load_font_from_file=True
    "font_candidates": ["Arial.ttf", "ArialMT.ttf"],
    "font_italic": ["Arial Italic.ttf", "Arial-Italic.ttf", "ArialMT-Italic.ttf"],
    "font_bold":   ["Arial Bold.ttf",   "Arial-Bold.ttf",   "ArialMT-Bold.ttf"],
    "set_mathtext_to_font_family": True, # makes mathtext use the same font family
    "text_usetex": False,
    "axes_unicode_minus": False,
    "axes_formatter_use_mathtext": False,

    # ---- Titles & labels ----
    "show_suptitle": False,
    "suptitle": f"",
    "suptitle_fontsize": 7,
    "suptitle_pad": 8.0,

    "xlabel": "Rank",
    "overlap_ylabel": "overlap",
    "eig_ylabel": "eig.",  # unused here
    "show_xlabel": True,
    "show_overlap_ylabel": True,
    "show_eig_ylabel": True,  # unused here
    "xlabel_fontsize": 7,
    "ylabel_fontsize": 7,

    # ---- Axes scales & ranges ----
    "x_scale": "log",          # "log" or "linear"
    "x_lim": (0.9, 30),        # (xmin, xmax)
    "overlap_y_scale": "linear",
    "eig_y_scale": "linear",   # unused here
    "overlap_y_lim": None,
    "eig_y_lim": None,         # unused here

    # ---- Ticks ----
    "xtick_labelsize": 7,
    "ytick_labelsize": 7,
    "xtick_pad": 2,
    "ytick_pad": 2,

    "x_ticks": None,
    "overlap_y_ticks": [0, 0.02],
    "eig_y_ticks": None,       # unused here

    "x_num_ticks": None,
    "x_log_base": 10,          # base used by LogLocator (when x_scale="log" and x_num_ticks is not None)
    "overlap_y_num_ticks": None,
    "eig_y_num_ticks": None,   # unused here

    # Optional: enable mathtext log formatter on x-axis (kept off by default)
    "x_use_tex_log_formatter": False,

    # ---- Per-model markers/colors/LINESTYLES (you control these) ----
    "per_model_style": {
        "C+R": {"marker": "o", "color": "tab:blue",
                "overlap_linestyle": "None", "eig_linestyle": "-"},
        "C":   {"marker": "x", "color": "tab:orange",
                "overlap_linestyle": "None", "eig_linestyle": ":"},
    },

    # ---- Global defaults (used when per-model override not provided) ----
    "overlap_marker_size": 2.0,
    "overlap_linestyle": "None",
    "overlap_linewidth": 0.5,

    "eig_marker_size": 2.0,    # unused here
    "eig_linestyle": "-",      # unused here
    "eig_linewidth": 0.5,      # unused here

    # ---- Error bars ----
    "error_cap_length": 1.2,
    "error_cap_linewidth": 0.6,
    "error_line_linewidth": 0.5,
    "error_ecolor": "k",

    # ---- Z-order so error bars appear ON TOP of markers ----
    "marker_zorder": 2.0,
    "error_zorder": 3.5,

    # ---- PR vertical lines ----
    "show_pr_lines": False,
    "pr_linestyle": "--",
    "pr_linewidth": 0.8,
    "pr_alpha": 0.9,

    # ---- Grid & spines ----
    "show_grid": True,
    "grid_axis": "both",
    "grid_alpha": 0.5,
    "grid_linestyle": ":",
    "spines_visible": {"top": False, "right": False, "left": True, "bottom": True},

    # ---- Legend ----
    "overlap_legend": {
        "show": True,
        "loc": "upper right",
        "ncols": 1,
        "fontsize": 7,
        "frameon": False,
        "handlelength": 1.2,
        "handleheight": 0.6,
        "handletextpad": 0.4,
        "labelspacing": 0.2,
        "borderaxespad": 0.2,
        "columnspacing": 0.8,
        "markerscale": 1.0,
    },
    "eig_legend": {  # unused here
        "show": True,
        "loc": "upper right",
        "ncols": 1,
        "fontsize": 7,
        "frameon": False,
        "handlelength": 1.2,
        "handleheight": 0.6,
        "handletextpad": 0.4,
        "labelspacing": 0.2,
        "borderaxespad": 0.2,
        "columnspacing": 0.8,
        "markerscale": 1.0,
    },
}

# ---- Font setup (Arial) ----
def _add_font_if_exists(path):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return True
    return False

rcParams["text.usetex"] = bool(PLOT_CFG["text_usetex"])
rcParams["axes.unicode_minus"] = bool(PLOT_CFG["axes_unicode_minus"])
rcParams["axes.formatter.use_mathtext"] = bool(PLOT_CFG["axes_formatter_use_mathtext"])

if PLOT_CFG["try_load_font_from_file"]:
    regular_path = None
    for fn in PLOT_CFG["font_candidates"]:
        p = os.path.join(PLOT_CFG["font_dir"], fn)
        if _add_font_if_exists(p):
            regular_path = p
            break
    for fn in PLOT_CFG["font_italic"] + PLOT_CFG["font_bold"]:
        _add_font_if_exists(os.path.join(PLOT_CFG["font_dir"], fn))

    if regular_path:
        family_name = fm.FontProperties(fname=regular_path).get_name()
    else:
        family_name = PLOT_CFG["font_family"]
else:
    family_name = PLOT_CFG["font_family"]

rcParams.update({
    "font.family": family_name,
    "font.sans-serif": [family_name, "DejaVu Sans", "Liberation Sans", "sans-serif"],
})

if PLOT_CFG["set_mathtext_to_font_family"]:
    rcParams.update({
        "mathtext.default": "regular",
        "mathtext.fontset": "custom",
        "mathtext.rm": family_name,
        "mathtext.it": f"{family_name}:italic",
        "mathtext.bf": f"{family_name}:bold",
        "mathtext.sf": family_name,
    })

# ---------- Build figure (1 row x 1 col): Overlaps ----------
fig, ax_ov = plt.subplots(
    1, 1, figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"],
    constrained_layout=PLOT_CFG["constrained_layout"],
)

# Helper: color/marker/linestyles per model
prop_cycle_colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', [])
def style_for_model(idx, name):
    style = PLOT_CFG["per_model_style"].get(name, {})
    color = style.get("color", (prop_cycle_colors[idx % len(prop_cycle_colors)] if prop_cycle_colors else None))
    marker = style.get("marker", "o")
    ov_ls = style.get("overlap_linestyle", PLOT_CFG["overlap_linestyle"])
    eg_ls = style.get("eig_linestyle",     PLOT_CFG["eig_linestyle"])
    ov_lw = style.get("overlap_linewidth", PLOT_CFG["overlap_linewidth"])
    eg_lw = style.get("eig_linewidth",     PLOT_CFG["eig_linewidth"])
    return color, marker, ov_ls, eg_ls, ov_lw, eg_lw

def _raise_errorbar(container, z_error, cap_lw=None, bar_lw=None):
    try:
        for cap in container.caplines:
            cap.set_zorder(z_error)
            if cap_lw is not None:
                cap.set_linewidth(cap_lw)
    except Exception:
        pass
    try:
        for blc in container.barlinecols:
            blc.set_zorder(z_error)
            if bar_lw is not None:
                blc.set_linewidths(bar_lw)
    except Exception:
        pass

# Scales
ax_ov.set_yscale(PLOT_CFG["overlap_y_scale"])
ax_ov.set_xscale("log" if PLOT_CFG["x_scale"] == "log" else "linear")

# Grid & spines
if PLOT_CFG["show_grid"]:
    ax_ov.grid(True, axis=PLOT_CFG["grid_axis"],
               alpha=PLOT_CFG["grid_alpha"], linestyle=PLOT_CFG["grid_linestyle"])
else:
    ax_ov.grid(False)
for spine, vis in PLOT_CFG["spines_visible"].items():
    if spine in ax_ov.spines:
        ax_ov.spines[spine].set_visible(vis)

# X-limits (controllable)
max_n_global = max(stats["n_global"] for stats in per_model_stats.values()) if len(per_model_stats) else 1
default_xlim = (0.5, max_n_global + 0.5)
xlim = PLOT_CFG["x_lim"] if PLOT_CFG["x_lim"] is not None else default_xlim
ax_ov.set_xlim(*xlim)

handles_ov, labels_ov = [], []

for j, model_name in enumerate(model_names):
    stats = per_model_stats[model_name]
    c, m, ov_ls, eg_ls, ov_lw, eg_lw = style_for_model(j, model_name)
    xg = np.arange(1, stats["n_global"] + 1)

    h_ov = ax_ov.errorbar(
        xg, stats["ov_mean"][:stats["n_global"]],
        yerr=stats["ov_sem"][:stats["n_global"]],
        fmt=m, markersize=PLOT_CFG["overlap_marker_size"],
        linestyle=ov_ls, linewidth=ov_lw,
        color=c,
        ecolor=(PLOT_CFG["error_ecolor"] if PLOT_CFG["error_ecolor"] else c),
        elinewidth=PLOT_CFG["error_line_linewidth"],
        capsize=PLOT_CFG["error_cap_length"],
        capthick=PLOT_CFG["error_cap_linewidth"],
        zorder=PLOT_CFG["marker_zorder"],
    )
    _raise_errorbar(
        h_ov, PLOT_CFG["error_zorder"],
        cap_lw=PLOT_CFG["error_cap_linewidth"],
        bar_lw=PLOT_CFG["error_line_linewidth"],
    )

    if PLOT_CFG["show_pr_lines"]:
        ax_ov.axvline(stats["pr_mean"], linestyle=PLOT_CFG["pr_linestyle"],
                      linewidth=PLOT_CFG["pr_linewidth"], alpha=PLOT_CFG["pr_alpha"], color=c)

    handle_ov = h_ov.lines[0] if hasattr(h_ov, "lines") and h_ov.lines else h_ov
    handles_ov.append(handle_ov)
    labels_ov.append(model_name)

# Axis labels
if PLOT_CFG["show_overlap_ylabel"]:
    ax_ov.set_ylabel(PLOT_CFG["overlap_ylabel"], fontsize=PLOT_CFG["ylabel_fontsize"])
if PLOT_CFG["show_xlabel"]:
    ax_ov.set_xlabel(PLOT_CFG["xlabel"], fontsize=PLOT_CFG["xlabel_fontsize"])

# Tick params
ax_ov.tick_params(axis="x", labelsize=PLOT_CFG["xtick_labelsize"], pad=PLOT_CFG["xtick_pad"])
ax_ov.tick_params(axis="y", labelsize=PLOT_CFG["ytick_labelsize"], pad=PLOT_CFG["ytick_pad"])

# Optional explicit ticks (X)
if PLOT_CFG["x_ticks"] is not None:
    ax_ov.xaxis.set_major_locator(FixedLocator(PLOT_CFG["x_ticks"]))
    ax_ov.xaxis.set_major_formatter(FixedFormatter([str(t) for t in PLOT_CFG["x_ticks"]]))
elif PLOT_CFG["x_num_ticks"] is not None and PLOT_CFG["x_scale"] == "log":
    ax_ov.xaxis.set_major_locator(LogLocator(base=PLOT_CFG["x_log_base"], numticks=PLOT_CFG["x_num_ticks"]))
    if PLOT_CFG["x_use_tex_log_formatter"]:
        ax_ov.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: tex_log_formatter(x, pos, base=PLOT_CFG["x_log_base"])))

# Optional explicit ticks (Y)
if PLOT_CFG["overlap_y_ticks"] is not None:
    ax_ov.yaxis.set_major_locator(FixedLocator(PLOT_CFG["overlap_y_ticks"]))
    ax_ov.yaxis.set_major_formatter(FixedFormatter([str(t) for t in PLOT_CFG["overlap_y_ticks"]]))
elif PLOT_CFG["overlap_y_num_ticks"] is not None:
    ax_ov.yaxis.set_major_locator(MaxNLocator(PLOT_CFG["overlap_y_num_ticks"]))

# Y limits (controllable)
if PLOT_CFG["overlap_y_lim"] is not None:
    ax_ov.set_ylim(*PLOT_CFG["overlap_y_lim"])

# Legend
ov_leg_cfg = PLOT_CFG["overlap_legend"]
if ov_leg_cfg.get("show", True) and handles_ov:
    ax_ov.legend(
        handles_ov, labels_ov,
        loc=ov_leg_cfg.get("loc", "upper right"),
        ncol=ov_leg_cfg.get("ncols", 1),
        fontsize=ov_leg_cfg.get("fontsize", 8),
        frameon=ov_leg_cfg.get("frameon", False),
        handlelength=ov_leg_cfg.get("handlelength", 1.2),
        handleheight=ov_leg_cfg.get("handleheight", 0.6),
        handletextpad=ov_leg_cfg.get("handletextpad", 0.4),
        labelspacing=ov_leg_cfg.get("labelspacing", 0.2),
        borderaxespad=ov_leg_cfg.get("borderaxespad", 0.2),
        columnspacing=ov_leg_cfg.get("columnspacing", 0.8),
        markerscale=ov_leg_cfg.get("markerscale", 1.0),
    )

# Suptitle
if PLOT_CFG["show_suptitle"]:
    fig.suptitle(PLOT_CFG["suptitle"], fontsize=PLOT_CFG["suptitle_fontsize"], y=0.98)

# Layout & Save
if PLOT_CFG["tight_layout"]:
    fig.tight_layout()
if PLOT_CFG["save_path"]:
    fig.savefig(PLOT_CFG["save_path"], bbox_inches=PLOT_CFG["bbox_inches"],
                transparent=PLOT_CFG["save_transparent"])

plt.show()
