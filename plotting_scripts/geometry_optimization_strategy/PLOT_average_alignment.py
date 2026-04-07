#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------- #
# 1) INSERT YOUR FILES HERE
# --------------------------------------------------------------------------- #

results_files = {
    "C": "PATH/TO/C/alignment_results",
    "CR": "PATH/TO/CR/alignment_results",
}

# --------------------------------------------------------------------------- #
# 2) COMPUTE AVG ALIGNMENT + STD/sqrt(N)
# --------------------------------------------------------------------------- #
model_ids = list(results_files.keys())
AVG_align_per_model_id = []
STD_align_per_model_id = []

# Key inside each pickle
ALIGNMENT_KEY = "alignment_normalized"

for (model_id, file_path) in results_files.items():
    # Unified handling: if file_path is a directory, pool across its .pkl files;
    # otherwise treat it as a single .pkl. Then compute stats once from the pooled vals.
    if os.path.isdir(file_path) or file_path.endswith(".pkl"):
        if os.path.isdir(file_path):
            pkl_paths = sorted(
                os.path.join(file_path, fn)
                for fn in os.listdir(file_path)
                if fn.endswith(".pkl")
            )
        else:
            pkl_paths = [file_path]

        vals = []
        for p in pkl_paths:
            with open(p, "rb") as f:
                data = pickle.load(f)
            # gather pair-wise normalized alignments (per-file structure produced upstream)
            vals.extend(float(v) for v in data[ALIGNMENT_KEY].values())

        print(len(vals))
        mean = float(np.mean(vals))
        std = float(np.std(vals)) / np.sqrt(len(vals))  # SEM
        print(std)

        # collect for bar plot
        AVG_align_per_model_id.append(mean)
        STD_align_per_model_id.append(std)
        continue


# --------------------------------------------------------------------------- #
# 3) PLOT (full control; Arial everywhere; no TeX; computation unchanged)
# --------------------------------------------------------------------------- #
from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter, AutoMinorLocator, LogLocator, FuncFormatter
import textwrap

def tex_log_formatter(x, pos=None, base=10):
    # Optional log tick formatter (mathtext). Will still be Arial because we set mathtext.*.
    if not np.isfinite(x) or x <= 0:
        return ""
    k = int(round(np.log(x) / np.log(base)))
    return r"$10^{%+d}$" % k

PLOT_CFG = {
    # ---- Figure & export ----
    "figsize": (0.5, 0.85),           # inches
    "dpi": 300,
    "tight_layout": False,
    "constrained_layout": False,
    "save_path": "Global_alignment.svg",
    "save_transparent": True,
    "bbox_inches": "tight",
    "show_fig": True,

    # ---- Fonts: Arial everywhere (text + mathtext) ----
    "font_family": "Arial",
    "try_load_font_from_file": False,     # set True if you want to add Arial .ttf manually
    "font_paths": [
        # e.g. "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        # e.g. "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        # e.g. "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
    ],
    "set_mathtext_to_font_family": True,
    "text_usetex": False,
    "axes_unicode_minus": False,
    "axes_formatter_use_mathtext": False,

    # ---- Global font size (optional convenience) ----
    # If you want "everything 7pt", set global_fontsize=7 and also set the individual
    # sizes below to 7 (or leave them and set them all to 7 explicitly).
    "global_fontsize": 7,

    # ---- Title ----
    "show_title": True,
    "title": "avg. alignment",
    "title_fontsize": 7,
    "title_pad": 8.0,

    # ---- Axis labels ----
    "x_label": "Network",
    "y_label": r"avg. $A_{\mu \nu}$",
    "show_x_label": True,
    "show_y_label": True,
    "x_label_fontsize": 7,
    "y_label_fontsize": 7,
    "x_label_pad": None,     # None -> matplotlib default; else number
    "y_label_pad": None,

    # ---- Axes scale & range ----
    "y_scale": "linear",     # "linear" or "log"
    "y_lim": None,           # (ymin, ymax) or None
    "ylim_pad": 0.06,        # headroom if auto
    "y_log_base": 10,

    "x_lim": None,           # (xmin, xmax) or None
    "x_margin": None,        # None or number; applied via ax.margins(x=...)

    # ---- Tick controls ----
    "show_xticks": True,
    "show_yticks": True,

    "xtick_labelsize": 7,
    "ytick_labelsize": 7,
    "xtick_rotation": 0,
    "xtick_ha": "center",
    "xtick_pad": 2,
    "ytick_pad": 2,

    "tick_length": 2.0,
    "tick_width": 0.6,

    # ---- Y ticks ----
    "y_ticks": None,             # explicit ticks if desired (list)
    "y_ticklabels": None,        # custom tick strings (list)
    "y_num_ticks": 4,            # approx count if y_ticks None
    "y_minor_num_ticks": None,   # AutoMinorLocator n for linear scale only

    # ---- Optional formatter settings ----
    "use_log_tex_formatter": True,   # only used if y_scale == "log"

    # ---- X tick labels (bar names) ----
    "x_tick_labels": None,       # list[str] or None to use model_ids
    "wrap_xtick_labels": True,
    "max_model_id_chars_per_line": 12,

    # ---- Spines ----
    "spines_visible": {"top": False, "right": False, "left": True, "bottom": True},
    "spines_linewidth": None,    # None -> leave default; else number

    # ---- Grid ----
    "show_grid": True,
    "grid_axis": "y",
    "grid_alpha": 0.6,
    "grid_linestyle": ":",
    "grid_linewidth": None,

    # ---- Bars ----
    "bar_width": 0.4,
    "cluster_sep": 0.40,         # distance between bars
    "bar_alpha": 0.95,
    "bar_edgecolor": "black",
    "bar_linewidth": 0.2,

    # Control per-bar colors: either a single color or a list of colors length n
    "bar_color": "tab:blue",
    "bar_colors": None,          # e.g. ["tab:blue","tab:orange",...] or None

    # ---- Errorbar styling ----
    "show_errorbars": True,
    "error_cap_size": 1,
    "error_linewidth": 0.5,
    "error_color": "black",
    "error_capthick": None,      # None -> matplotlib default; else number
    "error_zorder": None,        # None -> default; else number
    "bars_zorder": None,

    # ---- Legend (optional; single series) ----
    "show_legend": False,
    "legend_label": "Average normalised alignment",
    "legend_loc": "upper left",
    "legend_ncols": 1,
    "legend_fontsize": 7,
    "legend_frameon": False,
    "legend_handlelength": 0.6,
    "legend_handleheight": 0.4,
    "legend_handletextpad": 0.4,
    "legend_labelspacing": 0.05,
    "legend_borderaxespad": 0.2,
    "legend_columnspacing": 0.6,
    "legend_markerscale": 1.0,

    # ---- Background ----
    "figure_facecolor": "white",
    "axes_facecolor": "white",
}

# ---- Arial everywhere (including mathtext) ----
def _add_font_if_exists(path):
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        return True
    return False

# Core rcParams
rcParams["text.usetex"] = bool(PLOT_CFG["text_usetex"])
rcParams["axes.unicode_minus"] = bool(PLOT_CFG["axes_unicode_minus"])
rcParams["axes.formatter.use_mathtext"] = bool(PLOT_CFG["axes_formatter_use_mathtext"])

# Optional: add font files
if PLOT_CFG["try_load_font_from_file"]:
    for p in PLOT_CFG["font_paths"]:
        _add_font_if_exists(p)

family_name = PLOT_CFG["font_family"]

rcParams.update({
    "font.family": family_name,
    "font.sans-serif": [family_name, "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": PLOT_CFG["global_fontsize"],
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

# ---- Data to plot ----
labels = model_ids
means  = np.asarray(AVG_align_per_model_id, dtype=float)
stds   = np.asarray(STD_align_per_model_id, dtype=float)

# ---- X positions ----
n = len(labels)
bar_width   = PLOT_CFG["bar_width"]
cluster_sep = PLOT_CFG["cluster_sep"]
x = np.arange(n) * (bar_width + cluster_sep)

# ---- Errorbars: omit when std == 0 (globally or per-bar) ----
def _yerr_or_none(std_arr):
    if std_arr is None:
        return None
    std_arr = np.asarray(std_arr, dtype=float)
    if np.all(np.isclose(std_arr, 0.0, rtol=0, atol=1e-15)):
        return None
    return np.ma.masked_where(np.isclose(std_arr, 0.0, rtol=0, atol=1e-15), std_arr)

yerr = _yerr_or_none(stds) if PLOT_CFG["show_errorbars"] else None

err_kw = dict(
    capsize=PLOT_CFG["error_cap_size"],
    elinewidth=PLOT_CFG["error_linewidth"],
)
if PLOT_CFG["error_color"] is not None:
    err_kw["ecolor"] = PLOT_CFG["error_color"]
if PLOT_CFG["error_capthick"] is not None:
    err_kw["capthick"] = PLOT_CFG["error_capthick"]

# ---- Figure/Axes ----
fig, ax = plt.subplots(figsize=PLOT_CFG["figsize"], dpi=PLOT_CFG["dpi"],
                       constrained_layout=PLOT_CFG["constrained_layout"])
fig.patch.set_facecolor(PLOT_CFG["figure_facecolor"])
ax.set_facecolor(PLOT_CFG["axes_facecolor"])

ax.set_yscale(PLOT_CFG["y_scale"])

# ---- Bars ----
bar_colors = PLOT_CFG["bar_colors"] if PLOT_CFG["bar_colors"] is not None else PLOT_CFG["bar_color"]

bars = ax.bar(
    x, means, yerr=yerr,
    color=bar_colors,
    width=bar_width,
    edgecolor=PLOT_CFG["bar_edgecolor"],
    linewidth=PLOT_CFG["bar_linewidth"],
    alpha=PLOT_CFG["bar_alpha"],
    error_kw=err_kw,
    zorder=PLOT_CFG["bars_zorder"],
)

# ---- X ticks & labels ----
xticklabels = PLOT_CFG["x_tick_labels"] if PLOT_CFG["x_tick_labels"] is not None else labels
if PLOT_CFG["wrap_xtick_labels"]:
    width_wrap = PLOT_CFG["max_model_id_chars_per_line"]
    xticklabels = ["\n".join(textwrap.wrap(str(lbl), width_wrap)) for lbl in xticklabels]

ax.set_xticks(x)
ax.set_xticklabels(xticklabels, rotation=PLOT_CFG["xtick_rotation"], ha=PLOT_CFG["xtick_ha"])

if not PLOT_CFG["show_xticks"]:
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
else:
    ax.tick_params(
        axis="x",
        labelsize=PLOT_CFG["xtick_labelsize"],
        pad=PLOT_CFG["xtick_pad"],
        length=PLOT_CFG["tick_length"],
        width=PLOT_CFG["tick_width"],
    )

# ---- Y label & ticks ----
if PLOT_CFG["show_y_label"]:
    ax.set_ylabel(PLOT_CFG["y_label"], fontsize=PLOT_CFG["y_label_fontsize"], labelpad=PLOT_CFG["y_label_pad"])
else:
    ax.set_ylabel("")

if not PLOT_CFG["show_yticks"]:
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
else:
    ax.tick_params(
        axis="y",
        labelsize=PLOT_CFG["ytick_labelsize"],
        pad=PLOT_CFG["ytick_pad"],
        length=PLOT_CFG["tick_length"],
        width=PLOT_CFG["tick_width"],
    )

# ---- X label ----
if PLOT_CFG["show_x_label"]:
    ax.set_xlabel(PLOT_CFG["x_label"], fontsize=PLOT_CFG["x_label_fontsize"], labelpad=PLOT_CFG["x_label_pad"])
else:
    ax.set_xlabel("")

# ---- Title ----
if PLOT_CFG["show_title"]:
    ax.set_title(PLOT_CFG["title"], fontsize=PLOT_CFG["title_fontsize"], pad=PLOT_CFG["title_pad"])

# ---- X/Y limits ----
if PLOT_CFG["x_lim"] is not None:
    ax.set_xlim(*PLOT_CFG["x_lim"])
if PLOT_CFG["x_margin"] is not None:
    ax.margins(x=float(PLOT_CFG["x_margin"]))

if PLOT_CFG["y_lim"] is not None:
    ax.set_ylim(*PLOT_CFG["y_lim"])
else:
    if n:
        if PLOT_CFG["y_scale"] == "linear":
            ymax = float(np.max(means + (stds if stds.size else 0.0))) if n else 1.0
            ax.set_ylim(0.0, ymax * (1 + PLOT_CFG["ylim_pad"]))
        else:
            pos = means[means > 0]
            if pos.size == 0:
                pos = np.array([1e-8])
            ymax = float(np.max(pos))
            ymin = float(np.min(pos))
            ax.set_ylim(max(ymin * 0.8, 1e-12), ymax * (1 + PLOT_CFG["ylim_pad"]))

# ---- Y ticks: explicit or approx count ----
if PLOT_CFG.get("y_ticks") is not None:
    ax.yaxis.set_major_locator(FixedLocator(PLOT_CFG["y_ticks"]))
    if PLOT_CFG.get("y_ticklabels") is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(PLOT_CFG["y_ticklabels"]))
elif PLOT_CFG.get("y_num_ticks") is not None:
    if PLOT_CFG["y_scale"] == "log":
        ax.yaxis.set_major_locator(LogLocator(base=PLOT_CFG.get("y_log_base", 10),
                                              numticks=PLOT_CFG["y_num_ticks"]))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=PLOT_CFG["y_num_ticks"]))

if PLOT_CFG.get("y_minor_num_ticks") and PLOT_CFG["y_scale"] == "linear":
    ax.yaxis.set_minor_locator(AutoMinorLocator(PLOT_CFG["y_minor_num_ticks"]))

# ---- Log tick formatter (when y is log) ----
if PLOT_CFG.get("y_scale", "linear") == "log" and PLOT_CFG["use_log_tex_formatter"]:
    ax.yaxis.set_major_formatter(FuncFormatter(tex_log_formatter))

# ---- Grid ----
if PLOT_CFG["show_grid"]:
    grid_kw = dict(
        axis=PLOT_CFG["grid_axis"],
        alpha=PLOT_CFG["grid_alpha"],
        linestyle=PLOT_CFG["grid_linestyle"],
    )
    if PLOT_CFG["grid_linewidth"] is not None:
        grid_kw["linewidth"] = float(PLOT_CFG["grid_linewidth"])

    ax.grid(True, **grid_kw)
else:
    ax.grid(False)


# ---- Spines ----
for spine, vis in PLOT_CFG["spines_visible"].items():
    if spine in ax.spines:
        ax.spines[spine].set_visible(vis)
        if PLOT_CFG["spines_linewidth"] is not None:
            ax.spines[spine].set_linewidth(PLOT_CFG["spines_linewidth"])

# ---- Legend (optional) ----
if PLOT_CFG["show_legend"]:
    ax.legend(
        handles=[bars[0]],
        labels=[PLOT_CFG["legend_label"]],
        loc=PLOT_CFG["legend_loc"],
        ncol=PLOT_CFG["legend_ncols"],
        fontsize=PLOT_CFG["legend_fontsize"],
        frameon=PLOT_CFG["legend_frameon"],
        handlelength=PLOT_CFG["legend_handlelength"],
        handleheight=PLOT_CFG["legend_handleheight"],
        handletextpad=PLOT_CFG["legend_handletextpad"],
        labelspacing=PLOT_CFG["legend_labelspacing"],
        borderaxespad=PLOT_CFG["legend_borderaxespad"],
        columnspacing=PLOT_CFG["legend_columnspacing"],
        markerscale=PLOT_CFG["legend_markerscale"],
    )

# ---- Layout / Save / Show ----
if PLOT_CFG["tight_layout"]:
    fig.tight_layout()

if PLOT_CFG["save_path"]:
    fig.savefig(PLOT_CFG["save_path"], bbox_inches=PLOT_CFG["bbox_inches"],
                transparent=PLOT_CFG["save_transparent"])

if PLOT_CFG["show_fig"]:
    plt.show()
else:
    plt.close(fig)
