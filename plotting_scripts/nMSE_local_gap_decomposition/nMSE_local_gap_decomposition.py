import pickle
import numpy as np
import LIB_plot_utility as myplt
import os
from pathlib import Path

import matplotlib.pyplot as plt

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


def _apply_global_style():
    """Set global matplotlib style (Arial, white background, etc.)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "sans-serif",
    ]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    # Use mathtext for axes tick formatting (matches other scripts)
    plt.rcParams["axes.formatter.use_mathtext"] = True


# ============================================================
# ======================== DATA CONFIG =======================
# ============================================================

# Networks (models) to compare.
# Replace the example relative paths below with the paths used in your local copy of the repo/results.
paths = {
    "CR": {
        "global_L2_DimReduced_path": "PATH/TO/CR/global_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/CR/local_results",
    },
    "C": {
        "global_L2_DimReduced_path": "PATH/TO/C/global_results.pkl",
        "local_Gamma_DimReduced_folder": "PATH/TO/C/local_results",
    },
}
# All 4 bbox features
FEATURES = ["bbox_center_x", "bbox_center_y", "bbox_x_length", "bbox_y_length"]
FEATURE_TICK_LABELS = [r"$C_h$", r"$C_v$", r"$L_h$", r"$L_v$"]

# If True, pool SEM across ALL outer-split values from ALL files (when directories)
STD_OF_AVG_SPLIT_AND_FILE = True

# ============================================================
# ======================== PLOT CONFIG =======================
# ============================================================

SAVE_CFG = {
    "save": True,
    "filename": "MSE_local_gap_decomposition.svg",
    "transparent": True,
    "bbox_inches": "tight",
    "pad_inches": 0.02,
    "dpi": 300,  # save DPI
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
    "title": r"nMSE decomposition (local + gap)",
    "x_label": None,                      # None or "" to suppress
    "y_label": "nMSE",
    "y2_label": r"$\sqrt{\mathrm{nMSE}}$",  # used only if secondary axis is enabled

    "title_size": 7,
    "x_label_size": 7,
    "y_label_size": 7,
    "y2_label_size": 7,

    "x_tick_size": 7,
    "y_tick_size": 5,
    "y2_tick_size": 5,

    "x_tick_rotation": 0,   # rotation in degrees

    "figsize": (2.2, 1.3),  # (width, height) in inches
    "dpi": 150,             # on-screen DPI
    "tight_layout": True,
    "constrained_layout": False,
}

AXES_CFG = {
    # Primary (left) axis = nMSE
    "y_lim": (0.0, 0.22),                   # (ymin, ymax) or None for auto
    "x_lim": None,
    "y_ticks": None,                 # None => automatic
    "y_ticklabels": None,            # None => automatic
    # Secondary (right) axis = sqrt(nMSE), OPTIONAL
    "use_secondary_yaxis": False,    # <-- default: NO secondary axis
    "y2_ticks": None,                # ticks in sqrt(nMSE) units
    "y2_ticklabels": None,

    "show_grid": True,
    "grid_axis": "y",
    "grid_alpha": 0.25,
    "grid_linestyle": "-",
    "grid_linewidth": 0.8,
}

# Group/bars layout
GROUP_CFG = {
    "bar_width": 0.36,        # width of each bar
    "intra_group_gap": 0.06,  # gap between features within the same model group
    "inter_group_gap": 0.60,  # gap between model groups
}

BAR_CFG = {
    "show_error": True,
    "local_color": "tab:orange",      # color for local nMSE
    "gap_color": "tab:blue",        # color for gap nMSE (global - local)
    "bar_edgecolor": None,
    "bar_linewidth": 0.4,
    "bar_alpha": 0.95,

    "error_cap_size": 2,
    "error_linewidth": 0.6,
    "error_color": "#333333",
}

LEGEND_CFG = {
    "show": True,
    "labels": {
        "local": r"$E_{\mathrm{loc}}$",
        "gap":   r"$\Delta E$",
    },
    "loc": "upper left",
    "fontsize": 7,
    "frameon": False,
    "ncols": 1,
}

# ============================================================
# ======================== DATA LOADING ======================
# ============================================================

def compute_nmse_for_model_and_feature(global_path_str, local_folder_str, feature, std_of_avg_split_and_file):
    """
    Compute global and local nMSE (mean and std) for a single model-feature pair.
    """
    # ---------- GLOBAL ----------
    print(f"  feature: {feature}")
    print("\tProcessing global...")

    path = Path(global_path_str).expanduser()
    multi_file_global = False

    if path.is_file() and path.suffix.lower() == ".pkl":
        print("\t\tProcessing a single global file")
        mse_avg, mse_avg_std, labels_variance = myplt.retrieve_global_mse(global_path_str, feature)

    elif path.is_dir():
        print("\t\tProcessing multiple global files")
        multi_file_global = True
        mse_avg_per_file = []
        mse_avg_std_per_file = []
        mse_per_outer_split_per_file = []

        pkl_files = sorted(path.glob("*.pkl"))
        for i, pkl_file in enumerate(pkl_files):
            print(f"\t\t  global file {i+1}/{len(pkl_files)}")
            if pkl_file.is_file():
                mse_avg, mse_avg_std, labels_variance, mse_per_outer_split = myplt.retrieve_global_mse(
                    pkl_file, feature, return_mse_per_outer_split=True
                )
                mse_avg_per_file.append(mse_avg)
                mse_avg_std_per_file.append(mse_avg_std)
                mse_per_outer_split_per_file.append(mse_per_outer_split)

        mse_avg = np.mean(mse_avg_per_file)
        mse_avg_std_avg = np.mean(mse_avg_std_per_file)
        std_across_files = np.std(mse_avg_per_file)
        mse_avg_std = np.sqrt((mse_avg_std_avg ** 2) + (std_across_files ** 2))

        mse_per_outer_split_per_file = np.stack(mse_per_outer_split_per_file, axis=0)
        mse_std_of_avg_split_and_file = np.std(mse_per_outer_split_per_file) / np.sqrt(
            mse_per_outer_split_per_file.size
        )
        nmse_std_of_avg_split_and_file = mse_std_of_avg_split_and_file / labels_variance
    else:
        raise FileNotFoundError(f"Global path does not exist or isn’t a .pkl file/folder: {path}")

    nmse_avg = mse_avg / labels_variance
    nmse_avg_std = mse_avg_std / labels_variance

    if std_of_avg_split_and_file and multi_file_global:
        global_nmse_std = nmse_std_of_avg_split_and_file
    else:
        global_nmse_std = nmse_avg_std

    # ---------- LOCAL ----------
    print("\tProcessing local...")
    path_local = Path(local_folder_str).expanduser()
    multi_file_local = False

    if path_local.is_dir() and any(p.is_file() and p.suffix.lower() == ".pkl" for p in path_local.iterdir()):
        print("\t\tProcessing a single local folder")
        AvgLocalMse_avg, AvgLocalMse_avg_std = myplt.retrieve_avg_local_mse_gamma_regularization_fast(
            local_folder_str, feature
        )

    elif path_local.is_dir() and any(d.is_dir() for d in path_local.iterdir()):
        print("\t\tProcessing multiple local folders")
        multi_file_local = True

        AvgLocalMse_avg_per_file = []
        AvgLocalMse_avg_std_per_file = []
        LocalMse_per_outer_split_per_file = []

        run_directories = sorted([d for d in path_local.iterdir() if d.is_dir()])
        for i, run_dir in enumerate(run_directories):
            print(f"\t\t  local folder {i+1}/{len(run_directories)}")

            AvgLocalMse_avg, AvgLocalMse_avg_std, LocalMse_per_outer_split = \
                myplt.retrieve_avg_local_mse_gamma_regularization_fast(
                    run_dir, feature, return_mse_per_outer_split=True
                )

            AvgLocalMse_avg_per_file.append(AvgLocalMse_avg)
            AvgLocalMse_avg_std_per_file.append(AvgLocalMse_avg_std)
            LocalMse_per_outer_split_per_file.append(LocalMse_per_outer_split)

        AvgLocalMse_avg = np.mean(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std_avg = np.mean(AvgLocalMse_avg_std_per_file)
        std_across_files = np.std(AvgLocalMse_avg_per_file)
        AvgLocalMse_avg_std = np.sqrt((AvgLocalMse_avg_std_avg ** 2) + (std_across_files ** 2))

        LocalMse_per_outer_split_per_file = np.stack(LocalMse_per_outer_split_per_file, axis=0)
        LocalMse_std_of_avg_split_and_file = np.std(LocalMse_per_outer_split_per_file) / np.sqrt(
            LocalMse_per_outer_split_per_file.size
        )
        LocalNMSE_std_of_avg_split_and_file = LocalMse_std_of_avg_split_and_file / labels_variance

    else:
        raise FileNotFoundError(f"Local folder does not exist or isn’t a .pkl file/folder: {path_local}")

    AvgLocalNMSE_avg = AvgLocalMse_avg / labels_variance
    AvgLocalNMSE_avg_std = AvgLocalMse_avg_std / labels_variance

    if std_of_avg_split_and_file and multi_file_local:
        local_nmse_std = LocalNMSE_std_of_avg_split_and_file
    else:
        local_nmse_std = AvgLocalNMSE_avg_std

    return nmse_avg, global_nmse_std, AvgLocalNMSE_avg, local_nmse_std


# ============================================================
# ========================= MAIN PLOT ========================
# ============================================================

def main():
    if FONT_CFG.get("use_global_rc", True):
        _apply_global_style()

    model_ids = list(paths.keys())
    n_models = len(model_ids)
    n_features = len(FEATURES)

    # Arrays: [n_features, n_models]
    global_nmse_avg = np.zeros((n_features, n_models), dtype=float)
    global_nmse_std = np.zeros((n_features, n_models), dtype=float)
    local_nmse_avg = np.zeros((n_features, n_models), dtype=float)
    local_nmse_std = np.zeros((n_features, n_models), dtype=float)

    # ----- Load data -----
    for m_idx, (model_name, model_cfg) in enumerate(paths.items()):
        print(f"\nModel: {model_name}")
        g_path = model_cfg["global_L2_DimReduced_path"]
        l_folder = model_cfg["local_Gamma_DimReduced_folder"]

        for f_idx, feat in enumerate(FEATURES):
            g_mean, g_std, l_mean, l_std = compute_nmse_for_model_and_feature(
                g_path,
                l_folder,
                feat,
                STD_OF_AVG_SPLIT_AND_FILE,
            )
            global_nmse_avg[f_idx, m_idx] = g_mean
            global_nmse_std[f_idx, m_idx] = g_std
            local_nmse_avg[f_idx, m_idx] = l_mean
            local_nmse_std[f_idx, m_idx] = l_std

    # ----- Local + gap (global - local) -----
    g_means = global_nmse_avg
    g_stds = global_nmse_std
    l_means = local_nmse_avg
    l_stds = local_nmse_std

    gap_means = g_means - l_means
    gap_stds = np.sqrt(g_stds**2 + l_stds**2)

    # =======================================================
    # ========== X positions with inverted grouping =========
    # =======================================================
    # We now group outer by MODEL (CR, then C), and within each model
    # we show the 4 FEATURES as separate (stacked) bars.
    bar_width   = GROUP_CFG["bar_width"]
    feature_gap = GROUP_CFG["intra_group_gap"]   # small gap between features within a model
    model_gap   = GROUP_CFG["inter_group_gap"]   # larger gap between models

    nF, nM = n_features, n_models

    # Width occupied by one model block (all 4 features)
    model_width = nF * bar_width + (nF - 1) * feature_gap

    # Centers of each model group along x
    model_centers = np.arange(nM, dtype=float) * (model_width + model_gap)

    x_flat            = []
    local_flat        = []
    local_std_flat    = []
    gap_flat          = []
    gap_std_flat      = []
    default_xticklabels = []

    for m_idx, mid in enumerate(model_ids):
        center = model_centers[m_idx]
        left_edge = center - model_width / 2.0

        for f_idx, feat_label in enumerate(FEATURE_TICK_LABELS):
            # position of this feature bar inside this model
            x = left_edge + f_idx * (bar_width + feature_gap)

            x_flat.append(x)
            local_flat.append(l_means[f_idx, m_idx])
            local_std_flat.append(l_stds[f_idx, m_idx])
            gap_flat.append(gap_means[f_idx, m_idx])
            gap_std_flat.append(gap_stds[f_idx, m_idx])

            # Feature name on top, model label (C/CR) on bottom
            default_xticklabels.append(f"{feat_label}\n{mid}")

    x_flat         = np.asarray(x_flat, dtype=float)
    local_flat     = np.asarray(local_flat, dtype=float)
    local_std_flat = np.asarray(local_std_flat, dtype=float)
    gap_flat       = np.asarray(gap_flat, dtype=float)
    gap_std_flat   = np.asarray(gap_std_flat, dtype=float)

    # ----- Figure & Axes -----
    fig = plt.figure(
        figsize=TEXT_CFG["figsize"],
        dpi=TEXT_CFG["dpi"],
        constrained_layout=TEXT_CFG["constrained_layout"],
    )
    ax = fig.add_subplot(111)

    # Spines
    if STYLE_CFG.get("hide_top_right_spines", True):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(STYLE_CFG.get("spine_color", "#333333"))
        ax.spines[spine].set_linewidth(STYLE_CFG.get("spine_linewidth", 1.0))

    # Grid
    if AXES_CFG["show_grid"]:
        ax.grid(
            True,
            axis=AXES_CFG["grid_axis"],
            alpha=AXES_CFG["grid_alpha"],
            linestyle=AXES_CFG["grid_linestyle"],
            linewidth=AXES_CFG["grid_linewidth"],
        )

    # ----- Bars (stacked: local + gap) -----
    err_kw = dict(
        capsize=BAR_CFG["error_cap_size"],
        lw=BAR_CFG["error_linewidth"],
        ecolor=BAR_CFG["error_color"],
    )

    if BAR_CFG["show_error"]:
        local_yerr = local_std_flat
        gap_yerr = gap_std_flat
    else:
        local_yerr = None
        gap_yerr = None

    local_bars = ax.bar(
        x_flat,
        local_flat,
        width=bar_width,
        color=BAR_CFG["local_color"],
        alpha=BAR_CFG["bar_alpha"],
        edgecolor=BAR_CFG["bar_edgecolor"],
        linewidth=BAR_CFG["bar_linewidth"],
        yerr=local_yerr,
        error_kw=err_kw if local_yerr is not None else None,
        label=LEGEND_CFG["labels"]["local"],
    )

    gap_bars = ax.bar(
        x_flat,
        gap_flat,
        width=bar_width,
        color=BAR_CFG["gap_color"],
        alpha=BAR_CFG["bar_alpha"],
        edgecolor=BAR_CFG["bar_edgecolor"],
        linewidth=BAR_CFG["bar_linewidth"],
        bottom=local_flat,
        yerr=gap_yerr,
        error_kw=err_kw if gap_yerr is not None else None,
        label=LEGEND_CFG["labels"]["gap"],
    )

    # ----- X ticks -----
    ax.set_xticks(x_flat)
    ax.set_xticklabels(
        default_xticklabels,
        fontsize=TEXT_CFG["x_tick_size"],
        rotation=TEXT_CFG["x_tick_rotation"],
    )

    # ----- Y axis formatting -----
    ax.tick_params(axis="y", labelsize=TEXT_CFG["y_tick_size"])

    if AXES_CFG["y_ticks"] is not None:
        ax.set_yticks(AXES_CFG["y_ticks"])
        if AXES_CFG["y_ticklabels"] is not None:
            ax.set_yticklabels(AXES_CFG["y_ticklabels"])

    # y limits
    if AXES_CFG["y_lim"] is not None:
        ax.set_ylim(*AXES_CFG["y_lim"])
    else:
        # auto from data with small padding
        total_height = local_flat + gap_flat
        total_err = np.sqrt(local_std_flat**2 + gap_std_flat**2)
        ymax = float(np.max(total_height + total_err))
        ax.set_ylim(0.0, ymax * 1.05 if ymax > 0 else 1.0)

    if AXES_CFG["x_lim"] is not None:
        ax.set_xlim(*AXES_CFG["x_lim"])

    # ----- Labels & title -----
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

    # ----- Optional secondary y-axis: sqrt(nMSE) -----
    secax = None
    if AXES_CFG.get("use_secondary_yaxis", False):
        def _nmse_to_sqrt(y):
            return np.sqrt(y)

        def _sqrt_to_nmse(y):
            return y**2

        secax = ax.secondary_yaxis("right", functions=(_nmse_to_sqrt, _sqrt_to_nmse))
        secax.set_ylabel(TEXT_CFG["y2_label"], fontsize=TEXT_CFG["y2_label_size"])
        secax.tick_params(axis="y", labelsize=TEXT_CFG["y2_tick_size"])

        if AXES_CFG.get("y2_ticks") is not None:
            secax.set_yticks(AXES_CFG["y2_ticks"])
            if AXES_CFG.get("y2_ticklabels") is not None:
                secax.set_yticklabels(AXES_CFG["y2_ticklabels"])

    # ----- Legend -----
    if LEGEND_CFG["show"]:
        ax.legend(
            handles=[local_bars[0], gap_bars[0]],
            labels=[LEGEND_CFG["labels"]["local"], LEGEND_CFG["labels"]["gap"]],
            loc=LEGEND_CFG["loc"],
            fontsize=LEGEND_CFG["fontsize"],
            frameon=LEGEND_CFG["frameon"],
            ncols=LEGEND_CFG["ncols"],
        )

    # ----- Layout & save -----
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
