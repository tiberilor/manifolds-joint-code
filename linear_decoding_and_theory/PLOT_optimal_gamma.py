import numpy as np
import LIB_plot_utility as myplt
import matplotlib.pyplot as plt

local_Gamma_DimReduced_folder = "path/to/folder/containing/local_regression_results_for_multiple_gamma_values"

# CHOOSE THE FEATURE HERE
feature = "bbox_center_x"
# feature = "bbox_center_y"
# feature = "bbox_x_length"
# feature = "bbox_y_length"

# Gamma-selection method used for theory computation.
# "outer" is the recommended option for reproducing the paper results.
best_gamma_theory_method = "outer"  # "inner", "outer"

# END DATA CONFIG

# Retrieve the local cross-validated NMSE for each gamma.
# The utility function returns a signed quantity, so we flip the sign to plot local CV NMSE.
print("cv gap at fixed gamma")
gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, cv_nmse_gap_fixed_gamma_std \
    = myplt.retrieve_avg_local_nmse_gap_fixed_gamma_list(local_Gamma_DimReduced_folder, feature, 0, 0, 1)
cv_nmse_fixed_gamma_avg = []
for nmse in cv_nmse_gap_fixed_gamma_avg:
    cv_nmse_fixed_gamma_avg.append(-nmse)
cv_nmse_fixed_gamma_std = cv_nmse_gap_fixed_gamma_std

# determine best gamma for the theory
if best_gamma_theory_method == "inner":
    # CV: local_Gamma_DimReduced
    print("Computing local cv mse")
    cv_local_mse_avg, cv_local_mse_avg_std, best_gamma_per_outer_split = myplt.retrieve_avg_local_mse_gamma_regularization(
        local_Gamma_DimReduced_folder, feature, return_best_gammas_per_outer_split=True)
    best_gamma_theory = np.mean(best_gamma_per_outer_split)
elif best_gamma_theory_method == "outer":
    best_gamma_theory_idx = np.argmax(cv_nmse_gap_fixed_gamma_avg)
    best_gamma_theory = gamma_cv_list[best_gamma_theory_idx]
else:
    print(f"method {best_gamma_theory_method} is not a valid method for determining the best gamma for theory")
    exit()

print(f"Best gamma theory: {best_gamma_theory}")

# =======================
# Plot configuration
# =======================
plot_errorbars = False
log_x = True
symlog_y = False
symlog_linthresh = 1e-5
figsize = (9, 6)
legend_loc = "best"
linewidth = 1.9
markersize = 4.5
capsize = 3

main_title = f"Local CV NMSE vs γ — {feature}"
top_title = "Gamma selection for theory computation"
x_label = "γ"
y_label = "Local CV NMSE"

# Colors (edit any)
colors = {
    "cv_fixed_gamma":      "tab:green",
    "best_gamma_line":     "gray",
}

# Optionally save the figure (set to a path or keep None)
savefig_path = None  # e.g., "nmse_gap_vs_gamma.png"

idx_cv = np.argsort(gamma_cv_list)
g_cv = np.asarray(gamma_cv_list, dtype=float)[idx_cv]
cv_nmse_fixed_gamma_avg_arr = np.asarray(cv_nmse_fixed_gamma_avg, dtype=float)[idx_cv]
cv_nmse_fixed_gamma_std_arr = np.asarray(cv_nmse_fixed_gamma_std, dtype=float)[idx_cv]

# =======================
# Build the figure
# =======================
fig, ax_top = plt.subplots(1, 1, figsize=figsize)
fig.suptitle(main_title)

if log_x:
    ax_top.set_xscale("log")

if symlog_y:
    ax_top.set_yscale("symlog", linthresh=symlog_linthresh)

# ----- TOP PLOT -----
ax_top.set_title(top_title)

# CV at fixed gamma (with error bars)
if plot_errorbars:
    ax_top.errorbar(
        g_cv, cv_nmse_fixed_gamma_avg_arr,
        yerr=cv_nmse_fixed_gamma_std_arr,
        label="CV NMSE gap (fixed γ)",
        color=colors["cv_fixed_gamma"], marker="o", linewidth=linewidth, markersize=markersize, capsize=capsize
    )
else:
    ax_top.errorbar(
        g_cv, cv_nmse_fixed_gamma_avg_arr,
        label="Local CV NMSE",
        color=colors["cv_fixed_gamma"], marker="o", linewidth=linewidth, markersize=markersize, capsize=capsize
    )

# Vertical line: best gamma
ax_top.axvline(best_gamma_theory, color=colors["best_gamma_line"], linestyle="-",
               alpha=0.65, label=f"selected γ = {best_gamma_theory}")

ax_top.set_xlabel(x_label)
ax_top.set_ylabel(y_label)
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc=legend_loc)

plt.tight_layout()

if savefig_path is not None:
    fig.savefig(savefig_path, dpi=300, bbox_inches="tight")

plt.show()
