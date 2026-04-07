import pickle
import numpy as np
import os
from scipy.optimize import root_scalar
import warnings


def retrieve_cv_classification(file_path):
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    categories = []
    balanced_accuracies_per_category_per_split = []

    for category, res in results["balanced_accuracy"].items():
        categories.append(category)
        balanced_accuracies_per_split = []
        for split, acc in res.items():
            balanced_accuracies_per_split.append(acc)
        balanced_accuracies_per_category_per_split.append(balanced_accuracies_per_split)

    balanced_accuracies_per_category_per_split = np.array(balanced_accuracies_per_category_per_split)
    # shape (catgories, splits)

    return balanced_accuracies_per_category_per_split, categories


def retrieve_global_mse(file_path, feature, return_labels_variance=True, return_mse_per_outer_split=False):
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    n_outer_splits = results["args"]["n_cv_splits"]

    mse_avg = results["mse_test_avg"][feature]
    mse_std = results["mse_test_std"][feature]
    mse_avg_std = mse_std / np.sqrt(n_outer_splits)

    labels_variance = results["labels_variance_full"][feature]

    result = (mse_avg, mse_avg_std)
    if return_labels_variance:
        result += (labels_variance,)
    if return_mse_per_outer_split:
        mse_per_outer_split = []
        for split in range(n_outer_splits):
            mse_per_outer_split.append(results["mse_test"][feature][split])
        result += (np.array(mse_per_outer_split),)
    return result


def find_optimal_gammas(folder, feature):

    # collect .pkl files from the folder
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(".pkl")
    ]

    gammas = []
    avg_inner_mse_per_gamma = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        # retrieve gamma
        gammas.append(results["args"]["gamma"])
        # retrieve categories and their number
        categories = list(results["local"].keys())
        n_categories = len(categories)
        # retrieve number of outer splits
        n_outer_splits = results["args"]["n_cv_splits"]

        # initialize numpy container
        res_vector = np.zeros((n_outer_splits, n_categories))
        # fill
        for cat_num, category in enumerate(categories):
            for outer_split in range(n_outer_splits):
                res_vector[outer_split, cat_num] = results["local"][category]["inner_mse_test_avg"][feature][outer_split]
        # append
        avg_inner_mse_per_gamma.append(res_vector)

    # stack
    avg_inner_mse_per_gamma = np.stack(avg_inner_mse_per_gamma, axis=0)
    # shape: (n_gammas, n_outer_splits, n_categories)

    # average over categories
    avg_mse = np.mean(avg_inner_mse_per_gamma, axis=2)  # shape: (n_gammas, n_outer_splits)

    # determine index of best gamma per outer split
    best_gammas = []
    for outer_split in range(np.shape(avg_mse)[1]):
        best_gamma_idx = np.argmin(avg_mse[:, outer_split])
        best_gammas.append(gammas[best_gamma_idx])

    return np.array(best_gammas)


def retrieve_local_mses_per_outer_splits_gamma_regularization(folder, feature):
    # retrieve optimal gammas
    best_gammas_per_outer_split = find_optimal_gammas(folder, feature)

    # collect .pkl files from the folder
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(".pkl")
    ]

    n_outer_splits = len(best_gammas_per_outer_split)

    # for each outer split, find the file corresponding to the optimal gamma and append it
    best_outer_mses_per_outer_split_per_category = []
    for outer_split in range(n_outer_splits):
        found = False
        # find the file corresponding to the optimal gamma
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                results = pickle.load(f)
            gamma_current = results["args"]["gamma"]
            if gamma_current == best_gammas_per_outer_split[outer_split]:
                found = True
            else:
                continue
            # Once the file is found, append the test mse for the current outer split
            categories = list(results["local"].keys())
            n_categories = len(categories)
            best_outer_mses_per_category = []
            for category in categories:
                best_outer_mses_per_category.append(results["local"][category]["mse_test"][feature][outer_split])
            best_outer_mses_per_category = np.stack(best_outer_mses_per_category, axis=0)
            best_outer_mses_per_outer_split_per_category.append(best_outer_mses_per_category)
            break

        if not found:
            print(f"Error: no file found with the desired gamma {best_gammas_per_outer_split[outer_split]} for split {outer_split}")
            exit()

    best_outer_mses_per_outer_split_per_category = np.stack(best_outer_mses_per_outer_split_per_category, axis=0)
    # shape (split, category)
    best_outer_mses_per_category_per_outer_split = best_outer_mses_per_outer_split_per_category.T

    return best_outer_mses_per_category_per_outer_split, best_gammas_per_outer_split, categories


def retrieve_avg_local_mse_gamma_regularization(folder, feature, return_best_gammas_per_outer_split=False, return_mse_per_outer_split=False):

    best_outer_mses_per_category_per_outer_split, best_gammas_per_outer_split, categories \
        = retrieve_local_mses_per_outer_splits_gamma_regularization(folder, feature)  # shape (category, split)

    best_outer_mse = np.mean(best_outer_mses_per_category_per_outer_split, axis=0)

    n_outer_splits = np.shape(best_outer_mse)[0]

    best_outer_mse_avg = np.mean(best_outer_mse)
    best_outer_mse_std = np.std(best_outer_mse)
    best_outer_mse_avg_std = best_outer_mse_std / np.sqrt(n_outer_splits)

    result = (best_outer_mse_avg, best_outer_mse_avg_std)

    if return_best_gammas_per_outer_split:
        result += (best_gammas_per_outer_split,)

    if return_mse_per_outer_split:
        result += (best_outer_mse,)

    return result


def _list_result_files(folder):
    return [os.path.join(folder, f)
            for f in os.listdir(folder)
            if not f.startswith(".") and f.lower().endswith(".pkl")]


def _load_minimal_metrics(fp, feature, ref_categories=None):
    """Return (gamma, categories, inner_avg[S], mse_test[C,S]) for one file."""
    with open(fp, "rb") as f:
        R = pickle.load(f)
    g = R["args"]["gamma"]
    S = R["args"]["n_cv_splits"]
    local = R["local"]

    # Make category order deterministic & consistent across files
    cats = list(local.keys()) if ref_categories is None else ref_categories
    if ref_categories is None:
        cats = sorted(cats)

    C = len(cats)
    inner = np.empty((C, S), dtype=np.float64)
    mse   = np.empty((C, S), dtype=np.float64)

    for ci, cat in enumerate(cats):
        inner[ci, :] = [local[cat]["inner_mse_test_avg"][feature][s] for s in range(S)]
        mse[ci,   :] = [local[cat]["mse_test"][feature][s]            for s in range(S)]

    inner_avg = inner.mean(axis=0)                   # shape (S,)
    return g, cats, inner_avg, mse                   # mse shape (C,S)


def _build_cache(folder, feature):
    files = sorted(_list_result_files(folder))
    gammas, inner_rows, mse_blocks = [], [], []
    categories = None

    for fp in files:
        g, cats, inner_avg, mse = _load_minimal_metrics(fp, feature, ref_categories=categories)
        if categories is None:
            categories = cats
        gammas.append(g)
        inner_rows.append(inner_avg)                 # (S,)
        mse_blocks.append(mse)                       # (C,S)

    G = len(gammas)
    inner_matrix = np.stack(inner_rows, axis=0)      # (G,S)
    mse_stack = np.stack(mse_blocks, axis=0)      # (G,C,S)
    return np.array(gammas), categories, inner_matrix, mse_stack


def retrieve_local_mses_per_outer_splits_gamma_regularization_fast(folder, feature):
    gammas, categories, inner_matrix, mse_stack = _build_cache(folder, feature)
    # Best gamma index per split using inner-CV average
    best_idx = np.argmin(inner_matrix, axis=0)       # (S,)
    best_gammas = gammas[best_idx]                   # (S,)

    # Gather per-category test MSEs for the chosen gamma in each split
    S = inner_matrix.shape[1]
    best_mse_S_C = mse_stack[best_idx, :, np.arange(S)]  # (S,C) via advanced indexing
    best_mse_C_S = np.swapaxes(best_mse_S_C, 0, 1)       # (C,S) to match your original
    return best_mse_C_S, best_gammas, categories


def retrieve_avg_local_mse_gamma_regularization_fast(folder, feature,
                                                     return_best_gammas_per_outer_split=False,
                                                     return_mse_per_outer_split=False):
    per_cat_per_split, best_gammas, categories = (
        retrieve_local_mses_per_outer_splits_gamma_regularization_fast(folder, feature)
    )
    per_split = per_cat_per_split.mean(axis=0)       # average over categories → (S,)
    mu = float(per_split.mean())
    sd = float(per_split.std())
    se = sd / np.sqrt(per_split.size)

    result = (mu, se)
    if return_best_gammas_per_outer_split:
        result += (best_gammas,)
    if return_mse_per_outer_split:
        result += (per_split,)
    return result


def retrieve_local_mses_per_outer_split_closest_gamma(folder, target_gamma, feature):
    # find the file with gamma closest to best_gamma_avg
    # collect .pkl files from the folder
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(".pkl")
    ]
    smallest_absolute_distance = np.inf
    closest_path = None
    closest_gamma = None
    for file_path in file_paths:
            with open(file_path, "rb") as f:
                results = pickle.load(f)
            current_gamma = results["args"]["gamma"]
            distance_current = np.abs(target_gamma - current_gamma)

            if distance_current < smallest_absolute_distance:
                closest_path = file_path
                smallest_absolute_distance = distance_current
                closest_gamma = current_gamma
    print(f"Target gamma: {target_gamma}, Closest gamma: {closest_gamma}")

    with open(closest_path, "rb") as f:
        results = pickle.load(f)

    gamma = results["args"]["gamma"]
    n_outer_splits = results["args"]["n_cv_splits"]
    categories = list(results["local"].keys())
    n_categories = len(categories)

    local_mses_per_category_per_outer_split = np.zeros((n_outer_splits, n_categories))

    for cat_num, category in enumerate(categories):
        for outer_split in range(n_outer_splits):
            local_mses_per_category_per_outer_split[outer_split, cat_num] \
                = results["local"][category]["mse_test"][feature][outer_split]

    return local_mses_per_category_per_outer_split, categories, gamma


def retrieve_local_mses_per_outer_splits_fixed_gamma(file_path, feature):
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    gamma = results["args"]["gamma"]
    n_outer_splits = results["args"]["n_cv_splits"]
    categories = list(results["local"].keys())
    n_categories = len(categories)

    local_mses_per_category_per_outer_split = np.zeros((n_outer_splits, n_categories))

    for cat_num, category in enumerate(categories):
        for outer_split in range(n_outer_splits):
            local_mses_per_category_per_outer_split[outer_split, cat_num] \
                = results["local"][category]["mse_test"][feature][outer_split]

    return local_mses_per_category_per_outer_split, categories, gamma


def retrieve_avg_local_mse_fixed_gamma(path, feature):
    local_mses_per_category_per_outer_split, categories, gamma \
        = retrieve_local_mses_per_outer_splits_fixed_gamma(path, feature)

    local_mses_per_outer_split = np.mean(local_mses_per_category_per_outer_split, axis=0)
    n_outer_splits = np.shape(local_mses_per_outer_split)[0]

    mse_avg = np.mean(local_mses_per_outer_split)
    mse_std = np.std(local_mses_per_outer_split)
    mse_avg_std = mse_std / np.sqrt(n_outer_splits)

    return mse_avg, mse_avg_std, gamma


def retrieve_avg_local_nmse_gap_fixed_gamma_list(folder, feature, cv_global_mse_avg, cv_global_mse_avg_std, labels_variance):
    gamma_cv_list = []
    cv_nmse_gap_fixed_gamma_avg = []
    cv_nmse_gap_fixed_gamma_std = []
    cv_gamma_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(".pkl")
    ]
    for path in cv_gamma_paths:
        AvgLocalMse_avg, AvgLocalMse_avg_std, gamma = retrieve_avg_local_mse_fixed_gamma(path, feature)

        cv_nmse_gap_fixed_gamma_avg.append((cv_global_mse_avg - AvgLocalMse_avg) / labels_variance)  # TODO: plot
        cv_nmse_gap_fixed_gamma_std.append(np.sqrt(
            (cv_global_mse_avg_std ** 2) + (AvgLocalMse_avg_std ** 2)) / labels_variance)  # TODO: plot
        gamma_cv_list.append(gamma)
    return gamma_cv_list, cv_nmse_gap_fixed_gamma_avg, cv_nmse_gap_fixed_gamma_std


def retrieve_local_mses_per_outer_splits(file_path, feature):
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    n_outer_splits = results["args"]["n_cv_splits"]
    categories = list(results["local"].keys())
    n_categories = len(categories)

    local_mses_per_category_per_outer_split = np.zeros((n_categories, n_outer_splits))
    best_l2_per_category_per_outer_split = np.zeros((n_categories, n_outer_splits))

    for cat_num, category in enumerate(categories):
        for outer_split in range(n_outer_splits):
            local_mses_per_category_per_outer_split[cat_num, outer_split] \
                = results["local"][category]["mse_test"][feature][outer_split]
            best_l2_per_category_per_outer_split[cat_num, outer_split] \
                = results["local"][category]["best_l2_strength"][feature][outer_split]

    return local_mses_per_category_per_outer_split, best_l2_per_category_per_outer_split, categories


def retrieve_average_local_mse(file_path, feature, return_mse_per_outer_split=False):
    local_mses_per_category_per_outer_split, best_l2_per_category_per_outer_split, categories \
        = retrieve_local_mses_per_outer_splits(file_path, feature)

    local_mses_per_outer_split = np.mean(local_mses_per_category_per_outer_split, axis=0)
    n_outer_splits = np.shape(local_mses_per_outer_split)[0]

    mse_avg = np.mean(local_mses_per_outer_split)
    mse_std = np.std(local_mses_per_outer_split)
    mse_avg_std = mse_std / np.sqrt(n_outer_splits)

    result = (mse_avg, mse_avg_std)

    if return_mse_per_outer_split:
        result += (local_mses_per_outer_split,)

    return result


def retrieve_theory_gap_closest_gamma(folder, feature, best_gamma_avg, return_gamma=False, return_detailed_results=False):
    # find the file with gamma closest to best_gamma_avg
    # collect .pkl files from the folder
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith(".") and f.lower().endswith(".pkl")
    ]
    smallest_absolute_distance = np.inf
    closest_path = None
    closest_gamma = None
    for file_path in file_paths:
            with open(file_path, "rb") as f:
                results = pickle.load(f)
            current_gamma = results["args"]["gamma"]
            distance_current = np.abs(best_gamma_avg - current_gamma)

            if distance_current < smallest_absolute_distance:
                closest_path = file_path
                smallest_absolute_distance = distance_current
                closest_gamma = current_gamma
    print(f"Best gamma: {best_gamma_avg}, Closest gamma: {closest_gamma}")

    # now compute the theory using this file
    return retrieve_theory_gap(closest_path, feature, return_gamma=return_gamma, return_detailed_results=return_detailed_results)


def retrieve_theory_gap(file_path, feature, return_gamma=False, return_detailed_results=False):
    # define a detailed results dictionary to return on request
    detailed_results = {}

    with open(file_path, "rb") as fh:
        data_dict = pickle.load(fh)

    gamma = data_dict["args"]["gamma"]

    results_local = data_dict["local"]
    results_global = data_dict["global"]

    # ============================
    # Retrieve all necessary quantities
    # ============================

    # retrieve global quantities
    var_y = results_global["labels_variance"][feature]  # i.e. [<y_mu^2>]

    detailed_results["labels_variance"] = var_y

    hat_var_y = results_global["linearized_labels_variance"][feature]  # i.e. [<\hat{y}_mu^2>]
    hat_var_y_1 = results_global["linearized_labels_variance_fluctuations"][feature]

    # Handle both old (no 'feature' key) and new (per-feature) formats
    C_raw = results_global["covariance"]
    C1_raw = results_global["covariance_fluctuations"]
    C = C_raw[feature] if isinstance(C_raw, dict) else C_raw
    C1 = C1_raw[feature] if isinstance(C1_raw, dict) else C1_raw

    d = results_global["io_covariance_linearized"][feature]
    d1 = results_global["io_covariance_linearized_fluctuations"][feature]

    # retrieve local quantities
    r_mu_list = []
    u1_mu_list = []
    # b_mu_list = []
    var1_mu_list = []
    var_tot_mu_list = []
    x0_mu_list = []
    y0_mu_list = []
    var_delta_y_hat_mu_list = []
    # for detailed results
    PR_per_category = []
    radius_per_category = []
    centroid_norm_per_category = []
    overlaps_per_category = []  # shape (category, overlaps)
    eigenvalues_per_category = []  # shape (category, eigenvalues)

    categories = list(results_local["centroid"].keys())
    for category in categories:
        w_mu = results_local["regression_vector"][category][feature]
        r_mu_single = np.linalg.norm(w_mu)
        u1_mu = w_mu / r_mu_single
        r_mu_list.append(r_mu_single)
        u1_mu_list.append(u1_mu)
        var1_mu_list.append(results_local["variance_along_linear_rule"][category][feature])

        # total_variance can be a scalar or a {feature: scalar} dict
        _tv = results_local["total_variance"][category]
        var_tot_mu_list.append(_tv[feature] if isinstance(_tv, dict) else _tv)

        x0_mu_list.append(results_local["centroid"][category])
        y0_mu_list.append(results_local["label_mean"][category][feature])
        var_delta_y_hat_mu_list.append(results_local["linearized_label_variance"][category][feature])
        # for detailed results:
        covariance_eigenvalues = results_local["covariance_eigenvalues"][category]
        eigenvalues_per_category.append(covariance_eigenvalues)
        PR_per_category.append((covariance_eigenvalues.sum() ** 2) / (np.square(covariance_eigenvalues).sum()))
        radius_per_category.append(np.sqrt(np.mean(covariance_eigenvalues)))
        centroid_norm_per_category.append(results_local["centroid_norm"][category])
        overlaps_per_category.append(results_local["linear_rule_vs_PC_overlap"][category][feature])

    mean_r = np.mean(r_mu_list)
    var_r = np.var(r_mu_list)
    var_1 = np.mean(var1_mu_list)
    var_tot = np.mean(var_tot_mu_list)
    # b2 = np.mean(np.array(b_mu_list)**2)
    r_mu_list = np.array(r_mu_list)
    u1_mu_list = np.array(u1_mu_list)
    cu = np.mean(u1_mu_list, axis=0)
    c = np.linalg.norm(cu)
    u = cu / c
    N = np.shape(u1_mu_list)[1]
    # for var_0
    x0_mu = np.stack(x0_mu_list, axis=0)  # Shape: (P, N)
    P = np.shape(x0_mu)[0]
    y0_mu = np.array(y0_mu_list)

    # for detailed results
    overlaps_per_category = np.stack(overlaps_per_category, axis=0)  # shape (category, overlaps)
    eigenvalues_per_category = np.stack(eigenvalues_per_category, axis=0)  # shape (category, eigenvalues)

    # ============================
    # Compute MSEs
    # ============================

    # ---> MSE GAP
    empirical_mse_gap = results_global["regression_error_gap"][feature]

    # ---> MSE GAP FLUCTUATIONS
    C1_pinv = np.linalg.pinv(C1, hermitian=True)
    empirical_mse_gap_fluctuations = hat_var_y_1 - d1 @ C1_pinv @ d1

    # ---> MSE GAP CENTROIDS
    C0 = np.cov(x0_mu, rowvar=False)
    X0_C1pinv_X0 = x0_mu @ C1_pinv @ x0_mu.T
    y_tilde_0 = y0_mu - d1 @ C1_pinv @ x0_mu.T
    try:
        X0_C1pinv_X0_inv_Y0 = np.linalg.solve(X0_C1pinv_X0, y_tilde_0)
    except np.linalg.LinAlgError as e:
        warnings.warn(
            "\n" + "=" * 80 +
            "\n⚠️  WARNING: Singular matrix while solving for centroids contribution."
            f"\n     Feature: {feature if 'feature' in locals() else 'unknown'}"
            f"\n     A.shape={getattr(X0_C1pinv_X0, 'shape', None)}, "
            f"b.shape={getattr(y_tilde_0, 'shape', None)}"
            "\n     Action: Setting centroids contribution to 0 and continuing."
            f"\n     Details: {e}"
            "\n" + "=" * 80 + "\n",
            stacklevel=1
        )
        # Fallback: force a zero contribution path
        X0_C1pinv_X0_inv_Y0 = np.zeros_like(y_tilde_0)
    mse_gap_centroids = y_tilde_0 @ X0_C1pinv_X0_inv_Y0

    # ---> MSE GAP INFINITE THEORY AND CONTRIBUTIONS
    mse_gap_scale_contribution = var_r / (mean_r ** 2)
    mse_gap_orientation_contribution = (var_tot / var_1) * ((1 - (c ** 2)) / (c ** 2)) / N
    detailed_results["mse_gap_scale_contribution"] = mse_gap_scale_contribution
    detailed_results["mse_gap_orientation_contribution"] = mse_gap_orientation_contribution
    detailed_results["SNR"] = var_1 / var_tot
    detailed_results["var_1"] = var_1
    detailed_results["var_tot"] = var_tot
    detailed_results["c_factor"] = (1 - (c ** 2)) / (c ** 2)
    detailed_results["c"] = c
    detailed_results["PR_per_category"] = np.array(PR_per_category)
    detailed_results["radius_per_category"] = np.array(radius_per_category)
    detailed_results["centroid_norm_per_category"] = np.array(centroid_norm_per_category)
    detailed_results["overlaps_per_category"] = overlaps_per_category  # shape (category, overlaps)
    detailed_results["eigenvalues_per_category"] = eigenvalues_per_category  # shape (category, eigenvalues)
    detailed_results["categories"] = categories
    detailed_results["centroids"] = x0_mu  # Shape: (P, N)
    detailed_results["hat_var_y_1"] = hat_var_y_1
    hat_var_y_1_theory = var_1 * np.mean(r_mu_list ** 2)
    detailed_results["hat_var_y_1_theory"] = hat_var_y_1_theory

    # DEBUG: Multiple options for what we use the hat_var_y_1 below (UNCOMMENT THE DESIRED ONE):
    # (a) the one derived from theory
    # reg_feat_var = hat_var_y_1_theory
    # , or (b) the empirical one
    reg_feat_var = hat_var_y_1
    # store the used value
    detailed_results["hat_var_y_1_used"] = reg_feat_var
    # OPTIONALLY: print difference between opt (a) and (b):
    print(f"hat_var_y_1. Theory: {hat_var_y_1_theory}, Empirical: {hat_var_y_1}")
    # END DEBUG

    mse_gap_fluctuations_theory_infinite = (1 - 1 / (
            (1 + mse_gap_scale_contribution) * (1 + mse_gap_orientation_contribution))) * reg_feat_var

    # print('DEBUG:')
    # print(f"labels_var inside: {var_y}")
    # print(f"hat_var_y_1: {hat_var_y_1}")
    # print(f"hat_var_y_1 (theory): {var_1 * np.mean(r_mu_list ** 2)}")
    # print(f"hat_var_y: {hat_var_y}")
    # print(f"hat_var_y_1 (measured): {hat_var_y_1}")
    # print(f"var_y (measured): {var_y}")
    # print(f"ratio hat_var_y_1 (theory)/ var_y (measured): {(var_1 * np.mean(r_mu_list ** 2))/var_y}")
    # print('END DEBUG:')

    mse_gap_scale_contribution_expanded = mse_gap_scale_contribution * reg_feat_var
    mse_gap_orientation_contribution_expanded = mse_gap_orientation_contribution * reg_feat_var

    # ---> FINITE THEORY
    # collect quantities
    r_mu = r_mu_list
    var_1_mu = np.array(var1_mu_list)

    # finite-P empirical mixed moments
    mean_r_var1 = np.mean(r_mu * var_1_mu)
    mean_r2_var1 = np.mean((r_mu ** 2) * var_1_mu)

    var_d_mu = []  # size: P, N-1
    Sigma_1d_mu = []  # size: P, N-1
    for category in categories:
        var_d_mu.append(results_local["variances_orthogonal"][category][feature])
        Sigma_1d_mu.append(results_local["covariance_cross_terms"][category][feature])

    # convert to numpy arrays
    var_d_mu = np.stack(var_d_mu, axis=0)
    Sigma_1d_mu = np.stack(Sigma_1d_mu, axis=0)

    # compute order parameter
    oparam = solve_self_consistent_equation(var_d_mu, var_1_mu, Sigma_1d_mu, c, N, P)

    print(f"oparam: {oparam}")

    # compute derived quantities stage 1
    U_d_mu = var_d_mu + oparam
    U_1_mu = var_1_mu + (oparam / (1 - c ** 2))
    S_d_mu = (Sigma_1d_mu ** 2) / U_d_mu

    # compute derived quantities stage 2
    U_d_mu_inv_sum = np.sum(1 / U_d_mu, axis=1)
    S_d_mu_sum_d = np.sum(S_d_mu, axis=1)
    U_1_mu_corrected = U_1_mu - S_d_mu_sum_d
    var_1_mu_corrected = var_1_mu - S_d_mu_sum_d

    # useful per-category objects
    correction_sum_mu = np.sum((Sigma_1d_mu / U_d_mu) ** 2, axis=1)
    last_factor_mu = (var_1_mu_corrected ** 2) / U_1_mu_corrected

    # compute coefficients
    a_coefficient = (+ var_1 * (c ** 2)
                     + oparam / P
                     - ((N - 1) * (c ** 2) * oparam) / N
                     - (c ** 2) * np.mean(S_d_mu_sum_d)
                     + (c ** 2) * (oparam ** 2) * np.mean(U_d_mu_inv_sum) / N
                     - (c ** 2) * np.mean(last_factor_mu)
                     # DEBUG: add neglected term
                     # + (c ** 2) * (oparam ** 2) * np.mean(np.sum((Sigma_1d_mu / U_d_mu) ** 2, axis=1) / U_1_mu_corrected) / N
                     # END DEBUG
                     )

    b_coefficient = (- 2 * c * mean_r_var1
                     + 2 * c * np.mean(S_d_mu_sum_d * r_mu)
                     + 2 * c * np.mean(last_factor_mu * r_mu)
                     )

    c_coefficient = (+ mean_r2_var1
                     - np.mean(S_d_mu_sum_d * (r_mu ** 2))
                     - np.mean(last_factor_mu * (r_mu ** 2))
                     )

    mse_gap_fluctuations_theory_finite = c_coefficient - (b_coefficient ** 2) / (4 * a_coefficient)

    print("DEBUG:")
    print(f"mse_gap_fluctuations_theory_finite: {mse_gap_fluctuations_theory_finite}")
    print("END DEBUG:")

    returns = (empirical_mse_gap, empirical_mse_gap_fluctuations, mse_gap_centroids,
               mse_gap_fluctuations_theory_infinite,
               mse_gap_scale_contribution_expanded, mse_gap_orientation_contribution_expanded,
               mse_gap_fluctuations_theory_finite)

    if return_gamma:
        returns += (gamma,)
    if return_detailed_results:
        returns += (detailed_results,)

    return returns


def self_consistent_equation(oparam, var_d_mu, var_1_mu, Sigma_1d_mu, c, N, P):
    alpha = P / N

    # compute derived quantities stage 1
    U_d_mu = var_d_mu + oparam
    U_1_mu = var_1_mu + (oparam / (1 - c ** 2))
    S_d_mu = (Sigma_1d_mu ** 2) / U_d_mu

    # compute derived quantities stage 2
    U_d_mu_inv_sum = np.sum(1 / U_d_mu, axis=1)
    S_d_mu_sum_d = np.sum(S_d_mu, axis=1)
    U_1_mu_corrected = U_1_mu - S_d_mu_sum_d

    # compute various terms in the self-consistent equation
    term_d = np.mean(U_d_mu_inv_sum) / N
    term_1 = np.mean(1 / U_1_mu_corrected) / (N * (1 - c ** 2))
    term_correction = np.mean(np.sum((Sigma_1d_mu / U_d_mu) ** 2, axis=1) / U_1_mu_corrected) / N

    out = oparam * (term_1 + term_d + term_correction) - 1 + 1 / (N * alpha)

    return out


def solve_self_consistent_equation(var_d_mu, var_1_mu, Sigma_1d_mu, c, N, P, lower_bound=1e-6, upper_bound=1e8):
    """
    Find the zero of self_consistent_equation in the interval (0, infinity).

    Returns:
        float: The value of oparam where the function is zero.
    """

    # Wrap the function to accept only oparam
    def f(oparam):
        return self_consistent_equation(oparam, var_d_mu, var_1_mu, Sigma_1d_mu, c, N, P)

    # Use root_scalar to find the zero in the specified interval
    result = root_scalar(f, bracket=(lower_bound, upper_bound), method='brentq')

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge.")


# NEW: local_global_lambda
def retrieve_avg_local_mse_ridge_from_pkl(
    file_path,
    feature,
    return_mse_per_outer_split=False,
    return_best_l2_per_outer_split=False,
):
    """
    Reads the *single* ridge local-regression .pkl produced by the new local script.

    Reconstructs CV test MSE as:
      for each outer split:
        choose best lambda by minimizing (avg across categories) inner_mse_test_avg over lambdas
        then evaluate outer-test MSE (avg across categories) at that chosen lambda
      then average across outer splits + SEM across outer splits

    Returns:
      (mse_avg, mse_sem)
      + optionally (mse_per_outer_split,)
      + optionally (best_l2_per_outer_split,)
    """

    with open(file_path, "rb") as f:
        results = pickle.load(f)

    local = results["local"]

    # categories
    categories = list(local.keys())
    if len(categories) == 0:
        raise ValueError(f"No categories found in local pkl: {file_path}")

    # outer splits (read from first category)
    any_cat = categories[0]
    outer_splits = sorted(local[any_cat]["mse_test"][feature].keys())
    n_outer = len(outer_splits)
    if n_outer == 0:
        raise ValueError(f"No outer splits found for feature={feature} in {file_path}")

    # lambda grid (robust: reconstruct from dict keys)
    lam_keys = list(local[any_cat]["mse_test"][feature][outer_splits[0]].keys())
    l2_strengths = sorted([float(x) for x in lam_keys])
    L = len(l2_strengths)
    if L == 0:
        raise ValueError(f"No lambdas found in {file_path}")

    mse_per_outer = np.zeros(n_outer, dtype=float)
    best_l2_per_outer = np.zeros(n_outer, dtype=float)

    for oi, outer_split in enumerate(outer_splits):
        # inner CV criterion: avg across categories of inner_mse_test_avg
        inner_avg_over_cats = np.zeros(L, dtype=float)
        for li, l2 in enumerate(l2_strengths):
            vals = []
            for c in categories:
                vals.append(local[c]["inner_mse_test_avg"][feature][outer_split][l2])
            inner_avg_over_cats[li] = float(np.mean(vals))

        best_idx = int(np.argmin(inner_avg_over_cats))
        best_l2 = float(l2_strengths[best_idx])
        best_l2_per_outer[oi] = best_l2

        # outer test MSE at best l2, averaged across categories
        vals = []
        for c in categories:
            vals.append(local[c]["mse_test"][feature][outer_split][best_l2])
        mse_per_outer[oi] = float(np.mean(vals))

    mse_avg = float(np.mean(mse_per_outer))
    mse_std = float(np.std(mse_per_outer))
    mse_sem = float(mse_std / np.sqrt(n_outer)) if n_outer > 1 else 0.0

    out = (mse_avg, mse_sem)
    if return_mse_per_outer_split:
        out += (mse_per_outer,)
    if return_best_l2_per_outer_split:
        out += (best_l2_per_outer,)
    return out


def retrieve_theory_gap_kind(
    folder,
    feature,
    kind,
    return_gamma=False,
    return_detailed_results=False,
):
    """
    Selects the theory .pkl inside `folder` corresponding to a specific ridge-theory kind:
      kind in {"opt","min","max","mid"}.

    The updated theory script saves:
      ..._TheoryRegression_{kind}_... .pkl
    and also stores:
      data_dict["args"]["l2_theory_kind"] = kind

    We search .pkl files in the folder, prefer exact args match, fallback to filename match.
    If multiple matches exist, we take the most recently modified file.
    Then we call retrieve_theory_gap(...) on that file.

    return_gamma is kept for backward compatibility, but ridge-theory files may not be meaningful in "gamma".
    """

    # collect .pkl files from the folder
    file_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if (not f.startswith(".")) and f.lower().endswith(".pkl")
    ]
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No theory .pkl files found in: {folder}")

    matches = []

    # pass 1: match via args["l2_theory_kind"]
    for fp in file_paths:
        try:
            with open(fp, "rb") as fh:
                dd = pickle.load(fh)
            k = dd.get("args", {}).get("l2_theory_kind", None)
            if k == kind:
                matches.append(fp)
        except Exception:
            pass

    # pass 2: fallback via filename pattern
    if len(matches) == 0:
        for fp in file_paths:
            base = os.path.basename(fp)
            if f"_TheoryRegression_{kind}_" in base:
                matches.append(fp)

    if len(matches) == 0:
        raise FileNotFoundError(f"No theory file found for kind='{kind}' in folder={folder}")

    # pick most recent
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = matches[0]

    return retrieve_theory_gap(
        chosen,
        feature,
        return_gamma=return_gamma,
        return_detailed_results=return_detailed_results,
    )
# END NEW: local global lambda















