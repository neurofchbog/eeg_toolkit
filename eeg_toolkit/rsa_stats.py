"""
RSA statistical testing — 1D cluster-based permutation tests.

Tests for time-resolved RSA model fits (Spearman correlations):
    1. test_model_fit       — is model correlation > 0? (one-sample)
    2. compare_model_fits   — is model A > model B? (paired)

Visualization:
    3. plot_model_fits      — time-resolved model fit with SEM + sig bars

Design notes
------------
- 1D cluster permutation over time only (same approach as mvpa_stats).
- Cluster mass = sum of absolute t-values within a cluster.
- Two-tailed by default.
- Self-contained: cluster engine is included here (not imported from
  mvpa_stats) so the RSA module has no dependency on the MVPA module.
"""

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Core: find clusters in a 1D stat map
# ===================================================================

def find_clusters_1d(stat_values, threshold, times):
    """
    Find contiguous clusters of suprathreshold values in a 1D array.

    Parameters
    ----------
    stat_values : array (n_times,)
        Test statistic (typically t-values).
    threshold : float
        Absolute threshold for cluster formation.
    times : array (n_times,)
        Time points (used for reporting).

    Returns
    -------
    clusters : list of dict
        Each dict has: indices, size, sum_stat (cluster mass),
        max_stat, time_range, start_idx, end_idx.
    """
    significant = np.abs(stat_values) >= threshold

    if not np.any(significant):
        return []

    clusters = []
    cluster_start = None

    for i, is_sig in enumerate(significant):
        if is_sig and cluster_start is None:
            cluster_start = i
        elif not is_sig and cluster_start is not None:
            _append_cluster(clusters, stat_values, times,
                            cluster_start, i - 1)
            cluster_start = None

    # Cluster extends to end
    if cluster_start is not None:
        _append_cluster(clusters, stat_values, times,
                        cluster_start, len(significant) - 1)

    return clusters


def _append_cluster(clusters, stat_values, times, start, end):
    """Helper: build a cluster dict and append to list."""
    idx = np.arange(start, end + 1)
    clusters.append({
        "indices": idx,
        "size": len(idx),
        "sum_stat": float(np.sum(np.abs(stat_values[idx]))),
        "max_stat": float(np.max(np.abs(stat_values[idx]))),
        "time_range": (float(times[start]), float(times[end])),
        "start_idx": int(start),
        "end_idx": int(end),
    })


# ===================================================================
# Core: permutation engines
# ===================================================================

def _run_permutation_onesample(data, cluster_threshold,
                               n_permutations, seed, verbose):
    """
    Build the null distribution for a one-sample test by randomly
    sign-flipping subjects' values.

    Parameters
    ----------
    data : array (n_subjects, n_times)
        Values to test against zero.
    """
    rng = np.random.RandomState(seed)
    n_subjects = data.shape[0]
    n_times = data.shape[1]
    times_dummy = np.arange(n_times)

    null_max_masses = np.empty(n_permutations)

    for p in range(n_permutations):
        signs = rng.choice([-1, 1], size=n_subjects)
        perm_data = data * signs[:, None]
        perm_t, _ = sp_stats.ttest_1samp(perm_data, 0, axis=0)
        perm_clusters = find_clusters_1d(perm_t, cluster_threshold,
                                         times_dummy)
        if perm_clusters:
            null_max_masses[p] = max(c["sum_stat"] for c in perm_clusters)
        else:
            null_max_masses[p] = 0.0

        if verbose and (p + 1) % 2000 == 0:
            print(f"    permutation {p + 1}/{n_permutations}")

    return null_max_masses


def _run_permutation_paired(data_a, data_b, cluster_threshold,
                            n_permutations, seed, verbose):
    """
    Build the null distribution for a paired test by randomly
    swapping condition labels within subjects (sign-flip).

    Parameters
    ----------
    data_a, data_b : array (n_subjects, n_times)
    """
    rng = np.random.RandomState(seed)
    n_subjects = data_a.shape[0]
    n_times = data_a.shape[1]
    times_dummy = np.arange(n_times)

    null_max_masses = np.empty(n_permutations)

    for p in range(n_permutations):
        swap = rng.choice([True, False], size=n_subjects)
        perm_a = np.where(swap[:, None], data_b, data_a)
        perm_b = np.where(swap[:, None], data_a, data_b)

        perm_t, _ = sp_stats.ttest_rel(perm_a, perm_b, axis=0)
        perm_clusters = find_clusters_1d(perm_t, cluster_threshold,
                                         times_dummy)
        if perm_clusters:
            null_max_masses[p] = max(c["sum_stat"] for c in perm_clusters)
        else:
            null_max_masses[p] = 0.0

        if verbose and (p + 1) % 2000 == 0:
            print(f"    permutation {p + 1}/{n_permutations}")

    return null_max_masses


def _evaluate_clusters(real_clusters, null_max_masses, alpha):
    """
    Determine which real clusters are significant given the null
    distribution. Adds 'p_value' and 'significant' to each cluster dict.

    Returns the list of significant clusters.
    """
    for cluster in real_clusters:
        mass = cluster["sum_stat"]
        cluster["p_value"] = float((null_max_masses >= mass).mean())
        cluster["significant"] = cluster["p_value"] < alpha

    sig = [c for c in real_clusters if c["significant"]]
    return sig


# ===================================================================
# Public API: test types
# ===================================================================

def test_model_fit(fits, times, model_name=None,
                   n_permutations=10000, cluster_threshold=None,
                   alpha=0.05, seed=42, verbose=True):
    """
    Test whether a model's Spearman correlation is significantly > 0.

    One-sample cluster permutation test (sign-flip).

    Parameters
    ----------
    fits : np.ndarray (n_subjects, n_times)
        Per-subject Spearman correlations at each time point
        (from ``rsa.compute_model_fits``).
    times : np.ndarray (n_times,)
    model_name : str, optional
        Label for printing.
    n_permutations : int
    cluster_threshold : float or None
        Cluster-forming t-threshold. None = t for p=0.05 two-tailed.
    alpha : float
        Cluster significance threshold.
    seed : int
    verbose : bool

    Returns
    -------
    results : dict
        t_values, clusters, significant_clusters, null_max_masses,
        times, n_subjects, n_permutations, cluster_threshold, alpha,
        test_type, model_name.
    """
    n_subj = fits.shape[0]
    label = model_name or "model"

    if cluster_threshold is None:
        cluster_threshold = sp_stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

    if verbose:
        print(f"Test: {label} vs zero")
        print(f"  {n_subj} subjects, threshold t={cluster_threshold:.2f}, "
              f"{n_permutations} permutations")

    # Real t-map
    real_t, _ = sp_stats.ttest_1samp(fits, 0, axis=0)
    real_clusters = find_clusters_1d(real_t, cluster_threshold, times)

    if verbose:
        print(f"  Found {len(real_clusters)} clusters in real data")

    # Permutation null
    null_max = _run_permutation_onesample(
        fits, cluster_threshold, n_permutations, seed, verbose)

    # Evaluate
    sig = _evaluate_clusters(real_clusters, null_max, alpha)

    if verbose:
        _print_results(sig, real_clusters, null_max, alpha)

    return _pack_results(
        real_t, real_clusters, sig, null_max,
        times, n_subj, n_permutations,
        cluster_threshold, alpha,
        test_type="model_vs_zero",
        model_name=label,
    )


def compare_model_fits(fits_a, fits_b, times,
                       label_a="Model A", label_b="Model B",
                       n_permutations=10000, cluster_threshold=None,
                       alpha=0.05, seed=42, verbose=True):
    """
    Test whether model A fits better than model B.

    Paired cluster permutation test on Spearman correlations
    (same subjects in both).

    Parameters
    ----------
    fits_a, fits_b : np.ndarray (n_subjects, n_times)
    times : np.ndarray
    label_a, label_b : str
    n_permutations : int
    cluster_threshold : float or None
    alpha : float
    seed : int
    verbose : bool

    Returns
    -------
    results : dict
    """
    n_subj = fits_a.shape[0]

    if fits_a.shape != fits_b.shape:
        raise ValueError(
            f"Fit shapes don't match: {fits_a.shape} vs {fits_b.shape}"
        )

    if cluster_threshold is None:
        cluster_threshold = sp_stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

    if verbose:
        print(f"Test: {label_a} vs {label_b}")
        print(f"  {n_subj} subjects, threshold t={cluster_threshold:.2f}, "
              f"{n_permutations} permutations")

    # Real t-map (paired)
    real_t, _ = sp_stats.ttest_rel(fits_a, fits_b, axis=0)
    real_clusters = find_clusters_1d(real_t, cluster_threshold, times)

    if verbose:
        print(f"  Found {len(real_clusters)} clusters in real data")

    # Permutation null
    null_max = _run_permutation_paired(
        fits_a, fits_b, cluster_threshold,
        n_permutations, seed, verbose)

    # Evaluate
    sig = _evaluate_clusters(real_clusters, null_max, alpha)

    if verbose:
        _print_results(sig, real_clusters, null_max, alpha)

    return _pack_results(
        real_t, real_clusters, sig, null_max,
        times, n_subj, n_permutations,
        cluster_threshold, alpha,
        test_type="compare_models",
        model_name=f"{label_a} vs {label_b}",
    )


# ===================================================================
# Result packing and printing
# ===================================================================

def _pack_results(real_t, real_clusters, sig_clusters, null_max,
                  times, n_subj, n_permutations,
                  cluster_threshold, alpha, test_type,
                  model_name=None):
    """Assemble a standard results dict."""
    return {
        "t_values": real_t,
        "clusters": real_clusters,
        "significant_clusters": sig_clusters,
        "null_max_masses": null_max,
        "times": times,
        "n_subjects": n_subj,
        "n_permutations": n_permutations,
        "cluster_threshold": cluster_threshold,
        "alpha": alpha,
        "test_type": test_type,
        "model_name": model_name,
    }


def _print_results(sig, all_clusters, null_max, alpha):
    """Pretty-print test results."""
    print(f"\n  Null distribution: mean={null_max.mean():.3f}, "
          f"std={null_max.std():.3f}, "
          f"95th={np.percentile(null_max, 95):.3f}, "
          f"99th={np.percentile(null_max, 99):.3f}")

    if not sig:
        print(f"  No significant clusters (alpha={alpha})")
    else:
        print(f"  {len(sig)}/{len(all_clusters)} significant clusters:")
        for i, c in enumerate(sig, 1):
            print(f"    Cluster {i}: {c['time_range'][0]:.3f}s – "
                  f"{c['time_range'][1]:.3f}s, mass={c['sum_stat']:.2f}, "
                  f"p={c['p_value']:.4f}")
    print()


# ===================================================================
# Visualization
# ===================================================================

def plot_model_fits(fits_dict, times, stats_dict=None,
                    colors=None, figsize=(10, 5), title=None,
                    show_zero=True, xlabel="Time (s)",
                    ylabel="Spearman r"):
    """
    Plot time-resolved model fits with SEM and significance markers.

    Parameters
    ----------
    fits_dict : dict
        {model_name: np.ndarray (n_subjects, n_times)}. Output of
        ``rsa.compute_model_fits``.
    times : np.ndarray
    stats_dict : dict or None
        {model_name: results_dict} from ``test_model_fit``.
        If provided, significant clusters are marked.
    colors : dict or None
        {model_name: color_string}. None = auto cycle.
    figsize : tuple
    title : str or None
    show_zero : bool
        Show the zero baseline.
    xlabel, ylabel : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    # Default color cycle
    default_colors = [
        "#2166AC", "#B2182B", "#4DAF4A", "#FF7F00",
        "#984EA3", "#A65628",
    ]
    if colors is None:
        colors = {}
        for i, name in enumerate(fits_dict):
            colors[name] = default_colors[i % len(default_colors)]

    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(fits_dict)

    for name, fit_arr in fits_dict.items():
        n_subj = fit_arr.shape[0]
        mean_r = np.nanmean(fit_arr, axis=0)
        sem_r = np.nanstd(fit_arr, axis=0) / np.sqrt(n_subj)
        col = colors.get(name, "#333333")

        ax.plot(times, mean_r, color=col, lw=2, label=name)
        ax.fill_between(times, mean_r - sem_r, mean_r + sem_r,
                        color=col, alpha=0.2)

    # Zero line
    if show_zero:
        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.5)

    # Cue onset
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.5)

    # Significance markers
    if stats_dict is not None:
        # Compute y positions for sig bars (stacked below plot)
        y_min = ax.get_ylim()[0]
        y_range = ax.get_ylim()[1] - y_min
        bar_height = y_range * 0.015
        bar_gap = y_range * 0.025

        for mi, (name, results) in enumerate(stats_dict.items()):
            sig_clusters = results.get("significant_clusters", [])
            col = colors.get(name, "#333333")
            y_pos = y_min - bar_gap * (mi + 1)

            for cluster in sig_clusters:
                t0 = times[cluster["start_idx"]]
                t1 = times[cluster["end_idx"]]
                ax.plot([t0, t1], [y_pos, y_pos], color=col, lw=3.5,
                        solid_capstyle="butt", alpha=0.85)

        # Adjust ylim to show bars
        if any(stats_dict[n].get("significant_clusters", [])
               for n in stats_dict):
            new_ymin = y_min - bar_gap * (len(stats_dict) + 1)
            ax.set_ylim(bottom=new_ymin)

    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9, framealpha=0.85)

    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        n_subj = next(iter(fits_dict.values())).shape[0]
        ax.set_title(f"RSA model fits (N={n_subj})", fontsize=11)

    fig.tight_layout()
    return fig, ax