"""
MVPA statistical testing — 1D cluster-based permutation tests.

Three test types, all built on the same core cluster engine:
    1. real_vs_chance    — is decoding above 0.5? (one-sample)
    2. real_vs_null      — is real better than label-shuffled null? (paired)
    3. compare_analyses  — is decoding A better than decoding B? (paired)

Workflow
--------
1. Load group data via ``mvpa.load_all_scores``
2. Call the appropriate test function
3. Results dict includes significant clusters, null distribution, t-map
4. Plot with ``plot_decoding_with_stats``

Design notes
------------
- 1D cluster permutation over time only (no spatial adjacency — scores
  are scalar per time point).
- Cluster mass = sum of absolute t-values within a cluster.
- Two-tailed by default (uses abs(t) for cluster formation and mass).
- Adapted from the Colab implementation with cleaner API and toolkit
  conventions.
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
# Core: permutation engine
# ===================================================================

def _run_permutation_paired(data_a, data_b, cluster_threshold,
                            n_permutations, seed, verbose):
    """
    Build the null distribution of maximum cluster mass by randomly
    swapping condition labels within subjects (paired sign-flip).

    Parameters
    ----------
    data_a, data_b : array (n_subjects, n_times)
    cluster_threshold : float
    n_permutations : int
    seed : int
    verbose : bool

    Returns
    -------
    null_max_masses : array (n_permutations,)
    """
    rng = np.random.RandomState(seed)
    n_subjects = data_a.shape[0]
    n_times = data_a.shape[1]
    times_dummy = np.arange(n_times)  # only used for find_clusters_1d

    null_max_masses = np.empty(n_permutations)

    for p in range(n_permutations):
        # Random sign-flip: swap a↔b for a random subset of subjects
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


def _run_permutation_onesample(data, cluster_threshold,
                               n_permutations, seed, verbose):
    """
    Build the null distribution for a one-sample test by randomly
    sign-flipping subjects' difference scores.

    Parameters
    ----------
    data : array (n_subjects, n_times)
        Difference scores (e.g., accuracy − 0.5).
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


def _evaluate_clusters(real_clusters, null_max_masses, alpha):
    """
    Determine which real clusters are significant given the null
    distribution. Adds 'p_value' and 'significant' to each cluster dict.

    Returns the list of significant clusters.
    """
    for cluster in real_clusters:
        mass = cluster["sum_stat"]
        # p = proportion of null distribution ≥ observed mass
        cluster["p_value"] = float((null_max_masses >= mass).mean())
        cluster["significant"] = cluster["p_value"] < alpha

    sig = [c for c in real_clusters if c["significant"]]
    return sig


# ===================================================================
# Public API: three test types
# ===================================================================

def test_real_vs_chance(group, chance=0.5,
                        n_permutations=10000, cluster_threshold=None,
                        alpha=0.05, seed=42, verbose=True):
    """
    Test whether decoding accuracy is significantly above chance.

    One-sample cluster permutation test on (scores − chance).

    Parameters
    ----------
    group : dict
        Output of ``mvpa.load_all_scores``.
    chance : float
        Chance-level accuracy (default 0.5 for binary).
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
    """
    scores = group["subject_scores"]  # (n_subj, n_times)
    times = group["times"]
    n_subj = scores.shape[0]

    diff = scores - chance

    if cluster_threshold is None:
        cluster_threshold = sp_stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

    if verbose:
        print(f"Test: real vs chance ({chance})")
        print(f"  {n_subj} subjects, threshold t={cluster_threshold:.2f}, "
              f"{n_permutations} permutations")

    # Real t-map
    real_t, _ = sp_stats.ttest_1samp(diff, 0, axis=0)
    real_clusters = find_clusters_1d(real_t, cluster_threshold, times)

    if verbose:
        print(f"  Found {len(real_clusters)} clusters in real data")

    # Permutation null
    null_max = _run_permutation_onesample(
        diff, cluster_threshold, n_permutations, seed, verbose)

    # Evaluate
    sig = _evaluate_clusters(real_clusters, null_max, alpha)

    if verbose:
        _print_results(sig, real_clusters, null_max, alpha)

    return _pack_results(real_t, real_clusters, sig, null_max,
                         times, n_subj, n_permutations,
                         cluster_threshold, alpha,
                         test_type="real_vs_chance")


def test_real_vs_null(group, n_permutations=10000, cluster_threshold=None,
                      alpha=0.05, seed=42, verbose=True):
    """
    Test whether real decoding is better than label-shuffled null.

    Paired cluster permutation test: real scores vs null scores.

    Parameters
    ----------
    group : dict
        Output of ``mvpa.load_all_scores``. Must contain
        ``subject_null_scores``.

    Returns
    -------
    results : dict
    """
    scores = group["subject_scores"]   # (n_subj, n_times)
    null = group["subject_null_scores"]  # (n_subj, n_perms, n_times)
    times = group["times"]
    n_subj = scores.shape[0]

    if null is None:
        raise ValueError("No null scores in group data. "
                         "Rerun decoding with n_null_permutations > 0.")

    # Average null across permutations per subject
    null_mean = null.mean(axis=1)  # (n_subj, n_times)

    if cluster_threshold is None:
        cluster_threshold = sp_stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

    if verbose:
        print(f"Test: real vs null")
        print(f"  {n_subj} subjects, threshold t={cluster_threshold:.2f}, "
              f"{n_permutations} permutations")
        print(f"  Mean real: {scores.mean():.4f}, "
              f"mean null: {null_mean.mean():.4f}")

    # Real t-map (paired)
    real_t, _ = sp_stats.ttest_rel(scores, null_mean, axis=0)
    real_clusters = find_clusters_1d(real_t, cluster_threshold, times)

    if verbose:
        print(f"  Found {len(real_clusters)} clusters in real data")

    # Permutation null
    null_max = _run_permutation_paired(
        scores, null_mean, cluster_threshold, n_permutations, seed, verbose)

    # Evaluate
    sig = _evaluate_clusters(real_clusters, null_max, alpha)

    if verbose:
        _print_results(sig, real_clusters, null_max, alpha)

    return _pack_results(real_t, real_clusters, sig, null_max,
                         times, n_subj, n_permutations,
                         cluster_threshold, alpha,
                         test_type="real_vs_null")


def compare_analyses(group_a, group_b,
                     n_permutations=10000, cluster_threshold=None,
                     alpha=0.05, seed=42, verbose=True):
    """
    Test whether decoding accuracy differs between two analyses.

    Paired cluster permutation test on scores_A vs scores_B
    (same subjects in both).

    Parameters
    ----------
    group_a, group_b : dict
        Output of ``mvpa.load_all_scores`` for two different analyses.
        Must have the same subjects in the same order.

    Returns
    -------
    results : dict
    """
    scores_a = group_a["subject_scores"]
    scores_b = group_b["subject_scores"]
    times = group_a["times"]
    n_subj = scores_a.shape[0]

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"Score shapes don't match: {scores_a.shape} vs {scores_b.shape}. "
            f"Ensure both analyses use the same window and resample settings."
        )

    if cluster_threshold is None:
        cluster_threshold = sp_stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

    label_a = group_a.get("analysis_name", "A")
    label_b = group_b.get("analysis_name", "B")

    if verbose:
        print(f"Test: {label_a} vs {label_b}")
        print(f"  {n_subj} subjects, threshold t={cluster_threshold:.2f}, "
              f"{n_permutations} permutations")

    # Real t-map (paired)
    real_t, _ = sp_stats.ttest_rel(scores_a, scores_b, axis=0)
    real_clusters = find_clusters_1d(real_t, cluster_threshold, times)

    if verbose:
        print(f"  Found {len(real_clusters)} clusters in real data")

    # Permutation null
    null_max = _run_permutation_paired(
        scores_a, scores_b, cluster_threshold,
        n_permutations, seed, verbose)

    # Evaluate
    sig = _evaluate_clusters(real_clusters, null_max, alpha)

    if verbose:
        _print_results(sig, real_clusters, null_max, alpha)

    return _pack_results(real_t, real_clusters, sig, null_max,
                         times, n_subj, n_permutations,
                         cluster_threshold, alpha,
                         test_type="compare_analyses")


# ===================================================================
# Result packing and printing
# ===================================================================

def _pack_results(real_t, real_clusters, sig_clusters, null_max,
                  times, n_subj, n_permutations,
                  cluster_threshold, alpha, test_type):
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

def plot_decoding_with_stats(group, results, ax=None,
                             color="#2E86C1", null_color="#999999",
                             sig_color="black", sig_y=None,
                             label="Real", show_null=True,
                             show_chance=True, title=None):
    """
    Plot group decoding accuracy with SEM and significance markers.

    Parameters
    ----------
    group : dict
        From ``mvpa.load_all_scores``.
    results : dict
        From any of the test functions.
    ax : matplotlib.axes.Axes or None
        If None, creates a new figure.
    color : str
        Color for the real accuracy trace.
    null_color : str
        Color for the null trace (if applicable).
    sig_color : str
        Color for significance dot markers.
    sig_y : float or None
        Y position for significance markers. None = auto (bottom of plot).
    label : str
        Legend label for the real trace.
    show_null : bool
        Show null distribution trace (only for real_vs_null tests).
    show_chance : bool
        Show the 0.5 chance line.
    title : str or None

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    times = group["times"]
    scores = group["subject_scores"]
    n_subj = scores.shape[0]

    # Real accuracy ± SEM
    mean = scores.mean(axis=0)
    sem = scores.std(axis=0) / np.sqrt(n_subj)

    ax.plot(times, mean, color=color, lw=2, label=label)
    ax.fill_between(times, mean - sem, mean + sem,
                    color=color, alpha=0.2)

    # Null trace (for real_vs_null tests)
    if show_null and group.get("subject_null_scores") is not None:
        null = group["subject_null_scores"]
        null_mean_per_subj = null.mean(axis=1)
        null_mean = null_mean_per_subj.mean(axis=0)
        null_sem = null_mean_per_subj.std(axis=0) / np.sqrt(n_subj)

        ax.plot(times, null_mean, color=null_color, lw=1.5, ls="--",
                label="Null")
        ax.fill_between(times, null_mean - null_sem, null_mean + null_sem,
                        color=null_color, alpha=0.15)

    # Chance line
    if show_chance:
        ax.axhline(0.5, color="k", lw=0.8, ls=":", alpha=0.5)

    # Cue onset
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.5)

    # Significance markers
    sig_clusters = results.get("significant_clusters", [])
    if sig_clusters:
        if sig_y is None:
            # Place markers just below the data
            all_vals = np.concatenate([mean - sem])
            sig_y = all_vals.min() - (mean.max() - all_vals.min()) * 0.08

        for cluster in sig_clusters:
            t_start = cluster["start_idx"]
            t_end = cluster["end_idx"]
            cluster_times = times[t_start:t_end + 1]
            ax.scatter(cluster_times, np.full_like(cluster_times, sig_y),
                       s=8, c=sig_color, alpha=0.8, marker="o",
                       zorder=5)

    # Formatting
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best", fontsize=9)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        test_type = results.get("test_type", "")
        n_sig = len(sig_clusters)
        ax.set_title(f"Decoding (N={n_subj}, {n_sig} sig. cluster"
                     f"{'s' if n_sig != 1 else ''}, "
                     f"{test_type})", fontsize=11)

    ax.tick_params(labelsize=9)
    return ax