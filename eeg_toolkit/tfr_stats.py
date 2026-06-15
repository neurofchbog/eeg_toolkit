"""
Spatiotemporal cluster-based permutation testing for TFR band power.

Workflow
--------
1. build_adjacency        — channel adjacency from montage × time
2. load_band_data         — per-subject band-averaged power (n_subj, n_ch, n_times)
3. run_cluster_test       — 1-sample cluster permutation on paired differences
4. describe_clusters      — pretty-print cluster channels + timing
5. plot_cluster_results   — condition traces + difference wave + topomap inset

Design notes
------------
- Band-average first, then cluster over channels × time. This is the standard
  approach for EEG time-frequency analysis and exploits electrode topology.
- Two-config pattern: ``cfg`` (pipeline) + ``cfg_tfr`` (TFR analysis).
- Adjacency is built once and reused across tests.
"""

import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test

from eeg_toolkit.io import find_subjects
from eeg_toolkit.tfr import (
    load_subject_tfrs,
    _resolve_conditions,
    _resolve_contrasts,
)


# ===================================================================
# Channel helpers
# ===================================================================

def get_eeg_channels(cfg, cfg_tfr, window_name):
    """
    Get the list of common EEG channels from the first subject's TFR.

    Returns only channels present in the standard_1020 montage,
    excluding non-EEG channels (EOG, AUX, MISC) and Iz.
    """
    subjects = find_subjects(cfg)
    tfrs = load_subject_tfrs(cfg, cfg_tfr, subjects[0], window_name)
    first_tfr = next(iter(tfrs.values()))
    ch_names = first_tfr.ch_names

    def _is_eeg(ch):
        up = ch.upper()
        return (up != "IZ" and
                not any(x in up for x in ("AUX", "HEOG", "VEOG", "EOG", "MISC")))

    return sorted([ch for ch in ch_names if _is_eeg(ch)])


# ===================================================================
# Adjacency
# ===================================================================

def build_adjacency(eeg_channels, n_times, montage="standard_1020"):
    """
    Build spatiotemporal adjacency matrix (channels × time).

    Parameters
    ----------
    eeg_channels : list of str
        Channel names.
    n_times : int
        Number of time points in the analysis window.
    montage : str
        MNE montage name.

    Returns
    -------
    adjacency : scipy.sparse matrix
    """
    info = mne.create_info(eeg_channels, sfreq=1000, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage(montage), match_case=False
    )
    ch_adj, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    adjacency = mne.stats.combine_adjacency(ch_adj,n_times)
    print(
        f"Adjacency: {len(eeg_channels)} channels × {n_times} time points "
        f"= {adjacency.shape[0]} nodes"
    )
    return adjacency


# ===================================================================
# Data loading
# ===================================================================

def load_band_data(cfg, cfg_tfr, window_name, freq_band, eeg_channels,
                   tmin=None, tmax=None, conditions=None):
    """
    Load per-subject band-averaged TFR power.

    Averages power within a frequency band, picks common EEG channels,
    and optionally crops time.

    Parameters
    ----------
    freq_band : tuple (fmin, fmax)
    eeg_channels : list of str
    tmin, tmax : float, optional
    conditions : list of str, optional
        None = all conditions (no contrasts).

    Returns
    -------
    data : dict {condition: np.ndarray (n_subjects, n_channels, n_times)}
    times : np.ndarray
    subjects_used : list of str
    """
    subjects = find_subjects(cfg)
    fmin, fmax = freq_band

    if conditions is None:
        conditions = list(_resolve_conditions(cfg_tfr, window_name).keys())

    # First pass: collect data
    all_data = {c: [] for c in conditions}
    subjects_used = []
    ref_times = None

    for subject in subjects:
        try:
            tfrs = load_subject_tfrs(cfg, cfg_tfr, subject, window_name)
        except Exception:
            continue

        # Check all conditions present
        if not all(c in tfrs for c in conditions):
            continue

        subjects_used.append(subject)

        for cond in conditions:
            tfr = tfrs[cond].copy().pick(eeg_channels, verbose="WARNING")

            freqs = tfr.freqs
            times = tfr.times
            data = tfr.data  # (n_ch, n_freqs, n_times)

            # Band average
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_data = data[:, freq_mask, :].mean(axis=1)  # (n_ch, n_times)

            if ref_times is None:
                ref_times = times.copy()

            all_data[cond].append(band_data)

    # Stack into arrays
    out = {}
    for cond in conditions:
        out[cond] = np.array(all_data[cond])  # (n_subj, n_ch, n_times)

    # Time crop
    if tmin is not None or tmax is not None:
        time_mask = np.ones(len(ref_times), dtype=bool)
        if tmin is not None:
            time_mask &= ref_times >= tmin
        if tmax is not None:
            time_mask &= ref_times <= tmax
        ref_times = ref_times[time_mask]
        for cond in out:
            out[cond] = out[cond][:, :, time_mask]

    n_subj = len(subjects_used)
    shape = next(iter(out.values())).shape
    print(
        f"Loaded {n_subj} subjects × {len(conditions)} conditions, "
        f"shape per condition: {shape}"
    )
    return out, ref_times, subjects_used


# ===================================================================
# Cluster test
# ===================================================================

def run_cluster_test(data_a, data_b, adjacency, label="",
                     n_permutations=1024, threshold=None, tail=0,
                     alpha=0.05, seed=42):
    """
    1-sample spatiotemporal cluster permutation test on paired differences.

    Parameters
    ----------
    data_a, data_b : np.ndarray (n_subjects, n_channels, n_times)
    adjacency : scipy.sparse matrix
        From build_adjacency.
    label : str
        Descriptive label for printing.
    n_permutations : int
        1024 for exploration, 10000 for publication.
    threshold : float or None
        Cluster-forming t-threshold. None = p=0.05 two-tailed.
    tail : int
        0 = two-tailed, 1 = a > b, -1 = a < b.
    alpha : float
        Significance threshold for clusters.
    seed : int
        Reproducibility.

    Returns
    -------
    results : dict with keys:
        t_obs, clusters, cluster_p_values, significant_clusters,
        diff, n_subjects, label, threshold
    """
    from scipy.stats import t as t_dist

    diff = data_a - data_b  # (n_subj, n_ch, n_times)
    n_subj = diff.shape[0]

    if threshold is None:
        threshold = t_dist.ppf(1 - 0.05 / 2, df=n_subj - 1)

    print(f"Running: {label}")
    print(
        f"  {n_subj} subjects, threshold t={threshold:.2f}, "
        f"{n_permutations} permutations, tail={tail}"
    )

    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        diff,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        adjacency=adjacency,
        n_jobs=-1,
        seed=seed,
        out_type="mask",
        verbose="WARNING",
    )

    # Pack significant clusters
    sig = [
        (i, p, clusters[i])
        for i, p in enumerate(cluster_pv)
        if p < alpha
    ]

    print(f"  Found {len(clusters)} clusters, {len(sig)} significant (p < {alpha})")

    return {
        "t_obs": t_obs,
        "clusters": clusters,
        "cluster_p_values": cluster_pv,
        "significant_clusters": sig,
        "diff": diff,
        "n_subjects": n_subj,
        "label": label,
        "threshold": threshold,
        "n_permutations": n_permutations,
    }


# ===================================================================
# Cluster description
# ===================================================================

def describe_clusters(results, eeg_channels, times, label=None):
    """
    Pretty-print significant cluster details with three effect size
    estimates following FieldTrip/Oostenveld recommendations.

    Effect sizes reported (all paired Cohen's d):
        1. d_cluster   — averaged over the cluster mask (most closely
                          related to the cluster inference)
        2. d_rectangle — averaged over the circumscribed rectangle
                          (conservative lower bound, easy to report)
        3. d_max       — maximum single channel-timepoint effect
                          (upper bound)

    Parameters
    ----------
    results : dict
        Output of run_cluster_test.
    eeg_channels : list of str
    times : np.ndarray
    label : str, optional
        Override results['label'].

    Returns
    -------
    list of dict
        One dict per significant cluster with all reporting metrics.
    """
    label = label or results.get("label", "")
    sig = results.get("significant_clusters", [])
    t_obs = results.get("t_obs")       # (n_ch, n_times)
    diff = results.get("diff")         # (n_subj, n_ch, n_times)

    print(f"\n{'=' * 60}")
    print(f"CLUSTERS: {label}")
    if "n_permutations" in results:
        print(f"  {results['n_subjects']} subjects, "
              f"threshold t = {results.get('threshold', '?'):.2f}, "
              f"{results['n_permutations']} permutations")
    print(f"{'=' * 60}")

    if not sig:
        print("No significant clusters")
        return []

    cluster_info = []

    for i, (idx, p, mask) in enumerate(sig, 1):
        ch_idx, t_idx = np.where(mask)
        ch_names = sorted(set(eeg_channels[c] for c in ch_idx))
        ch_unique = np.unique(ch_idx)
        t_unique = np.unique(t_idx)
        t0, t1 = times[t_unique[0]], times[t_unique[-1]]
        size = int(mask.sum())

        # Cluster mass: sum of t-values within the cluster
        tmass = float(t_obs[mask].sum()) if t_obs is not None else None

        d_cluster = None
        d_rectangle = None
        d_max = None
        max_ch = None
        max_time = None

        if diff is not None:
            n_subj = diff.shape[0]

            # 1. Cohen's d over cluster mask
            subj_means = np.array([
                diff[s][mask].mean() for s in range(n_subj)
            ])
            d_cluster = float(subj_means.mean() / subj_means.std())

            # 2. Cohen's d over circumscribed rectangle
            rect_diff = diff[:, ch_unique.min():ch_unique.max() + 1,
                             t_unique.min():t_unique.max() + 1]
            rect_subj = rect_diff.mean(axis=(1, 2))  # (n_subj,)
            d_rectangle = float(rect_subj.mean() / rect_subj.std())

            # 3. Maximum Cohen's d (single channel-timepoint)
            # Compute paired d at every ch × time within the cluster span
            rect_d = rect_diff.mean(axis=0) / rect_diff.std(axis=0)
            max_idx = np.unravel_index(
                np.nanargmax(np.abs(rect_d)), rect_d.shape
            )
            d_max = float(rect_d[max_idx])
            max_ch = eeg_channels[ch_unique.min() + max_idx[0]]
            max_time = float(times[t_unique.min() + max_idx[1]])

        info = {
            "channels": ch_names,
            "n_channels": len(ch_names),
            "time_start": float(t0),
            "time_end": float(t1),
            "duration": float(t1 - t0),
            "size": size,
            "tmass": tmass,
            "d_cluster": d_cluster,
            "d_rectangle": d_rectangle,
            "d_max": d_max,
            "d_max_channel": max_ch,
            "d_max_time": max_time,
            "p_value": float(p),
        }
        cluster_info.append(info)

        print(f"\n--- Cluster {i} (p = {p:.4f}) ---")
        print(f"Channels ({len(ch_names)}): {', '.join(ch_names)}")
        print(f"Time span: {t0:.3f} s to {t1:.3f} s ({t1 - t0:.3f} s)")
        print(f"Cluster size: {size} points")
        if tmass is not None:
            print(f"Cluster mass (tmass): {tmass:.3f}")
        if d_cluster is not None:
            print(f"Effect sizes (paired Cohen's d):")
            print(f"  d_cluster    = {d_cluster:.3f}  "
                  f"(averaged over cluster mask)")
            print(f"  d_rectangle  = {d_rectangle:.3f}  "
                  f"(circumscribed rectangle — lower bound)")
            print(f"  d_max        = {d_max:.3f}  "
                  f"(max at {max_ch}, {max_time:.3f} s — upper bound)")

    return cluster_info


# ===================================================================
# Visualization
# ===================================================================

def plot_cluster_results(data_a, data_b, results, eeg_channels, times,
                         label_a="A", label_b="B",
                         color_a="#2E86C1", color_b="#E74C3C",
                         roi=None, montage="standard_1020",
                         tmin=None, tmax=None,
                         figsize=(10, 8), title=None,
                         topo_pos=None):
    """
    Stacked panel figure: condition traces (top) + difference wave (bottom)
    with topomap inset and cluster time bar.
 
    Parameters
    ----------
    data_a, data_b : np.ndarray (n_subjects, n_channels, n_times)
    results : dict
        Output of run_cluster_test.
    eeg_channels : list of str
    times : np.ndarray
    label_a, label_b : str
        Condition labels for the legend.
    color_a, color_b : str
        Hex colors.
    roi : list of str, optional
        Channels for the timecourse. None = use significant cluster channels
        intersected with posterior sites, fallback to all channels.
    montage : str
    tmin, tmax : float, optional
        Plot window. None = full range.
    topo_pos : tuple of 4 floats, optional
        (x, y, width, height) in axes fraction coordinates for the topomap
        inset. Default places it upper-right. Example: (0.05, 0.55, 0.18, 0.35)
        to place it in the upper-left area.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
 
    # Build info for topomap
    info = mne.create_info(list(eeg_channels), sfreq=1000, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage(montage), match_case=False
    )
 
    # Determine ROI
    sig_channels = set()
    for _, _, mask in results.get("significant_clusters", []):
        for c in np.unique(np.where(mask)[0]):
            sig_channels.add(eeg_channels[c])
    sig_channels = sorted(sig_channels)
 
    if roi is None:
        posterior = ["P3", "P4", "Pz", "PO3", "PO4", "PO7", "PO8",
                     "O1", "O2", "Oz", "P7", "P8"]
        roi_intersect = [ch for ch in posterior if ch in sig_channels and ch in eeg_channels]
        roi = roi_intersect if roi_intersect else list(eeg_channels)
 
    roi_idx = [list(eeg_channels).index(ch) for ch in roi if ch in eeg_channels]
 
    # Time mask for plotting
    plot_mask = np.ones(len(times), dtype=bool)
    if tmin is not None:
        plot_mask &= times >= tmin
    if tmax is not None:
        plot_mask &= times <= tmax
    t_plot = times[plot_mask]
 
    # Helper: mean ± SEM over ROI
    def mean_sem(arr):
        roi_data = arr[:, roi_idx, :][:, :, plot_mask].mean(axis=1)
        m = roi_data.mean(axis=0)
        s = roi_data.std(axis=0) / np.sqrt(roi_data.shape[0])
        return m, s
 
    # Difference wave (paired SEM)
    def diff_wave(a, b):
        da = a[:, roi_idx, :][:, :, plot_mask].mean(axis=1)
        db = b[:, roi_idx, :][:, :, plot_mask].mean(axis=1)
        d = da - db
        return d.mean(axis=0), d.std(axis=0) / np.sqrt(d.shape[0])
 
    # Topomap: A-B averaged over time
    topo_a = data_a.mean(axis=(0, 2))  # (n_ch,)
    topo_b = data_b.mean(axis=(0, 2))
    topo_diff = topo_a - topo_b
 
    # Cluster time union
    tspan = None
    sig_list = results.get("significant_clusters", [])
    if sig_list:
        all_t_idx = np.concatenate(
            [np.unique(np.where(mask)[1]) for _, _, mask in sig_list]
        )
        tspan = (times[all_t_idx.min()], times[all_t_idx.max()])
 
    # ---- Figure ----
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
 
    # Top panel: condition traces
    ma, sa = mean_sem(data_a)
    mb, sb = mean_sem(data_b)
    ax_top.plot(t_plot, ma, color=color_a, lw=2.2, label=label_a)
    ax_top.fill_between(t_plot, ma - sa, ma + sa, color=color_a, alpha=0.22)
    ax_top.plot(t_plot, mb, color=color_b, lw=2.2, label=label_b)
    ax_top.fill_between(t_plot, mb - sb, mb + sb, color=color_b, alpha=0.22)
    ax_top.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_top.axvline(0, color="k", ls=":", lw=1, alpha=0.7)
    if tmin is not None or tmax is not None:
        ax_top.set_xlim(tmin, tmax)
    ax_top.set_ylabel("Power (log-ratio)")
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc="best", fontsize=9, framealpha=0.85)
 
    # Topomap inset
    if topo_pos is not None:
        inset = inset_axes(
            ax_top, width="100%", height="100%",
            bbox_to_anchor=topo_pos,
            bbox_transform=ax_top.transAxes,
            loc="center",
        )
    else:
        inset = inset_axes(
            ax_top, width="18%", height="35%", loc="upper right",
            bbox_to_anchor=(-0.22, -0.02, 1.0, 1.0),
            bbox_transform=ax_top.transAxes,
        )
    topo_mask = None
    if sig_channels:
        topo_mask = np.array([ch in sig_channels for ch in eeg_channels])
    vmax = np.abs(topo_diff).max()
    mne.viz.plot_topomap(
        topo_diff, info, axes=inset, cmap="RdBu_r",
        vlim=(-vmax, vmax), sensors=False,
        mask=topo_mask,
        mask_params=dict(
            marker="o", markerfacecolor="w",
            markeredgecolor="k", markersize=3.5,
        ) if topo_mask is not None else None,
        show=False,
    )
 
    # Bottom panel: difference wave
    md, sd = diff_wave(data_a, data_b)
    ax_bot.plot(t_plot, md, color="black", lw=1.8,
                label=f"{label_a} − {label_b}")
    ax_bot.fill_between(t_plot, md - sd, md + sd, color="black", alpha=0.18)
    ax_bot.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_bot.axvline(0, color="k", ls=":", lw=1, alpha=0.7)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Difference")
 
    # Cluster time bar
    if tspan is not None:
        y0, y1 = ax_bot.get_ylim()
        y_bar = y0 + 0.06 * (y1 - y0)
        ax_bot.plot(tspan, [y_bar, y_bar], color="black", lw=3.5,
                    solid_capstyle="butt")
 
    if title is None:
        n_sig = len(sig_list)
        roi_label = ", ".join(roi) if len(roi) <= 4 else f"{len(roi)} ch"
        title = f"{results.get('label', '')} — ROI: {roi_label} ({n_sig} sig. clusters)"
    ax_top.set_title(title, fontsize=12, fontweight="bold")
 
    fig.align_ylabels()
    return fig