"""
Time-frequency analysis — induced power via Morlet wavelets.

Workflow
--------
1. compute_tfr_subject / compute_tfr_all
   Load final epochs → subtract ERP (induced) → Morlet TFR → baseline (dB) → save
2. compute_grand_average_tfr
   Average per-subject TFRs across the group
3. Visualization helpers for exploration notebooks

Design notes
------------
- Two-config pattern: ``cfg`` (pipeline) + ``cfg_tfr`` (this analysis).
- Respects ``analysis_name`` suffix for filename disambiguation.
- One HDF5 per subject per window, all conditions stored via comment field.
- Grand averages: one HDF5 per condition under group_results/tfr/.
"""

from pathlib import Path

import numpy as np
import mne
from mne.time_frequency import read_tfrs, write_tfrs

from eeg_toolkit.io import find_subjects, get_subject_dir, get_analysis_dir
from eeg_toolkit.artifacts import get_final_epochs_path


# ===================================================================
# Path helpers
# ===================================================================

def _analysis_suffix(cfg_tfr):
    """Return '_<name>' suffix if analysis_name is set, else ''."""
    name = getattr(cfg_tfr.tfr, "analysis_name", None)
    return f"_{name}" if name else ""


def get_tfr_path(cfg, subject, window_name, cfg_tfr=None):
    """Per-subject TFR: subj01/subj01_cue_tfr.h5"""
    suffix = _analysis_suffix(cfg_tfr) if cfg_tfr else ""
    subj_dir = get_subject_dir(cfg, subject)
    return subj_dir / f"{subject}_{window_name}{suffix}_tfr.h5"


def get_grand_average_tfr_dir(cfg):
    """group_results/tfr/ — created on first use."""
    d = get_analysis_dir(cfg) / "group_results" / "tfr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_grand_average_tfr_path(cfg, window_name, condition, cfg_tfr=None):
    """Grand-average TFR per condition: grand_average_cue_spatial_tfr.h5"""
    suffix = _analysis_suffix(cfg_tfr) if cfg_tfr else ""
    d = get_grand_average_tfr_dir(cfg)
    return d / f"grand_average_{window_name}{suffix}_{condition}_tfr.h5"


# ===================================================================
# Config helpers
# ===================================================================

def _build_freqs(cfg_tfr):
    """Frequency vector from YAML: np.arange(start, stop+1, step)."""
    f = cfg_tfr.tfr.freqs
    return np.arange(f.start, f.stop + 1, f.step, dtype=float)


def _build_n_cycles(freqs, cfg_tfr):
    """Adaptive n_cycles = max(freqs / 2, n_cycles_min)."""
    mode = getattr(cfg_tfr.tfr, "n_cycles_mode", "adaptive")
    n_min = getattr(cfg_tfr.tfr, "n_cycles_min", 3)
    if mode == "adaptive":
        return np.maximum(freqs / 2.0, n_min)
    return np.full_like(freqs, float(n_min))


def _resolve_conditions(cfg_tfr, window_name):
    """Return {name: [event_ids]} — supports global or per-window layout."""
    conds = cfg_tfr.conditions
    if hasattr(conds, window_name):
        window_conds = getattr(conds, window_name)
        return {k: v for k, v in vars(window_conds).items()}
    return {k: v for k, v in vars(conds).items()}


def _resolve_contrasts(cfg_tfr, window_name):
    """Return {name: {cond: weight}} for a window, or empty dict."""
    contrasts = getattr(cfg_tfr, "contrasts", None)
    if contrasts is None:
        return {}
    if hasattr(contrasts, window_name):
        return {k: v for k, v in vars(getattr(contrasts, window_name)).items()}
    return {}


# ===================================================================
# Epoch selection
# ===================================================================

def _select_epochs(epochs, event_ids):
    """Select epochs matching a list of integer event IDs."""
    names = [name for name, eid in epochs.event_id.items() if eid in event_ids]
    if not names:
        raise ValueError(
            f"No epochs match event IDs {event_ids}. "
            f"Available: {epochs.event_id}"
        )
    return epochs[names]


# ===================================================================
# Core computation
# ===================================================================

def _subtract_evoked(epochs):
    """Subtract condition-average ERP from each trial → induced activity."""
    evoked = epochs.average()
    epochs_ind = epochs.copy()
    epochs_ind._data -= evoked.data[np.newaxis, :, :]
    return epochs_ind


def compute_tfr_subject(cfg, cfg_tfr, subject, overwrite=False):
    """
    Compute induced TFR for one subject, all configured windows.

    For each window × condition:
        1. Load final epochs, pick channels
        2. Select condition epochs
        3. Subtract ERP (induced decomposition)
        4. Compute Morlet TFR (average power across trials)
        5. Apply baseline correction (log-ratio / dB)
    Then compute contrasts (weighted sums of condition TFRs).
    Saves one HDF5 per window with all conditions + contrasts.

    Returns
    -------
    success : bool
    trial_counts : dict  {window: {condition: n_trials}}
    """
    # Unpack config
    freqs = _build_freqs(cfg_tfr)
    n_cycles = _build_n_cycles(freqs, cfg_tfr)
    decim = getattr(cfg_tfr.tfr, "decim", 1)
    n_jobs = getattr(cfg_tfr.tfr, "n_jobs", 1)
    decomposition = getattr(cfg_tfr.tfr, "decomposition", "induced")
    baseline_mode = getattr(cfg_tfr.tfr, "baseline_mode", "logratio")
    min_trials = getattr(cfg_tfr.tfr, "min_trials_per_condition", 10)
    picks = getattr(cfg_tfr.tfr, "picks", "eeg")

    trial_counts = {}

    for window_cfg in cfg_tfr.tfr.windows:
        wname = window_cfg.name
        out_path = get_tfr_path(cfg, subject, wname, cfg_tfr)

        # -- Overwrite guard --
        if not overwrite and out_path.exists():
            print(f"  {subject} [{wname}]: exists, skipping")
            continue

        # -- Load final epochs --
        epochs_path = get_final_epochs_path(cfg, subject, wname)
        if not epochs_path.exists():
            print(f"  {subject} [{wname}]: final epochs not found, skipping")
            return False, {}

        epochs = mne.read_epochs(str(epochs_path), verbose="WARNING")
        epochs.pick(picks, verbose="WARNING")

        # -- Per-condition TFR --
        conditions = _resolve_conditions(cfg_tfr, wname)
        baseline = tuple(window_cfg.baseline)
        tfr_list = []
        window_counts = {}

        for cond_name, event_ids in conditions.items():
            try:
                epochs_cond = _select_epochs(epochs, event_ids)
            except ValueError as e:
                print(f"  {subject} [{wname}] {cond_name}: {e}")
                continue

            n_trials = len(epochs_cond)
            window_counts[cond_name] = n_trials

            if n_trials < min_trials:
                print(
                    f"  {subject} [{wname}] {cond_name}: "
                    f"{n_trials} trials < min {min_trials}, skipping"
                )
                continue

            # Induced: subtract ERP
            if decomposition == "induced":
                epochs_cond = _subtract_evoked(epochs_cond)

            # Morlet TFR — average power across trials
            tfr = epochs_cond.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=True,
                decim=decim,
                n_jobs=n_jobs,
                verbose="WARNING",
            )

            # Baseline correction
            tfr.apply_baseline(
                baseline=baseline, mode=baseline_mode, verbose="WARNING"
            )

            tfr.comment = cond_name
            tfr_list.append(tfr)

        if not tfr_list:
            print(f"  {subject} [{wname}]: no valid conditions, skipping")
            continue

        # -- Contrasts (weighted sums) --
        contrasts = _resolve_contrasts(cfg_tfr, wname)
        cond_tfrs = {t.comment: t for t in tfr_list}

        for contrast_name, weights in contrasts.items():
            w = vars(weights) if hasattr(weights, "__dict__") else weights
            if not all(c in cond_tfrs for c in w):
                missing = [c for c in w if c not in cond_tfrs]
                print(
                    f"  {subject} [{wname}] contrast '{contrast_name}': "
                    f"missing {missing}, skipping"
                )
                continue
            contrast_data = sum(
                weight * cond_tfrs[c].data for c, weight in w.items()
            )
            contrast_tfr = cond_tfrs[list(w.keys())[0]].copy()
            contrast_tfr.data = contrast_data
            contrast_tfr.comment = contrast_name
            tfr_list.append(contrast_tfr)

        # -- Save --
        write_tfrs(str(out_path), tfr_list, overwrite=True)
        print(
            f"  {subject} [{wname}]: saved {len(tfr_list)} TFRs "
            f"({out_path.name})"
        )
        trial_counts[wname] = window_counts

    return True, trial_counts


def compute_tfr_all(cfg, cfg_tfr, overwrite=False):
    """Batch TFR computation with try/except + summary."""
    subjects = find_subjects(cfg)
    all_counts = {}
    results = {}

    print(f"Computing TFRs for {len(subjects)} subjects")
    print("=" * 50)

    for subject in subjects:
        print(f"\n{subject}")
        try:
            ok, counts = compute_tfr_subject(
                cfg, cfg_tfr, subject, overwrite=overwrite
            )
            results[subject] = "ok" if ok else "skipped"
            all_counts[subject] = counts
        except Exception as e:
            results[subject] = f"ERROR: {e}"
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'=' * 50}")
    n_ok = sum(1 for v in results.values() if v == "ok")
    n_err = sum(1 for v in results.values() if str(v).startswith("ERROR"))
    n_skip = len(subjects) - n_ok - n_err
    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_err} errors")

    return results, all_counts


# ===================================================================
# Grand averages
# ===================================================================

def _ensure_tfr_list(tfrs):
    """Normalize read_tfrs output — newer MNE returns object, older returns list."""
    if isinstance(tfrs, list):
        return tfrs
    return [tfrs]


def load_subject_tfrs(cfg, cfg_tfr, subject, window_name):
    """Load all condition TFRs for one subject + window → {comment: AverageTFR}."""
    fpath = get_tfr_path(cfg, subject, window_name, cfg_tfr)
    tfr_list = _ensure_tfr_list(read_tfrs(str(fpath), verbose="WARNING"))
    return {t.comment: t for t in tfr_list}


def compute_grand_average_tfr(cfg, cfg_tfr):
    """
    Average per-subject TFRs across the group.

    Saves one HDF5 per condition (and contrast) per window.
    """
    subjects = find_subjects(cfg)

    for window_cfg in cfg_tfr.tfr.windows:
        wname = window_cfg.name

        # Collect all labels: conditions + contrasts
        conditions = _resolve_conditions(cfg_tfr, wname)
        contrasts = _resolve_contrasts(cfg_tfr, wname)
        all_labels = list(conditions.keys()) + list(contrasts.keys())

        for label in all_labels:
            subject_tfrs = []

            for subject in subjects:
                try:
                    tfrs = load_subject_tfrs(cfg, cfg_tfr, subject, wname)
                except Exception:
                    continue
                if label not in tfrs:
                    continue
                subject_tfrs.append(tfrs[label])

            if len(subject_tfrs) < 2:
                print(
                    f"  [{wname}] {label}: "
                    f"{len(subject_tfrs)} subjects, skipping grand average"
                )
                continue

            # Average across subjects
            ga = subject_tfrs[0].copy()
            ga.data = np.mean([t.data for t in subject_tfrs], axis=0)
            ga.comment = f"grand_average_{label} (N={len(subject_tfrs)})"

            out_path = get_grand_average_tfr_path(cfg, wname, label, cfg_tfr)
            ga.save(str(out_path), overwrite=True)
            print(
                f"  [{wname}] {label}: grand average saved "
                f"(N={len(subject_tfrs)})"
            )


# ===================================================================
# Loading helpers (for notebooks)
# ===================================================================

def load_grand_average_tfr(cfg, cfg_tfr, window_name, condition):
    """Load a single grand-average TFR by condition name."""
    fpath = get_grand_average_tfr_path(cfg, window_name, condition, cfg_tfr)
    tfrs = _ensure_tfr_list(read_tfrs(str(fpath), verbose="WARNING"))
    return tfrs[0]


def load_all_grand_averages(cfg, cfg_tfr, window_name):
    """Load all available grand-average TFRs for a window → {label: AverageTFR}."""
    conditions = _resolve_conditions(cfg_tfr, window_name)
    contrasts = _resolve_contrasts(cfg_tfr, window_name)
    all_labels = list(conditions.keys()) + list(contrasts.keys())

    result = {}
    for label in all_labels:
        fpath = get_grand_average_tfr_path(cfg, window_name, label, cfg_tfr)
        if fpath.exists():
            tfrs = _ensure_tfr_list(read_tfrs(str(fpath), verbose="WARNING"))
            result[label] = tfrs[0]
    return result


# ===================================================================
# Visualization
# ===================================================================

def plot_tfr_heatmap(tfr, title=None, vmin=None, vmax=None, cmap="RdBu_r",
                     fmin=None, fmax=None, tmin=None, tmax=None, axes=None):
    """
    Time-frequency heatmap for an AverageTFR.

    If the TFR has multiple channels they are averaged together.
    Color scale is symmetric around 0 by default (appropriate for
    log-ratio baseline).

    Parameters
    ----------
    tfr : AverageTFR
    title : str, optional
    vmin, vmax : float, optional
        Explicit color limits. If both None, symmetric around 0.
    cmap : str
    fmin, fmax : float, optional
        Frequency crop.
    tmin, tmax : float, optional
        Time crop.
    axes : matplotlib Axes, optional
        Draw into existing axes. If None, creates a new figure.

    Returns
    -------
    fig : Figure or None (if axes was provided)
    """
    import matplotlib.pyplot as plt

    data = tfr.data.copy()  # (n_ch, n_freqs, n_times)
    freqs = tfr.freqs.copy()
    times = tfr.times.copy()

    # Average across channels if multi-channel
    if data.ndim == 3:
        data = data.mean(axis=0)

    # Crop frequencies
    freq_mask = np.ones(len(freqs), dtype=bool)
    if fmin is not None:
        freq_mask &= freqs >= fmin
    if fmax is not None:
        freq_mask &= freqs <= fmax
    data = data[freq_mask, :]
    freqs = freqs[freq_mask]

    # Crop time
    time_mask = np.ones(len(times), dtype=bool)
    if tmin is not None:
        time_mask &= times >= tmin
    if tmax is not None:
        time_mask &= times <= tmax
    data = data[:, time_mask]
    times = times[time_mask]

    # Symmetric color scale
    if vmin is None and vmax is None:
        vlim = np.max(np.abs(data))
        vmin, vmax = -vlim, vlim

    fig = None
    if axes is None:
        fig, axes = plt.subplots(figsize=(8, 4))

    im = axes.pcolormesh(
        times, freqs, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    if tmin is not None or tmax is not None:
        axes.set_xlim(tmin, tmax)
    if fmin is not None or fmax is not None:
        axes.set_ylim(fmin, fmax)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Frequency (Hz)")
    if title:
        axes.set_title(title)

    cb = plt.colorbar(im, ax=axes)
    cb.set_label("Power (log-ratio)")

    if fig is not None:
        fig.tight_layout()
    return fig


def plot_tfr_conditions(cfg, cfg_tfr, window_name, channels,
                        conditions=None, vmin=None, vmax=None,
                        fmin=None, fmax=None, tmin=None, tmax=None,
                        cmap="RdBu_r", figsize=None):
    """
    Side-by-side TFR heatmaps — one panel per condition — for a channel
    or ROI (averaged if multiple channels given).

    Parameters
    ----------
    channels : str or list of str
        Channel name(s). Multiple = ROI average.
    conditions : list of str, optional
        Subset of labels to plot. None = all available.
    """
    import matplotlib.pyplot as plt

    ga = load_all_grand_averages(cfg, cfg_tfr, window_name)
    labels = conditions or list(ga.keys())
    labels = [l for l in labels if l in ga]

    if isinstance(channels, str):
        channels = [channels]

    ncols = len(labels)
    if figsize is None:
        figsize = (5 * ncols, 4)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)

    for i, label in enumerate(labels):
        tfr = ga[label].copy()
        tfr.pick(channels, verbose="WARNING")
        plot_tfr_heatmap(
            tfr, title=label, vmin=vmin, vmax=vmax, cmap=cmap,
            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            axes=axes[0, i],
        )

    roi_label = ", ".join(channels) if len(channels) <= 3 else f"{len(channels)} ch"
    fig.suptitle(f"TFR — {roi_label} [{window_name}]", fontsize=12)
    fig.tight_layout()
    return fig


def plot_tfr_topomaps(cfg, cfg_tfr, window_name, conditions=None,
                      fmin=4, fmax=8, tmin=0.0, tmax=0.5,
                      vmin=None, vmax=None, cmap="RdBu_r", figsize=None):
    """
    Topographic maps of mean power in a time-frequency window.

    One topomap per condition, useful for identifying spatial patterns.

    Parameters
    ----------
    fmin, fmax : float
        Frequency band (Hz).
    tmin, tmax : float
        Time window (s).
    """
    import matplotlib.pyplot as plt

    ga = load_all_grand_averages(cfg, cfg_tfr, window_name)
    labels = conditions or list(ga.keys())
    labels = [l for l in labels if l in ga]

    ncols = len(labels)
    if figsize is None:
        figsize = (4 * ncols, 4)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)

    for i, label in enumerate(labels):
        tfr = ga[label]
        vlim = (vmin, vmax) if vmin is not None or vmax is not None else (None, None)
        tfr.plot_topomap(
            tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
            vlim=vlim, cmap=cmap,
            axes=axes[0, i], show=False, colorbar=(i == ncols - 1),
            
        )
        axes[0, i].set_title(label)

    band_label = f"{fmin}-{fmax} Hz, {tmin}-{tmax} s"
    fig.suptitle(f"TFR topomaps — {band_label} [{window_name}]", fontsize=12)
    fig.tight_layout()
    return fig


def plot_tfr_timecourse(cfg, cfg_tfr, window_name, channels, fmin, fmax,
                        conditions=None, tmin=None, tmax=None,
                        with_ci=True, ci_alpha=0.25,
                        colors=None, figsize=(8, 4), title=None):
    """
    Band power time course with optional SEM shading.

    Loads per-subject TFRs, averages across channels and the frequency
    band, then plots the mean ± SEM time course for each condition.

    Parameters
    ----------
    channels : str or list of str
        Channel name(s). Multiple = ROI average.
    fmin, fmax : float
        Frequency band to average (Hz).
    conditions : list of str, optional
        Subset of labels to plot. None = all conditions (excludes contrasts).
    tmin, tmax : float, optional
        Time crop.
    with_ci : bool
        If True (default), plot SEM shading around the mean.
    ci_alpha : float
        Opacity of the SEM shading.
    colors : dict, optional
        {condition: color}. Defaults to tab10.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt
    from scipy.stats import sem

    subjects = find_subjects(cfg)

    # Default to conditions only (no contrasts)
    if conditions is None:
        cond_keys = list(_resolve_conditions(cfg_tfr, window_name).keys())
        conditions = list(cond_keys)
    if isinstance(channels, str):
        channels = [channels]

    # Default colors
    if colors is None:
        tab10 = plt.cm.tab10.colors
        colors = {c: tab10[i % 10] for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=figsize)
    times = None

    for cond_name in conditions:
        # Collect band power timecourse per subject
        subject_traces = []
        for subject in subjects:
            try:
                tfrs = load_subject_tfrs(cfg, cfg_tfr, subject, window_name)
            except Exception:
                continue
            if cond_name not in tfrs:
                continue

            tfr = tfrs[cond_name].copy()
            tfr.pick(channels, verbose="WARNING")

            freqs = tfr.freqs
            data = tfr.data.mean(axis=0)  # avg channels → (n_freqs, n_times)

            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = data[freq_mask, :].mean(axis=0)  # (n_times,)
            subject_traces.append(band_power)

            if times is None:
                times = tfr.times.copy()

        if not subject_traces:
            continue

        traces = np.array(subject_traces)  # (n_subjects, n_times)
        mean_power = traces.mean(axis=0)
        sem_power = sem(traces, axis=0)

        # Crop time
        time_mask = np.ones(len(times), dtype=bool)
        if tmin is not None:
            time_mask &= times >= tmin
        if tmax is not None:
            time_mask &= times <= tmax

        t = times[time_mask]
        m = mean_power[time_mask]
        s = sem_power[time_mask]
        color = colors.get(cond_name)

        ax.plot(t, m, label=cond_name, color=color)
        if with_ci:
            ax.fill_between(t, m - s, m + s, alpha=ci_alpha, color=color)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    if tmin is not None or tmax is not None:
        ax.set_xlim(tmin, tmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (log-ratio)")
    ax.legend()

    roi_label = ", ".join(channels) if len(channels) <= 3 else f"{len(channels)} ch"
    if title is None:
        title = f"{fmin}-{fmax} Hz — {roi_label} [{window_name}]"
    ax.set_title(title)

    fig.tight_layout()
    return fig


# ===================================================================
# Amplitude extraction for statistics
# ===================================================================

def extract_mean_band_power(cfg, cfg_tfr, window_name, rois, freq_bands,
                            time_windows, conditions=None):
    """
    Extract mean band power per subject × condition × ROI × band × time window.

    Analogous to erp_stats.extract_mean_amplitudes but for TFR data.

    Parameters
    ----------
    rois : dict
        {roi_name: [channel_names]}
    freq_bands : dict
        {band_name: (fmin, fmax)}
    time_windows : dict
        {tw_name: (tmin, tmax)}
    conditions : list of str, optional
        Subset of conditions. None = all conditions (no contrasts).

    Returns
    -------
    df : pandas.DataFrame
        Long format with columns: subject, condition, roi, channels,
        freq_band, fmin, fmax, time_window, tmin, tmax, mean_power
    """
    import pandas as pd

    subjects = find_subjects(cfg)

    if conditions is None:
        conditions = list(_resolve_conditions(cfg_tfr, window_name).keys())

    rows = []

    for subject in subjects:
        try:
            tfrs = load_subject_tfrs(cfg, cfg_tfr, subject, window_name)
        except Exception:
            print(f"  {subject}: file missing, skipping")
            continue

        for cond_name in conditions:
            if cond_name not in tfrs:
                continue
            tfr = tfrs[cond_name]

            for roi_name, ch_list in rois.items():
                tfr_roi = tfr.copy().pick(ch_list, verbose="WARNING")
                data = tfr_roi.data.mean(axis=0)  # avg channels → (n_freqs, n_times)
                freqs = tfr_roi.freqs
                times = tfr_roi.times

                for band_name, (fmin, fmax) in freq_bands.items():
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)

                    for tw_name, (tmin, tmax) in time_windows.items():
                        time_mask = (times >= tmin) & (times <= tmax)

                        mean_power = data[np.ix_(freq_mask, time_mask)].mean()

                        rows.append({
                            "subject": subject,
                            "condition": cond_name,
                            "roi": roi_name,
                            "channels": ", ".join(ch_list),
                            "freq_band": band_name,
                            "fmin": fmin,
                            "fmax": fmax,
                            "time_window": tw_name,
                            "tmin": tmin,
                            "tmax": tmax,
                            "mean_power": mean_power,
                        })

    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} rows: {len(subjects)} subjects × "
          f"{len(conditions)} conditions × {len(rois)} ROIs × "
          f"{len(freq_bands)} bands × {len(time_windows)} time windows")
    return df

def run_tfr_paired_tests(df, contrast, alpha_fdr=0.05):
    """
    Paired t-tests + Wilcoxon for a two-condition contrast across
    all ROI × freq_band × time_window combinations.

    Parameters
    ----------
    df : DataFrame
        Output of extract_mean_band_power.
    contrast : tuple of str
        (condition_a, condition_b). Effect = a - b.
    alpha_fdr : float
        FDR threshold.

    Returns
    -------
    stats_df : DataFrame
        One row per ROI × band × time window with t, p, Wilcoxon,
        Cohen's d, FDR-corrected p-values.
    """
    import pandas as pd
    from scipy.stats import ttest_rel, wilcoxon
    from statsmodels.stats.multitest import multipletests

    cond_a, cond_b = contrast
    df_a = df[df["condition"] == cond_a]
    df_b = df[df["condition"] == cond_b]

    groups = ["roi", "freq_band", "time_window"]
    rows = []

    for keys, grp_a in df_a.groupby(groups):
        roi, band, tw = keys
        grp_b = df_b[
            (df_b["roi"] == roi) &
            (df_b["freq_band"] == band) &
            (df_b["time_window"] == tw)
        ]

        # Align by subject
        merged = pd.merge(
            grp_a[["subject", "mean_power"]],
            grp_b[["subject", "mean_power"]],
            on="subject", suffixes=("_a", "_b"),
        )

        if len(merged) < 3:
            continue

        vals_a = merged["mean_power_a"].values
        vals_b = merged["mean_power_b"].values
        diff = vals_a - vals_b

        # Paired t-test
        t_stat, p_ttest = ttest_rel(vals_a, vals_b)

        # Wilcoxon signed-rank
        try:
            w_stat, p_wilcoxon = wilcoxon(diff)
        except ValueError:
            w_stat, p_wilcoxon = np.nan, np.nan

        # Cohen's d (paired)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

        # Get metadata from group
        meta = grp_a.iloc[0]

        rows.append({
            "roi": roi,
            "freq_band": band,
            "fmin": meta["fmin"],
            "fmax": meta["fmax"],
            "time_window": tw,
            "tmin": meta["tmin"],
            "tmax": meta["tmax"],
            "n": len(merged),
            "mean_a": vals_a.mean(),
            "mean_b": vals_b.mean(),
            "mean_diff": diff.mean(),
            "cohens_d": d,
            "t_stat": t_stat,
            "p_ttest": p_ttest,
            "w_stat": w_stat,
            "p_wilcoxon": p_wilcoxon,
        })

    stats_df = pd.DataFrame(rows)

    if len(stats_df) == 0:
        return stats_df

    # FDR correction (Benjamini-Hochberg) on t-test p-values
    reject, p_fdr, _, _ = multipletests(
        stats_df["p_ttest"], alpha=alpha_fdr, method="fdr_bh"
    )
    stats_df["p_fdr"] = p_fdr
    stats_df["sig_fdr"] = reject

    # Label
    stats_df["contrast"] = f"{cond_a}_vs_{cond_b}"

    # Sort for readability
    stats_df = stats_df.sort_values(["freq_band", "roi", "time_window"]).reset_index(drop=True)

    n_sig = stats_df["sig_fdr"].sum()
    print(f"Paired tests: {len(stats_df)} comparisons, {n_sig} significant after FDR")

    return stats_df

# ===================================================================
# Cluster-based permutation testing
# ===================================================================

def run_tfr_cluster_test(cfg, cfg_tfr, window_name, channels,
                         contrast=("spatial", "symbolic"),
                         n_permutations=1024, threshold=None, tail=0,
                         fmin=None, fmax=None, tmin=None, tmax=None):
    """
    Cluster-based permutation test on TFR data (frequency × time).

    Averages across channels (ROI), computes per-subject condition
    differences, then runs a 1-sample cluster test on the 2D maps.

    Parameters
    ----------
    channels : str or list of str
        Channel name(s). Multiple = ROI average.
    contrast : tuple of str
        (condition_a, condition_b). Effect direction = a − b.
    n_permutations : int
        Number of permutations (1024 for exploration, 10000 for publication).
    threshold : float or None
        Cluster-forming threshold (t-value). If None, uses p=0.05 two-tailed.
    tail : int
        0 = two-tailed, 1 = a > b, -1 = a < b.
    fmin, fmax : float, optional
        Frequency crop before testing.
    tmin, tmax : float, optional
        Time crop before testing.

    Returns
    -------
    results : dict with keys:
        T_obs       : (n_freqs, n_times) — observed T-statistics
        clusters    : list of boolean arrays marking each cluster
        cluster_pv  : array of p-values per cluster
        freqs       : frequency vector (after crop)
        times       : time vector (after crop)
        n_subjects  : number of subjects included
    """
    from mne.stats import permutation_cluster_1samp_test
    from scipy.stats import t as t_dist

    if isinstance(channels, str):
        channels = [channels]

    subjects = find_subjects(cfg)
    cond_a, cond_b = contrast

    # Collect per-subject differences
    diffs = []
    ref_freqs = None
    ref_times = None

    for subject in subjects:
        try:
            tfrs = load_subject_tfrs(cfg, cfg_tfr, subject, window_name)
        except Exception:
            continue
        if cond_a not in tfrs or cond_b not in tfrs:
            continue

        tfr_a = tfrs[cond_a].copy().pick(channels, verbose="WARNING")
        tfr_b = tfrs[cond_b].copy().pick(channels, verbose="WARNING")

        # Average across channels → (n_freqs, n_times)
        data_a = tfr_a.data.mean(axis=0)
        data_b = tfr_b.data.mean(axis=0)

        if ref_freqs is None:
            ref_freqs = tfr_a.freqs.copy()
            ref_times = tfr_a.times.copy()

        diffs.append(data_a - data_b)

    diffs = np.array(diffs)  # (n_subjects, n_freqs, n_times)
    n_subj = len(diffs)
    print(f"Loaded {n_subj} subjects")

    # Crop frequencies
    freq_mask = np.ones(len(ref_freqs), dtype=bool)
    if fmin is not None:
        freq_mask &= ref_freqs >= fmin
    if fmax is not None:
        freq_mask &= ref_freqs <= fmax
    diffs = diffs[:, freq_mask, :]
    ref_freqs = ref_freqs[freq_mask]

    # Crop time
    time_mask = np.ones(len(ref_times), dtype=bool)
    if tmin is not None:
        time_mask &= ref_times >= tmin
    if tmax is not None:
        time_mask &= ref_times <= tmax
    diffs = diffs[:, :, time_mask]
    ref_times = ref_times[time_mask]

    print(f"Test dimensions: {len(ref_freqs)} freqs × {len(ref_times)} time points")

    # Default threshold: t critical at p=0.05 two-tailed
    if threshold is None:
        threshold = t_dist.ppf(1 - 0.05 / 2, df=n_subj - 1)
        print(f"Cluster-forming threshold: t = {threshold:.2f} (p=0.05, df={n_subj - 1})")

    # Run cluster test
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        diffs,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        verbose="WARNING",
    )

    # Summary
    n_sig = np.sum(cluster_pv < 0.05)
    print(f"Found {len(clusters)} clusters, {n_sig} significant (p < 0.05)")
    for i, (cl, pv) in enumerate(zip(clusters, cluster_pv)):
        if pv < 0.1:
            freq_idx, time_idx = np.where(cl)
            f_range = (ref_freqs[freq_idx.min()], ref_freqs[freq_idx.max()])
            t_range = (ref_times[time_idx.min()], ref_times[time_idx.max()])
            print(f"  Cluster {i+1}: p={pv:.4f}, "
                  f"{f_range[0]:.0f}-{f_range[1]:.0f} Hz, "
                  f"{t_range[0]:.3f}-{t_range[1]:.3f} s")

    return {
        "T_obs": T_obs,
        "clusters": clusters,
        "cluster_pv": cluster_pv,
        "freqs": ref_freqs,
        "times": ref_times,
        "n_subjects": n_subj,
        "contrast": f"{cond_a} − {cond_b}",
    }


def plot_tfr_clusters(results, alpha=0.05, vmax=None, cmap="RdBu_r",
                      figsize=(9, 5), title=None):
    """
    Plot T-statistic heatmap with significant cluster contours.

    Parameters
    ----------
    results : dict
        Output of run_tfr_cluster_test.
    alpha : float
        Significance threshold for cluster p-values.
    vmax : float, optional
        Symmetric color limit. If None, auto from data.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    T_obs = results["T_obs"]
    clusters = results["clusters"]
    cluster_pv = results["cluster_pv"]
    freqs = results["freqs"]
    times = results["times"]

    if vmax is None:
        vmax = np.max(np.abs(T_obs))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        times, freqs, T_obs, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto"
    )
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(freqs[0], freqs[-1])

    # Overlay significant clusters as contours
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for cl, pv in zip(clusters, cluster_pv):
        if pv < alpha:
            sig_mask |= cl

    if sig_mask.any():
        ax.contour(
            times, freqs, sig_mask.astype(float),
            levels=[0.5], colors="black", linewidths=2,
        )

    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("T-statistic")

    if title is None:
        n_sig = np.sum(cluster_pv < alpha)
        title = (f"Cluster permutation: {results['contrast']} "
                 f"(N={results['n_subjects']}, {n_sig} sig. clusters)")
    ax.set_title(title)

    fig.tight_layout()
    return fig