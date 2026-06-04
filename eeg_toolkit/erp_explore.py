"""
Exploratory ERP visualization.

Loads grand-average evokeds and produces three types of plots to help
identify ROIs and time windows of interest.

All plot functions accept an optional `cfg_erp` argument so they can find
analysis-specific grand averages (e.g., when analysis_name='position' is
set in the YAML). When cfg_erp is None, falls back to default filenames.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne

from eeg_toolkit.io import find_subjects
from eeg_toolkit.evoked import (
    get_grand_average_path,
    get_evoked_path,
    _get_erp_dir,
    _resolve_windows,
)


# ============================================================================
# Path helpers
# ============================================================================

def _get_figures_dir(cfg):
    """group_results/erp/figures (created if missing)."""
    fig_dir = _get_erp_dir(cfg) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save_or_show(fig, save, fname, cfg, verbose=True):
    """Save figure if save=True, otherwise show."""
    if save:
        fpath = _get_figures_dir(cfg) / fname
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f"   saved: {fpath.name}")
    else:
        plt.show()


# ============================================================================
# Loading helpers
# ============================================================================

def load_grand_averages(cfg, window_name, cfg_erp=None):
    """
    Load grand-average evokeds for one window as a dict keyed by comment.
    """
    path = get_grand_average_path(cfg, window_name, cfg_erp)
    if not path.exists():
        raise FileNotFoundError(
            f"Grand average not found: {path.name}. "
            f"Run compute_grand_averages first."
        )
    evokeds = mne.read_evokeds(path, verbose="WARNING")
    return {evo.comment: evo for evo in evokeds}


# ============================================================================
# Plot 1: Butterfly + topomap join plot
# ============================================================================

def plot_butterfly(cfg, window_name, conditions=None,
                   times="peaks", save=False, verbose=True, cfg_erp=None):
    """Butterfly plot of grand-average ERPs with topomaps at peak times."""
    gas = load_grand_averages(cfg, window_name, cfg_erp)

    if conditions is None:
        conditions = list(gas.keys())

    for label in conditions:
        if label not in gas:
            print(f"   WARNING: '{label}' not in grand averages - skipping")
            continue

        evo = gas[label]
        fig = evo.plot_joint(
            times=times,
            title=f"{window_name} - {label}",
            show=False,
        )
        _save_or_show(
            fig, save,
            f"{window_name}_butterfly_{label}.png",
            cfg, verbose,
        )


# ============================================================================
# Plot 2: Topographic maps at sampled timepoints
# ============================================================================

def plot_topomaps(cfg, window_name, times,
                  conditions=None, save=False, verbose=True,
                  vlim=(None, None), cfg_erp=None):
    """Topographic maps at user-specified timepoints, one row per condition."""
    gas = load_grand_averages(cfg, window_name, cfg_erp)

    if conditions is None:
        conditions = list(gas.keys())

    for label in conditions:
        if label not in gas:
            print(f"   WARNING: '{label}' not in grand averages - skipping")
            continue

        evo = gas[label]
        fig = evo.plot_topomap(
            times=times,
            ch_type="eeg",
            colorbar=True,
            vlim=vlim,
            show=False,
        )
        fig.suptitle(f"{window_name} - {label}", y=1.02)
        _save_or_show(
            fig, save,
            f"{window_name}_topomap_{label}.png",
            cfg, verbose,
        )


# ============================================================================
# Plot 3: Channel overlays
# ============================================================================

def plot_channel_overlay(cfg, window_name, channels,
                         conditions=None, save=False, verbose=True,
                         cfg_erp=None):
    """Overlay conditions at specified channels (one figure per channel)."""
    if isinstance(channels, str):
        channels = [channels]

    gas = load_grand_averages(cfg, window_name, cfg_erp)

    if conditions is None:
        conditions = [k for k in gas.keys() if "_vs_" not in k]

    for ch in channels:
        first_evo = next(iter(gas.values()))
        if ch not in first_evo.ch_names:
            print(f"   WARNING: channel '{ch}' not found - skipping")
            continue

        evokeds_to_plot = {label: gas[label] for label in conditions
                           if label in gas}
        if not evokeds_to_plot:
            continue

        fig = mne.viz.plot_compare_evokeds(
            evokeds_to_plot,
            picks=ch,
            title=f"{window_name} - {ch}",
            show=False,
            ci=False,
        )
        if isinstance(fig, list):
            fig = fig[0]

        _save_or_show(
            fig, save,
            f"{window_name}_overlay_{ch}.png",
            cfg, verbose,
        )


# ============================================================================
# Plot 4: Channel overlays with confidence intervals
# ============================================================================

def plot_channel_overlay_with_ci(cfg, cfg_erp_or_none, window_name, channels,
                                 conditions=None, save=False, verbose=True):
    """
    Like plot_channel_overlay, but with 95% CI across subjects.

    Note: second positional arg is cfg_erp (can be None) — kept positional
    for backwards compatibility with paired-analysis notebooks.
    """
    if isinstance(channels, str):
        channels = [channels]

    cfg_erp = cfg_erp_or_none
    subjects = find_subjects(cfg)

    evokeds_per_cond = {}
    for subject in subjects:
        evo_path = get_evoked_path(cfg, subject, window_name, cfg_erp)
        if not evo_path.exists():
            continue
        subj_evokeds = mne.read_evokeds(evo_path, verbose="WARNING")
        for evo in subj_evokeds:
            evokeds_per_cond.setdefault(evo.comment, []).append(evo)

    if conditions is None:
        conditions = [k for k in evokeds_per_cond.keys() if "_vs_" not in k]

    if not evokeds_per_cond:
        print("   No subject evokeds found.")
        return

    for ch in channels:
        first_list = next(iter(evokeds_per_cond.values()))
        if ch not in first_list[0].ch_names:
            print(f"   WARNING: channel '{ch}' not found - skipping")
            continue

        evokeds_to_plot = {label: evokeds_per_cond[label]
                           for label in conditions
                           if label in evokeds_per_cond}
        if not evokeds_to_plot:
            continue

        fig = mne.viz.plot_compare_evokeds(
            evokeds_to_plot,
            picks=ch,
            title=f"{window_name} - {ch} (mean ± 95% CI, N={len(subjects)})",
            show=False,
            ci=0.95,
        )
        if isinstance(fig, list):
            fig = fig[0]

        _save_or_show(
            fig, save,
            f"{window_name}_overlay_ci_{ch}.png",
            cfg, verbose,
        )


# ============================================================================
# Plot 5: ROI overlays
# ============================================================================

def plot_roi_overlay(cfg, window_name, rois,
                     conditions=None, with_ci=True,
                     save=False, verbose=True, cfg_erp=None):
    """
    Overlay conditions averaged across channels within named ROIs.

    Parameters
    ----------
    cfg : SimpleNamespace
    window_name : str
    rois : dict {roi_name: [channel_list]}
    conditions : list of str or None
    with_ci : bool
        If True, plots 95% CI across subjects.
    save : bool
    verbose : bool
    cfg_erp : SimpleNamespace or None
        Optional. If provided, used to find analysis-specific evokeds.
    """
    for roi_name, channels in rois.items():
        if with_ci:
            subjects = find_subjects(cfg)
            evokeds_per_cond = {}
            for subject in subjects:
                evo_path = get_evoked_path(cfg, subject, window_name, cfg_erp)
                if not evo_path.exists():
                    continue
                subj_evokeds = mne.read_evokeds(evo_path, verbose="WARNING")
                for evo in subj_evokeds:
                    evokeds_per_cond.setdefault(evo.comment, []).append(evo)

            if conditions is None:
                conditions = [k for k in evokeds_per_cond.keys()
                              if "_vs_" not in k]

            evokeds_to_plot = {label: evokeds_per_cond[label]
                               for label in conditions
                               if label in evokeds_per_cond}
            ci = 0.95
            n_subj = len(next(iter(evokeds_to_plot.values())))
            title_suffix = f" (mean ± 95% CI, N={n_subj})"
        else:
            gas = load_grand_averages(cfg, window_name, cfg_erp)
            if conditions is None:
                conditions = [k for k in gas.keys() if "_vs_" not in k]
            evokeds_to_plot = {label: gas[label]
                               for label in conditions if label in gas}
            ci = False
            title_suffix = ""

        first = next(iter(evokeds_to_plot.values()))
        if isinstance(first, list):
            first = first[0]
        valid_channels = [c for c in channels if c in first.ch_names]
        if len(valid_channels) < len(channels):
            missing = set(channels) - set(valid_channels)
            print(f"   WARNING: ROI '{roi_name}' missing channels {missing}")
        if not valid_channels:
            print(f"   WARNING: ROI '{roi_name}' has no valid channels - skipping")
            continue

        fig = mne.viz.plot_compare_evokeds(
            evokeds_to_plot,
            picks=valid_channels,
            combine="mean",
            title=f"{window_name} - ROI: {roi_name} "
                  f"({', '.join(valid_channels)}){title_suffix}",
            show=False,
            ci=ci,
        )
        if isinstance(fig, list):
            fig = fig[0]

        _save_or_show(
            fig, save,
            f"{window_name}_roi_{roi_name}.png",
            cfg, verbose,
        )