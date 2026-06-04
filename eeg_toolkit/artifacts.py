"""
Trial rejection: threshold-based auto-detection + interactive confirmation.

Two-tier design:
1. Generic single-window inspection (any experiment, any window).
   Auto-flags bad epochs via peak-to-peak and flatline thresholds, opens
   the MNE Qt5 epochs browser showing ONLY the flagged epochs. User clicks
   to RESCUE (keep) any they disagree with. Unclicked = rejected.

2. Cross-window propagation. Maps rejections from a parent window to
   one or more child windows by matching event sample times.

Workflow (call in order):
   1. inspect_trials_subject(cfg, subject)
   2. propagate_rejections_subject(cfg, subject, parent, children)
   3. apply_rejection_subject(cfg, subject)

YAML config under analysis.artifacts:
    analysis:
      artifacts:
        p2p_threshold: 150e-6
        flatline_threshold: 1e-6
        inspect_window: trial
        propagate_to:
          - cue
"""

from pathlib import Path
import json
import numpy as np
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_dir,
    update_status,
)
from eeg_toolkit.ica import get_clean_epochs_path
from eeg_toolkit.epoching import get_epochs_path


# ============================================================================
# Path helpers
# ============================================================================

def _get_rejected_indices_path(cfg, subject, window_name):
    """Path to the JSON log of rejected trial indices for one window."""
    return (get_subject_dir(cfg, subject)
            / f"{subject}_{window_name}_rejected_indices.json")


def get_final_epochs_path(cfg, subject, window_name):
    """Path to the final, ready-to-analyze epochs (post-rejection)."""
    return (get_subject_dir(cfg, subject)
            / f"{subject}_{window_name}_final-epo.fif")


# ============================================================================
# Config helpers
# ============================================================================

def _get_artifact_cfg(cfg):
    """Read artifact parameters from the config, with sensible defaults."""
    analysis = getattr(cfg, "analysis", None)
    art = getattr(analysis, "artifacts", None) if analysis else None

    return {
        "p2p_threshold": float(getattr(art, "p2p_threshold", 150e-6)),
        "flatline_threshold": float(getattr(art, "flatline_threshold", 1e-6)),
        "inspect_window": getattr(art, "inspect_window", None),
        "propagate_to": list(getattr(art, "propagate_to", [])),
    }


# ============================================================================
# Shared helpers (defined BEFORE batch functions that use them)
# ============================================================================

def _print_summary(summary):
    """Shared summary printer for batch functions."""
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k.capitalize():12s}: {len(v)}")
    if summary.get("failed"):
        print("\nFailures:")
        for subj, err in summary["failed"]:
            print(f"  {subj}: {err}")
    print("=" * 60)


# ============================================================================
# Automatic bad-epoch detection
# ============================================================================

def find_bad_epochs(epochs, p2p_thresh=150e-6, flat_thresh=1e-6):
    """
    Identify epochs that violate peak-to-peak or flatline thresholds.

    Parameters
    ----------
    epochs : mne.Epochs (preloaded)
    p2p_thresh : float
        Maximum allowed peak-to-peak amplitude (Volts).
    flat_thresh : float
        Minimum required peak-to-peak amplitude.

    Returns
    -------
    dict with keys: bad_indices, details, all_max_ptp.
    """
    data = epochs.get_data(picks="eeg", copy=False)
    ptp = np.ptp(data, axis=-1)
    max_ptp = np.max(ptp, axis=1)
    min_ptp = np.min(ptp, axis=1)

    bad_indices = []
    details = []

    for i in range(len(epochs)):
        reasons = []
        if max_ptp[i] > p2p_thresh:
            reasons.append(
                f"p2p={max_ptp[i]*1e6:.1f}uV > {p2p_thresh*1e6:.0f}uV"
            )
        if min_ptp[i] < flat_thresh:
            reasons.append(
                f"flat={min_ptp[i]*1e6:.2f}uV < {flat_thresh*1e6:.1f}uV"
            )
        if reasons:
            bad_indices.append(i)
            details.append({
                "index": i,
                "max_ptp_uv": round(max_ptp[i] * 1e6, 1),
                "min_ptp_uv": round(min_ptp[i] * 1e6, 2),
                "reasons": reasons,
            })

    return {
        "bad_indices": bad_indices,
        "details": details,
        "all_max_ptp": max_ptp,
    }


# ============================================================================
# Tier 1: Generic single-window interactive inspection
# ============================================================================

def inspect_trials_subject(cfg, subject, window_name=None,
                           overwrite=False, verbose=True):
    """
    Auto-flag + interactive confirmation on ONE epoching window.

    1. Auto-detect bad epochs via thresholds.
    2. Show ONLY the flagged epochs in the Qt5 browser.
    3. User clicks epochs to RESCUE (keep). Unclicked = reject.
    """
    import matplotlib.pyplot as plt

    art_cfg = _get_artifact_cfg(cfg)
    p2p_thresh = art_cfg["p2p_threshold"]
    flat_thresh = art_cfg["flatline_threshold"]

    if window_name is None:
        window_name = art_cfg["inspect_window"]
    if window_name is None:
        raise ValueError(
            "No window_name provided and analysis.artifacts.inspect_window "
            "is not set in the config."
        )

    out_json = _get_rejected_indices_path(cfg, subject, window_name)

    if not overwrite and out_json.exists():
        if verbose:
            print(f"[{subject}] {window_name}: rejection JSON already exists "
                  f"- skipping (use overwrite=True to redo)")
        return False

    in_path = get_clean_epochs_path(cfg, subject, window_name)
    if not in_path.exists():
        print(f"[{subject}] clean epochs not found ({in_path.name}) - skipping")
        return False

    epochs = mne.read_epochs(in_path, preload=True, verbose="WARNING")
    n_total = len(epochs)

    if verbose:
        print(f"[{subject}] inspecting '{window_name}' "
              f"({n_total} epochs, {epochs.info['sfreq']} Hz)")

    # Auto-detect
    result = find_bad_epochs(epochs, p2p_thresh, flat_thresh)
    auto_bad = result["bad_indices"]
    details = result["details"]
    all_max_ptp = result["all_max_ptp"]

    if verbose:
        print(f"\n   Auto-detection (p2p > {p2p_thresh*1e6:.0f}uV, "
              f"flat < {flat_thresh*1e6:.1f}uV):")
        if auto_bad:
            print(f"   ! {len(auto_bad)} / {n_total} epochs flagged "
                  f"({100*len(auto_bad)/n_total:.1f}%):")
            for d in details:
                print(f"      epoch {d['index']:3d}: {', '.join(d['reasons'])}")
        else:
            print(f"   OK No epochs exceeded thresholds.")

        pctl = np.percentile(all_max_ptp * 1e6, [50, 90, 95, 99])
        print(f"\n   Max p2p distribution (uV): "
              f"median={pctl[0]:.1f}, P90={pctl[1]:.1f}, "
              f"P95={pctl[2]:.1f}, P99={pctl[3]:.1f}")

    # No bad epochs: skip plot, save empty log
    if not auto_bad:
        if verbose:
            print(f"\n   No epochs to review - saving empty rejection log.")

        log = {
            "subject": subject,
            "window": window_name,
            "thresholds": {"p2p_uv": p2p_thresh * 1e6,
                           "flat_uv": flat_thresh * 1e6},
            "n_total_epochs": n_total,
            "auto_flagged_indices": [],
            "rescued_indices": [],
            "final_rejected_indices": [],
            "n_rejected": 0,
            "pct_rejected": 0.0,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if verbose:
            print(f"   Saved: {out_json.name}")
        update_status(cfg, subject, trials_inspected=True,
                      n_rejected_trials=0)
        return True

    # Show only flagged epochs - click to RESCUE
    if verbose:
        print(f"\n   Showing {len(auto_bad)} auto-flagged epochs.")
        print(f"   Plot position -> original index mapping:")
        for pos, idx in enumerate(auto_bad):
            print(f"      position {pos:3d} -> epoch {idx}")
        print(f"\n   Click an epoch to RESCUE it (= keep it).")
        print(f"   Unclicked epochs will be REJECTED.")
        print(f"   Close the window when done.\n")

    flagged_epochs = epochs[auto_bad]
    flagged_epochs.plot(
        n_epochs=10,
        n_channels=20,
        scalings=dict(eeg=100e-6),
        title=f"{subject} - {len(auto_bad)} flagged epochs "
              f"(click to RESCUE, unclicked = reject)",
        block=True,
    )
    plt.show(block=True)

    # Map positional indices to MNE's internal selection indices
    selection_bad = [epochs.selection[i] for i in auto_bad]
    
    # After the plot closes, flagged_epochs drops the clicked (grey) epochs.
    # Therefore, flagged_epochs.selection contains the UNCLICKED epochs.
    kept_selection = set(flagged_epochs.selection)
    
    # Clicked (grey) = dropped from flagged_epochs = RESCUED
    rescued_selection = set(selection_bad) - kept_selection
    
    # Map back to the positional indices of the full epochs object
    rescued_original = [i for i in auto_bad if epochs.selection[i] in rescued_selection]
    final_bad = [i for i in auto_bad if epochs.selection[i] in kept_selection]

    if verbose:
        print(f"\n   Summary:")
        print(f"      auto-flagged:   {len(auto_bad)}")
        if rescued_original:
            print(f"      rescued (kept): {len(rescued_original)} -> "
                  f"{sorted(rescued_original)}")
        else:
            print(f"      rescued (kept): 0")
        print(f"      final rejected: {len(final_bad)} / {n_total} "
              f"({100*len(final_bad)/n_total:.1f}%)")

    log = {
        "subject": subject,
        "window": window_name,
        "thresholds": {"p2p_uv": p2p_thresh * 1e6,
                       "flat_uv": flat_thresh * 1e6},
        "n_total_epochs": n_total,
        "auto_flagged_indices": auto_bad,
        "rescued_indices": rescued_original,
        "final_rejected_indices": final_bad,
        "n_rejected": len(final_bad),
        "pct_rejected": round(100 * len(final_bad) / n_total, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    if verbose:
        print(f"   Saved: {out_json.name}")

    update_status(cfg, subject, trials_inspected=True,
                  n_rejected_trials=len(final_bad))
    return True


# ============================================================================
# Tier 2: Cross-window propagation
# ============================================================================

def propagate_rejections_subject(cfg, subject,
                                 parent_window=None, child_windows=None,
                                 overwrite=False, verbose=True):
    """
    Map parent-window rejections to child windows by event timing.

    For each rejected parent epoch, any child epoch whose trigger sample
    falls within the parent epoch's time range is also rejected.
    """
    art_cfg = _get_artifact_cfg(cfg)

    if parent_window is None:
        parent_window = art_cfg["inspect_window"]
    if child_windows is None:
        child_windows = art_cfg["propagate_to"]
    if not child_windows:
        if verbose:
            print(f"[{subject}] no child windows to propagate to - skipping")
        return False

    parent_json = _get_rejected_indices_path(cfg, subject, parent_window)
    if not parent_json.exists():
        print(f"[{subject}] parent rejection JSON not found "
              f"({parent_json.name}) - run inspect first")
        return False

    with open(parent_json, "r", encoding="utf-8") as f:
        parent_log = json.load(f)
    parent_rejected = parent_log["final_rejected_indices"]

    if not parent_rejected:
        if verbose:
            print(f"[{subject}] no rejected epochs in '{parent_window}' "
                  f"- propagating empty list to children")

    parent_epo_path = get_clean_epochs_path(cfg, subject, parent_window)
    if not parent_epo_path.exists():
        print(f"[{subject}] parent clean epochs not found - skipping")
        return False

    parent_epochs = mne.read_epochs(parent_epo_path, preload=False,
                                    verbose="WARNING")
    parent_events = parent_epochs.events
    parent_tmin = parent_epochs.tmin
    parent_tmax = parent_epochs.tmax
    sfreq = parent_epochs.info["sfreq"]

    rejected_ranges = []
    for idx in parent_rejected:
        s = parent_events[idx, 0]
        r_start = s + int(parent_tmin * sfreq)
        r_end = s + int(parent_tmax * sfreq)
        rejected_ranges.append((r_start, r_end))

    n_done = 0
    for child_name in child_windows:
        child_json = _get_rejected_indices_path(cfg, subject, child_name)
        if not overwrite and child_json.exists():
            if verbose:
                print(f"   {child_name}: rejection JSON already exists "
                      f"- skipping")
            continue

        child_epo_path = get_clean_epochs_path(cfg, subject, child_name)
        if not child_epo_path.exists():
            print(f"   {child_name}: clean epochs not found - skipping")
            continue

        child_epochs = mne.read_epochs(child_epo_path, preload=False,
                                       verbose="WARNING")
        child_events = child_epochs.events

        child_rejected = []
        for ci, c_ev in enumerate(child_events):
            c_sample = c_ev[0]
            for r_start, r_end in rejected_ranges:
                if r_start <= c_sample <= r_end:
                    child_rejected.append(ci)
                    break

        log = {
            "subject": subject,
            "window": child_name,
            "propagated_from": parent_window,
            "n_parent_rejected": len(parent_rejected),
            "final_rejected_indices": child_rejected,
            "n_rejected": len(child_rejected),
            "n_total_epochs": len(child_events),
            "pct_rejected": round(
                100 * len(child_rejected) / len(child_events), 1
            ) if len(child_events) > 0 else 0.0,
        }
        with open(child_json, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if verbose:
            print(f"   {child_name}: {len(child_rejected)} / "
                  f"{len(child_events)} epochs rejected "
                  f"(propagated from {parent_window})")
            print(f"   saved: {child_json.name}")

        n_done += 1

    return n_done > 0


# ============================================================================
# Apply rejections - produce final epochs
# ============================================================================

def apply_rejection_subject(cfg, subject, overwrite=False, verbose=True):
    """Apply saved rejections and save *_final-epo.fif for every window."""
    n_done = 0

    for window_cfg in cfg.epoching_windows:
        wname = window_cfg.name
        in_path = get_clean_epochs_path(cfg, subject, wname)
        json_path = _get_rejected_indices_path(cfg, subject, wname)
        out_path = get_final_epochs_path(cfg, subject, wname)

        if not in_path.exists() or not json_path.exists():
            continue

        if not overwrite and out_path.exists():
            if verbose:
                print(f"   [{subject}] {wname}: final epochs already exist "
                      f"- skipping")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        bad_indices = log.get("final_rejected_indices", [])

        epochs = mne.read_epochs(in_path, preload=True, verbose="WARNING")
        n_before = len(epochs)

        if bad_indices:
            epochs.drop(bad_indices, reason="ARTIFACT_REJECTION")

        epochs.save(out_path, overwrite=True, verbose="WARNING")

        if verbose:
            print(f"   [{subject}] {wname}: dropped {len(bad_indices)} -> "
                  f"{len(epochs)} / {n_before} epochs kept. "
                  f"Saved {out_path.name}")
        n_done += 1

    if n_done > 0:
        update_status(cfg, subject, rejections_applied=True)

    return n_done > 0


# ============================================================================
# Batch wrappers
# ============================================================================

def inspect_all_trials(cfg, window_name=None, overwrite=False, verbose=True):
    """Run interactive single-window inspection for every included subject."""
    subjects = find_subjects(cfg)
    summary = {"inspected": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Inspecting trials for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = inspect_trials_subject(
                cfg, subject, window_name=window_name,
                overwrite=overwrite, verbose=verbose,
            )
            if ok:
                summary["inspected"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    _print_summary(summary)
    return summary


def propagate_rejections_all(cfg, parent_window=None, child_windows=None,
                             overwrite=False, verbose=True):
    """Propagate rejections from parent to children for every included subject."""
    subjects = find_subjects(cfg)
    summary = {"propagated": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Propagating rejections for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = propagate_rejections_subject(
                cfg, subject,
                parent_window=parent_window,
                child_windows=child_windows,
                overwrite=overwrite, verbose=verbose,
            )
            if ok:
                summary["propagated"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    _print_summary(summary)
    return summary


def apply_rejection_all(cfg, overwrite=False, verbose=True):
    """Apply rejections and save final epochs for every included subject."""
    subjects = find_subjects(cfg)
    summary = {"applied": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Applying rejections for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = apply_rejection_subject(
                cfg, subject,
                overwrite=overwrite, verbose=verbose,
            )
            if ok:
                summary["applied"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    _print_summary(summary)