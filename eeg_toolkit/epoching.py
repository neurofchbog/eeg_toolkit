"""
Epoching module.

For each subject, iterates over the epoching windows defined in the YAML
config and produces one .fif file per window. Reads from:
    - <subject>_preprocessed_clean_raw.fif  (post bad-channel interpolation)
    - <subject>_derived_events_eve.fif      (post event-code recoding)

For each window, builds an MNE Epochs object using the parameters from the
config (events, tmin, tmax, baseline, rejection, decim) and saves it as
<subject>_epo_<window_name>.fif.
"""

from pathlib import Path
import numpy as np
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_path,
    get_subject_dir,
    update_status,
)
from eeg_toolkit.event_codes import _get_derived_events_path
from eeg_toolkit.bad_channels import _get_clean_raw_path


# ============================================================================
# Path helpers
# ============================================================================

def get_epochs_path(cfg, subject, window_name):
    """
    Path to the epochs file for a given subject and window name.

    Convention: <subject>_epo_<window_name>.fif
    """
    return get_subject_dir(cfg, subject) / f"{subject}_{window_name}-epo.fif"

# ============================================================================
# Per-window epoching
# ============================================================================

def create_window(cfg, subject, window_cfg, raw=None, events=None,
                  overwrite=False, verbose=True):
    """
    Build and save one epochs file for a given subject and window definition.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration.
    subject : str
    window_cfg : SimpleNamespace
        One entry from cfg.epoching_windows (with name, events, tmin, tmax,
        baseline, reject_peak_to_peak, decim, use_for_ica).
    raw : mne.io.Raw or None
        If provided, use this Raw object instead of loading from disk
        (useful when calling create_all_windows so we don't reload N times).
    events : np.ndarray or None
        If provided, use these events instead of loading from disk.
    overwrite : bool
        Re-create even if the output file exists.
    verbose : bool

    Returns
    -------
    bool
        True on success, False if skipped.
    """
    name = window_cfg.name
    out_path = get_epochs_path(cfg, subject, name)

    if not overwrite and out_path.exists():
        if verbose:
            print(f"   window '{name}': already exists — skipping")
        return False

    # Load raw if not provided
    if raw is None:
        raw_in = _get_clean_raw_path(cfg, subject)
        if not raw_in.exists():
            print(f"[{subject}] clean raw not found ({raw_in.name}) — skipping")
            return False
        raw = mne.io.read_raw_fif(raw_in, preload=True, verbose="WARNING")

    # Load events if not provided
    if events is None:
        events_in = _get_derived_events_path(cfg, subject)
        if not events_in.exists():
            print(f"[{subject}] derived events not found ({events_in.name}) — skipping")
            return False
        events = mne.read_events(events_in)

    # Build event_id dict expected by mne.Epochs (name -> code)
    # We just use the integer code as the name (as a string) for transparency.
    target_codes = list(window_cfg.events)
    event_id = {str(c): int(c) for c in target_codes}

    # Filter events to only those that match this window's target codes
    n_total = len(events)
    matching = np.isin(events[:, 2], target_codes)
    n_match = matching.sum()
    if verbose:
        print(f"   window '{name}': {n_match} / {n_total} events match "
              f"target codes {target_codes if len(target_codes) <= 6 else f'(n={len(target_codes)})'}")

    if n_match == 0:
        print(f"   [warn] no events match window '{name}' — skipping")
        return False

    # Build Epochs object
    baseline = window_cfg.baseline  # None or [tmin, tmax]
    if baseline is not None:
        baseline = tuple(baseline)

    reject = None
    if window_cfg.reject_peak_to_peak is not None:
        reject = {k: float(v) for k, v in vars(window_cfg.reject_peak_to_peak).items()}

    decim = int(getattr(window_cfg, "decim", 1))

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=window_cfg.tmin,
        tmax=window_cfg.tmax,
        baseline=baseline,
        reject=reject,
        decim=decim,
        preload=True,
        on_missing="warn",
        verbose="WARNING",
    )

    if verbose:
        print(f"   created: {len(epochs)} epoch(s), "
              f"{len(epochs.times)} samples per epoch, "
              f"sfreq={epochs.info['sfreq']} Hz")
        print(f"   per-condition counts:")
        for code in target_codes:
            label = str(code)
            try:
                n = len(epochs[label])
                print(f"      code {code}: {n}")
            except KeyError:
                pass  # not present in this subject

    epochs.save(out_path, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"   saved: {out_path.name}")

    return True


# ============================================================================
# Per-subject (all windows)
# ============================================================================

def create_all_windows(cfg, subject, overwrite=False, verbose=True):
    """
    Generate epochs for all windows defined in cfg.epoching_windows.
    Loads raw and events once and reuses them across windows.
    """
    raw_in = _get_clean_raw_path(cfg, subject)
    events_in = _get_derived_events_path(cfg, subject)

    if not raw_in.exists():
        print(f"[{subject}] clean raw not found ({raw_in.name}) — skipping")
        return False
    if not events_in.exists():
        print(f"[{subject}] derived events not found ({events_in.name}) — skipping")
        return False

    if verbose:
        print(f"[{subject}] epoching all windows")

    raw    = mne.io.read_raw_fif(raw_in, preload=True, verbose="WARNING")
    events = mne.read_events(events_in)

    n_done = 0
    for window_cfg in cfg.epoching_windows:
        ok = create_window(cfg, subject, window_cfg,
                           raw=raw, events=events,
                           overwrite=overwrite, verbose=verbose)
        if ok:
            n_done += 1

    if n_done > 0:
        update_status(
            cfg, subject,
            epochs_created=True,
            n_epoching_windows=n_done,
        )

    return n_done > 0


# ============================================================================
# Batch
# ============================================================================

def epoch_all_subjects(cfg, overwrite=False, verbose=True):
    """
    Run epoching on every included subject.
    """
    subjects = find_subjects(cfg)
    summary = {"epoched": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Epoching {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = create_all_windows(cfg, subject,
                                    overwrite=overwrite, verbose=verbose)
            if ok:
                summary["epoched"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    if verbose:
        print("=" * 60)
        print(f"Epoched: {len(summary['epoched'])}")
        print(f"Skipped: {len(summary['skipped'])}")
        print(f"Failed:  {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    return summary