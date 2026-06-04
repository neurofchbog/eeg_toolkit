"""
Preprocessing module for raw EEG data.

Applies a standard pipeline to each subject's raw FIF:
    1. Band-pass filter (Butterworth IIR by default)
    2. Mark candidate channels as EOG if present
    3. Apply electrode montage (standard_1020 by default)
    4. Resample to target sample rate (events are adjusted accordingly)
    5. Save as <subj>_preprocessed_raw.fif and <subj>_preprocessed_events.fif

All parameters come from the YAML config (under `preprocessing`).
"""

from pathlib import Path
import numpy as np
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_path,
    update_status,
)


# ============================================================================
# Per-step helpers
# ============================================================================

def apply_filter(raw, cfg, verbose=True):
    """
    Apply a band-pass filter to a Raw object using the parameters in cfg.

    Reads:
        cfg.preprocessing.filter.low_freq
        cfg.preprocessing.filter.high_freq
        cfg.preprocessing.filter.method  ('iir' or 'fir')
        cfg.preprocessing.filter.iir_order
        cfg.preprocessing.filter.iir_ftype
    """
    f = cfg.preprocessing.filter
    if f.method == "iir":
        iir_params = dict(order=f.iir_order, ftype=f.iir_ftype)
        raw.filter(
            l_freq=f.low_freq,
            h_freq=f.high_freq,
            method="iir",
            iir_params=iir_params,
            verbose="WARNING",
        )
    else:
        raw.filter(
            l_freq=f.low_freq,
            h_freq=f.high_freq,
            method="fir",
            verbose="WARNING",
        )
    if verbose:
        print(f"   filter: {f.low_freq}-{f.high_freq} Hz "
              f"({f.method}, order {f.iir_order})")
    return raw


def assign_eog_channels(raw, cfg, verbose=True):
    """
    Relabel channels listed in cfg.preprocessing.eog_candidates as 'eog'
    if they are present in the data.
    """
    candidates = list(cfg.preprocessing.eog_candidates) \
        if hasattr(cfg.preprocessing, "eog_candidates") else []
    found = [ch for ch in candidates if ch in raw.ch_names]

    if found:
        mapping = {ch: "eog" for ch in found}
        raw.set_channel_types(mapping, verbose="WARNING")
        if verbose:
            print(f"   EOG channels assigned: {found}")
    else:
        if verbose:
            print(f"   EOG channels: none found in this recording "
                  f"(searched for: {candidates})")
    return raw, found


def apply_montage(raw, cfg, verbose=True):
    """
    Apply the electrode montage specified in cfg.preprocessing.montage.

    Non-EEG channels (EOG, ECG, etc.) are kept but only EEG channels get
    coordinates from the montage.
    """
    montage_name = cfg.preprocessing.montage

    # Pick only EEG channels for montage application; keep others intact
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False)
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found — cannot apply montage.")

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn",
                    verbose="WARNING")
    if verbose:
        print(f"   montage applied: {montage_name} "
              f"({len(eeg_picks)} EEG channels)")
    return raw


def resample_with_events(raw, events, cfg, verbose=True):
    """
    Resample the Raw object to the target sample rate and rescale the
    events array to match the new time axis.

    Returns
    -------
    raw_resampled : mne.io.Raw
    events_resampled : np.ndarray of shape (n_events, 3)
    """
    target_sfreq = cfg.preprocessing.resample_sfreq
    sfreq_old = raw.info["sfreq"]

    if target_sfreq == sfreq_old:
        if verbose:
            print(f"   resample: skipped (already at {sfreq_old} Hz)")
        return raw, events

    raw_resampled = raw.copy().resample(target_sfreq, npad="auto",
                                        verbose="WARNING")
    factor = target_sfreq / sfreq_old

    if events is not None and len(events) > 0:
        events_resampled = events.copy()
        events_resampled[:, 0] = np.round(events[:, 0] * factor).astype(int)
    else:
        events_resampled = events

    if verbose:
        print(f"   resample: {sfreq_old} -> {target_sfreq} Hz "
              f"(factor {factor:.4f})")
    return raw_resampled, events_resampled


# ============================================================================
# Per-subject pipeline
# ============================================================================

def preprocess_subject(cfg, subject, overwrite=False, verbose=True):
    """
    Run the full preprocessing pipeline on a single subject.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration.
    subject : str
        Subject ID (e.g., 'subj10').
    overwrite : bool
        If False (default), skip subjects whose output files already exist.
    verbose : bool
        Print per-step progress messages.

    Returns
    -------
    bool
        True on success, False if skipped or failed.
    """
    raw_in   = get_subject_path(cfg, subject, "raw")
    events_in = get_subject_path(cfg, subject, "events")
    raw_out   = get_subject_path(cfg, subject, "preprocessed_raw")
    events_out = get_subject_path(cfg, subject, "preprocessed_events")

    if not raw_in.exists():
        print(f"[{subject}] raw FIF not found ({raw_in.name}) — skipping")
        return False

    if not overwrite and raw_out.exists() and events_out.exists():
        if verbose:
            print(f"[{subject}] already preprocessed — skipping "
                  f"(use overwrite=True to redo)")
        return False

    if verbose:
        print(f"[{subject}] preprocessing")

    # 1. Load
    raw = mne.io.read_raw_fif(raw_in, preload=True, verbose="WARNING")
    events = mne.read_events(events_in) if events_in.exists() else None

    # 2. Filter
    raw = apply_filter(raw, cfg, verbose=verbose)

    # 3. EOG assignment
    raw, eog_channels = assign_eog_channels(raw, cfg, verbose=verbose)

    # 4. Montage
    raw = apply_montage(raw, cfg, verbose=verbose)

    # 5. Resample
    raw, events = resample_with_events(raw, events, cfg, verbose=verbose)

    # 6. Save
    raw.save(raw_out, overwrite=True, verbose="WARNING")
    if events is not None:
        mne.write_events(events_out, events, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"   saved: {raw_out.name}")
        if events is not None:
            print(f"   saved: {events_out.name}")

    # 7. Update status
    update_status(
        cfg, subject,
        preprocessed=True,
        sfreq_preprocessed=raw.info["sfreq"],
        n_eog_channels=len(eog_channels),
    )
    return True


# ============================================================================
# Batch
# ============================================================================

def preprocess_all(cfg, overwrite=False, verbose=True):
    """
    Run preprocessing on every included subject (after exclusions).

    Returns
    -------
    summary : dict
        {'preprocessed': [...], 'skipped': [...], 'failed': [...]}
    """
    subjects = find_subjects(cfg)  # already filters excluded subjects
    summary = {"preprocessed": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Preprocessing {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = preprocess_subject(cfg, subject,
                                    overwrite=overwrite, verbose=verbose)
            if ok:
                summary["preprocessed"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    if verbose:
        print("=" * 60)
        print(f"Preprocessed: {len(summary['preprocessed'])}")
        print(f"Skipped:      {len(summary['skipped'])}")
        print(f"Failed:       {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    return summary