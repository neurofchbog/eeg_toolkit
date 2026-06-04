"""
Bad channel detection, manual inspection, and interpolation.

Workflow per subject:
    1. Load preprocessed raw FIF
    2. Run automatic bad-channel detection (LOF) to seed `raw.info['bads']`
    3. Open MNE's interactive plot + PSD plot for manual confirmation
    4. After the user closes the plots, interpolate flagged channels
       (spherical spline) and apply average reference
    5. Save as <subj>_preprocessed_clean_raw.fif

The list of bad channels (auto + manual final) is also saved to a JSON file
per subject for reproducibility and reporting.
"""

from pathlib import Path
import json
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_path,
    get_subject_dir,
    update_status,
)


# ============================================================================
# Output path helpers
# ============================================================================

def _get_clean_raw_path(cfg, subject):
    """Path for the post-interpolation, average-referenced raw."""
    return get_subject_dir(cfg, subject) / f"{subject}_preprocessed_clean_raw.fif"


def _get_bads_log_path(cfg, subject):
    """Path for the JSON log of bad channels (auto + manual)."""
    return get_subject_dir(cfg, subject) / f"{subject}_bad_channels.json"


# ============================================================================
# Automatic detection
# ============================================================================

def detect_bad_channels_auto(raw, picks="eeg", n_neighbors=8, threshold=1.5,
                             verbose=True):
    """
    Detect bad channels using MNE's Local Outlier Factor (LOF) method.

    Parameters
    ----------
    raw : mne.io.Raw
    picks : str | list
        Channel selection (default 'eeg').
    n_neighbors : int
        Number of neighbors used by the LOF algorithm.
    threshold : float
        LOF score threshold above which a channel is flagged. Higher
        threshold = more conservative (fewer channels flagged).
    verbose : bool

    Returns
    -------
    list of str
        Channel names flagged as bad.
    """
    bads = mne.preprocessing.find_bad_channels_lof(
        raw,
        picks=picks,
        n_neighbors=n_neighbors,
        threshold=threshold,
        return_scores=False,
        verbose="WARNING",
    )
    if verbose:
        if bads:
            print(f"   auto-detection (LOF): flagged {len(bads)} channel(s) -> {bads}")
        else:
            print(f"   auto-detection (LOF): no channels flagged")
    return list(bads)


# ============================================================================
# Per-subject interactive workflow
# ============================================================================

def inspect_subject_bads(cfg, subject, threshold=1.5, overwrite=False,
                         verbose=True):
    """
    Interactive bad-channel inspection for a single subject.

    Steps:
        1. Load preprocessed raw
        2. Auto-detect bads via LOF (pre-populates raw.info['bads'])
        3. Open interactive plots (time-domain + PSD)
        4. Wait for the user to confirm/edit and close the windows
        5. Interpolate, re-reference (average), save

    Parameters
    ----------
    cfg : SimpleNamespace
    subject : str
    threshold : float
        LOF threshold for auto-detection seeding.
    overwrite : bool
        If False (default), skip subjects whose clean output already exists.
    verbose : bool

    Returns
    -------
    bool
        True on success, False if skipped.
    """
    raw_in   = get_subject_path(cfg, subject, "preprocessed_raw")
    raw_out  = _get_clean_raw_path(cfg, subject)
    bads_log = _get_bads_log_path(cfg, subject)

    if not raw_in.exists():
        print(f"[{subject}] preprocessed FIF not found ({raw_in.name}) — skipping")
        return False

    if not overwrite and raw_out.exists():
        if verbose:
            print(f"[{subject}] already has clean raw — skipping "
                  f"(use overwrite=True to redo)")
        return False

    if verbose:
        print(f"[{subject}] inspecting bad channels")

    # Load
    raw = mne.io.read_raw_fif(raw_in, preload=True, verbose="WARNING")

    # Auto-detect and seed raw.info['bads']
    auto_bads = detect_bad_channels_auto(raw, threshold=threshold,
                                         verbose=verbose)
    raw.info["bads"] = list(auto_bads)

    # Interactive inspection
    print(f"   [{subject}] opening interactive plots...")
    print(f"   - Click channel names in the time plot to toggle bad status.")
    print(f"   - Inspect the PSD for spectral outliers.")
    print(f"   - Close BOTH windows when done to continue.")

    # Time-domain plot — block=True so the function waits until user closes it
    fig_time = raw.plot(
        scalings=dict(eeg=100e-6),
        duration=20.0,
        n_channels=len(raw.ch_names),
        show_scrollbars=True,
        show_scalebars=True,
        block=False,
        title=f"{subject} — time-domain (click channel names to toggle bad)",
    )

    # PSD plot — non-blocking, just informational
    psd = raw.compute_psd(fmax=60, picks="eeg", verbose="WARNING")
    fig_psd = psd.plot(show=False)
    fig_psd.canvas.manager.set_window_title(f"{subject} — PSD before interpolation")
    fig_psd.show()

    # Block on the time-domain plot (user closes it when done)
    import matplotlib.pyplot as plt
    plt.show(block=True)

    final_bads = list(raw.info["bads"])
    if verbose:
        print(f"   final bad channels (manual): {final_bads}")

    # Save the bads log BEFORE interpolation, in case the user wants to inspect
    log = {
        "subject": subject,
        "auto_detected": auto_bads,
        "final_bads": final_bads,
        "manually_added":   [b for b in final_bads  if b not in auto_bads],
        "manually_removed": [b for b in auto_bads   if b not in final_bads],
        "n_total_channels": len(raw.ch_names),
    }
    with open(bads_log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    if verbose:
        print(f"   saved: {bads_log.name}")

    # Interpolate + re-reference
    if final_bads:
        raw.interpolate_bads(reset_bads=True, verbose="WARNING")
        if verbose:
            print(f"   interpolated {len(final_bads)} channel(s)")
    else:
        if verbose:
            print(f"   no channels to interpolate")

    raw.set_eeg_reference("average", projection=False, verbose="WARNING")
    if verbose:
        print(f"   re-referenced to average")

    # Save
    raw.save(raw_out, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"   saved: {raw_out.name}")

    # Update central status
    update_status(
        cfg, subject,
        bad_channels_marked=True,
        n_bad_channels=len(final_bads),
        bad_channel_names=",".join(final_bads) if final_bads else "",
    )
    return True


# ============================================================================
# Batch (interactive)
# ============================================================================

def inspect_all_subjects(cfg, threshold=1.5, overwrite=False, verbose=True):
    """
    Run interactive bad-channel inspection for every included subject,
    one after the other. The user must close each subject's plots before
    the next one opens.
    """
    subjects = find_subjects(cfg)  # excludes filtered subjects
    summary = {"inspected": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Inspecting bad channels for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = inspect_subject_bads(cfg, subject,
                                      threshold=threshold,
                                      overwrite=overwrite,
                                      verbose=verbose)
            if ok:
                summary["inspected"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    if verbose:
        print("=" * 60)
        print(f"Inspected: {len(summary['inspected'])}")
        print(f"Skipped:   {len(summary['skipped'])}")
        print(f"Failed:    {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    return summary