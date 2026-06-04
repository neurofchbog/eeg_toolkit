"""
ICA fitting, automatic labeling (ICLabel), interactive inspection,
and application to all epoching windows.

Workflow per subject (call the four functions in order):

    1. fit_ica_subject(cfg, subject)
       Fits ICA on the window flagged with `use_for_ica=true`.
       Internally re-filters epochs at 1 Hz HP for ICA decomposition,
       but does NOT modify the saved epochs (only the ICA solution
       reflects the higher HP). Saves <subject>_ica.fif.

    2. label_ica_subject(cfg, subject)
       Runs ICLabel on the fitted ICA. Pre-marks components for
       rejection based on classification probabilities and the
       threshold defined in cfg.ica.iclabel.prob_threshold.
       Saves <subject>_ica.fif (overwritten with marks) +
       <subject>_iclabel.json with the full classification.

    3. inspect_ica_subject(cfg, subject)
       Opens MNE's interactive plots (components topomap + sources)
       for manual confirmation/adjustment. Saves the final ICA.

    4. apply_ica_subject(cfg, subject)
       Applies the (cleaned) ICA solution to ALL epoching windows.
       Produces <subject>_<window>_clean-epo.fif for each window.
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
from eeg_toolkit.bad_channels import _get_clean_raw_path
from eeg_toolkit.epoching import get_epochs_path


# ============================================================================
# Path helpers
# ============================================================================

def _get_ica_path(cfg, subject):
    """Path to the fitted ICA solution."""
    return get_subject_dir(cfg, subject) / f"{subject}_ica.fif"


def _get_iclabel_path(cfg, subject):
    """Path to the ICLabel classification JSON."""
    return get_subject_dir(cfg, subject) / f"{subject}_iclabel.json"


def get_clean_epochs_path(cfg, subject, window_name):
    """Path to the post-ICA clean epochs for a given window."""
    return get_subject_dir(cfg, subject) / f"{subject}_{window_name}_clean-epo.fif"


def _get_ica_window_cfg(cfg):
    """Return the epoching window flagged with use_for_ica=true."""
    for w in cfg.epoching_windows:
        if getattr(w, "use_for_ica", False):
            return w
    raise ValueError("No epoching window has `use_for_ica: true` in the config.")


# ============================================================================
# Step 1: Fit ICA
# ============================================================================

def fit_ica_subject(cfg, subject, overwrite=False, verbose=True):
    """
    Fit ICA on the subject's `use_for_ica` window epochs.

    Internally applies a 1 Hz high-pass filter on a copy of the epochs
    before fitting, since ICA decomposition is much more stable with
    higher HP cutoff. The saved epochs themselves are NOT modified.

    Parameters
    ----------
    cfg : SimpleNamespace
    subject : str
    overwrite : bool
    verbose : bool

    Returns
    -------
    bool
    """
    out_path = _get_ica_path(cfg, subject)
    if not overwrite and out_path.exists():
        if verbose:
            print(f"[{subject}] ICA already fitted — skipping "
                  f"(use overwrite=True to redo)")
        return False

    window_cfg = _get_ica_window_cfg(cfg)
    epochs_path = get_epochs_path(cfg, subject, window_cfg.name)
    if not epochs_path.exists():
        print(f"[{subject}] epochs file not found ({epochs_path.name}) — skipping")
        return False

    if verbose:
        print(f"[{subject}] fitting ICA on '{window_cfg.name}' window")

    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")

    # Re-filter for ICA decomposition only (does not modify saved epochs)
    epochs_for_ica = epochs.copy().filter(
        l_freq=1.0, h_freq=None, verbose="WARNING"
    )
    if verbose:
        print(f"   re-filtered to 1 Hz HP for ICA decomposition only")

    ica_cfg = cfg.ica
    fit_params = vars(ica_cfg.fit_params) if hasattr(ica_cfg, "fit_params") else {}

    ica = mne.preprocessing.ICA(
        n_components=ica_cfg.n_components,
        method=ica_cfg.method,
        fit_params=fit_params,
        random_state=ica_cfg.random_state,
        verbose="WARNING",
    )

    if verbose:
        print(f"   running ICA: method={ica_cfg.method}, "
              f"n_components={ica_cfg.n_components}, "
              f"random_state={ica_cfg.random_state}")

    ica.fit(epochs_for_ica)

    ica.save(out_path, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"   saved: {out_path.name}")

    update_status(
        cfg, subject,
        ica_fitted=True,
        n_ica_components=int(ica.n_components_),
    )
    return True


# ============================================================================
# Step 2: ICLabel automatic classification
# ============================================================================

def label_ica_subject(cfg, subject, overwrite=False, verbose=True):
    """
    Run ICLabel on the fitted ICA. Pre-marks components classified as
    artifacts (with probability above threshold) for rejection.

    Saves a JSON log of all classifications regardless of marking.
    """
    from mne_icalabel import label_components

    ica_path = _get_ica_path(cfg, subject)
    if not ica_path.exists():
        print(f"[{subject}] ICA file not found ({ica_path.name}) — skipping")
        return False

    iclabel_log_path = _get_iclabel_path(cfg, subject)
    if not overwrite and iclabel_log_path.exists():
        if verbose:
            print(f"[{subject}] ICLabel already done — skipping "
                  f"(use overwrite=True to redo)")
        return False

    if verbose:
        print(f"[{subject}] running ICLabel")

    # Load the epochs that ICA was fit on, for ICLabel input
    window_cfg = _get_ica_window_cfg(cfg)
    epochs_path = get_epochs_path(cfg, subject, window_cfg.name)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")

    # ICLabel needs the same HP filter as the fitted ICA
    epochs_for_label = epochs.copy().filter(
        l_freq=1.0, h_freq=None, verbose="WARNING"
    )

    ica = mne.preprocessing.read_ica(ica_path, verbose="WARNING")

    # Run ICLabel
    result = label_components(epochs_for_label, ica, method="iclabel")
    labels = result["labels"]                  # list of str, length n_components
    probs  = result["y_pred_proba"]            # array (n_components,)

    # Apply threshold rule from config
    iclabel_cfg = cfg.ica.iclabel
    reject_labels = list(iclabel_cfg.reject_labels)
    threshold = iclabel_cfg.prob_threshold

    auto_excluded = []
    for i, (lbl, p) in enumerate(zip(labels, probs)):
        if lbl in reject_labels and p > threshold:
            auto_excluded.append(i)

    ica.exclude = list(auto_excluded)
    ica.save(ica_path, overwrite=True, verbose="WARNING")

    # Save the full classification log
    log = {
        "subject": subject,
        "n_components": int(ica.n_components_),
        "threshold": float(threshold),
        "reject_labels": reject_labels,
        "auto_excluded_indices": [int(i) for i in auto_excluded],
        "components": [
            {"index": int(i),
             "label": str(lbl),
             "probability": float(p),
             "auto_excluded": int(i) in auto_excluded}
            for i, (lbl, p) in enumerate(zip(labels, probs))
        ],
    }
    with open(iclabel_log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    if verbose:
        print(f"   classified {len(labels)} components")
        print(f"   auto-excluded {len(auto_excluded)} component(s) "
              f"(threshold = {threshold}):")
        for i in auto_excluded:
            print(f"      IC{i:02d}: {labels[i]} (p = {probs[i]:.2f})")
        print(f"   saved: {iclabel_log_path.name}")
        print(f"   updated: {ica_path.name}")

    update_status(
        cfg, subject,
        iclabel_done=True,
        n_iclabel_excluded=len(auto_excluded),
    )
    return True


# ============================================================================
# Step 3: Interactive inspection
# ============================================================================

def inspect_ica_subject(cfg, subject, verbose=True):
    """
    Open interactive ICA plots for manual confirmation/adjustment.

    Two windows open:
      - Component topographies (click to toggle exclusion)
      - Sources / time courses

    The ICLabel pre-markings are loaded as the starting set. After
    closing the windows, the final ica.exclude is saved.
    """
    import matplotlib.pyplot as plt

    ica_path = _get_ica_path(cfg, subject)
    if not ica_path.exists():
        print(f"[{subject}] ICA file not found ({ica_path.name}) — skipping")
        return False

    window_cfg = _get_ica_window_cfg(cfg)
    epochs_path = get_epochs_path(cfg, subject, window_cfg.name)
    epochs = mne.read_epochs(epochs_path, preload=True, verbose="WARNING")
    ica = mne.preprocessing.read_ica(ica_path, verbose="WARNING")

    # Load ICLabel info if available, to display labels in the title
    iclabel_log_path = _get_iclabel_path(cfg, subject)
    label_str = ""
    if iclabel_log_path.exists():
        with open(iclabel_log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        label_str = "\n".join(
            f"IC{c['index']:02d}: {c['label']} (p={c['probability']:.2f})"
            f"{'  [AUTO-EXCL]' if c['auto_excluded'] else ''}"
            for c in log["components"]
        )

    if verbose:
        print(f"[{subject}] interactive ICA inspection")
        print(f"   pre-excluded by ICLabel: {ica.exclude}")
        print(f"   - Click on component topographies to toggle exclusion.")
        print(f"   - Use the sources plot to see time courses.")
        print(f"   - Close BOTH windows when done to save.\n")
        if label_str:
            print("ICLabel classification:")
            print(label_str)
            print()

    # Components topography
    fig_comp = ica.plot_components(
        title=f"{subject} — components (click to toggle)",
        show=False,
    )
    if isinstance(fig_comp, list):
        for fig in fig_comp:
            fig.show()
    else:
        fig_comp.show()

    # Sources / time courses (block=True to wait for user)
    fig_src = ica.plot_sources(
        epochs,
        title=f"{subject} — sources (close when done)",
        show_scrollbars=True,
        block=True,
    )

    plt.show(block=True)

    # Save the final exclusion list
    ica.save(ica_path, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"\n   final excluded components: {ica.exclude}")
        print(f"   saved: {ica_path.name}")

    update_status(
        cfg, subject,
        ica_inspected=True,
        n_ica_excluded_final=len(ica.exclude),
    )
    return True


# ============================================================================
# Step 4: Apply ICA to all windows
# ============================================================================

def apply_ica_subject(cfg, subject, overwrite=False, verbose=True):
    """
    Apply the (cleaned) ICA to every epoching window's data.

    For each window, loads <subject>_<window>-epo.fif, applies
    ica.apply (zeroing out ica.exclude components), and saves
    <subject>_<window>_clean-epo.fif.
    """
    ica_path = _get_ica_path(cfg, subject)
    if not ica_path.exists():
        print(f"[{subject}] ICA file not found ({ica_path.name}) — skipping")
        return False

    ica = mne.preprocessing.read_ica(ica_path, verbose="WARNING")
    if verbose:
        print(f"[{subject}] applying ICA (excluding {len(ica.exclude)} component(s): "
              f"{ica.exclude})")

    n_done = 0
    for window_cfg in cfg.epoching_windows:
        in_path = get_epochs_path(cfg, subject, window_cfg.name)
        out_path = get_clean_epochs_path(cfg, subject, window_cfg.name)

        if not in_path.exists():
            print(f"   [warn] {window_cfg.name}: input not found, skipping")
            continue
        if not overwrite and out_path.exists():
            if verbose:
                print(f"   {window_cfg.name}: clean epochs already exist — skipping")
            continue

        epochs = mne.read_epochs(in_path, preload=True, verbose="WARNING")
        ica.apply(epochs, verbose="WARNING")
        epochs.save(out_path, overwrite=True, verbose="WARNING")
        if verbose:
            print(f"   {window_cfg.name}: saved {out_path.name}")
        n_done += 1

    if n_done > 0:
        update_status(
            cfg, subject,
            ica_applied=True,
        )
    return n_done > 0


# ============================================================================
# Batch wrappers
# ============================================================================

def fit_ica_all(cfg, overwrite=False, verbose=True):
    """Fit ICA for every included subject."""
    return _batch(cfg, fit_ica_subject, "fit ICA", overwrite, verbose)


def label_ica_all(cfg, overwrite=False, verbose=True):
    """Run ICLabel for every included subject."""
    return _batch(cfg, label_ica_subject, "label ICA (ICLabel)", overwrite, verbose)


def apply_ica_all(cfg, overwrite=False, verbose=True):
    """Apply ICA to every included subject's windows."""
    return _batch(cfg, apply_ica_subject, "apply ICA", overwrite, verbose)


def inspect_ica_all(cfg, verbose=True):
    """
    Run interactive inspection for every included subject in series.
    The user must close each subject's plots before the next one opens.
    """
    subjects = find_subjects(cfg)
    summary = {"inspected": [], "skipped": [], "failed": []}
    if verbose:
        print(f"Inspecting ICA for {len(subjects)} subject(s)\n")
    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = inspect_ica_subject(cfg, subject, verbose=verbose)
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


def _batch(cfg, fn, label, overwrite, verbose):
    """Generic batch runner for non-interactive ICA steps."""
    subjects = find_subjects(cfg)
    summary = {"done": [], "skipped": [], "failed": []}
    if verbose:
        print(f"Running {label} for {len(subjects)} subject(s)\n")
    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = fn(cfg, subject, overwrite=overwrite, verbose=verbose)
            if ok:
                summary["done"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()
    _print_summary(summary)
    return summary


def _print_summary(summary):
    print("=" * 60)
    for k in ("done", "inspected", "skipped", "failed"):
        if k in summary:
            print(f"{k.capitalize():10s}: {len(summary[k])}")
    if summary.get("failed"):
        print("\nFailures:")
        for subj, err in summary["failed"]:
            print(f"  {subj}: {err}")
    print("=" * 60)