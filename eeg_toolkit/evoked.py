"""
Evoked (ERP) analysis module.

Paradigm-agnostic: reads conditions and windows from a separate ERP config,
computes per-subject condition averages, optional contrasts (weighted linear
combinations), and group-level grand averages.

Conditions and contrasts can be defined either globally (same across windows)
or per-window (different event codes per window):

    # Global (same across windows)
    conditions:
      spatial: [12, 13, ...]

    # Per-window (different codes per window)
    conditions:
      trial:
        spatial: [1020]
      cue:
        spatial: [12, 13, ...]
"""

from pathlib import Path
import json
import numpy as np
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_dir,
    get_analysis_dir,
    update_status,
)
from eeg_toolkit.artifacts import get_final_epochs_path


# ============================================================================
# Path helpers
# ============================================================================

def _get_analysis_suffix(cfg_erp):
    """Optional suffix from cfg_erp.erp.analysis_name (e.g., 'position')."""
    if cfg_erp is None:
        return ""
    name = getattr(cfg_erp.erp, "analysis_name", None)
    return f"_{name}" if name else ""


def get_evoked_path(cfg, subject, window_name, cfg_erp=None):
    """Per-subject averaged ERPs (one file per window, all conditions inside)."""
    suffix = _get_analysis_suffix(cfg_erp)
    return get_subject_dir(cfg, subject) / f"{subject}_{window_name}{suffix}_ave.fif"


def get_grand_average_path(cfg, window_name, cfg_erp=None):
    """Group-level grand average ERPs."""
    suffix = _get_analysis_suffix(cfg_erp)
    return _get_erp_dir(cfg) / f"grand_average_{window_name}{suffix}_ave.fif"


def get_trial_counts_path(cfg, window_name, cfg_erp=None):
    """JSON log of trial counts per subject per condition."""
    suffix = _get_analysis_suffix(cfg_erp)
    return _get_erp_dir(cfg) / f"trial_counts_{window_name}{suffix}.json"

def _get_erp_dir(cfg):
    """group_results/erp directory (created if missing)."""
    erp_dir = get_analysis_dir(cfg) / "group_results" / "erp"
    erp_dir.mkdir(parents=True, exist_ok=True)
    return erp_dir

# ============================================================================
# Config helpers
# ============================================================================

def _resolve_windows(cfg_erp):
    """Read window definitions from the ERP config."""
    windows = getattr(cfg_erp.erp, "windows", None)
    if windows is None:
        raise ValueError("cfg_erp.erp.windows is not defined.")
    return windows


def _resolve_conditions_for_window(cfg_erp, window_name):
    """
    Return conditions dict {name: [event_codes]} for a specific window.

    Supports both global and per-window structure:
        conditions:
          spatial: [12, 13]                 # global
        # OR
        conditions:
          trial: {spatial: [1020]}          # per-window
          cue:   {spatial: [12, 13]}
    """
    conditions = getattr(cfg_erp, "conditions", None)
    if conditions is None:
        raise ValueError("cfg_erp.conditions is not defined.")

    cond_dict = vars(conditions)

    # Per-window: top-level keys are window names
    if window_name in cond_dict:
        per_window = cond_dict[window_name]
        if hasattr(per_window, "__dict__"):
            return vars(per_window)

    # Global: assume top-level keys are condition names
    # Validate by checking values are lists, not nested namespaces
    first_val = next(iter(cond_dict.values()), None)
    if first_val is not None and not isinstance(first_val, list):
        raise ValueError(
            f"Conditions structure unclear for window '{window_name}'. "
            f"Expected either global (cond_name: [codes]) or per-window "
            f"(window_name: {{cond_name: [codes]}})."
        )
    return cond_dict


def _resolve_contrasts_for_window(cfg_erp, window_name):
    """Return contrasts dict for a specific window, or empty dict if none."""
    contrasts = getattr(cfg_erp, "contrasts", None)
    if contrasts is None:
        return {}

    contrast_dict = vars(contrasts)

    # Per-window
    if window_name in contrast_dict:
        per_window = contrast_dict[window_name]
        if hasattr(per_window, "__dict__"):
            return {n: vars(w) for n, w in vars(per_window).items()}

    # Global: each top-level value should be a namespace of weights
    first_val = next(iter(contrast_dict.values()), None)
    if first_val is None:
        return {}
    if hasattr(first_val, "__dict__"):
        # Check if it's window-keyed (values are dicts of dicts) or global
        first_inner = next(iter(vars(first_val).values()), None)
        if hasattr(first_inner, "__dict__"):
            # Looks per-window but our window_name isn't in it: no contrasts for this window
            return {}
        return {n: vars(w) for n, w in contrast_dict.items()}
    return {}


def _get_window_param(window_cfg, key, default=None):
    """Safely get a per-window parameter."""
    return getattr(window_cfg, key, default)


# ============================================================================
# Per-subject averaging
# ============================================================================

def compute_evokeds_subject(cfg, cfg_erp, subject,
                            overwrite=False, verbose=True):
    """
    Compute condition-averaged ERPs for one subject across all windows
    defined in the ERP config.

    For each window:
      1. Load final epochs.
      2. Apply baseline correction (if specified).
      3. Apply analysis-time lowpass filter (if specified).
      4. Pick only requested channel types.
      5. Optionally equalize trial counts across conditions.
      6. Average per condition.
      7. Compute contrasts as weighted linear combinations of condition averages.
      8. Save all evokeds (conditions + contrasts) to one .fif per window.
    """
    windows = _resolve_windows(cfg_erp)
    erp_cfg = cfg_erp.erp
    picks = getattr(erp_cfg, "picks", "eeg")
    equalize = getattr(erp_cfg, "equalize_counts", False)
    min_trials = getattr(erp_cfg, "min_trials_per_condition", 0)

    n_done = 0
    trial_counts_all = {}

    for window_cfg in windows:
        wname = window_cfg.name
        out_path = get_evoked_path(cfg, subject, wname, cfg_erp)

        if not overwrite and out_path.exists():
            if verbose:
                print(f"   [{subject}] {wname}: evokeds already exist - skipping")
            continue

        in_path = get_final_epochs_path(cfg, subject, wname)
        if not in_path.exists():
            if verbose:
                print(f"   [{subject}] {wname}: final epochs not found - skipping")
            continue

        # Resolve conditions/contrasts for THIS window
        conditions = _resolve_conditions_for_window(cfg_erp, wname)
        contrasts = _resolve_contrasts_for_window(cfg_erp, wname)

        epochs = mne.read_epochs(in_path, preload=True, verbose="WARNING")

        # Baseline
        baseline = _get_window_param(window_cfg, "baseline", None)
        if baseline is not None:
            epochs.apply_baseline(tuple(baseline), verbose="WARNING")

        # Lowpass for analysis
        lowpass = _get_window_param(window_cfg, "lowpass", None)
        if lowpass is not None:
            epochs.filter(l_freq=None, h_freq=float(lowpass),
                          method="iir", verbose="WARNING")

        # Pick channels
        epochs.pick(picks)

        # Build per-condition epoch sets
        cond_epochs = {}
        cond_counts = {}
        for cond_name, codes in conditions.items():
            valid_codes = [str(c) for c in codes if str(c) in epochs.event_id]
            if not valid_codes:
                if verbose:
                    print(f"   [{subject}] {wname}: condition '{cond_name}' "
                          f"has no matching events - skipping")
                continue
            cond_epo = epochs[valid_codes]
            cond_epochs[cond_name] = cond_epo
            cond_counts[cond_name] = len(cond_epo)

        if not cond_epochs:
            if verbose:
                print(f"   [{subject}] {wname}: no conditions matched - skipping window")
            continue

        # Skip window if any condition is below min_trials
        if min_trials > 0:
            below = [c for c, n in cond_counts.items() if n < min_trials]
            if below:
                if verbose:
                    print(f"   [{subject}] {wname}: insufficient trials "
                          f"({below}) - skipping window")
                continue

        # Equalize counts
        if equalize and len(cond_epochs) > 1:
            cond_to_codes = {
                name: [str(c) for c in conditions[name]
                       if str(c) in epochs.event_id]
                for name in cond_epochs.keys()
            }
            all_codes = []
            for codes in cond_to_codes.values():
                all_codes.extend(codes)
            combined = epochs[all_codes].copy()
            groups = list(cond_to_codes.values())
            combined.equalize_event_counts(groups)  # no verbose= kwarg

            cond_epochs = {
                name: combined[codes]
                for name, codes in cond_to_codes.items()
            }
            cond_counts = {name: len(e) for name, e in cond_epochs.items()}

        if verbose:
            counts_str = ", ".join(f"{k}={v}" for k, v in cond_counts.items())
            print(f"   [{subject}] {wname}: {counts_str}"
                  f"{' (equalized)' if equalize else ''}")

        # Average per condition
        evokeds = []
        cond_avgs = {}
        for cond_name, cond_epo in cond_epochs.items():
            evo = cond_epo.average()
            evo.comment = cond_name
            evokeds.append(evo)
            cond_avgs[cond_name] = evo

        # Contrasts
        for contrast_name, weights in contrasts.items():
            missing = [c for c in weights if c not in cond_avgs]
            if missing:
                if verbose:
                    print(f"   [{subject}] {wname}: contrast '{contrast_name}' "
                          f"missing conditions {missing} - skipping")
                continue
            evo_list = [cond_avgs[c] for c in weights]
            weight_list = [float(weights[c]) for c in weights]
            contrast_evo = mne.combine_evoked(evo_list, weights=weight_list)
            contrast_evo.comment = contrast_name
            evokeds.append(contrast_evo)

        if not evokeds:
            if verbose:
                print(f"   [{subject}] {wname}: no evokeds produced - skipping")
            continue

        mne.write_evokeds(out_path, evokeds, overwrite=True, verbose="WARNING")
        if verbose:
            n_conds = len(cond_avgs)
            n_contrasts = len(evokeds) - n_conds
            print(f"   [{subject}] {wname}: saved {len(evokeds)} evokeds "
                  f"({n_conds} conditions + {n_contrasts} contrasts)")

        trial_counts_all[wname] = cond_counts
        n_done += 1

    if n_done > 0:
        update_status(cfg, subject, evokeds_computed=True)

    return n_done > 0, trial_counts_all


# ============================================================================
# Group grand average
# ============================================================================

def compute_grand_averages(cfg, cfg_erp, overwrite=False, verbose=True):
    """Compute grand-average ERPs across all subjects, per condition and contrast."""
    windows = _resolve_windows(cfg_erp)
    subjects = find_subjects(cfg)

    n_done = 0
    for window_cfg in windows:
        wname = window_cfg.name
        out_path = get_grand_average_path(cfg, wname, cfg_erp)

        if not overwrite and out_path.exists():
            if verbose:
                print(f"[{wname}] grand averages exist - skipping")
            continue

        evokeds_by_label = {}
        n_loaded = 0

        for subject in subjects:
            evo_path = get_evoked_path(cfg, subject, wname, cfg_erp)
            if not evo_path.exists():
                continue

            subj_evokeds = mne.read_evokeds(evo_path, verbose="WARNING")
            for evo in subj_evokeds:
                evokeds_by_label.setdefault(evo.comment, []).append(evo)
            n_loaded += 1

        if n_loaded == 0:
            if verbose:
                print(f"[{wname}] no subjects had evokeds - skipping")
            continue

        grand_averages = []
        for label, evo_list in evokeds_by_label.items():
            ga = mne.grand_average(evo_list)
            ga.comment = label
            grand_averages.append(ga)
            if verbose:
                print(f"[{wname}] {label}: averaged across {len(evo_list)} subjects")

        mne.write_evokeds(out_path, grand_averages,
                          overwrite=True, verbose="WARNING")
        if verbose:
            print(f"[{wname}] saved grand averages: {out_path.name}\n")
        n_done += 1

    return n_done > 0


# ============================================================================
# Batch wrapper
# ============================================================================

def compute_evokeds_all(cfg, cfg_erp, overwrite=False, verbose=True):
    """Compute evokeds for every included subject + log trial counts."""
    subjects = find_subjects(cfg)
    summary = {"computed": [], "skipped": [], "failed": []}
    all_counts = {}

    if verbose:
        print(f"Computing evokeds for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok, trial_counts = compute_evokeds_subject(
                cfg, cfg_erp, subject,
                overwrite=overwrite, verbose=verbose,
            )
            if ok:
                summary["computed"].append(subject)
                for wname, counts in trial_counts.items():
                    all_counts.setdefault(wname, {})[subject] = counts
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    for wname, subj_counts in all_counts.items():
        log_path = get_trial_counts_path(cfg, wname, cfg_erp)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(subj_counts, f, indent=2)
        if verbose:
            print(f"Trial counts log saved: {log_path.name}")

    _print_summary(summary)
    return summary


def _print_summary(summary):
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k.capitalize():12s}: {len(v)}")
    if summary.get("failed"):
        print("\nFailures:")
        for subj, err in summary["failed"]:
            print(f"  {subj}: {err}")
    print("=" * 60)