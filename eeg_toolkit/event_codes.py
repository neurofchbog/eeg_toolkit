"""
Event code manipulation module.

Reads raw events (as produced by raw2fif.py) and applies declarative rules
from the YAML config to produce a derived events array. The derived events
are what downstream epoching uses.

Rule types currently supported:
    - recodify_by_next : look ahead from a source event to the next event
      whose code falls in a target list, and emit a new event with a derived
      code at the location of the source event.

Add more rule types as needed for new paradigms (e.g., recodify_by_previous,
collapse_range, etc.).
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


# ============================================================================
# Path helpers
# ============================================================================

def _get_derived_events_path(cfg, subject):
    """Path to the derived events file (post-recoding)."""
    return get_subject_dir(cfg, subject) / f"{subject}_derived_events_eve.fif"


def _get_event_log_path(cfg, subject):
    """Path to a human-readable log of how events were recoded."""
    return get_subject_dir(cfg, subject) / f"{subject}_event_recoding_log.txt"


# ============================================================================
# Individual rule appliers
# ============================================================================

def _apply_recodify_by_next(events, rule, verbose=True):
    """
    For each event whose code matches rule.source_code, look ahead to find
    the next event whose code falls in one of the `look_ahead_codes.next_in`
    lists, and emit a new event with the corresponding `output_code` at the
    same sample as the source event.

    Returns the new events to add (does NOT modify the input array).
    """
    source = rule.source_code
    look_ahead = rule.look_ahead_codes  # list of (next_in, output_code)

    new_events = []
    counts = {la.output_code: 0 for la in look_ahead}
    n_unmatched = 0

    for i, ev in enumerate(events):
        if ev[2] != source:
            continue

        # Look ahead for a matching event
        matched = False
        for j in range(i + 1, len(events)):
            next_code = events[j, 2]
            for la in look_ahead:
                if next_code in la.next_in:
                    new_events.append([ev[0], 0, la.output_code])
                    counts[la.output_code] += 1
                    matched = True
                    break
            if matched:
                break
            # Stop if we encounter another source event before a match
            if events[j, 2] == source:
                break

        if not matched:
            n_unmatched += 1

    if verbose:
        print(f"   rule '{rule.name}':")
        for code, n in counts.items():
            print(f"      {source} -> {code}: {n} event(s)")
        if n_unmatched > 0:
            print(f"      {source} unmatched: {n_unmatched} event(s)")

    return np.array(new_events, dtype=int) if new_events else np.empty((0, 3), dtype=int)


# Dispatcher: maps rule type name -> implementation function
RULE_APPLIERS = {
    "recodify_by_next": _apply_recodify_by_next,
}


# ============================================================================
# Per-subject pipeline
# ============================================================================

def derive_events_subject(cfg, subject, overwrite=False, verbose=True):
    """
    Apply all event rules from the config to one subject's raw events.

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration (must have `event_rules`).
    subject : str
    overwrite : bool
        Re-run if derived events already exist.
    verbose : bool

    Returns
    -------
    bool
        True on success, False if skipped or failed.
    """
    # Use the resampled events (after preprocessing) so timestamps match
    # the preprocessed raw on which we'll epoch.
    events_in = get_subject_path(cfg, subject, "preprocessed_events")
    if not events_in.exists():
        print(f"[{subject}] preprocessed events not found ({events_in.name}) — skipping")
        return False

    derived_out = _get_derived_events_path(cfg, subject)
    log_out     = _get_event_log_path(cfg, subject)

    if not overwrite and derived_out.exists():
        if verbose:
            print(f"[{subject}] derived events already exist — skipping "
                  f"(use overwrite=True to redo)")
        return False

    if verbose:
        print(f"[{subject}] deriving events")

    # Load raw events
    events = mne.read_events(events_in)
    if verbose:
        unique, counts = np.unique(events[:, 2], return_counts=True)
        print(f"   raw events: {len(events)} total, {len(unique)} unique codes")

    # Apply each rule
    rules = getattr(cfg, "event_rules", None)
    if rules is None or len(rules) == 0:
        if verbose:
            print(f"   no event_rules defined in config — derived = raw")
        derived = events.copy()
    else:
        all_new_events = [events]  # start with raw events (rule.replace=False is default)
        for rule in rules:
            applier = RULE_APPLIERS.get(rule.type)
            if applier is None:
                raise ValueError(
                    f"Unknown rule type '{rule.type}'. "
                    f"Supported: {list(RULE_APPLIERS.keys())}"
                )
            new_events = applier(events, rule, verbose=verbose)

            replace = getattr(rule, "replace", False)
            if replace and len(new_events) > 0:
                # Replace source events with new events
                source = rule.source_code
                kept = all_new_events[0][all_new_events[0][:, 2] != source]
                all_new_events[0] = kept

            if len(new_events) > 0:
                all_new_events.append(new_events)

        derived = np.vstack(all_new_events)
        # Sort by sample index so the events array is monotonic
        derived = derived[np.argsort(derived[:, 0], kind="stable")]

    # Save derived events
    mne.write_events(derived_out, derived, overwrite=True, verbose="WARNING")

    # Save a small log
    with open(log_out, "w", encoding="utf-8") as f:
        f.write(f"Event recoding log for {subject}\n")
        f.write(f"Source: {events_in.name}\n")
        f.write(f"Rules applied: {len(rules) if rules else 0}\n\n")
        f.write(f"Raw events: {len(events)} total\n")
        unique, counts = np.unique(events[:, 2], return_counts=True)
        for c, n in zip(unique, counts):
            f.write(f"  code {c}: {n}\n")
        f.write(f"\nDerived events: {len(derived)} total\n")
        unique_d, counts_d = np.unique(derived[:, 2], return_counts=True)
        for c, n in zip(unique_d, counts_d):
            new_marker = " [NEW]" if c not in unique else ""
            f.write(f"  code {c}: {n}{new_marker}\n")

    if verbose:
        n_added = len(derived) - len(events)
        print(f"   derived events: {len(derived)} total ({n_added:+d} vs raw)")
        print(f"   saved: {derived_out.name}")
        print(f"   saved: {log_out.name}")

    update_status(
        cfg, subject,
        events_derived=True,
        n_derived_events=len(derived),
    )
    return True


# ============================================================================
# Batch
# ============================================================================

def derive_events_all(cfg, overwrite=False, verbose=True):
    """
    Apply event rules to every included subject.
    """
    subjects = find_subjects(cfg)
    summary = {"derived": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Deriving events for {len(subjects)} subject(s)\n")

    for i, subject in enumerate(subjects, start=1):
        if verbose:
            print(f"--- [{i}/{len(subjects)}] {subject} ---")
        try:
            ok = derive_events_subject(cfg, subject,
                                       overwrite=overwrite, verbose=verbose)
            if ok:
                summary["derived"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    if verbose:
        print("=" * 60)
        print(f"Derived: {len(summary['derived'])}")
        print(f"Skipped: {len(summary['skipped'])}")
        print(f"Failed:  {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    return summary