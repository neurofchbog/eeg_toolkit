"""
XDF -> FIF conversion module.

Converts raw Lab Recorder XDF files into MNE Raw FIF format,
preserving the original numeric event codes as-is.

This module is paradigm-agnostic: it does not assume any meaning
for the marker values. Event-code documentation lives in the project
README and is consumed downstream (e.g., during epoching).
"""

from pathlib import Path
import numpy as np
import pyxdf
import mne

from eeg_toolkit.io import (
    find_raw_files,
    get_subject_dir,
    get_subject_path,
    update_status,
)


# ============================================================================
# Stream identification
# ============================================================================

def _find_stream(streams, stype, require_data=True):
    """
    Find the first stream of a given type in an XDF file.

    Parameters
    ----------
    streams : list
        Streams returned by pyxdf.load_xdf.
    stype : str
        Stream type to look for, e.g. 'EEG' or 'Markers'.
    require_data : bool
        If True, skip streams of the right type but with no samples
        (some recordings include empty placeholder marker streams).

    Returns
    -------
    dict or None
        The matching stream, or None if not found.
    """
    for s in streams:
        s_type = s["info"]["type"][0] if s["info"]["type"] else ""
        if s_type != stype:
            continue
        n_samples = len(s["time_series"]) if s["time_series"] is not None else 0
        if require_data and n_samples == 0:
            continue
        return s
    return None


def _get_channel_names(eeg_stream):
    """
    Extract channel names from an EEG stream's metadata.

    Falls back to generic 'Ch1', 'Ch2', ... if metadata is missing.
    """
    try:
        channels = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ch_names = [ch["label"][0] for ch in channels]
        if ch_names:
            return ch_names
    except (TypeError, KeyError, IndexError):
        pass
    n_ch = int(eeg_stream["info"]["channel_count"][0])
    return [f"Ch{i + 1}" for i in range(n_ch)]


# ============================================================================
# Building MNE objects
# ============================================================================

def _eeg_stream_to_raw(eeg_stream, data_unit_to_volts=1e-6):
    """
    Convert an XDF EEG stream into an MNE Raw object.

    Parameters
    ----------
    eeg_stream : dict
        EEG stream returned by pyxdf.
    data_unit_to_volts : float
        Multiplicative factor to convert raw XDF samples to Volts
        (MNE's expected unit). Use 1e-6 if the amplifier streams in
        microvolts (typical for ActiCHamp + Lab Recorder), or 1.0 if
        the data is already in Volts.

    Returns
    -------
    raw : mne.io.RawArray
    """
    # pyxdf returns (n_samples, n_channels); MNE expects (n_channels, n_samples)
    data = np.asarray(eeg_stream["time_series"]).T * data_unit_to_volts
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    ch_names = _get_channel_names(eeg_stream)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="WARNING")
    return raw


def _markers_to_events(marker_stream, eeg_stream):
    """
    Convert an XDF marker stream into an MNE events array.

    The MNE events array has shape (n_events, 3):
        [sample_index, 0, event_code]

    Parameters
    ----------
    marker_stream : dict
        Marker stream returned by pyxdf.
    eeg_stream : dict
        EEG stream (used to align marker times to EEG samples).

    Returns
    -------
    events : np.ndarray of shape (n_valid_events, 3)
    event_mapping : dict[int, str]
        Mapping from numeric event code -> original marker string.
        Useful for documentation; saved alongside the events file.
    """
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    n_samples_eeg = np.asarray(eeg_stream["time_series"]).shape[0]
    eeg_start_time = eeg_stream["time_stamps"][0]

    marker_times = np.asarray(marker_stream["time_stamps"])
    raw_values = marker_stream["time_series"]

    # Unwrap markers (pyxdf returns them as length-1 lists/arrays)
    flat_values = []
    for v in raw_values:
        if isinstance(v, (list, np.ndarray)) and len(v) >= 1:
            flat_values.append(v[0])
        else:
            flat_values.append(v)

    # Convert each marker to an integer code, preserving the original value
    codes = []
    mapping = {}
    for val in flat_values:
        s_val = str(val)
        try:
            code = int(float(s_val))
        except (ValueError, TypeError):
            # Non-numeric marker: hash to a positive integer code
            code = (hash(s_val) % 100000) + 100000  # offset to avoid collisions
        codes.append(code)
        mapping[code] = s_val

    codes = np.asarray(codes, dtype=int)

    # Align marker times to EEG samples
    sample_indices = ((marker_times - eeg_start_time) * sfreq).round().astype(int)

    # Keep only markers that fall within the EEG recording
    valid = (sample_indices >= 0) & (sample_indices < n_samples_eeg)
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        print(f"   [warn] {n_dropped} markers fell outside the EEG window and were dropped")

    sample_indices = sample_indices[valid]
    codes = codes[valid]

    events = np.column_stack([
        sample_indices,
        np.zeros(len(sample_indices), dtype=int),
        codes,
    ])
    return events, mapping


# ============================================================================
# Per-subject conversion
# ============================================================================

def _subject_id_from_xdf(xdf_path, prefix):
    """
    Derive a subject ID from an XDF filename.

    Examples
    --------
    >>> _subject_id_from_xdf(Path("subj10.xdf"), "subj")
    'subj10'
    """
    stem = xdf_path.stem  # 'subj10' from 'subj10.xdf'
    return stem


def convert_xdf_to_fif(cfg, xdf_path, subject=None, overwrite=False, verbose=True):
    """
    Convert a single XDF file into MNE FIF files for one subject.

    Outputs (in the subject's analysis folder):
        - <subject>_raw.fif
        - <subject>_events_eve.fif
        - <subject>_event_mapping.txt

    Parameters
    ----------
    cfg : SimpleNamespace
        Loaded configuration.
    xdf_path : str or Path
        Path to the source XDF file.
    subject : str or None
        Subject ID. If None, derived from the XDF filename.
    overwrite : bool
        If False (default), skip subjects whose output files already exist.
    verbose : bool
        Print per-step progress messages.

    Returns
    -------
    bool
        True if conversion succeeded, False if skipped or failed.
    """
    xdf_path = Path(xdf_path)
    if subject is None:
        subject = _subject_id_from_xdf(xdf_path, cfg.paths.subject_prefix)

    raw_out = get_subject_path(cfg, subject, "raw")
    events_out = get_subject_path(cfg, subject, "events")
    mapping_out = raw_out.parent / f"{subject}_event_mapping.txt"

    if not overwrite and raw_out.exists() and events_out.exists():
        if verbose:
            print(f"[{subject}] already converted — skipping (use overwrite=True to redo)")
        return False

    if verbose:
        print(f"[{subject}] reading {xdf_path.name}")

    streams, _ = pyxdf.load_xdf(str(xdf_path))

    eeg_stream = _find_stream(streams, "EEG", require_data=True)
    if eeg_stream is None:
        print(f"[{subject}] no EEG stream found — skipping")
        return False

    marker_stream = _find_stream(streams, "Markers", require_data=True)

    # Build Raw object
    raw = _eeg_stream_to_raw(eeg_stream, data_unit_to_volts=1e-6)
    if verbose:
        print(f"[{subject}]   EEG: {len(raw.ch_names)} ch, "
              f"{raw.info['sfreq']} Hz, {raw.times[-1]:.1f} s")

    # Build events
    events = None
    mapping = {}
    if marker_stream is not None:
        events, mapping = _markers_to_events(marker_stream, eeg_stream)
        if verbose:
            unique_codes = np.unique(events[:, 2])
            print(f"[{subject}]   markers: {len(events)} events, "
                  f"{len(unique_codes)} unique codes")

    # Make sure the subject folder exists
    get_subject_dir(cfg, subject, create=True)

    # Save Raw
    raw.save(raw_out, overwrite=True, verbose="WARNING")
    if verbose:
        print(f"[{subject}]   saved: {raw_out.name}")

    # Save events + mapping
    if events is not None and len(events) > 0:
        mne.write_events(events_out, events, overwrite=True, verbose="WARNING")
        with open(mapping_out, "w", encoding="utf-8") as f:
            f.write("# Event-code mapping (numeric code -> original marker string)\n")
            f.write(f"# Source: {xdf_path.name}\n")
            for code in sorted(mapping.keys()):
                f.write(f"{code}\t{mapping[code]}\n")
        if verbose:
            print(f"[{subject}]   saved: {events_out.name}, {mapping_out.name}")

    # Update central status CSV
    update_status(
        cfg, subject,
        xdf_converted=True,
        n_eeg_channels=len(raw.ch_names),
        n_events=len(events) if events is not None else 0,
        sfreq_raw=raw.info["sfreq"],
    )

    return True


# ============================================================================
# Batch conversion
# ============================================================================

def convert_all_xdfs(cfg, overwrite=False, verbose=True):
    """
    Convert every XDF file in the raw directory to FIF format.

    Parameters
    ----------
    cfg : SimpleNamespace
    overwrite : bool
        Re-convert subjects that already have output files.
    verbose : bool
        Print per-subject progress.

    Returns
    -------
    summary : dict
        {'converted': [...], 'skipped': [...], 'failed': [...]}
    """
    xdf_files = find_raw_files(cfg, extension="xdf")
    summary = {"converted": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Found {len(xdf_files)} XDF files in {cfg.paths.raw_dir}\n")

    for i, xdf_path in enumerate(xdf_files, start=1):
        subject = _subject_id_from_xdf(xdf_path, cfg.paths.subject_prefix)
        if verbose:
            print(f"--- [{i}/{len(xdf_files)}] {subject} ---")
        try:
            ok = convert_xdf_to_fif(cfg, xdf_path, subject=subject,
                                    overwrite=overwrite, verbose=verbose)
            if ok:
                summary["converted"].append(subject)
            else:
                summary["skipped"].append(subject)
        except Exception as e:
            print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
            summary["failed"].append((subject, str(e)))
        if verbose:
            print()

    if verbose:
        print("=" * 60)
        print(f"Converted: {len(summary['converted'])}")
        print(f"Skipped:   {len(summary['skipped'])}")
        print(f"Failed:    {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    return summary