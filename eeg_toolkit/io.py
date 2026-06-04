"""
I/O utilities for the EEG toolkit.

Handles subject discovery, path construction, and status tracking
(checkpointing) so subjects already processed can be skipped.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime



# ============================================================================
# Directory and path helpers
# ============================================================================

def get_raw_dir(cfg):
    """
    Return the directory where the raw recordings live (read-only).

    This is the source folder containing XDF, EDF, ASC, etc. — files
    that should never be modified by the toolkit.
    """
    return Path(cfg.paths.raw_dir)


def get_analysis_dir(cfg):
    """
    Return the root analysis directory where subject folders live.

    The path is built as:
        <base_dir> / <analysis_folder>

    Note: `paradigm` is treated as a label (used in reports/logs)
    and is NOT included in the path.
    """
    return Path(cfg.paths.base_dir) / cfg.paths.analysis_folder


def find_raw_files(cfg, extension="xdf"):
    """
    List all raw files in `raw_dir` matching a given extension.

    Parameters
    ----------
    cfg : SimpleNamespace
    extension : str
        File extension to look for, without the dot (e.g., 'xdf', 'edf').

    Returns
    -------
    list of pathlib.Path
        Sorted list of matching files.
    """
    raw_dir = get_raw_dir(cfg)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Check `paths.raw_dir` in your config."
        )

    files = sorted(raw_dir.glob(f"*.{extension}"))
    return files


def find_subjects(cfg, sort_numerically=True, apply_exclusions=True):
    """
    Find all subject folders under the analysis directory.

    Parameters
    ----------
    cfg : SimpleNamespace
    sort_numerically : bool
        If True, sort by numeric suffix (subj1, subj2, ..., subj10)
        instead of alphabetically.
    apply_exclusions : bool
        If True (default), filter out subjects listed in `cfg.excluded_subjects`.
        Set to False to get the full list of all subject folders on disk.

    Returns
    -------
    list of str
    """
    analysis_dir = get_analysis_dir(cfg)
    prefix = cfg.paths.subject_prefix

    if not analysis_dir.exists():
        raise FileNotFoundError(
            f"Analysis directory not found: {analysis_dir}\n"
            f"Check `paths.base_dir` and `paths.analysis_folder` in your config."
        )

    subjects = [
        p.name for p in analysis_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith(prefix.lower())
    ]

    # Apply config-based exclusions
    if apply_exclusions:
        excluded = get_excluded_subjects(cfg)
        if excluded:
            n_before = len(subjects)
            actually_excluded = sorted(set(subjects) & excluded)
            subjects = [s for s in subjects if s not in excluded]
            if actually_excluded:
                print(f"[find_subjects] excluded {len(actually_excluded)} subject(s): "
                      f"{actually_excluded}")

    if sort_numerically:
        def _numeric_key(name):
            digits = "".join(c for c in name if c.isdigit())
            return int(digits) if digits else 0
        subjects.sort(key=_numeric_key)
    else:
        subjects.sort()

    return subjects


def get_excluded_subjects(cfg):
    """
    Return the set of subjects to exclude based on the config.

    Reads `cfg.excluded_subjects` (a list in the YAML). Returns an empty
    set if the field is not defined.

    Returns
    -------
    set of str
    """
    excluded = getattr(cfg, "excluded_subjects", None)
    if excluded is None:
        return set()
    return set(excluded)

def get_subject_dir(cfg, subject, create=False):
    """
    Return the directory of a given subject under the analysis folder.

    Parameters
    ----------
    create : bool
        If True, create the directory if it doesn't exist.
    """
    subj_dir = get_analysis_dir(cfg) / subject
    if create:
        subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir


def get_subject_path(cfg, subject, kind):
    """
    Build a standardized file path for a subject's output file.

    Parameters
    ----------
    kind : str
        One of: 'raw', 'events', 'preprocessed_raw', 'preprocessed_events',
        'epochs', 'ica', 'clean_epochs', 'report'.
    """
    subj_dir = get_subject_dir(cfg, subject)

    filename_map = {
        "raw":                  f"{subject}_raw.fif",
        "events":               f"{subject}_events_eve.fif",
        "preprocessed_raw":     f"{subject}_preprocessed_raw.fif",
        "preprocessed_events":  f"{subject}_preprocessed_events_eve.fif",
        "epochs":               f"{subject}_epo.fif",
        "ica":                  f"{subject}_ica.fif",
        "clean_epochs":         f"{subject}_clean_epo.fif",
        "report":               f"{subject}_report.html",
    }

    if kind not in filename_map:
        raise ValueError(
            f"Unknown file kind '{kind}'. "
            f"Valid options: {list(filename_map.keys())}"
        )

    return subj_dir / filename_map[kind]

# ============================================================================
# Status tracking (checkpointing)
# ============================================================================

STATUS_COLUMNS = [
    "subject",
    "preprocessed",
    "bad_channels_marked",
    "epochs_created",
    "ica_fitted",
    "ica_applied",
    "n_bad_channels",
    "n_epochs_total",
    "n_epochs_rejected",
    "n_ica_excluded",
    "last_updated",
]


def get_status_path(cfg):
    """Path to the central subjects-status CSV."""
    return get_analysis_dir(cfg) / "subjects_status.csv"


def load_status(cfg):
    """Load the subjects-status CSV. Returns an empty DataFrame if missing."""
    status_path = get_status_path(cfg)
    if status_path.exists():
        return pd.read_csv(status_path)
    return pd.DataFrame(columns=STATUS_COLUMNS)


def update_status(cfg, subject, **fields):
    """
    Update the status row for a given subject.

    Examples
    --------
    >>> update_status(cfg, 'subj1', preprocessed=True, n_bad_channels=2)
    """
    df = load_status(cfg)

    fields["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure all needed columns exist with object dtype to avoid
    # pandas refusing to mix bools/ints/floats into a fixed-dtype column.
    for col in fields.keys():
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype=object)
        elif df[col].dtype != object:
            df[col] = df[col].astype(object)

    if subject in df["subject"].values:
        for col, val in fields.items():
            df.loc[df["subject"] == subject, col] = val
    else:
        new_row = {"subject": subject, **fields}
        for col in STATUS_COLUMNS:
            new_row.setdefault(col, None)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(get_status_path(cfg), index=False)
    return df


def is_step_done(cfg, subject, step):
    """
    Check whether a given pipeline step has been completed for a subject.

    Parameters
    ----------
    step : str
        Boolean column. One of: 'preprocessed', 'bad_channels_marked',
        'epochs_created', 'ica_fitted', 'ica_applied'.
    """
    df = load_status(cfg)
    if subject not in df["subject"].values:
        return False
    val = df.loc[df["subject"] == subject, step].values[0]
    return bool(val) and str(val).lower() not in ("nan", "none", "false", "")