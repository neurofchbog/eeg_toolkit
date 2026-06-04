"""
Representational Similarity Analysis (RSA) — time-resolved.

Paradigm-agnostic time-resolved RSA using crossnobis distance via
rsatoolbox. Conditions are passed directly in code (no YAML config)
so that users can define any set of conditions from the notebook.

Workflow
--------
1. compute_rdms_subject / compute_rdms_all
   Load final epochs → optional resample → select conditions → balance
   trials → build rsatoolbox TemporalDataset → crossnobis RDM movie
   → save per-subject .npz

2. load_subject_rdms / load_all_rdms
   Reload saved RDMs for model comparison and stats.

3. compute_model_fits
   Time-resolved Spearman correlation between neural RDMs and
   theoretical model RDMs.

Design notes
------------
- Single-config pattern: only ``cfg`` (pipeline) is needed. Conditions
  and model RDMs are passed as function arguments in the notebook.
- Saves one .npz per subject per analysis, keyed by ``analysis_name``.
- Follows toolkit conventions: overwrite guard, verbose prints, batch
  summary dict, path helpers.
- Requires ``rsatoolbox`` (pip-installable).
"""

from pathlib import Path
import numpy as np
import mne

from eeg_toolkit.io import (
    find_subjects,
    get_subject_dir,
    get_analysis_dir,
)
from eeg_toolkit.artifacts import get_final_epochs_path


# ===================================================================
# Path helpers
# ===================================================================

def _get_rsa_dir(cfg):
    """group_results/rsa/ — created on first use."""
    d = get_analysis_dir(cfg) / "group_results" / "rsa"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rdm_path(cfg, subject, window_name, analysis_name):
    """
    Per-subject RDM results: subj01/subj01_cue_rsa_spatial.npz

    Parameters
    ----------
    analysis_name : str
        User-chosen label for this RSA analysis (e.g., 'spatial',
        'symbolic', 'all'). Becomes part of the filename.
    """
    subj_dir = get_subject_dir(cfg, subject)
    return subj_dir / f"{subject}_{window_name}_rsa_{analysis_name}.npz"


def get_group_rdms_path(cfg, window_name, analysis_name):
    """Group-level stacked RDMs: group_results/rsa/..._rdms.npz"""
    return _get_rsa_dir(cfg) / f"{window_name}_rsa_{analysis_name}_rdms.npz"


# ===================================================================
# Epoch selection helpers
# ===================================================================

def _select_and_label_epochs(epochs, conditions):
    """
    Select epochs matching conditions and assign condition labels.

    Parameters
    ----------
    epochs : mne.Epochs
    conditions : dict
        {condition_label: [event_codes]}

    Returns
    -------
    data : np.ndarray (n_trials, n_channels, n_times)
    labels : list of str (n_trials,)
    """
    all_data = []
    all_labels = []

    for cond_name, codes in conditions.items():
        names = [name for name, eid in epochs.event_id.items()
                 if eid in codes]
        if not names:
            continue
        cond_epochs = epochs[names]
        all_data.append(cond_epochs.get_data(copy=True))
        all_labels.extend([cond_name] * len(cond_epochs))

    if not all_data:
        raise ValueError(
            "No epochs matched any condition. "
            f"Available event_id: {epochs.event_id}"
        )

    data = np.concatenate(all_data, axis=0)
    return data, all_labels


def _balance_trials(data, labels, seed=42):
    """
    Balance trial counts across conditions by random subsampling.

    Parameters
    ----------
    data : np.ndarray (n_trials, n_channels, n_times)
    labels : list of str
    seed : int

    Returns
    -------
    data_balanced : np.ndarray
    labels_balanced : list of str
    """
    rng = np.random.RandomState(seed)
    unique_labels = sorted(set(labels))
    labels_arr = np.array(labels)

    min_count = min(np.sum(labels_arr == lab) for lab in unique_labels)

    indices = []
    for lab in unique_labels:
        lab_idx = np.where(labels_arr == lab)[0]
        chosen = rng.choice(lab_idx, size=min_count, replace=False)
        indices.extend(chosen)

    indices = sorted(indices)
    return data[indices], [labels[i] for i in indices]


# ===================================================================
# Per-subject RDM computation
# ===================================================================

def compute_rdms_subject(cfg, subject, window_name, conditions,
                         analysis_name,
                         resample_sfreq=None, baseline=None,
                         picks="eeg", time_range=None,
                         method="crossnobis", balance=True, seed=42,
                         overwrite=False, verbose=True):
    """
    Compute time-resolved RDMs for one subject.

    Parameters
    ----------
    cfg : SimpleNamespace
        Pipeline config (paths, subjects).
    subject : str
        Subject ID.
    window_name : str
        Epoch window to load (e.g., 'cue').
    conditions : dict
        {condition_label: [event_codes]}. All conditions are included
        simultaneously in the RDM. Example::

            {'d1p2': [12], 'd1p3': [13], 'd2p1': [21], ...}

    analysis_name : str
        Label for this analysis — used in the output filename.
    resample_sfreq : float or None
        If set, resample epochs to this frequency before computation.
    baseline : tuple or None
        Baseline window (e.g., (-0.2, 0.0)).
    picks : str
        Channel selection (default 'eeg').
    time_range : tuple or None
        (tmin, tmax) to crop epochs after loading.
    method : str
        Distance metric for rsatoolbox. Default 'crossnobis' (cross-
        validated Mahalanobis). Also supports 'correlation', 'euclidean'.
    balance : bool
        If True, subsample to equalize trial counts across conditions.
    seed : int
        Random seed for trial balancing.
    overwrite : bool
    verbose : bool

    Returns
    -------
    success : bool
    result : dict or None
        Keys: dissimilarities, times, condition_names, n_conditions,
        subject, window_name, analysis_name, method.
    """
    import gc
    from rsatoolbox.data import TemporalDataset
    from rsatoolbox.rdm.calc import calc_rdm_movie

    out_path = get_rdm_path(cfg, subject, window_name, analysis_name)

    if not overwrite and out_path.exists():
        if verbose:
            print(f"   [{subject}] RDMs already exist — skipping")
        return True, _load_subject_result(out_path)

    # ── Load epochs ──
    in_path = get_final_epochs_path(cfg, subject, window_name)
    if not in_path.exists():
        if verbose:
            print(f"   [{subject}] final epochs not found — skipping")
        return False, None

    epochs = mne.read_epochs(in_path, preload=True, verbose="WARNING")

    # ── Pick channels ──
    epochs.pick(picks, verbose="WARNING")

    # Drop non-EEG channels that might slip through
    non_eeg = [ch for ch in epochs.ch_names
               if ch.upper() in ("HEOG", "VEOG", "EOG", "AUX_1",
                                  "ECG", "EMG", "MISC")]
    if non_eeg:
        epochs.drop_channels(non_eeg)

    # ── Baseline ──
    if baseline is not None:
        epochs.apply_baseline(baseline, verbose="WARNING")

    # ── Resample ──
    if resample_sfreq is not None:
        epochs.resample(resample_sfreq, verbose="WARNING")

    # ── Crop ──
    if time_range is not None:
        epochs.crop(tmin=time_range[0], tmax=time_range[1])

    # ── Select and label epochs ──
    data, labels = _select_and_label_epochs(epochs, conditions)
    times = epochs.times.copy()
    ch_names = list(epochs.ch_names)

    del epochs
    gc.collect()

    if verbose:
        unique, counts = np.unique(labels, return_counts=True)
        counts_str = ", ".join(f"{u}={c}" for u, c in zip(unique, counts))
        print(f"   [{subject}] {len(data)} trials: {counts_str}")

    # ── Balance trials ──
    if balance:
        data, labels = _balance_trials(data, labels, seed=seed)
        if verbose:
            n_per = len(data) // len(set(labels))
            print(f"   [{subject}] balanced to {n_per} per condition")

    # ── Build TemporalDataset ──
    dataset = TemporalDataset(
        measurements=data,
        descriptors={"subject": subject},
        obs_descriptors={"condition": labels},
        channel_descriptors={"channel": ch_names},
        time_descriptors={"time": times},
    )

    # ── Compute RDM movie ──
    if verbose:
        print(f"   [{subject}] computing {method} RDM movie "
              f"({len(set(labels))} conditions, "
              f"{data.shape[1]} channels, "
              f"{data.shape[2]} time points)...")

    rdms = calc_rdm_movie(
        dataset,
        method=method,
        descriptor="condition",
    )

    # ── Extract results ──
    dissimilarities = rdms.dissimilarities      # (n_times, n_pairs)
    rdm_times = np.array(rdms.rdm_descriptors["time"])
    condition_names = list(rdms.pattern_descriptors["condition"])

    result = {
        "dissimilarities": dissimilarities,
        "times": rdm_times,
        "condition_names": condition_names,
        "n_conditions": len(condition_names),
        "subject": subject,
        "window_name": window_name,
        "analysis_name": analysis_name,
        "method": method,
    }

    _save_subject_result(out_path, result)

    if verbose:
        print(f"   [{subject}] saved: {out_path.name} "
              f"({dissimilarities.shape[0]} time points, "
              f"{dissimilarities.shape[1]} pairs)")

    del data, dataset, rdms
    gc.collect()

    return True, result


# ===================================================================
# Save / load helpers
# ===================================================================

def _save_subject_result(out_path, result):
    """Save result dict as .npz."""
    save_dict = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, (list, tuple)):
            save_dict[key] = np.array(val, dtype=object)
        else:
            save_dict[key] = np.array(val)
    np.savez_compressed(out_path, **save_dict)


def _load_subject_result(path):
    """Load .npz back to dict."""
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        if val.ndim == 0:
            result[key] = val.item()
        else:
            result[key] = val
    return result


def load_subject_rdms(cfg, subject, window_name, analysis_name):
    """Load one subject's RDM results."""
    path = get_rdm_path(cfg, subject, window_name, analysis_name)
    if not path.exists():
        raise FileNotFoundError(f"RDMs not found: {path}")
    return _load_subject_result(path)


def load_all_rdms(cfg, window_name, analysis_name, verbose=True):
    """
    Load and stack all subjects' RDMs into a group dict.

    Returns
    -------
    group : dict
        dissimilarities : np.ndarray (n_subjects, n_times, n_pairs)
        times : np.ndarray (n_times,)
        condition_names : list of str
        n_conditions : int
        subjects : list of str
        analysis_name : str
    """
    subjects = find_subjects(cfg)

    all_dissim = []
    subjects_loaded = []
    ref_times = None
    ref_conditions = None

    for subject in subjects:
        path = get_rdm_path(cfg, subject, window_name, analysis_name)
        if not path.exists():
            if verbose:
                print(f"   [{subject}] not found — skipping")
            continue

        result = _load_subject_result(path)
        dissim = result["dissimilarities"]
        times = result["times"]
        cond_names = list(result["condition_names"])

        if ref_times is None:
            ref_times = times
            ref_conditions = cond_names
        else:
            if len(times) != len(ref_times):
                if verbose:
                    print(f"   [{subject}] time mismatch "
                          f"({len(times)} vs {len(ref_times)}) — skipping")
                continue

        all_dissim.append(dissim)
        subjects_loaded.append(subject)

    if not all_dissim:
        raise RuntimeError("No subjects loaded.")

    group = {
        "dissimilarities": np.array(all_dissim),
        "times": ref_times,
        "condition_names": ref_conditions,
        "n_conditions": len(ref_conditions),
        "subjects": subjects_loaded,
        "analysis_name": analysis_name,
    }

    if verbose:
        shape = group["dissimilarities"].shape
        print(f"Loaded {len(subjects_loaded)} subjects, "
              f"shape: {shape}")

    return group


# ===================================================================
# RDM manipulation helpers
# ===================================================================

def vec_to_square(vec, n_conditions):
    """Convert upper-triangle vector to full symmetric matrix."""
    mat = np.zeros((n_conditions, n_conditions))
    iu = np.triu_indices(n_conditions, k=1)
    mat[iu] = vec
    mat = mat + mat.T
    return mat


def square_to_vec(mat):
    """Extract upper-triangle vector from a symmetric matrix."""
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


# ===================================================================
# Model comparison
# ===================================================================

def compute_model_fits(group, models, verbose=True):
    """
    Time-resolved Spearman correlation between neural and model RDMs.

    For each subject and time point, computes the Spearman rank
    correlation between the upper triangle of the neural RDM and
    each model RDM.

    Parameters
    ----------
    group : dict
        Output of ``load_all_rdms``. Contains 'dissimilarities' with
        shape (n_subjects, n_times, n_pairs).
    models : dict
        {model_name: model_rdm}. Each model_rdm can be:

        - A 1D array of length n_pairs (upper triangle).
        - A 2D square matrix (n_conditions × n_conditions), which
          will be converted to upper-triangle automatically.

    verbose : bool
        Print summary statistics.

    Returns
    -------
    fits : dict
        {model_name: np.ndarray (n_subjects, n_times)} of Spearman r.
    """
    from scipy.stats import spearmanr

    dissim = group["dissimilarities"]  # (n_subj, n_times, n_pairs)
    n_subj, n_times, n_pairs = dissim.shape

    # Prepare model vectors
    model_vecs = {}
    for name, model in models.items():
        m = np.asarray(model, dtype=float)
        if m.ndim == 2:
            m = square_to_vec(m)
        if len(m) != n_pairs:
            raise ValueError(
                f"Model '{name}' has {len(m)} pairs, "
                f"expected {n_pairs}. Check that the model RDM has "
                f"the same number of conditions as the neural RDMs."
            )
        model_vecs[name] = m

    # Compute Spearman at each time point for each subject
    fits = {name: np.full((n_subj, n_times), np.nan) for name in models}

    for si in range(n_subj):
        for ti in range(n_times):
            neural_vec = dissim[si, ti, :]
            for name, mvec in model_vecs.items():
                mask = ~(np.isnan(neural_vec) | np.isnan(mvec))
                if mask.sum() < 3:
                    continue
                r, _ = spearmanr(neural_vec[mask], mvec[mask])
                fits[name][si, ti] = r

    if verbose:
        print("Model fit summary (mean ± peak across time):")
        for name, fit_arr in fits.items():
            mean_r = np.nanmean(fit_arr)
            mean_across_subj = np.nanmean(fit_arr, axis=0)
            peak_r = np.nanmax(mean_across_subj)
            peak_t = group["times"][np.nanargmax(mean_across_subj)]
            print(f"   {name:25s}: mean r = {mean_r:.4f}, "
                  f"peak r = {peak_r:.4f} at {peak_t:.3f}s")

    return fits


# ===================================================================
# Batch wrapper
# ===================================================================

def _compute_rdms_worker(cfg, subject, window_name, conditions,
                         analysis_name, kwargs):
    """Wrapper for joblib parallelism."""
    try:
        ok, _ = compute_rdms_subject(
            cfg, subject, window_name, conditions, analysis_name,
            verbose=False, **kwargs,
        )
        return subject, ok, None
    except Exception as e:
        return subject, False, f"{type(e).__name__}: {e}"


def compute_rdms_all(cfg, window_name, conditions, analysis_name,
                     n_jobs=1, overwrite=False, verbose=True, **kwargs):
    """
    Compute RDMs for all subjects.

    Parameters
    ----------
    cfg : SimpleNamespace
    window_name : str
    conditions : dict
        {condition_label: [event_codes]}.
    analysis_name : str
    n_jobs : int
        Number of parallel jobs. 1 = sequential (safest).
        Note: rsatoolbox's crossnobis can use memory; monitor if
        parallelizing.
    overwrite : bool
    verbose : bool
    **kwargs
        Passed to ``compute_rdms_subject`` (resample_sfreq, baseline,
        time_range, method, balance, seed, picks).

    Returns
    -------
    summary : dict
        {'computed': [...], 'skipped': [...], 'failed': [...]}
    group : dict or None
        Output of ``load_all_rdms`` if at least one subject succeeded.
    """
    subjects = find_subjects(cfg)
    summary = {"computed": [], "skipped": [], "failed": []}

    if verbose:
        print(f"Computing RDMs for {len(subjects)} subject(s)\n")

    if n_jobs == 1:
        # Sequential — cleaner output
        for i, subject in enumerate(subjects, start=1):
            if verbose:
                print(f"--- [{i}/{len(subjects)}] {subject} ---")
            try:
                ok, result = compute_rdms_subject(
                    cfg, subject, window_name, conditions, analysis_name,
                    overwrite=overwrite, verbose=verbose, **kwargs,
                )
                if ok:
                    summary["computed"].append(subject)
                else:
                    summary["skipped"].append(subject)
            except Exception as e:
                print(f"[{subject}] ERROR: {type(e).__name__}: {e}")
                summary["failed"].append((subject, str(e)))
            if verbose:
                print()
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_rdms_worker)(
                cfg, subject, window_name, conditions, analysis_name,
                {**kwargs, "overwrite": overwrite},
            )
            for subject in subjects
        )

        for subject, ok, err in results:
            if err:
                summary["failed"].append((subject, err))
                if verbose:
                    print(f"[{subject}] FAILED: {err}")
            elif ok:
                summary["computed"].append(subject)
            else:
                summary["skipped"].append(subject)

    # Print summary
    if verbose:
        print("=" * 60)
        print(f"Computed: {len(summary['computed'])}")
        print(f"Skipped:  {len(summary['skipped'])}")
        print(f"Failed:   {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    # Load group data
    group = None
    try:
        group = load_all_rdms(
            cfg, window_name, analysis_name, verbose=verbose)
    except Exception:
        if verbose:
            print("Could not load group data.")

    return summary, group