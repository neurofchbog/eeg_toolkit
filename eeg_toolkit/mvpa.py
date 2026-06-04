"""
Multivariate pattern analysis (MVPA) — temporal decoding.

Paradigm-agnostic time-resolved classification of EEG epochs.
Conditions are passed directly in code (no YAML config) so that users
can define any binary comparison from the notebook without editing
config files.

Workflow
--------
1. decode_subject / decode_all
   Load final epochs → optional resample → select conditions → build X, y
   → SlidingEstimator (StandardScaler + classifier) → cross-validated accuracy
   → optional: decision function distances, temporal generalization
   → optional: null permutation(s) with shuffled labels
   → save per-subject .npz

2. load_subject_scores / load_all_scores
   Reload saved results for exploration, stats, and plotting.

Design notes
------------
- Single-config pattern: only ``cfg`` (pipeline) is needed. Conditions,
  classifier, and MVPA-specific options are passed as function arguments
  directly in the notebook.
- Saves one .npz per subject per analysis, keyed by ``analysis_name``.
- Follows toolkit conventions: overwrite guard, verbose prints, batch
  summary dict, path helpers.
"""

from pathlib import Path

import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore

from eeg_toolkit.io import (
    find_subjects,
    get_subject_dir,
    get_analysis_dir,
)
from eeg_toolkit.artifacts import get_final_epochs_path


# ===================================================================
# Path helpers
# ===================================================================

def _get_mvpa_dir(cfg):
    """group_results/mvpa/ — created on first use."""
    d = get_analysis_dir(cfg) / "group_results" / "mvpa"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_decode_path(cfg, subject, window_name, analysis_name):
    """
    Per-subject decoding results: subj01/subj01_cue_mvpa_spatial_vs_symbolic.npz

    Parameters
    ----------
    analysis_name : str
        User-chosen label for this decoding analysis (e.g.,
        'spatial_vs_symbolic', 'target_pos_1_vs_2'). Becomes part
        of the filename.
    """
    subj_dir = get_subject_dir(cfg, subject)
    return subj_dir / f"{subject}_{window_name}_mvpa_{analysis_name}.npz"


def get_group_scores_path(cfg, window_name, analysis_name):
    """Group-level stacked scores: group_results/mvpa/..._scores.npz"""
    return _get_mvpa_dir(cfg) / f"{window_name}_mvpa_{analysis_name}_scores.npz"


# ===================================================================
# Classifier factory
# ===================================================================

_CLASSIFIERS = {
    "linear_svc": lambda: LinearSVC(C=1, max_iter=2000, dual="auto"),
    "lda": lambda: LinearDiscriminantAnalysis(),
    "logistic": lambda: LogisticRegression(C=1, max_iter=1000, solver="lbfgs"),
}


def _make_classifier(name="linear_svc"):
    """Return a sklearn pipeline: StandardScaler + classifier."""
    if name not in _CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Supported: {list(_CLASSIFIERS.keys())}"
        )
    return make_pipeline(StandardScaler(), _CLASSIFIERS[name]())


# ===================================================================
# Epoch selection
# ===================================================================

def _select_condition_epochs(epochs, event_ids):
    """
    Select epochs matching a list of integer event IDs.

    Returns the subset of epochs whose event code is in ``event_ids``.
    """
    names = [name for name, eid in epochs.event_id.items()
             if eid in event_ids]
    if not names:
        raise ValueError(
            f"No epochs match event IDs {event_ids}. "
            f"Available: {epochs.event_id}"
        )
    return epochs[names]


# ===================================================================
# Per-subject decoding
# ===================================================================

def decode_subject(cfg, subject, window_name, conditions, analysis_name,
                   classifier="linear_svc", n_folds=5, n_repeats=1,
                   resample_sfreq=None, baseline=None, picks="eeg",
                   compute_distances=False, compute_temporal_gen=False,
                   n_null_permutations=0, null_seed=42,
                   overwrite=False, verbose=True):
    """
    Run time-resolved decoding for a single subject.

    Incremental: if a previous result exists on disk and ``overwrite=False``,
    only the newly requested features (distances, temporal generalization,
    null permutations) are computed.  Already-computed features are kept.
    Epochs are only loaded if there is actual work to do.

    Parameters
    ----------
    cfg : SimpleNamespace
        Pipeline config (paths, subjects).
    subject : str
        Subject ID.
    window_name : str
        Epoch window to load (e.g., 'cue', 'trial').
    conditions : dict
        Exactly two conditions: {label: [event_codes]}.
        Example: {'spatial': [12, 13, ...], 'symbolic': [112, 113, ...]}
    analysis_name : str
        Label for this analysis — used in the output filename.
    classifier : str
        'linear_svc', 'lda', or 'logistic'.
    n_folds : int
        Number of folds for stratified k-fold CV.
    n_repeats : int
        Number of times to repeat the k-fold CV with different random
        splits. The final score is averaged across all repeats × folds.
        Use 100 for publication-quality smoothness.
    resample_sfreq : float or None
        If set, resample epochs to this frequency before decoding.
        Useful to speed up analysis (e.g., 64 Hz).
    baseline : tuple or None
        Baseline window to apply (e.g., (-0.2, 0.0)). None = no baseline.
    picks : str
        Channel selection (default 'eeg').
    compute_distances : bool
        If True, also extract per-class decision function distances
        from the SVM/logistic hyperplane (not available for LDA).
    compute_temporal_gen : bool
        If True, also compute the full time × time generalization matrix
        using GeneralizingEstimator.
    n_null_permutations : int
        Number of label-shuffle permutations. 0 = skip null.
        1 = single null (fast, matches your Colab approach).
        >1 = multi-permutation null (slower, more rigorous).
    null_seed : int
        Base random seed for null permutations (incremented per perm).
    overwrite : bool
        If True, recompute everything from scratch.
        If False (default), only compute features not already on disk.
    verbose : bool

    Returns
    -------
    success : bool
        True if any computation was performed (or all features present).
    result : dict or None
        Keys: scores, times, conditions, n_trials, analysis_name.
        Optional keys: distances, temporal_gen_scores, null_scores.
    """
    import gc

    # --- Validate conditions ---
    if len(conditions) != 2:
        raise ValueError(
            f"Exactly 2 conditions required for binary decoding, "
            f"got {len(conditions)}: {list(conditions.keys())}"
        )
    cond_names = list(conditions.keys())
    cond_a, cond_b = cond_names

    out_path = get_decode_path(cfg, subject, window_name, analysis_name)

    # ==============================================================
    # Determine what needs computing
    # ==============================================================
    existing = None
    needs_scores = True
    needs_distances = compute_distances
    needs_temporal_gen = compute_temporal_gen
    needs_null = n_null_permutations > 0

    if not overwrite and out_path.exists():
        existing = _load_subject_result(out_path)

        # Scores already done?
        if "scores" in existing:
            needs_scores = False

        # Distances already done?
        if "distances" in existing:
            needs_distances = False
        elif not compute_distances:
            needs_distances = False

        # Temporal gen already done?
        if "temporal_gen_scores" in existing:
            needs_temporal_gen = False
        elif not compute_temporal_gen:
            needs_temporal_gen = False

        # Null: check if enough permutations exist
        if "null_scores" in existing and n_null_permutations > 0:
            existing_null = existing["null_scores"]
            if existing_null.ndim == 2 and existing_null.shape[0] >= n_null_permutations:
                needs_null = False
        elif n_null_permutations == 0:
            needs_null = False

        # Nothing to do?
        if not any([needs_scores, needs_distances,
                    needs_temporal_gen, needs_null]):
            if verbose:
                print(f"  [{subject}] {analysis_name}: all requested "
                      f"features already on disk — skipping")
            return False, existing

        # Report what will be computed
        if verbose:
            todo = []
            if needs_scores:
                todo.append("scores")
            if needs_distances:
                todo.append("distances")
            if needs_temporal_gen:
                todo.append("temporal_gen")
            if needs_null:
                todo.append(f"null({n_null_permutations})")
            print(f"  [{subject}] {analysis_name}: computing {', '.join(todo)} "
                  f"(keeping existing results)")

    # ==============================================================
    # Load epochs and build X, y (only if something needs computing)
    # ==============================================================
    epochs_path = get_final_epochs_path(cfg, subject, window_name)
    if not epochs_path.exists():
        if verbose:
            print(f"  [{subject}] final epochs not found — skipping")
        return False, None

    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose="WARNING")
    epochs.pick(picks, verbose="WARNING")

    # Optional baseline
    if baseline is not None:
        epochs.apply_baseline(tuple(baseline), verbose="WARNING")

    # Optional resample
    if resample_sfreq is not None:
        original_sfreq = epochs.info["sfreq"]
        if resample_sfreq != original_sfreq:
            epochs.resample(resample_sfreq, verbose="WARNING")
            if verbose:
                print(f"  [{subject}] resampled: {original_sfreq} → "
                      f"{resample_sfreq} Hz")

    # Select conditions
    try:
        epochs_a = _select_condition_epochs(epochs, conditions[cond_a])
        epochs_b = _select_condition_epochs(epochs, conditions[cond_b])
    except ValueError as e:
        if verbose:
            print(f"  [{subject}] {analysis_name}: {e}")
        del epochs; gc.collect()
        return False, None

    n_a, n_b = len(epochs_a), len(epochs_b)
    if n_a == 0 or n_b == 0:
        if verbose:
            print(f"  [{subject}] {analysis_name}: empty condition "
                  f"({cond_a}={n_a}, {cond_b}={n_b}) — skipping")
        del epochs, epochs_a, epochs_b; gc.collect()
        return False, None

    # Build X, y — then free epoch objects
    X = np.concatenate([epochs_a.get_data(copy=True),
                        epochs_b.get_data(copy=True)], axis=0)
    y = np.array([0] * n_a + [1] * n_b)
    times = epochs.times.copy()
    sfreq = epochs.info["sfreq"]

    del epochs, epochs_a, epochs_b
    gc.collect()

    if verbose:
        print(f"  [{subject}] {analysis_name}: {cond_a}={n_a}, "
              f"{cond_b}={n_b}, {X.shape[1]} ch × {X.shape[2]} times "
              f"({sfreq:.0f} Hz)")

    # ==============================================================
    # Start with existing result or empty metadata
    # ==============================================================
    result = dict(existing) if existing else {}

    # Always update metadata
    result.update({
        "times": times,
        "conditions": cond_names,
        "n_trials": {cond_a: n_a, cond_b: n_b},
        "analysis_name": analysis_name,
        "classifier": classifier,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "sfreq": sfreq,
    })

    # ==============================================================
    # Compute only what's needed
    # ==============================================================

    # --- Scores (sliding decoding with repeated CV) ---
    if needs_scores:
        all_fold_scores = []
        for rep in range(n_repeats):
            cv_rep = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=42 + rep)
            clf = _make_classifier(classifier)
            slider = SlidingEstimator(clf, scoring="accuracy", n_jobs=1,
                                      verbose=False)
            rep_scores = cross_val_multiscore(slider, X, y, cv=cv_rep,
                                              n_jobs=1, verbose=False)
            all_fold_scores.append(rep_scores)

        scores_all = np.vstack(all_fold_scores)  # (n_repeats*n_folds, n_times)
        result["scores"] = scores_all.mean(axis=0)
        result["scores_per_fold"] = scores_all
        del scores_all

        if verbose:
            total_folds = n_repeats * n_folds
            print(f"  [{subject}] scores: {n_repeats} repeats × {n_folds} folds "
                  f"= {total_folds} runs, mean accuracy = "
                  f"{result['scores'].mean():.3f} ± "
                  f"{result['scores'].std():.3f}")

    # --- Decision function distances (repeated CV) ---
    if needs_distances:
        if classifier == "lda":
            if verbose:
                print(f"  [{subject}] distances not supported for LDA — "
                      f"skipping")
        else:
            all_dist = []
            for rep in range(n_repeats):
                cv_rep = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                         random_state=42 + rep)
                clf_dist = _make_classifier(classifier)
                rep_dist = _compute_distances(X, y, clf_dist, cv_rep,
                                              n_a, n_b)
                all_dist.append(rep_dist)
            # Average across repeats: each is (2, n_times)
            result["distances"] = np.mean(all_dist, axis=0)
            del all_dist
            if verbose:
                print(f"  [{subject}] distances computed ({n_repeats} repeats): "
                      f"shape {result['distances'].shape}")

    # --- Temporal generalization (repeated CV) ---
    if needs_temporal_gen:
        all_gen = []
        for rep in range(n_repeats):
            cv_rep = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=42 + rep)
            gen_clf = _make_classifier(classifier)
            generalizer = GeneralizingEstimator(gen_clf, scoring="accuracy",
                                                n_jobs=1, verbose=False)
            gen_scores = cross_val_multiscore(generalizer, X, y, cv=cv_rep,
                                              n_jobs=1, verbose=False)
            all_gen.append(gen_scores.mean(axis=0))
            del gen_scores, generalizer

        result["temporal_gen_scores"] = np.mean(all_gen, axis=0)
        del all_gen
        gc.collect()
        if verbose:
            print(f"  [{subject}] temporal gen ({n_repeats} repeats): "
                  f"{result['temporal_gen_scores'].shape}")

    # --- Null permutations (repeated CV per permutation) ---
    if needs_null:
        null_all = []
        for p in range(n_null_permutations):
            rng = np.random.RandomState(null_seed + p)
            y_perm = rng.permutation(y)

            perm_fold_scores = []
            for rep in range(n_repeats):
                cv_rep = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                         random_state=42 + rep)
                null_clf = _make_classifier(classifier)
                null_slider = SlidingEstimator(null_clf, scoring="accuracy",
                                               n_jobs=1, verbose=False)
                rep_scores = cross_val_multiscore(null_slider, X, y_perm,
                                                  cv=cv_rep, n_jobs=1,
                                                  verbose=False)
                perm_fold_scores.append(rep_scores)
                del null_slider

            # Average across repeats × folds for this permutation
            null_all.append(np.vstack(perm_fold_scores).mean(axis=0))
            del perm_fold_scores

        result["null_scores"] = np.array(null_all)
        del null_all
        if verbose:
            null_mean = result["null_scores"].mean()
            print(f"  [{subject}] null ({n_null_permutations} perm × "
                  f"{n_repeats} repeats): mean = {null_mean:.3f}")

    # ==============================================================
    # Free data arrays and save
    # ==============================================================
    del X, y
    gc.collect()

    _save_subject_result(out_path, result)
    if verbose:
        print(f"  [{subject}] saved: {out_path.name}")

    return True, result


# ===================================================================
# Decision function distances helper
# ===================================================================

def _compute_distances(X, y, clf_template, cv, n_a, n_b):
    """
    Compute per-class mean decision function distances across CV folds.

    Returns shape (2, n_times): row 0 = class 0, row 1 = class 1.
    """
    from sklearn.base import clone

    n_times = X.shape[2]
    class0_folds = []
    class1_folds = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit sliding estimator manually
        slider = SlidingEstimator(clone(clf_template), scoring="accuracy",
                                   n_jobs=1, verbose=False)
        slider.fit(X_train, y_train)
        decision_vals = np.asarray(slider.decision_function(X_test))
        # decision_vals shape: (n_test_trials, n_times)

        class0_folds.append(decision_vals[y_test == 0])
        class1_folds.append(decision_vals[y_test == 1])

    # Stack across folds, average across trials
    class0_all = np.vstack(class0_folds).mean(axis=0)  # (n_times,)
    class1_all = np.vstack(class1_folds).mean(axis=0)

    return np.stack([class0_all, class1_all], axis=0)  # (2, n_times)


# ===================================================================
# Save / load helpers
# ===================================================================

def _save_subject_result(out_path, result):
    """Save a per-subject result dict as .npz."""
    save_dict = {}
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, dict):
            # Store dicts as strings (for n_trials, conditions, etc.)
            save_dict[key] = str(val)
        elif isinstance(val, list):
            save_dict[key] = np.array(val)
        else:
            save_dict[key] = np.array(val)

    np.savez_compressed(out_path, **save_dict)


def _load_subject_result(path):
    """Load a per-subject .npz into a dict, restoring types."""
    data = np.load(path, allow_pickle=True)
    result = {}
    for key in data.files:
        val = data[key]
        # Scalar arrays → Python scalars
        if val.ndim == 0:
            val_item = val.item()
            # Try to restore dicts/lists stored as strings
            if isinstance(val_item, str) and val_item.startswith("{"):
                import ast
                try:
                    val_item = ast.literal_eval(val_item)
                except (ValueError, SyntaxError):
                    pass
            result[key] = val_item
        else:
            result[key] = val
    return result


def load_subject_scores(cfg, subject, window_name, analysis_name):
    """
    Load decoding results for one subject.

    Returns
    -------
    dict with keys: scores, times, conditions, n_trials, etc.
    """
    path = get_decode_path(cfg, subject, window_name, analysis_name)
    if not path.exists():
        raise FileNotFoundError(f"No decoding results: {path}")
    return _load_subject_result(path)


def load_all_scores(cfg, window_name, analysis_name, verbose=True):
    """
    Load decoding scores for all subjects and stack into group arrays.

    Returns
    -------
    group : dict with keys:
        subject_scores : (n_subjects, n_times)
        times : (n_times,)
        subjects : list of str
        conditions : list of str
        subject_null_scores : (n_subjects, n_perms, n_times) or None
        subject_distances : (n_subjects, 2, n_times) or None
        subject_temporal_gen : (n_subjects, n_times, n_times) or None
    """
    subjects = find_subjects(cfg)

    all_scores = []
    all_null = []
    all_distances = []
    all_temporal_gen = []
    subjects_used = []
    ref_times = None
    ref_conditions = None

    for subject in subjects:
        path = get_decode_path(cfg, subject, window_name, analysis_name)
        if not path.exists():
            continue

        result = _load_subject_result(path)
        all_scores.append(result["scores"])
        subjects_used.append(subject)

        if ref_times is None:
            ref_times = result["times"]
            ref_conditions = result["conditions"]

        # Optional arrays
        if "null_scores" in result:
            all_null.append(result["null_scores"])
        if "distances" in result:
            all_distances.append(result["distances"])
        if "temporal_gen_scores" in result:
            all_temporal_gen.append(result["temporal_gen_scores"])

    if not all_scores:
        raise FileNotFoundError(
            f"No decoding results found for {window_name}/{analysis_name}"
        )

    group = {
        "subject_scores": np.array(all_scores),
        "times": ref_times,
        "subjects": subjects_used,
        "conditions": ref_conditions,
        "analysis_name": analysis_name,
        "window_name": window_name,
    }

    # Optional stacking
    if all_null and len(all_null) == len(all_scores):
        group["subject_null_scores"] = np.array(all_null)
    else:
        group["subject_null_scores"] = None

    if all_distances and len(all_distances) == len(all_scores):
        group["subject_distances"] = np.array(all_distances)
    else:
        group["subject_distances"] = None

    if all_temporal_gen and len(all_temporal_gen) == len(all_scores):
        group["subject_temporal_gen"] = np.array(all_temporal_gen)
    else:
        group["subject_temporal_gen"] = None

    if verbose:
        n = len(subjects_used)
        shape = group["subject_scores"].shape
        print(f"Loaded {n} subjects, scores shape: {shape}")
        if group["subject_null_scores"] is not None:
            print(f"  null scores: {group['subject_null_scores'].shape}")
        if group["subject_distances"] is not None:
            print(f"  distances: {group['subject_distances'].shape}")
        if group["subject_temporal_gen"] is not None:
            print(f"  temporal gen: {group['subject_temporal_gen'].shape}")

    return group


# ===================================================================
# Parallel worker (module-level so joblib can pickle it on Windows)
# ===================================================================

def _decode_worker(cfg, subject, window_name, conditions, analysis_name,
                   classifier, n_folds, n_repeats, resample_sfreq, baseline, picks,
                   compute_distances, compute_temporal_gen,
                   n_null_permutations, null_seed, overwrite):
    """
    Wrapper for decode_subject used by parallel dispatch.

    Returns a simple status tuple instead of the full result dict
    (results are saved to disk by decode_subject).

    Returns
    -------
    (subject, status, error_msg)
        status is 'decoded', 'skipped', or 'failed'.
    """
    try:
        ok, _ = decode_subject(
            cfg, subject, window_name, conditions, analysis_name,
            classifier=classifier, n_folds=n_folds, n_repeats=n_repeats,
            resample_sfreq=resample_sfreq, baseline=baseline,
            picks=picks,
            compute_distances=compute_distances,
            compute_temporal_gen=compute_temporal_gen,
            n_null_permutations=n_null_permutations,
            null_seed=null_seed,
            overwrite=overwrite,
            verbose=True,  # each worker prints its own progress
        )
        return (subject, "decoded" if ok else "skipped", "")
    except Exception as e:
        return (subject, "failed", f"{type(e).__name__}: {e}")


# ===================================================================
# Batch wrapper
# ===================================================================

def decode_all(cfg, window_name, conditions, analysis_name,
               classifier="linear_svc", n_folds=5, n_repeats=1,
               resample_sfreq=None, baseline=None, picks="eeg",
               compute_distances=False, compute_temporal_gen=False,
               n_null_permutations=0, null_seed=42,
               n_jobs=1,
               overwrite=False, verbose=True):
    """
    Run temporal decoding for every included subject.

    Parameters
    ----------
    cfg : SimpleNamespace
        Pipeline config.
    window_name : str
        Epoch window name.
    conditions : dict
        Two conditions: {label: [event_codes]}.
    analysis_name : str
        Label for filename.
    n_jobs : int
        Number of parallel workers for subject-level parallelism.
        1 = sequential (clean verbose output, good for debugging).
        -1 = use all available CPU cores.
        N > 1 = use N cores.
        Note: parallelism is at the SUBJECT level, not within-subject.
        Each subject runs on a single core, but multiple subjects run
        simultaneously.
    classifier, n_folds, n_repeats, resample_sfreq, baseline, picks,
    compute_distances, compute_temporal_gen, n_null_permutations,
    null_seed, overwrite, verbose :
        See decode_subject docstring.

    Returns
    -------
    summary : dict
        {'decoded': [...], 'skipped': [...], 'failed': [...]}
    group : dict or None
        Stacked group results from load_all_scores. None if no subjects
        succeeded.
    """
    subjects = find_subjects(cfg)
    summary = {"decoded": [], "skipped": [], "failed": []}

    cond_names = list(conditions.keys())
    if verbose:
        print(f"MVPA temporal decoding: {cond_names[0]} vs {cond_names[1]}")
        print(f"Window: {window_name}, Classifier: {classifier}, "
              f"Folds: {n_folds}, Repeats: {n_repeats}")
        if resample_sfreq:
            print(f"Resampling to: {resample_sfreq} Hz")
        if compute_distances:
            print(f"Computing decision distances: yes")
        if compute_temporal_gen:
            print(f"Computing temporal generalization: yes")
        if n_null_permutations > 0:
            print(f"Null permutations: {n_null_permutations}")
        print(f"Subjects: {len(subjects)}, n_jobs: {n_jobs}")
        print("=" * 60)

    # --- Common kwargs for decode_subject ---
    common_kwargs = dict(
        window_name=window_name,
        conditions=conditions,
        analysis_name=analysis_name,
        classifier=classifier,
        n_folds=n_folds,
        n_repeats=n_repeats,
        resample_sfreq=resample_sfreq,
        baseline=baseline,
        picks=picks,
        compute_distances=compute_distances,
        compute_temporal_gen=compute_temporal_gen,
        n_null_permutations=n_null_permutations,
        null_seed=null_seed,
        overwrite=overwrite,
    )

    if n_jobs == 1:
        # ---- Sequential: clean verbose output ----
        for i, subject in enumerate(subjects, start=1):
            if verbose:
                print(f"\n[{i}/{len(subjects)}] {subject}")
            try:
                ok, _ = decode_subject(
                    cfg, subject, verbose=verbose, **common_kwargs,
                )
                if ok:
                    summary["decoded"].append(subject)
                else:
                    summary["skipped"].append(subject)
            except Exception as e:
                print(f"  [{subject}] ERROR: {type(e).__name__}: {e}")
                summary["failed"].append((subject, str(e)))
    else:
        # ---- Parallel: subject-level via joblib ----
        from joblib import Parallel, delayed

        if verbose:
            print(f"\nRunning in parallel ({n_jobs} workers)...\n")

        # verbose=10 prints per-task completion: "Done 5 out of 30"
        joblib_verbosity = 10 if verbose else 0
        results = Parallel(n_jobs=n_jobs, backend="loky",
                           verbose=joblib_verbosity)(
            delayed(_decode_worker)(
                cfg, subject, **common_kwargs,
            )
            for subject in subjects
        )

        # Collect results
        for subject, status, err_msg in results:
            if status == "decoded":
                summary["decoded"].append(subject)
            elif status == "skipped":
                summary["skipped"].append(subject)
            else:
                summary["failed"].append((subject, err_msg))

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print(f"Decoded: {len(summary['decoded'])}")
        print(f"Skipped: {len(summary['skipped'])}")
        print(f"Failed:  {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    # Load group results if any subjects succeeded
    group = None
    if summary["decoded"]:
        try:
            group = load_all_scores(cfg, window_name, analysis_name,
                                     verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Warning: could not load group scores: {e}")

    return summary, group


# ===================================================================
# Cross-condition decoding
# ===================================================================

def get_cross_decode_path(cfg, subject, window_name, analysis_name):
    """Per-subject cross-decoding results."""
    subj_dir = get_subject_dir(cfg, subject)
    return subj_dir / f"{subject}_{window_name}_xdecode_{analysis_name}.npz"


def cross_decode_subject(cfg, subject, window_name,
                         train_conditions, test_conditions,
                         analysis_name,
                         classifier="linear_svc",
                         resample_sfreq=None, baseline=None, picks="eeg",
                         bidirectional=True,
                         n_null_permutations=0, null_seed=42,
                         overwrite=False, verbose=True):
    """
    Cross-condition temporal decoding for a single subject.

    Trains on one set of conditions, tests on a completely separate set.
    No cross-validation — the train/test split is defined by the paradigm.

    Parameters
    ----------
    cfg : SimpleNamespace
    subject : str
    window_name : str
    train_conditions : dict
        Two conditions for training: {label: [event_codes]}.
    test_conditions : dict
        Two conditions for testing (same labels, different codes):
        {label: [event_codes]}.
    analysis_name : str
    classifier : str
    resample_sfreq : float or None
    baseline : tuple or None
    picks : str
    bidirectional : bool
        If True (default), also train on test_conditions and test on
        train_conditions, then average both directions. Gives a
        symmetric cross-decoding estimate.
    n_null_permutations : int
        Shuffle train labels before fitting. 0 = skip.
    null_seed : int
    overwrite : bool
    verbose : bool

    Returns
    -------
    success : bool
    result : dict or None
    """
    import gc
    from sklearn.base import clone

    if len(train_conditions) != 2 or len(test_conditions) != 2:
        raise ValueError("Exactly 2 conditions required for both train and test.")

    if list(train_conditions.keys()) != list(test_conditions.keys()):
        raise ValueError(
            f"Train and test must use the same labels. "
            f"Train: {list(train_conditions.keys())}, "
            f"Test: {list(test_conditions.keys())}"
        )

    cond_names = list(train_conditions.keys())
    cond_a, cond_b = cond_names

    out_path = get_cross_decode_path(cfg, subject, window_name, analysis_name)

    # Overwrite guard
    if not overwrite and out_path.exists():
        if verbose:
            print(f"  [{subject}] {analysis_name}: exists, skipping")
        return False, None

    # Load epochs
    epochs_path = get_final_epochs_path(cfg, subject, window_name)
    if not epochs_path.exists():
        if verbose:
            print(f"  [{subject}] final epochs not found — skipping")
        return False, None

    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose="WARNING")
    epochs.pick(picks, verbose="WARNING")

    if baseline is not None:
        epochs.apply_baseline(tuple(baseline), verbose="WARNING")

    if resample_sfreq is not None:
        orig = epochs.info["sfreq"]
        if resample_sfreq != orig:
            epochs.resample(resample_sfreq, verbose="WARNING")
            if verbose:
                print(f"  [{subject}] resampled: {orig} → {resample_sfreq} Hz")

    # Build train set
    try:
        ep_train_a = _select_condition_epochs(epochs, train_conditions[cond_a])
        ep_train_b = _select_condition_epochs(epochs, train_conditions[cond_b])
    except ValueError as e:
        if verbose:
            print(f"  [{subject}] train: {e}")
        del epochs; gc.collect()
        return False, None

    X_train = np.concatenate([ep_train_a.get_data(copy=True),
                               ep_train_b.get_data(copy=True)], axis=0)
    y_train = np.array([0] * len(ep_train_a) + [1] * len(ep_train_b))
    n_train = {cond_a: len(ep_train_a), cond_b: len(ep_train_b)}

    # Build test set
    try:
        ep_test_a = _select_condition_epochs(epochs, test_conditions[cond_a])
        ep_test_b = _select_condition_epochs(epochs, test_conditions[cond_b])
    except ValueError as e:
        if verbose:
            print(f"  [{subject}] test: {e}")
        del epochs, X_train, y_train; gc.collect()
        return False, None

    X_test = np.concatenate([ep_test_a.get_data(copy=True),
                              ep_test_b.get_data(copy=True)], axis=0)
    y_test = np.array([0] * len(ep_test_a) + [1] * len(ep_test_b))
    n_test = {cond_a: len(ep_test_a), cond_b: len(ep_test_b)}

    times = epochs.times.copy()
    sfreq = epochs.info["sfreq"]

    del epochs, ep_train_a, ep_train_b, ep_test_a, ep_test_b
    gc.collect()

    if verbose:
        print(f"  [{subject}] {analysis_name}: "
              f"train {n_train}, test {n_test}, "
              f"{X_train.shape[1]} ch × {X_train.shape[2]} times "
              f"({sfreq:.0f} Hz)")

    # ── Decode: direction A→B ──
    def _score_direction(X_tr, y_tr, X_te, y_te):
        clf = _make_classifier(classifier)
        slider = SlidingEstimator(clf, scoring="accuracy", n_jobs=1,
                                  verbose=False)
        slider.fit(X_tr, y_tr)
        return slider.score(X_te, y_te)

    scores_fwd = _score_direction(X_train, y_train, X_test, y_test)

    if bidirectional:
        scores_bwd = _score_direction(X_test, y_test, X_train, y_train)
        scores = (np.array(scores_fwd) + np.array(scores_bwd)) / 2
        if verbose:
            print(f"  [{subject}] bidirectional: "
                  f"fwd={np.mean(scores_fwd):.3f}, "
                  f"bwd={np.mean(scores_bwd):.3f}, "
                  f"avg={np.mean(scores):.3f}")
    else:
        scores = np.array(scores_fwd)
        if verbose:
            print(f"  [{subject}] accuracy: {np.mean(scores):.3f}")

    result = {
        "scores": scores,
        "times": times,
        "conditions": cond_names,
        "n_train": n_train,
        "n_test": n_test,
        "analysis_name": analysis_name,
        "classifier": classifier,
        "bidirectional": bidirectional,
        "sfreq": sfreq,
    }

    # ── Null permutations ──
    if n_null_permutations > 0:
        null_all = []
        for p in range(n_null_permutations):
            rng = np.random.RandomState(null_seed + p)
            y_perm = rng.permutation(y_train)

            null_fwd = _score_direction(X_train, y_perm, X_test, y_test)

            if bidirectional:
                y_perm_bwd = rng.permutation(y_test)
                null_bwd = _score_direction(X_test, y_perm_bwd, X_train, y_train)
                null_scores = (np.array(null_fwd) + np.array(null_bwd)) / 2
            else:
                null_scores = np.array(null_fwd)

            null_all.append(null_scores)

        result["null_scores"] = np.array(null_all)
        if verbose:
            print(f"  [{subject}] null ({n_null_permutations} perm): "
                  f"mean = {result['null_scores'].mean():.3f}")

    # ── Save and cleanup ──
    del X_train, y_train, X_test, y_test
    gc.collect()

    _save_subject_result(out_path, result)
    if verbose:
        print(f"  [{subject}] saved: {out_path.name}")

    return True, result


def _cross_decode_worker(cfg, subject, window_name,
                         train_conditions, test_conditions,
                         analysis_name, classifier,
                         resample_sfreq, baseline, picks,
                         bidirectional, n_null_permutations,
                         null_seed, overwrite):
    """Parallel worker for cross_decode_all."""
    try:
        ok, _ = cross_decode_subject(
            cfg, subject, window_name,
            train_conditions, test_conditions, analysis_name,
            classifier=classifier,
            resample_sfreq=resample_sfreq, baseline=baseline,
            picks=picks, bidirectional=bidirectional,
            n_null_permutations=n_null_permutations,
            null_seed=null_seed, overwrite=overwrite,
            verbose=True,
        )
        return (subject, "decoded" if ok else "skipped", "")
    except Exception as e:
        return (subject, "failed", f"{type(e).__name__}: {e}")


def cross_decode_all(cfg, window_name,
                     train_conditions, test_conditions,
                     analysis_name,
                     classifier="linear_svc",
                     resample_sfreq=None, baseline=None, picks="eeg",
                     bidirectional=True,
                     n_null_permutations=0, null_seed=42,
                     n_jobs=1, overwrite=False, verbose=True):
    """
    Run cross-condition decoding for every included subject.

    Parameters
    ----------
    train_conditions, test_conditions : dict
        Each has two conditions with matching labels.
    bidirectional : bool
        Average both train→test and test→train directions.
    n_jobs : int
        Subject-level parallelism.

    See cross_decode_subject for other parameters.
    """
    subjects = find_subjects(cfg)
    summary = {"decoded": [], "skipped": [], "failed": []}

    cond_names = list(train_conditions.keys())
    if verbose:
        print(f"Cross-decoding: {cond_names[0]} vs {cond_names[1]}")
        print(f"  Train: {list(train_conditions.values())}")
        print(f"  Test:  {list(test_conditions.values())}")
        print(f"  Bidirectional: {bidirectional}")
        if resample_sfreq:
            print(f"  Resampling to: {resample_sfreq} Hz")
        print(f"  Subjects: {len(subjects)}, n_jobs: {n_jobs}")
        print("=" * 60)

    common_kwargs = dict(
        window_name=window_name,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        analysis_name=analysis_name,
        classifier=classifier,
        resample_sfreq=resample_sfreq,
        baseline=baseline,
        picks=picks,
        bidirectional=bidirectional,
        n_null_permutations=n_null_permutations,
        null_seed=null_seed,
        overwrite=overwrite,
    )

    if n_jobs == 1:
        for i, subject in enumerate(subjects, start=1):
            if verbose:
                print(f"\n[{i}/{len(subjects)}] {subject}")
            try:
                ok, _ = cross_decode_subject(
                    cfg, subject, verbose=verbose, **common_kwargs,
                )
                if ok:
                    summary["decoded"].append(subject)
                else:
                    summary["skipped"].append(subject)
            except Exception as e:
                print(f"  [{subject}] ERROR: {type(e).__name__}: {e}")
                summary["failed"].append((subject, str(e)))
    else:
        from joblib import Parallel, delayed

        if verbose:
            print(f"\nRunning in parallel ({n_jobs} workers)...\n")

        joblib_verbosity = 10 if verbose else 0
        results = Parallel(n_jobs=n_jobs, backend="loky",
                           verbose=joblib_verbosity)(
            delayed(_cross_decode_worker)(
                cfg, subject, **common_kwargs,
            )
            for subject in subjects
        )

        for subject, status, err_msg in results:
            if status == "decoded":
                summary["decoded"].append(subject)
            elif status == "skipped":
                summary["skipped"].append(subject)
            else:
                summary["failed"].append((subject, err_msg))

    if verbose:
        print("\n" + "=" * 60)
        print(f"Decoded: {len(summary['decoded'])}")
        print(f"Skipped: {len(summary['skipped'])}")
        print(f"Failed:  {len(summary['failed'])}")
        if summary["failed"]:
            print("\nFailures:")
            for subj, err in summary["failed"]:
                print(f"  {subj}: {err}")
        print("=" * 60)

    # Load group results
    group = None
    if summary["decoded"]:
        try:
            group = load_all_cross_scores(cfg, window_name, analysis_name,
                                           verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Warning: could not load group scores: {e}")

    return summary, group


def load_all_cross_scores(cfg, window_name, analysis_name, verbose=True):
    """
    Load cross-decoding scores for all subjects and stack.

    Same structure as load_all_scores but uses cross-decode paths.
    """
    subjects = find_subjects(cfg)

    all_scores = []
    all_null = []
    subjects_used = []
    ref_times = None
    ref_conditions = None

    for subject in subjects:
        path = get_cross_decode_path(cfg, subject, window_name, analysis_name)
        if not path.exists():
            continue

        result = _load_subject_result(path)
        all_scores.append(result["scores"])
        subjects_used.append(subject)

        if ref_times is None:
            ref_times = result["times"]
            ref_conditions = result["conditions"]

        if "null_scores" in result:
            all_null.append(result["null_scores"])

    if not all_scores:
        raise FileNotFoundError(
            f"No cross-decoding results for {window_name}/{analysis_name}"
        )

    group = {
        "subject_scores": np.array(all_scores),
        "times": ref_times,
        "subjects": subjects_used,
        "conditions": ref_conditions,
        "analysis_name": analysis_name,
        "window_name": window_name,
    }

    if all_null and len(all_null) == len(all_scores):
        group["subject_null_scores"] = np.array(all_null)
    else:
        group["subject_null_scores"] = None

    # No distances or temporal_gen for cross-decoding
    group["subject_distances"] = None
    group["subject_temporal_gen"] = None

    if verbose:
        n = len(subjects_used)
        print(f"Loaded {n} subjects, scores shape: {group['subject_scores'].shape}")
        if group["subject_null_scores"] is not None:
            print(f"  null scores: {group['subject_null_scores'].shape}")

    return group