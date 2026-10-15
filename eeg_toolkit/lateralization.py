# -*- coding: utf-8 -*-
"""
lateralization.py — índice contralateral − ipsilateral y prueba de clústeres 1-D.

Por qué existe
--------------
Cuando un estímulo lateralizado evoca una respuesta contralateral, probar
"izquierda vs derecha" **por separado en cada ROI** parte el efecto en dos
mitades de signo opuesto y las somete a corrección múltiple como si fueran
independientes. El contraste correcto combina las dos mitades en un solo índice:

    contra = ( ROI_derecho | estímulo izquierda  +  ROI_izquierdo | estímulo derecha ) / 2
    ipsi   = ( ROI_izquierdo | estímulo izquierda +  ROI_derecho  | estímulo derecha ) / 2
    LI     = contra − ipsi

que es equivalente a  0.5 · [ (R_izq − L_izq) + (L_der − R_der) ].

Un LI negativo significa mayor negatividad contralateral, la firma habitual de
la orientación atencional hacia el lado del estímulo.

Uso típico
----------
    from eeg_toolkit.lateralization import compute_lateralization, cluster_test_1d

    li, times, subjects = compute_lateralization(
        cfg, cfg_erp,
        left_roi=["P3", "P7", "O1"], right_roi=["P4", "P8", "O2"],
        groupings={
            "spatial_target": (["spatial_target_pos1", "spatial_target_pos2"],
                               ["spatial_target_pos3", "spatial_target_pos4"]),
        },
    )
    res = cluster_test_1d(li["spatial_target"], times)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "compute_lateralization",
    "cluster_test_1d",
    "describe_clusters_1d",
    "window_mean_test",
]


# ---------------------------------------------------------------- evocados
def _load_subject_evokeds(cfg, cfg_erp, subject, window_name):
    """Devuelve {comment: Evoked} para un participante."""
    import mne
    from eeg_toolkit.evoked import get_evoked_path

    path = get_evoked_path(cfg, subject, window_name, cfg_erp)
    evokeds = mne.read_evokeds(str(path), verbose="ERROR")
    return {ev.comment: ev for ev in evokeds}


def _combine(evokeds, names, weights):
    """Promedia varios evocados en uno. weights: 'nave' o 'equal'."""
    import mne

    picked = [evokeds[n] for n in names]
    if len(picked) == 1:
        return picked[0].copy()
    return mne.combine_evoked(picked, weights=weights)


def compute_lateralization(cfg, cfg_erp, left_roi, right_roi, groupings,
                           window_name="cue", subjects=None,
                           weights="nave", verbose=True):
    """
    Índice de lateralización contra − ipsi, resuelto en el tiempo.

    Parameters
    ----------
    cfg, cfg_erp : configuraciones cargadas con load_config.
    left_roi, right_roi : list[str]
        Electrodos de cada hemisferio. Deben ser homólogos y del mismo número.
    groupings : dict[str, tuple[list[str], list[str]]]
        etiqueta -> (condiciones con el estímulo a la IZQUIERDA,
                     condiciones con el estímulo a la DERECHA)
    window_name : str
        Ventana de épocas ("cue", "trial", ...).
    subjects : list[str] | None
        Por defecto, los que devuelve find_subjects (respeta las exclusiones
        del config).
    weights : {"nave", "equal"}
        Cómo promediar varias condiciones dentro de un lado. "nave" pondera por
        número de ensayos; "equal" no.

    Returns
    -------
    li : dict[str, np.ndarray]  (n_subjects, n_times) en microvoltios
    times : np.ndarray
    subjects : list[str]
    """
    from eeg_toolkit.io import find_subjects

    if len(left_roi) != len(right_roi):
        raise ValueError(
            f"Los ROI deben tener el mismo número de electrodos "
            f"({len(left_roi)} vs {len(right_roi)}); si no, el índice mezcla "
            f"una diferencia de lateralización con una diferencia de promediado."
        )
    if subjects is None:
        subjects = find_subjects(cfg)

    out = {k: [] for k in groupings}
    times = None
    kept = []

    for subj in subjects:
        try:
            evs = _load_subject_evokeds(cfg, cfg_erp, subj, window_name)
        except Exception as exc:  # participante sin evocados
            if verbose:
                print(f"  {subj}: sin evocados ({exc.__class__.__name__}), se omite")
            continue

        missing = {n for _, (a, b) in groupings.items() for n in a + b} - set(evs)
        if missing:
            if verbose:
                print(f"  {subj}: faltan condiciones {sorted(missing)}, se omite")
            continue

        for label, (left_conds, right_conds) in groupings.items():
            ev_l = _combine(evs, left_conds, weights)
            ev_r = _combine(evs, right_conds, weights)
            if times is None:
                times = ev_l.times.copy()

            iL = [ev_l.ch_names.index(c) for c in left_roi]
            iR = [ev_l.ch_names.index(c) for c in right_roi]

            L_left = ev_l.data[iL].mean(axis=0)   # ROI izq., estímulo izq.
            R_left = ev_l.data[iR].mean(axis=0)   # ROI der., estímulo izq.
            L_right = ev_r.data[iL].mean(axis=0)  # ROI izq., estímulo der.
            R_right = ev_r.data[iR].mean(axis=0)  # ROI der., estímulo der.

            contra = (R_left + L_right) / 2.0
            ipsi = (L_left + R_right) / 2.0
            out[label].append((contra - ipsi) * 1e6)  # V -> uV

        kept.append(subj)

    li = {k: np.asarray(v) for k, v in out.items()}
    if verbose:
        n = len(kept)
        print(f"Lateralización calculada sobre {n} participantes, "
              f"{len(times)} puntos temporales")
        print(f"  contra = ({'+'.join(right_roi)} | izq, {'+'.join(left_roi)} | der) / 2")
        print(f"  ipsi   = ({'+'.join(left_roi)} | izq, {'+'.join(right_roi)} | der) / 2")
    return li, times, kept


# ---------------------------------------------------------------- estadística
def cluster_test_1d(data, times, n_permutations=10000, alpha_form=0.05,
                    alpha_cluster=0.05, tail=0, seed=42, label="",
                    tmin=None, tmax=None, verbose=True):
    """
    Prueba de permutación por clústeres de una muestra, sobre el tiempo.

    Parameters
    ----------
    data : np.ndarray (n_subjects, n_times)
    times : np.ndarray
    alpha_form : umbral de formación de clúster (bilateral si tail=0).
    alpha_cluster : umbral a nivel de clúster.
    tmin, tmax : restringe la ventana analizada antes de agrupar.

    Returns
    -------
    dict con t_obs, times, clusters (lista de dicts), n, threshold
    """
    from mne.stats import permutation_cluster_1samp_test
    from scipy.stats import t as t_dist

    data = np.asarray(data)
    mask = np.ones(len(times), bool)
    if tmin is not None:
        mask &= times >= tmin
    if tmax is not None:
        mask &= times <= tmax
    X, tt = data[:, mask], times[mask]

    n = X.shape[0]
    p_form = alpha_form / 2 if tail == 0 else alpha_form
    threshold = t_dist.ppf(1 - p_form, df=n - 1)
    if tail == -1:
        threshold = -threshold

    if verbose:
        print(f"Clusters 1-D: {label}")
        print(f"  n = {n}, umbral t = {threshold:.3f} (alfa formación {alpha_form}, "
              f"cola {tail}), {n_permutations} permutaciones")

    t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        X, n_permutations=n_permutations, threshold=threshold, tail=tail,
        adjacency=None, seed=seed, out_type="mask", verbose="ERROR",
    )

    found = []
    for cl, p in zip(clusters, cluster_pv):
        # MNE devuelve cada clúster como máscara booleana, como tupla de una
        # máscara, o como tupla de un slice, según la versión y la forma de la
        # entrada. Normalizamos a índices enteros.
        if isinstance(cl, tuple):
            cl = cl[0]
        if isinstance(cl, slice):
            idx = np.arange(len(t_obs))[cl]
        else:
            idx = np.where(np.asarray(cl))[0]
        if idx.size == 0:
            continue
        found.append({
            "tmin": float(tt[idx[0]]),
            "tmax": float(tt[idx[-1]]),
            "mass": float(t_obs[idx].sum()),
            "p": float(p),
            "significant": bool(p < alpha_cluster),
            "mean_uv": float(X[:, idx].mean()),
        })
    found.sort(key=lambda c: c["p"])

    if verbose:
        sig = [c for c in found if c["significant"]]
        print(f"  {len(found)} clúster(es), {len(sig)} significativo(s) "
              f"(p < {alpha_cluster})")

    return {"t_obs": t_obs, "times": tt, "clusters": found,
            "n": n, "threshold": float(threshold), "label": label}


def describe_clusters_1d(results):
    """Imprime los clústeres de cluster_test_1d en una tabla legible."""
    print(f"\n{'=' * 64}\n{results['label']}   (n = {results['n']})\n{'=' * 64}")
    if not results["clusters"]:
        print("  Sin clústeres.")
        return
    print(f"  {'ventana (s)':<20}{'media uV':>10}{'masa t':>10}{'p':>10}")
    for c in results["clusters"]:
        star = " *" if c["significant"] else ""
        print(f"  {c['tmin']:+.3f} a {c['tmax']:+.3f}   "
              f"{c['mean_uv']:>10.3f}{c['mass']:>10.1f}{c['p']:>10.4f}{star}")


def window_mean_test(data, times, tmin, tmax):
    """t de una muestra sobre la media de una ventana. Devuelve dict."""
    from scipy import stats

    m = (times >= tmin) & (times <= tmax)
    x = np.asarray(data)[:, m].mean(axis=1)
    t, p = stats.ttest_1samp(x, 0)
    return {"tmin": tmin, "tmax": tmax, "n": len(x), "mean_uv": float(x.mean()),
            "sd_uv": float(x.std(ddof=1)), "t": float(t), "p": float(p),
            "dz": float(x.mean() / x.std(ddof=1)), "values": x}
