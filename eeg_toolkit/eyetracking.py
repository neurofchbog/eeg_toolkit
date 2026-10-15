# -*- coding: utf-8 -*-
"""
eyetracking.py — lectura de archivos .asc de EyeLink y medidas de mirada.

Pensado para registros simultáneos con EEG: los mensajes del .asc llevan los mismos
códigos que los disparadores del EEG, de modo que los ensayos se pueden emparejar uno
a uno entre las dos señales.

Qué hace y por qué
------------------
`parse_asc` devuelve, por participante, la tabla de ensayos, las muestras de mirada
epocadas a un evento, y los eventos de sacada, fijación y parpadeo que el propio
EyeLink calculó. Sobre eso:

* `baseline_correct` resta a cada ensayo su posición media en una ventana previa. Es
  imprescindible cuando la corrección de deriva entre ensayos deja un desfase residual
  apreciable: medir respecto del centro físico de la pantalla mezcla ese desfase con el
  efecto.
* `detect_microsaccades` implementa Engbert y Kliegl (2003) sobre las muestras crudas,
  que es el procedimiento habitual en la literatura de atención encubierta y detecta
  sacadas pequeñas con más sensibilidad que el algoritmo por defecto del EyeLink.
"""
from __future__ import annotations

import re
import numpy as np

__all__ = ["parse_asc", "epoch_samples", "baseline_correct",
           "detect_microsaccades", "saccade_rate_timecourse", "px2deg"]


# ----------------------------------------------------------------- utilidades
def px2deg(px, screen_w_px=1920, screen_w_cm=53.0, distance_cm=60.0):
    """Convierte píxeles a grados de ángulo visual (aproximación de ángulo pequeño)."""
    cm = np.asarray(px, dtype=float) * (screen_w_cm / screen_w_px)
    return np.degrees(np.arctan2(cm, distance_cm))


_RE_TRIAL = re.compile(r"TRIAL_START_BLOCK_bloque_(simbolico|espacial)_(\d+)\.csv_TRIAL_(\d+)")
_RE_CUE_SYM = re.compile(r"TRIGGER_CUE_SYMBOLIC_TGTnum(\d+)_DISpos(\d+)")
_RE_CUE_SPA = re.compile(r"TRIGGER_CUE_SPATIAL_TGTpos(\d+)_DISnum(\d+)")
_RE_ARRAY = re.compile(r"TRIGGER_ARRAY_START_STR_([A-Z]{4})")
_RE_PROBE = re.compile(r"TRIGGER_40_PROBE_(\w+)")
_RE_RESP = re.compile(r"TRIGGER_RESP_(\d)")
_RE_DRIFT = re.compile(r"OFFSET ([0-9.]+) deg")


def parse_asc(path, verbose=False):
    """
    Lee un .asc de EyeLink.

    Returns
    -------
    dict con:
      samples   : (n, 4) float32 — tiempo (ms), x, y, pupila. NaN donde falta.
      trials    : lista de dicts, uno por ensayo, con tiempos absolutos y condición.
      saccades  : (m, 10) — trial, t_ini, t_fin, dur, x_ini, y_ini, x_fin, y_fin, ampl_deg, vel_pico
      blinks    : (k, 3)  — trial, t_ini, t_fin
      fixations : (j, 6)  — trial, t_ini, t_fin, dur, x, y
      sfreq     : float
    """
    times, xs, ys, ps = [], [], [], []
    trials, sacc, blink, fixs = [], [], [], []
    sfreq = None
    pending_drift = np.nan
    cur = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            c = line[0]
            if c.isdigit():                                   # muestra
                p = line.split()
                if len(p) < 4:
                    continue
                try:
                    times.append(float(p[0]))
                except ValueError:
                    continue
                def _v(s):
                    return np.nan if s == "." else float(s)
                try:
                    xs.append(_v(p[1])); ys.append(_v(p[2])); ps.append(_v(p[3]))
                except ValueError:
                    xs.append(np.nan); ys.append(np.nan); ps.append(np.nan)
                continue

            if line.startswith("MSG"):
                p = line.split(None, 2)
                if len(p) < 3:
                    continue
                t, msg = float(p[1]), p[2]

                if sfreq is None and "RECCFG" in msg:
                    m = re.search(r"RECCFG \w+ (\d+)", msg)
                    if m:
                        sfreq = float(m.group(1))
                m = _RE_DRIFT.search(msg)
                if m:
                    pending_drift = float(m.group(1))
                    continue

                m = _RE_TRIAL.search(msg)
                if m:
                    cur = {"block_type": m.group(1), "block_number": int(m.group(2)),
                           "trial_number": int(m.group(3)), "t_start": t,
                           "drift_deg": pending_drift}
                    trials.append(cur)
                    pending_drift = np.nan
                    continue
                if cur is None:
                    continue

                if "TRIGGER_50_FIXATION" in msg:   cur["t_fix"] = t
                elif "TRIGGER_DELAY_1" in msg:     cur["t_delay1"] = t
                elif "TRIGGER_DELAY_2" in msg:     cur["t_delay2"] = t
                elif "TRIAL_END" in msg:           cur["t_end"] = t
                else:
                    m = _RE_ARRAY.search(msg)
                    if m: cur["t_array"] = t; cur["array"] = m.group(1); continue
                    m = _RE_CUE_SYM.search(msg)
                    if m:
                        cur.update(t_cue=t, cue_modality="symbolic",
                                   digit=int(m.group(1)), position=int(m.group(2)))
                        continue
                    m = _RE_CUE_SPA.search(msg)
                    if m:
                        cur.update(t_cue=t, cue_modality="spatial",
                                   position=int(m.group(1)), digit=int(m.group(2)))
                        continue
                    m = _RE_PROBE.search(msg)
                    if m: cur["t_probe"] = t; cur["validez"] = m.group(1); continue
                    m = _RE_RESP.search(msg)
                    if m: cur["acc"] = int(m.group(1)); cur["t_resp"] = t; continue
                continue

            ti = len(trials) - 1
            if line.startswith("ESACC"):
                p = line.split()
                try:
                    sacc.append([ti, float(p[2]), float(p[3]), float(p[4]),
                                 float(p[5]), float(p[6]), float(p[7]), float(p[8]),
                                 float(p[9]), float(p[10])])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("EBLINK"):
                p = line.split()
                try:
                    blink.append([ti, float(p[2]), float(p[3])])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("EFIX"):
                p = line.split()
                try:
                    fixs.append([ti, float(p[2]), float(p[3]), float(p[4]),
                                 float(p[5]), float(p[6])])
                except (ValueError, IndexError):
                    pass

    samples = np.column_stack([np.asarray(times, dtype=np.float64),
                               np.asarray(xs, dtype=np.float32),
                               np.asarray(ys, dtype=np.float32),
                               np.asarray(ps, dtype=np.float32)])
    out = {"samples": samples, "trials": trials,
           "saccades": np.asarray(sacc, dtype=np.float64).reshape(-1, 10),
           "blinks": np.asarray(blink, dtype=np.float64).reshape(-1, 3),
           "fixations": np.asarray(fixs, dtype=np.float64).reshape(-1, 6),
           "sfreq": sfreq or 500.0}
    if verbose:
        print(f"  {len(trials)} ensayos, {len(samples):,} muestras, "
              f"{len(sacc)} sacadas, {len(blink)} parpadeos, {out['sfreq']:.0f} Hz")
    return out


def epoch_samples(parsed, lock="t_cue", tmin=-0.5, tmax=2.0,
                  blink_pad_ms=50.0, mark_offscreen=True,
                  screen=(1920, 1080)):
    """
    Epoca las muestras alrededor de un evento de cada ensayo.

    Las muestras dentro de un parpadeo (±`blink_pad_ms`) y, si se pide, las que caen
    fuera de la pantalla, se marcan como NaN: en un parpadeo el rastreador informa
    coordenadas absurdas antes de declarar la pérdida.

    Returns
    -------
    data : (n_trials, n_times, 3) float32 — x, y, pupila
    times : (n_times,) en segundos, relativo al evento
    """
    s = parsed["samples"]
    sf = parsed["sfreq"]
    t_all, x, y, p = s[:, 0], s[:, 1].copy(), s[:, 2].copy(), s[:, 3].copy()

    bad = np.zeros(len(t_all), bool)
    for _, b0, b1 in parsed["blinks"]:
        bad |= (t_all >= b0 - blink_pad_ms) & (t_all <= b1 + blink_pad_ms)
    if mark_offscreen:
        bad |= (x < 0) | (x > screen[0]) | (y < 0) | (y > screen[1])
    bad |= ~np.isfinite(x) | ~np.isfinite(y)
    x[bad] = np.nan; y[bad] = np.nan; p[bad] = np.nan
    p[p <= 0] = np.nan

    step = 1000.0 / sf
    offs = np.arange(round(tmin * 1000), round(tmax * 1000) + step / 2, step)
    times = offs / 1000.0
    n = len(parsed["trials"])
    data = np.full((n, len(offs), 3), np.nan, dtype=np.float32)

    order = np.argsort(t_all)
    t_sorted = t_all[order]
    for i, tr in enumerate(parsed["trials"]):
        t0 = tr.get(lock, np.nan)
        if not np.isfinite(t0):
            continue
        idx = np.searchsorted(t_sorted, t0 + offs)
        idx = np.clip(idx, 0, len(t_sorted) - 1)
        ok = np.abs(t_sorted[idx] - (t0 + offs)) <= step
        sel = order[idx]
        data[i, ok, 0] = x[sel][ok]
        data[i, ok, 1] = y[sel][ok]
        data[i, ok, 2] = p[sel][ok]
    return data, times


def baseline_correct(data, times, baseline=(-0.2, 0.0), channels=(0, 1)):
    """Resta a cada ensayo su media en la ventana de línea base. Modifica una copia."""
    out = np.array(data, dtype=np.float32, copy=True)
    m = (times >= baseline[0]) & (times <= baseline[1])
    for c in channels:
        base = np.nanmean(out[:, m, c], axis=1)
        out[:, :, c] -= base[:, None]
    return out


def detect_microsaccades(data, times, sfreq, vfac=5.0, min_dur=6,
                         max_ampl_deg=1.0, screen_w_px=1920,
                         screen_w_cm=53.0, distance_cm=60.0):
    """
    Detección de (micro)sacadas según Engbert y Kliegl (2003), por ensayo.

    El umbral de velocidad es `vfac` veces una estimación robusta (mediana) de la
    desviación típica de la velocidad, calculada dentro de cada ensayo y para cada eje.

    Returns
    -------
    lista (una entrada por ensayo) de arrays (k, 5):
        índice de inicio, índice de fin, amplitud en grados, ángulo en radianes,
        velocidad de pico en grados/s
    """
    deg_per_px = np.degrees(np.arctan2(screen_w_cm / screen_w_px, distance_cm))
    out = []
    for tr in range(data.shape[0]):
        xy = data[tr, :, :2].astype(float) * deg_per_px
        v = np.full_like(xy, np.nan)
        v[2:-2] = (xy[4:] + xy[3:-1] - xy[1:-3] - xy[:-4]) * (sfreq / 6.0)
        good = np.isfinite(v).all(axis=1)
        if good.sum() < 20:
            out.append(np.empty((0, 5))); continue
        med = np.nanmedian(v[good], axis=0)
        sd = np.sqrt(np.nanmedian((v[good] - med) ** 2, axis=0))
        sd[sd < 1e-6] = 1e-6
        rad = vfac * sd
        test = (v[:, 0] / rad[0]) ** 2 + (v[:, 1] / rad[1]) ** 2
        over = np.where(np.nan_to_num(test) > 1, True, False) & good

        ev, i = [], 0
        while i < len(over):
            if over[i]:
                j = i
                while j + 1 < len(over) and over[j + 1]:
                    j += 1
                if j - i + 1 >= min_dur:
                    d = xy[j] - xy[i]
                    ampl = float(np.hypot(*d))
                    if ampl <= max_ampl_deg:
                        ev.append([i, j, ampl, float(np.arctan2(d[1], d[0])),
                                   float(np.nanmax(np.hypot(v[i:j + 1, 0], v[i:j + 1, 1])))])
                i = j + 1
            else:
                i += 1
        out.append(np.asarray(ev, dtype=float).reshape(-1, 5))
    return out


def saccade_rate_timecourse(events, n_times, sfreq, sign=None, smooth_ms=50.0):
    """
    Tasa de sacadas por segundo a lo largo del tiempo.

    `sign` : array (n_trials,) con +1 o −1. Si se pasa, devuelve la tasa de sacadas
    cuya componente horizontal va en el sentido indicado menos la del sentido opuesto
    (es decir, hacia el lado de interés menos en contra).
    """
    n_tr = len(events)
    toward = np.zeros((n_tr, n_times))
    away = np.zeros((n_tr, n_times))
    for tr, ev in enumerate(events):
        for i0, i1, ampl, ang, _ in ev:
            i0 = int(i0)
            if i0 >= n_times:
                continue
            horiz = np.cos(ang)
            if sign is None:
                toward[tr, i0] += 1
            elif horiz * sign[tr] > 0:
                toward[tr, i0] += 1
            else:
                away[tr, i0] += 1
    k = max(1, int(round(smooth_ms / 1000.0 * sfreq)))
    win = np.ones(k) / k
    sm = lambda a: np.apply_along_axis(lambda r: np.convolve(r, win, mode="same"), 1, a) * sfreq
    return (sm(toward), sm(away)) if sign is not None else sm(toward)
