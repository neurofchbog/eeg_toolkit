# Experiment: Simultaneous EEG + Eye-Tracking Working Memory Paradigm

This document describes the specific experiment for which the
`eye_eeg_simul.yaml` config was created. It is intended as a reference for
anyone working with this dataset.

## Acquisition setup

| Property | Value |
|---|---|
| EEG system | Brain Products ActiCHamp |
| Channels | 32 (10-20 standard layout) |
| Sample rate | 500 Hz |
| Recording software | Lab Recorder (XDF format) |
| EEG → trigger transmission | Serial port (`COM4`, 115200 baud, `WRITE <code>` command) |
| Eye-tracker | EyeLink (H3 calibration) |
| Behavioral software | PsychoPy |

Channel layout (acquisition order): Fp1, Fz, F3, F7, FT9, FC5, FC1, C3, T7,
TP9, CP5, CP1, Pz, P3, P7, O1, Oz, O2, P4, P8, TP10, CP6, CP2, Cz, C4, T8,
FT10, FC6, FC2, F4, F8, Fp2

XDF stream names (recorded by Lab Recorder):
- EEG stream: `actiCHamp-16090700` (type `EEG`, 500 Hz)
- Marker stream: `actiCHampMarkers-16090700` (type `Markers`, asynchronous)

## Data location

```
Experimentos Eye Tracker/
├── Datos/                         # Raw recordings (READ-ONLY)
│   ├── subj2.xdf                  # EEG + markers (Lab Recorder XDF)
│   ├── subj3.xdf
│   ├── ...
│   ├── Eye_XXX.edf                # Eye-tracker raw
│   └── dual_<id>_S<session>.csv  # Behavioral logs
└── analisis/
    └── analisis_eeg/              # Outputs of this toolkit
        ├── subj2/
        │   ├── subj2_raw.fif
        │   ├── subj2_events_eve.fif
        │   ├── subj2_event_mapping.txt
        │   ├── subj2_preprocessed_raw.fif
        │   ├── subj2_epo.fif
        │   ├── subj2_ica.fif
        │   ├── subj2_clean_epo.fif
        │   └── subj2_report.html
        └── subjects_status.csv
```

## Paradigm description

A dual-task retro-cue working-memory paradigm with simultaneous EEG and
eye-tracking. Each subject performs interleaved blocks of two cue types:

- **Spatial cue (`espacial`)**: a dot appears at one of 4 horizontal positions;
  subjects must retrieve the letter shown at that position. A distracting
  number is shown simultaneously at a different position.
- **Symbolic cue (`simbolico`)**: a number (1–4) is shown at one of 4
  horizontal positions; subjects must retrieve the letter at the position
  indicated by the number (regardless of the number's spatial location).
  The number's position acts as a spatial distractor.

Both cue types deliberately produce **conflict** between symbolic and spatial
information, supporting analysis of how each cue type accesses ordinal
working-memory representations (mental-whiteboard / SPoARC hypotheses).

Trial counts: 12 blocks × ~20 trials = ~240 trials per subject
(120 spatial + 120 symbolic, block order randomized).

## Trial timeline

| Time (s) | Event | Marker code | Constant in script |
|---|---|---|---|
| 0.0 | Fixation onset | `50` | `TRIG_FIJACION` |
| 0.5 | Memory array (4 letters) | `20` | `TRIG_ARRAY_START` |
| 1.0 | Delay 1 (fixation) | `51` | `TRIG_DELAY_1` |
| 2.5 | Cue onset (single cue with conflict) | trial-condition code | (computed) |
| 3.0 | Delay 2 (fixation) | `52` | `TRIG_DELAY_2` |
| 4.5 | Probe letter | `40` | `TRIG_PROBE` |
| 4.5–9.5 | Response window (5 s max) | `90` / `91` / `99` | `TRIG_RESP_*` |

PsychoPy duration constants:
- `DUR_ARRAY = 0.5`
- `DUR_FIJACION = 1.5` (Delay 1)
- `DUR_CLAVE = 0.5` (cue duration)
- `DELAY_POST_CLAVE = 1.5` (Delay 2)

## Cue event-code scheme

The cue code encodes both target identity and distractor location:

```python
code_eeg = (disp_num * 10) + disp_pos + offset_eeg
```

Where:
- `disp_num` = the number displayed in the cue (1–4)
- `disp_pos` = the position of the displayed number (1–4)
- `offset_eeg` = `0` for spatial blocks, `100` for symbolic blocks

| Block | offset | Target | Distractor | Code range |
|---|---|---|---|---|
| Spatial (`espacial`) | 0 | `disp_pos` | `disp_num` | `12`–`43` |
| Symbolic (`simbolico`) | 100 | `disp_num` | `disp_pos` | `112`–`143` |

Because `disp_num ≠ disp_pos` is enforced, codes `11`, `22`, `33`, `44`
and `111`, `122`, `133`, `144` never occur.

### Decoding cue codes

```python
if C >= 100:
    block = 'symbolic'
    target_position = (C - 100) // 10
    distractor_position = (C - 100) % 10
else:
    block = 'spatial'
    distractor_position = C // 10
    target_position = C % 10
```

Examples:
- `134` → symbolic, target=position 3, distractor at position 4
- `34` → spatial, distractor=number 3, target at position 4

## Within-trial event codes (fixed)

| Code | Event | Per-subject count |
|---|---|---|
| `50` | Fixation onset (trial start) | ~240 |
| `20` | Memory array onset | ~240 |
| `51` | Delay 1 onset | ~240 |
| `52` | Delay 2 onset | ~240 |
| `40` | Probe onset | ~240 |

## Response codes

| Code | Meaning |
|---|---|
| `90` | Correct response |
| `91` | Incorrect response |
| `99` | Timeout (no response within 5 s) |

## Behavioral CSV (`dual_<id>_S<session>.csv`)

| Column | Description |
|---|---|
| `letra_pos1`–`letra_pos4` | Letters shown at each of the 4 array positions |
| `clave` | Target position/number (1–4) |
| `letra_probe` | Probe letter |
| `validez` | `valido` if probe matches target letter, else `invalido` |
| `respuesta_correcta` | Correct response (`si`/`no`) |
| `rt` | Reaction time (s) |
| `acc` | Accuracy (1 = correct, 0 = error) |
| `resp_key` | Key pressed (`s`/`n`) |
| `disp_num` | Number shown in cue (1–4) |
| `disp_pos` | Position of cue on screen (1–4) |
| `eeg_code` | Trial-condition code sent to EEG amp |
| `tipo_bloque` | `espacial` or `simbolico` |
| `ID`, `Edad`, `Sexo` | Demographics |
