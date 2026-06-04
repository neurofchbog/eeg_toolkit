# EEG Toolkit — Eye-Tracking + EEG Working Memory Paradigm

Modular, reusable toolkit for preprocessing and analyzing EEG data from
a dual-task retro-cue working-memory paradigm with simultaneous EEG and
eye-tracking acquisition.

## Project context

This toolkit was built to streamline EEG analysis across multiple
experiments, replacing ad-hoc Jupyter notebooks with a configurable,
reproducible pipeline. It is paradigm-agnostic at the code level —
every experiment-specific parameter lives in a YAML config file.

## Folder structure

eeg_toolkit/
├── configs/                   # YAML config per experiment/analysis
│   └── eye_eeg_simul.yaml
├── eeg_toolkit/               # Python package (importable code)
│   ├── init.py
│   ├── config.py              # YAML loader
│   ├── io.py                  # subject discovery, paths, status tracking
│   └── ...                    # (more modules as we build them)
├── notebooks/                 # Working notebooks per pipeline step
├── setup.py                   # Package install metadata
└── README.md

Data lives outside the toolkit folder, in:

Experimentos Eye Tracker/
├── Datos/                       # Raw recordings (READ-ONLY)
│   ├── subj2.xdf                # EEG + markers (Lab Recorder XDF)
│   ├── subj3.xdf
│   ├── ...
│   ├── Eye_XXX.edf              # Eye-tracker raw (used in future analysis)
│   └── dual_<id>_S<session>.csv # Behavioral logs (used in future analysis)
└── analisis/
└── analisis_eeg/            # Outputs of THIS toolkit
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

Channel layout (acquisition order):Fp1, Fz, F3, F7, FT9, FC5, FC1, C3, T7, TP9,
CP5, CP1, Pz, P3, P7, O1, Oz, O2, P4, P8,
TP10, CP6, CP2, Cz, C4, T8, FT10, FC6, FC2, F4, F8, Fp2

XDF stream names (recorded by Lab Recorder):
- EEG stream: `actiCHamp-16090700` (type `EEG`, 500 Hz)
- Marker stream: `actiCHampMarkers-16090700` (type `Markers`, asynchronous)

## Paradigm description

A dual-task retro-cue working-memory paradigm with simultaneous EEG and
eye-tracking. Each subject performs interleaved blocks of two cue types:

- **Spatial cue (`espacial`)**: a dot appears at one of 4 horizontal
  positions; subjects must retrieve the letter shown at that position.
  A distracting *number* is shown simultaneously at a different position.
- **Symbolic cue (`simbolico`)**: a number (1–4) is shown at one of 4
  horizontal positions; subjects must retrieve the letter at the position
  indicated by the *number* (regardless of the number's spatial location).
  The number's position acts as a spatial distractor.

Both cue types deliberately produce **conflict** between symbolic and
spatial information, supporting analysis of how each cue type accesses
ordinal working-memory representations (mental-whiteboard / SPoARC
hypotheses).

Trial counts: 12 blocks × ~20 trials = ~240 trials per subject
(120 spatial + 120 symbolic, block order randomized).

### Trial timeline

| Time (s) | Event | Marker code | Constant in script |
|---|---|---|---|
| 0.0 | Fixation onset | `50` | `TRIG_FIJACION` |
| 0.5 | Memory array (4 letters) | `20` | `TRIG_ARRAY_START` |
| 1.0 | Delay 1 (fixation) | `51` | `TRIG_DELAY_1` |
| 2.5 | Cue onset (single cue with conflict) | trial-condition code (see below) | (computed) |
| 3.0 | Delay 2 (fixation) | `52` | `TRIG_DELAY_2` |
| 4.5 | Probe letter | `40` | `TRIG_PROBE` |
| 4.5–9.5 | Response window (5 s max) | `90` (correct), `91` (incorrect), `99` (timeout) | `TRIG_RESP_CORR/INCORR/TIMEOUT` |

Durations are set in the PsychoPy script:
- `DUR_ARRAY = 0.5`
- `DUR_FIJACION = 1.5` (Delay 1)
- `DUR_CLAVE = 0.5` (cue duration)
- `DELAY_POST_CLAVE = 1.5` (Delay 2)

### Cue event-code scheme

The cue code encodes both target identity and distractor location. From
the PsychoPy script (`experiment.py`):

```python
code_eeg = (disp_num * 10) + disp_pos + offset_eeg
```

Where:
- `disp_num` = the *number* displayed in the cue (1–4)
- `disp_pos` = the *position* of the displayed number (1–4)
- `offset_eeg` = `0` for spatial blocks, `100` for symbolic blocks

The two block types differ in **which dimension is the target**:

| Block | offset | Target | Distractor | Code range |
|---|---|---|---|---|
| Spatial (`espacial`) | 0 | `disp_pos` (position is the target) | `disp_num` (random, ≠ target) | `12`–`43` |
| Symbolic (`simbolico`) | 100 | `disp_num` (number is the target) | `disp_pos` (random, ≠ target) | `112`–`143` |

Because `disp_num ≠ disp_pos` is enforced, the codes `11`, `22`, `33`,
`44` and `111`, `122`, `133`, `144` never occur. Each block has 12 unique
cue codes.

### Decoding cue codes for analysis

Given a cue code `C`:

```python
if C >= 100:
    block = 'symbolic'
    target_position = (C - 100) // 10        # disp_num
    distractor_position = (C - 100) % 10     # disp_pos
else:
    block = 'spatial'
    distractor_position = C // 10            # disp_num
    target_position = C % 10                 # disp_pos
```

For example:
- `134` → symbolic, target=position 3, distractor at position 4
- `34` → spatial, distractor=number 3, target at position 4

### Within-trial event codes (fixed)

| Code | Event | Sent | Per-subject count |
|---|---|---|---|
| `50` | Fixation onset (trial start) | every trial | ~240 |
| `20` | Memory array onset | every trial | ~240 |
| `51` | Delay 1 onset (post-array fixation) | every trial | ~240 |
| `52` | Delay 2 onset (post-cue fixation) | every trial | ~240 |
| `40` | Probe onset | every trial | ~240 |

### Response codes

| Code | Meaning |
|---|---|
| `90` | Correct response (`TRIG_RESP_CORR`) |
| `91` | Incorrect response (`TRIG_RESP_INCORR`) |
| `99` | Timeout (no response within 5 s) (`TRIG_TIMEOUT`) |

### Behavioral CSV (`dual_<id>_S<session>.csv`)

| Column | Description |
|---|---|
| `letra_pos1`–`letra_pos4` | Letters shown at each of the 4 array positions |
| `clave` | Target position/number (1–4); meaning depends on block type |
| `letra_probe` | Probe letter |
| `validez` | `valido` if probe matches target letter, else `invalido` |
| `respuesta_correcta` | Correct response (`si`/`no`) |
| `rt` | Reaction time (s) |
| `acc` | Accuracy (1 = correct, 0 = error) |
| `resp_key` | Key pressed by participant (`s`/`n`) |
| `disp_num` | Number shown in cue (1–4) |
| `disp_pos` | Position of cue on screen (1–4) |
| `eeg_code` | Trial-condition code sent to EEG amp |
| `tipo_bloque` | `espacial` or `simbolico` |
| `ID`, `Edad`, `Sexo` | Demographics |

## Pipeline overview

The toolkit is being built incrementally. Status:

- [x] Configuration system (`config.py`)
- [x] Subject discovery and path management (`io.py`)
- [ ] XDF → FIF conversion (`raw2fif.py`)
- [ ] Preprocessing (filter, montage, resample) (`preprocessing.py`)
- [ ] Bad-channel detection (`bad_channels.py`)
- [ ] Epoching (`epoching.py`)
- [ ] ICA fitting + ICLabel auto-rejection (`ica.py`)
- [ ] ERP analysis (`erp.py`)
- [ ] HTML report generation (`reports.py`)

## Installation

From the toolkit root:

```bash
conda create -n eeg_toolkit python=3.11 -y
conda activate eeg_toolkit
pip install -e .
pip install pyxdf
```

## Quick start

```python
from eeg_toolkit import load_config, find_subjects

cfg = load_config('configs/eye_eeg_simul.yaml')
subjects = find_subjects(cfg)
print(subjects)
```

## Configuration

All experiment- and analysis-specific parameters live in
`configs/<config_name>.yaml`. To run a different analysis (e.g., a
cue-locked epoching window), create a new YAML file and load it instead.
The Python code never changes.

See `configs/eye_eeg_simul.yaml` for an annotated example.
