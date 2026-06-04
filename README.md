# EEG Toolkit

A modular, config-driven toolkit for preprocessing and analyzing EEG data.
Every experiment-specific parameter lives in a YAML config file — the Python
code never needs to change between experiments or paradigms.

## Features

- XDF → FIF conversion (Lab Recorder recordings)
- Preprocessing: filtering, montage assignment, resampling, EOG channel labeling
- Automated bad-channel detection with manual inspection support
- Flexible event recoding system (rule-based, defined in YAML)
- Multi-window epoching (define as many time-locked windows as needed)
- ICA artifact rejection with ICLabel auto-labeling
- Trial-level artifact rejection with cross-window propagation
- ERP analysis: evoked averages, grand averages, statistics, publication figures
- Time-frequency analysis (TFR): Morlet wavelets, cluster-based permutation tests
- MVPA: temporal decoding, cross-decoding
- RSA: representational dissimilarity matrices
- HTML preprocessing reports per subject

## Folder structure

```
eeg_toolkit/
├── configs/                        # YAML config files (one per experiment/analysis)
│   ├── README.md                   # Decision guide: which config to use
│   ├── template.yaml               # Pipeline config template (start here)
│   ├── template_erp_paired.yaml    # ERP: 2-condition comparison template
│   ├── template_erp_factorial.yaml # ERP: multi-factor design template
│   └── template_tfr.yaml           # Time-frequency analysis template
├── eeg_toolkit/               # Importable Python package
│   ├── config.py              # YAML loader (dot-accessible config object)
│   ├── io.py                  # Subject discovery, path management, status tracking
│   ├── raw2fif.py             # XDF → FIF conversion
│   ├── preprocessing.py       # Filtering, montage, resampling
│   ├── bad_channels.py        # Bad-channel detection and inspection
│   ├── event_codes.py         # Rule-based event recoding
│   ├── epoching.py            # Multi-window epoching
│   ├── ica.py                 # ICA fitting, ICLabel labeling, inspection, application
│   ├── artifacts.py           # Trial rejection and cross-window propagation
│   ├── evoked.py              # Per-subject averages and grand averages
│   ├── erp_explore.py         # ERP visualization
│   ├── erp_stats.py           # ERP statistics and publication figures
│   ├── tfr.py                 # Time-frequency analysis
│   ├── tfr_stats.py           # TFR cluster-based permutation tests
│   ├── mvpa.py                # Temporal decoding and cross-decoding
│   └── rsa.py                 # Representational similarity analysis
├── notebooks/                 # Jupyter notebooks, one per pipeline step
│   ├── preprocessing/         # Steps 01–07: raw → clean epochs
│   ├── erp/                   # ERP analysis (paired and factorial branches)
│   ├── tfr/                   # Time-frequency analysis
│   ├── mvpa/                  # Decoding analysis
│   └── rsa/                   # RSA analysis
├── docs/                      # Experiment-specific documentation
├── setup.py                   # Package install metadata
└── requirements.txt           # Pinned dependencies
```

Data lives **outside** the toolkit folder and is referenced via paths in the
config file. Raw recordings are never modified.

## Installation

```bash
conda create -n eeg_toolkit python=3.11 -y
conda activate eeg_toolkit
pip install -r requirements.txt
```

Or for a development install:

```bash
pip install -e .
```

## Quick start

```python
from eeg_toolkit import load_config, find_subjects

cfg = load_config('configs/your_experiment.yaml')
subjects = find_subjects(cfg)
print(subjects)
```

## How it works

### 1. Create a config file

Copy `configs/template.yaml` and fill in your experiment's parameters:
paths to your data, preprocessing settings, event codes, epoching windows,
ICA parameters, and analysis options.

```bash
cp configs/template.yaml configs/my_experiment.yaml
# edit configs/my_experiment.yaml
```

### 2. Run the notebooks in order

The `notebooks/preprocessing/` folder walks through every preprocessing step:

```
01_raw2fif.ipynb          → convert raw recordings to FIF
02_preprocessing.ipynb    → filter, montage, resample
03_bad_channels.ipynb     → detect and review bad channels
04_event_codes.ipynb      → recode events per YAML rules
05_epoching.ipynb         → create epoching windows
06_ica.ipynb              → fit ICA, auto-label, inspect, apply
07_trial_rejection.ipynb  → reject bad trials, propagate across windows
```

Then run the analysis notebooks for your analysis type (ERP, TFR, MVPA, RSA).

### 3. Change experiments

To apply the same pipeline to a different experiment, create a new YAML config.
The Python code and notebooks do not change.

## Configuration

All experiment-specific parameters are in the YAML config. Key sections:

| Section | What it controls |
|---|---|
| `paths` | Raw data location, analysis output folder |
| `excluded_subjects` | Subjects to skip in all analyses |
| `preprocessing` | Filter cutoffs, resample frequency, montage, EOG channels |
| `event_rules` | Rules to recode raw event codes into analysis-ready codes |
| `epoching_windows` | One or more time-locked windows (each saves a separate FIF) |
| `ica` | Number of components, method, ICLabel rejection thresholds |
| `analysis.artifacts` | Peak-to-peak threshold, cross-window rejection propagation |
| `reports` | HTML report generation |

See `configs/template.yaml` for a fully annotated pipeline config example, and `configs/README.md` for a guide on which config template to use for each analysis type.

## Documentation

Experiment-specific documentation (hardware setup, paradigm description,
event code schemes, behavioral data format) lives in `docs/`.

## Requirements

- Python >= 3.9
- MNE-Python >= 1.5
- See `requirements.txt` for the full pinned dependency list
