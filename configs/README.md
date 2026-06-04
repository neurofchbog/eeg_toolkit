# Config Guide

This folder contains all configuration files for the EEG toolkit.
Every experiment-specific parameter lives here — the Python code never changes.

## Config types

There are two kinds of config files:

| Type | Purpose | When to create one |
|---|---|---|
| **Pipeline config** | Controls preprocessing (paths, filtering, epoching, ICA) | Once per experiment or dataset |
| **Analysis config** | Controls a specific analysis on preprocessed data (ERP, TFR) | Once per analysis question |

A typical experiment uses one pipeline config and one or more analysis configs.

## Which template to copy?

```
New experiment or dataset
└── copy template.yaml

Running an ERP analysis?
├── Comparing 2–3 conditions (A vs B)?
│   └── copy template_erp_paired.yaml
└── Multi-factor design (e.g., 2×2 ANOVA)?
    └── copy template_erp_factorial.yaml

Running a time-frequency analysis?
└── copy template_tfr.yaml
```

## Templates

| File | Use for |
|---|---|
| `template.yaml` | New experiment — preprocessing pipeline |
| `template_erp_paired.yaml` | ERP: 2-condition comparison (A vs B) |
| `template_erp_factorial.yaml` | ERP: multi-factor design (e.g., 2×2 ANOVA) |
| `template_tfr.yaml` | Time-frequency analysis (alpha, beta, theta, etc.) |

## How configs relate to each other

A pipeline config and an analysis config are loaded together in the notebooks:

```python
from eeg_toolkit import load_config

cfg     = load_config('configs/my_experiment.yaml')   # pipeline config
erp_cfg = load_config('configs/my_erp_analysis.yaml') # analysis config
```

The pipeline config provides subject paths and preprocessed data locations.
The analysis config defines what to compute on that data.

## Running multiple analyses on the same data

You can run as many analysis configs as you want on the same preprocessed data.
Use the `analysis_name` field in the analysis config to give each one a unique
suffix, so output files don't overwrite each other:

```yaml
erp:
  analysis_name: paired     # outputs: subj*_cue_paired_ave.fif
```

```yaml
erp:
  analysis_name: factorial  # outputs: subj*_cue_factorial_ave.fif
```

## Files in this folder

| File | Description |
|---|---|
| `template.yaml` | Pipeline config template |
| `template_erp_paired.yaml` | ERP paired-branch template |
| `template_erp_factorial.yaml` | ERP factorial-branch template |
| `template_tfr.yaml` | TFR analysis template |

Experiment-specific configs (e.g., `eye_eeg_simul.yaml`) live in their
respective experiment repositories and are not part of the toolkit itself.
