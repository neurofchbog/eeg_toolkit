# ERP Analysis Notebooks

## Overview

This folder contains the ERP (Event-Related Potential) analysis pipeline for
the `eeg_toolkit`. It is organized into two independent analysis branches —
**paired** and **factorial** — each with its own YAML config, evoked files,
and notebook sequence.

Both branches read from the same preprocessed data (clean, rejected epochs
produced by the preprocessing pipeline) but compute separate evoked averages,
grand averages, and statistics. They coexist on disk thanks to an
`analysis_name` suffix in the YAML config.

## Folder structure

```
notebooks/erp/
├── README.md                 ← this file
├── paired/
│   ├── 01_compute_evokeds.ipynb
│   ├── 02_explore.ipynb
│   └── 03_analysis.ipynb
└── factorial/
    ├── 01_compute_evokeds.ipynb
    ├── 02_explore.ipynb
    └── 03_analysis.ipynb
```

## Config files

Both branches depend on the main toolkit config plus their own ERP config:

| Config | Location | Purpose |
|--------|----------|---------|
| `<experiment>.yaml` | `configs/` | Main pipeline config (paths, subjects, preprocessing) |
| ERP config (paired) | `configs/` | Conditions to compare, time windows, baseline, equalization |
| ERP config (factorial) | `configs/` | All condition levels, factor structure, no equalization |

## Paired branch

**Use case**: compare two (or more) conditions — e.g., condition A vs B —
on a single epoching window.

**Typical workflow**:
- Define two condition groups by aggregating event codes in the ERP config
- Equalize trial counts across conditions before averaging
- Compute per-subject averages and group grand averages

### Notebook sequence

**01_compute_evokeds.ipynb**
- Loads the ERP config
- Computes per-subject condition averages and contrasts
- Optionally equalizes trial counts
- Computes group-level grand averages
- Saves trial-count logs as JSON

**02_explore.ipynb**
- Exploratory visualization of grand averages (requires Qt5 backend)
- Butterfly + topomap joint plots to identify peak latencies
- Topographic maps at sampled timepoints
- Channel-by-channel and ROI-based condition overlays with 95% CI
- Goal: identify ROIs and time windows before confirmatory statistics

**03_analysis.ipynb**
- Defines ROIs and time windows based on exploration
- Extracts mean amplitudes per subject / condition / ROI / time window
- Runs paired t-tests and Wilcoxon signed-rank tests with FDR correction
- Reports Cohen's d effect sizes
- Generates publication-ready figures (SEM bands, significant windows shaded)
- Exports stats CSV

## Factorial branch

**Use case**: analyze a multi-factor design — e.g., cue type × spatial side —
where conditions are defined by crossing multiple factors.

**Typical workflow**:
- Define all condition levels in the ERP config (no equalization needed)
- Compute per-subject averages for every condition
- Collapse or contrast conditions inline in the notebooks

### Notebook sequence

**01_compute_evokeds.ipynb**
- Loads the ERP config
- Computes per-subject averages for all condition levels
- Computes group-level grand averages

**02_explore.ipynb**
- Collapses condition levels into factor-level groupings inline
- Visualizes condition overlays at relevant ROIs
- Goal: check for expected effects before running the ANOVA

**03_analysis.ipynb**
- Extracts mean amplitudes for all conditions
- Parses condition names into factors using pandas
- Runs repeated-measures ANOVA per ROI per time window with FDR correction
- Follows up with planned pairwise contrasts
- Exports long-format CSV for external software (R, JASP, jamovi, SPSS)

## Toolkit modules used

| Module | Purpose |
|--------|---------|
| `evoked.py` | Per-subject averaging, contrasts, grand averages |
| `erp_explore.py` | Butterfly plots, topomaps, channel/ROI overlays |
| `erp_stats.py` | Amplitude extraction, paired tests, ANOVA, publication figures |

All modules are paradigm-agnostic. Experiment-specific details live entirely
in the YAML configs and inline notebook definitions.

## How to adapt for a new experiment

1. **Create a new YAML** in `configs/` with your conditions, windows, and
   baseline. Add `analysis_name` if running multiple analyses on the same data.

2. **Copy the appropriate branch** (paired or factorial) depending on your
   design.

3. **Update the config path** in the setup cell of each notebook
   (`load_config('configs/your_experiment.yaml')`).

4. **Define your ROIs and time windows** in notebook 03 based on what you
   observe in notebook 02.

5. **Run the notebooks in order** (01 → 02 → 03). Each notebook is
   independent after 01 has produced the evoked files.

No toolkit code needs to change. The same modules handle any number of
conditions, any event codes, any channel montage, and any time windows.

## Dependencies

- Python >= 3.9
- MNE-Python >= 1.5
- statsmodels (for ANOVA and FDR correction)
- scipy (for paired tests)
- pandas, numpy, matplotlib
- PyQt5 (for interactive exploration in notebook 02)
