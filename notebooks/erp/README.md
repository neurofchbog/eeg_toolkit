# ERP Analysis Notebooks

## Overview

This folder contains the ERP (Event-Related Potential) analysis pipeline for the `eeg_toolkit`. It is organized into two independent analysis branches — **paired** and **factorial** — each with its own YAML config, evoked files, and notebook sequence.

Both branches read from the same preprocessed data (clean, rejected epochs produced by the preprocessing pipeline) but compute separate evoked averages, grand averages, and statistics. They coexist on disk thanks to an `analysis_name` suffix in the YAML config.

## Folder Structure

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

## Config Files

Both branches depend on the main toolkit config plus their own ERP config:

| Config | Location | Purpose |
|--------|----------|---------|
| `eye_eeg_simul.yaml` | `configs/` | Main pipeline config (paths, subjects, preprocessing) |
| `erp_analysis.yaml` | `configs/` | Paired analysis: spatial vs symbolic, equalized trial counts |
| `erp_position.yaml` | `configs/` | Factorial analysis: 16 position-level conditions, no equalization |

## Paired Branch

**Question**: Do spatial and symbolic retro-cues differ in their cue-locked ERP signatures?

**Conditions**: 2 (spatial, symbolic) — defined by aggregating all raw cue codes per block type.

**Event codes**:
- Trial window: `1020` (spatial memory array), `1120` (symbolic memory array)
- Cue window: 12 spatial codes (12–43), 12 symbolic codes (112–143)

**Config**: `erp_analysis.yaml` — `equalize_counts: true`, both trial and cue windows.

**Output files**: `subj*_cue_ave.fif`, `subj*_trial_ave.fif`, `grand_average_cue_ave.fif`, `grand_average_trial_ave.fif`

### Notebook sequence

**01_compute_evokeds.ipynb**
- Loads `erp_analysis.yaml`
- Computes per-subject condition averages (spatial, symbolic) and contrasts (spatial_vs_symbolic) for both trial and cue windows
- Equalizes trial counts across conditions before averaging
- Computes group-level grand averages
- Saves trial-count logs as JSON

**02_explore.ipynb**
- Exploratory visualization of grand averages (requires Qt5 backend)
- Butterfly + topomap joint plots to identify peak latencies
- Topographic maps at sampled timepoints to track scalp distribution
- Channel-by-channel condition overlays with 95% CI
- ROI-based condition overlays to identify regions of interest
- Goal: decide on ROIs and time windows before running confirmatory stats

**03_analysis.ipynb**
- Defines ROIs and time windows based on exploration
- Extracts mean amplitudes per subject per condition per ROI per time window
- Runs paired t-tests and Wilcoxon signed-rank tests with FDR correction
- Reports Cohen's d effect sizes
- Generates publication-ready multi-panel figure (SEM bands, significant time windows shaded)
- Exports stats CSV

## Factorial Branch

**Question**: Does the lateralization of cue-locked ERPs depend on cue type (spatial/symbolic), information role (target/distractor), and spatial side (left/right)?

**Conditions**: 16 position-level conditions (4 target positions × 2 cue types + 4 distractor positions × 2 cue types), collapsed inline to 8 left/right groups for analysis.

**Event codes** (cue window only):
- Spatial blocks: cue code = `digit*10 + position` (e.g., 12 = digit 1 at position 2)
  - Position = target dimension, digit = distractor dimension
- Symbolic blocks: cue code = `digit*10 + position + 100`
  - Digit = target dimension, position = distractor dimension

**Config**: `erp_position.yaml` — `analysis_name: position`, `equalize_counts: false`, cue window only.

**Output files**: `subj*_cue_position_ave.fif`, `grand_average_cue_position_ave.fif`

### Notebook sequence

**01_compute_evokeds.ipynb**
- Loads `erp_position.yaml`
- Computes per-subject averages for all 16 position-level conditions
- No equalization (too many conditions, ~30 trials each)
- Computes group-level grand averages

**02_explore.ipynb**
- Collapses position-level evokeds into left/right groupings inline:
  - pos1 + pos2 → left, pos3 + pos4 → right
- Visualizes left vs right overlays at posterior ROIs, separately for each cue type × role combination
- Goal: check whether crossed lateralization is present

**03_analysis.ipynb**
- Extracts mean amplitudes for all 16 conditions
- Collapses to 8 left/right groups using pandas
- Parses condition names into 3 factors: `cue_type`, `role`, `side`
- Runs 2×2×2 repeated-measures ANOVA (cue_type × role × side) per ROI per time window, with FDR correction
- Follows up with paired left-vs-right contrasts per cue_type × role, FDR-corrected
- Exports long-format CSV for external software (R, JASP, jamovi, SPSS)

## Toolkit Modules Used

| Module | Purpose |
|--------|---------|
| `evoked.py` | Per-subject averaging, contrasts, grand averages |
| `erp_explore.py` | Butterfly plots, topomaps, channel/ROI overlays |
| `erp_stats.py` | Amplitude extraction, paired tests, ANOVA, publication figures |

All modules are paradigm-agnostic. Experiment-specific details live entirely in the YAML configs and inline notebook definitions.

## How to Adapt for a New Experiment

1. **Create a new YAML** in `configs/` with your conditions, windows, and baseline. Add `analysis_name` if running multiple analyses on the same data.

2. **Copy one of the notebook branches** (paired or factorial) depending on your design.

3. **Update the YAML path** in the setup cell of each notebook (`load_config('path/to/your_config.yaml')`).

4. **Define your ROIs and time windows** in notebook 03 based on what you observe in notebook 02.

5. **Run the notebooks in order** (01 → 02 → 03). Each notebook is independent after 01 has produced the evoked files.

No toolkit code needs to change. The same modules handle any number of conditions, any event codes, any channel montage, and any time windows.

## Dependencies

- Python >= 3.9
- MNE-Python >= 1.5
- statsmodels (for ANOVA and FDR correction)
- scipy (for paired tests)
- pandas, numpy, matplotlib
- PyQt5 (for interactive exploration in notebook 02)
