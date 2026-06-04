"""
ERP statistics module.

Three core functions:
  - extract_mean_amplitudes: per-subject mean amplitude per ROI per time window
  - run_paired_tests: paired t-test + Wilcoxon for one or more contrasts,
                      with FDR correction across the whole family
  - run_anova: repeated-measures ANOVA for omnibus testing (N >= 3 conditions)

Plus plot_results: multi-panel publication figure that overlays an arbitrary
number of conditions, with FDR-significant time windows shaded.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import mne

from eeg_toolkit.io import find_subjects
from eeg_toolkit.evoked import (
    get_evoked_path,
    _get_erp_dir,
)


# ============================================================================
# Path helpers
# ============================================================================

def _get_stats_dir(cfg):
    stats_dir = _get_erp_dir(cfg) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    return stats_dir


def get_amplitudes_csv_path(cfg, window_name):
    return _get_stats_dir(cfg) / f"mean_amplitudes_{window_name}.csv"


def get_stats_csv_path(cfg, window_name):
    return _get_stats_dir(cfg) / f"stats_{window_name}.csv"


def get_anova_csv_path(cfg, window_name):
    return _get_stats_dir(cfg) / f"anova_{window_name}.csv"


# ============================================================================
# Mean amplitude extraction (unchanged)
# ============================================================================

def extract_mean_amplitudes(cfg, window_name, rois, time_windows,
                            conditions=None, save_csv=True, verbose=True):
    """
    Extract mean amplitude per subject per condition per ROI per time window.

    Returns a long-format DataFrame.
    """
    subjects = find_subjects(cfg)
    rows = []

    for subject in subjects:
        evo_path = get_evoked_path(cfg, subject, window_name)
        if not evo_path.exists():
            if verbose:
                print(f"   [{subject}] evokeds not found - skipping")
            continue

        subj_evokeds = mne.read_evokeds(evo_path, verbose="WARNING")
        evo_dict = {evo.comment: evo for evo in subj_evokeds}

        if conditions is None:
            conds_to_use = [k for k in evo_dict.keys() if "_vs_" not in k]
        else:
            conds_to_use = conditions

        for cond_name in conds_to_use:
            if cond_name not in evo_dict:
                if verbose:
                    print(f"   [{subject}] condition '{cond_name}' missing - skipping")
                continue
            evo = evo_dict[cond_name]

            for roi_name, channels in rois.items():
                valid_ch = [c for c in channels if c in evo.ch_names]
                if not valid_ch:
                    if verbose:
                        print(f"   [{subject}] ROI '{roi_name}' has no valid "
                              f"channels - skipping")
                    continue

                evo_roi = evo.copy().pick(valid_ch)
                roi_mean = evo_roi.data.mean(axis=0)

                for tw_name, (tmin, tmax) in time_windows.items():
                    mask = (evo.times >= tmin) & (evo.times <= tmax)
                    if not mask.any():
                        if verbose:
                            print(f"   [{subject}] time window '{tw_name}' "
                                  f"({tmin}-{tmax}) outside epoch range")
                        continue
                    mean_amp = roi_mean[mask].mean()

                    rows.append({
                        "subject": subject,
                        "window": window_name,
                        "roi": roi_name,
                        "channels": ",".join(valid_ch),
                        "condition": cond_name,
                        "time_window": tw_name,
                        "tmin": tmin,
                        "tmax": tmax,
                        "mean_amp_uv": mean_amp * 1e6,
                    })

    df = pd.DataFrame(rows)

    if verbose:
        n_subj = df["subject"].nunique()
        print(f"\nExtracted {len(df)} rows from {n_subj} subjects.")
        print(f"   ROIs: {list(rois.keys())}")
        print(f"   Time windows: {list(time_windows.keys())}")
        print(f"   Conditions: {df['condition'].unique().tolist()}")

    if save_csv:
        out_path = get_amplitudes_csv_path(cfg, window_name)
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"   saved: {out_path.name}")

    return df


# ============================================================================
# Paired tests (now accepts list of contrasts; FDR across full family)
# ============================================================================

def run_paired_tests(df, contrasts, cfg=None, window_name=None,
                     save_csv=True, verbose=True):
    """
    Paired t-test + Wilcoxon side-by-side for one or more 2-condition contrasts.
    FDR correction is applied across the entire family of tests.

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_mean_amplitudes.
    contrasts : tuple or list of tuples
        e.g., ('spatial', 'symbolic')                 # single contrast
        or:   [('spatial','symbolic'), ('a','b')]     # multi-contrast
        Each test = mean(cond_a) - mean(cond_b).
    cfg, window_name : optional
        Required if save_csv=True.
    save_csv : bool
    verbose : bool

    Returns
    -------
    pd.DataFrame
        contrast, roi, time_window, tmin, tmax, n,
        mean_diff_uv, sd_diff_uv, cohens_d,
        t, p_t, p_t_fdr, sig_t_fdr,
        w, p_w, p_w_fdr, sig_w_fdr
    """
    # Normalize contrasts to a list of tuples
    if isinstance(contrasts, tuple) and len(contrasts) == 2 \
            and isinstance(contrasts[0], str):
        contrasts = [contrasts]

    pairs = []
    for cond_a, cond_b in contrasts:
        for (roi, tw), group in df.groupby(["roi", "time_window"]):
            wide = group.pivot_table(
                index="subject", columns="condition", values="mean_amp_uv"
            )
            if cond_a not in wide.columns or cond_b not in wide.columns:
                if verbose:
                    print(f"   [{cond_a} vs {cond_b}] {roi}/{tw}: "
                          f"missing condition - skipping")
                continue

            paired = wide[[cond_a, cond_b]].dropna()
            if len(paired) < 3:
                if verbose:
                    print(f"   [{cond_a} vs {cond_b}] {roi}/{tw}: "
                          f"too few paired observations - skipping")
                continue

            a = paired[cond_a].values
            b = paired[cond_b].values
            diff = a - b

            t_stat, p_t = stats.ttest_rel(a, b)
            try:
                w_stat, p_w = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, p_w = np.nan, np.nan

            tmin_val = group["tmin"].iloc[0]
            tmax_val = group["tmax"].iloc[0]

            pairs.append({
                "contrast": f"{cond_a}_vs_{cond_b}",
                "roi": roi,
                "time_window": tw,
                "tmin": tmin_val,
                "tmax": tmax_val,
                "n": len(paired),
                "mean_diff_uv": diff.mean(),
                "sd_diff_uv": diff.std(ddof=1),
                "cohens_d": diff.mean() / diff.std(ddof=1)
                            if diff.std(ddof=1) > 0 else np.nan,
                "t": t_stat,
                "p_t": p_t,
                "w": w_stat,
                "p_w": p_w,
            })

    res = pd.DataFrame(pairs)

    if res.empty:
        if verbose:
            print("No tests could be run.")
        return res

    # FDR across the whole family (all contrasts × ROIs × time windows)
    _, res["p_t_fdr"], _, _ = multipletests(res["p_t"], method="fdr_bh")
    valid_w = res["p_w"].notna()
    res["p_w_fdr"] = np.nan
    if valid_w.any():
        _, res.loc[valid_w, "p_w_fdr"], _, _ = multipletests(
            res.loc[valid_w, "p_w"], method="fdr_bh"
        )

    res["sig_t_fdr"] = res["p_t_fdr"] < 0.05
    res["sig_w_fdr"] = res["p_w_fdr"] < 0.05

    res = res[["contrast", "roi", "time_window", "tmin", "tmax", "n",
               "mean_diff_uv", "sd_diff_uv", "cohens_d",
               "t", "p_t", "p_t_fdr", "sig_t_fdr",
               "w", "p_w", "p_w_fdr", "sig_w_fdr"]]

    if verbose:
        n_contrasts = res["contrast"].nunique()
        print(f"\nPaired tests: {n_contrasts} contrast(s), "
              f"FDR across {len(res)} tests total.\n")
        print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if save_csv and cfg is not None and window_name is not None:
        out_path = get_stats_csv_path(cfg, window_name)
        res.to_csv(out_path, index=False)
        if verbose:
            print(f"\n   saved: {out_path.name}")

    return res


# ============================================================================
# Repeated-measures ANOVA (omnibus test for N >= 3 conditions)
# ============================================================================

def run_anova(df, conditions, cfg=None, window_name=None,
              save_csv=True, verbose=True):
    """
    One-way repeated-measures ANOVA per ROI per time window.

    Tests the null that all `conditions` have equal mean amplitude.
    Uses Greenhouse-Geisser correction for sphericity.
    Applies FDR correction across the family of tests (ROIs × time windows).

    Parameters
    ----------
    df : pd.DataFrame
        Output of extract_mean_amplitudes.
    conditions : list of str
        Conditions to test (must be >= 3 for ANOVA to make sense).
    cfg, window_name : optional
        For saving the CSV.

    Returns
    -------
    pd.DataFrame
        roi, time_window, tmin, tmax, n, df1, df2,
        F, p, p_gg, p_gg_fdr, sig_gg_fdr, eps_gg, eta2_p
    """
    try:
        from statsmodels.stats.anova import AnovaRM
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for run_anova. Install with `pip install statsmodels`."
        ) from e

    if len(conditions) < 2:
        raise ValueError("Need at least 2 conditions for ANOVA.")

    rows = []
    for (roi, tw), group in df.groupby(["roi", "time_window"]):
        sub = group[group["condition"].isin(conditions)].copy()
        wide = sub.pivot_table(index="subject", columns="condition",
                               values="mean_amp_uv")
        wide = wide[conditions].dropna()  # require all conditions present

        if len(wide) < 3:
            if verbose:
                print(f"   [{roi}/{tw}] too few subjects with all "
                      f"conditions - skipping")
            continue

        # Long format for AnovaRM
        long = wide.reset_index().melt(id_vars="subject",
                                        var_name="condition",
                                        value_name="amp")

        try:
            aov = AnovaRM(long, depvar="amp", subject="subject",
                          within=["condition"]).fit()
            row = aov.anova_table.iloc[0]
            F = row["F Value"]
            df1 = row["Num DF"]
            df2 = row["Den DF"]
            p = row["Pr > F"]
        except Exception as e:
            if verbose:
                print(f"   [{roi}/{tw}] ANOVA failed: {e}")
            continue

        # Greenhouse-Geisser via manual computation
        eps_gg, p_gg = _greenhouse_geisser(wide.values, F, df1, df2)

        # Partial eta-squared: SS_effect / (SS_effect + SS_error)
        # For RM-ANOVA from F: eta2_p = (F * df1) / (F * df1 + df2)
        eta2_p = (F * df1) / (F * df1 + df2)

        rows.append({
            "roi": roi,
            "time_window": tw,
            "tmin": group["tmin"].iloc[0],
            "tmax": group["tmax"].iloc[0],
            "n": len(wide),
            "df1": df1,
            "df2": df2,
            "F": F,
            "p": p,
            "p_gg": p_gg,
            "eps_gg": eps_gg,
            "eta2_p": eta2_p,
        })

    res = pd.DataFrame(rows)

    if res.empty:
        if verbose:
            print("No ANOVAs could be run.")
        return res

    _, res["p_gg_fdr"], _, _ = multipletests(res["p_gg"], method="fdr_bh")
    res["sig_gg_fdr"] = res["p_gg_fdr"] < 0.05

    res = res[["roi", "time_window", "tmin", "tmax", "n",
               "df1", "df2", "F", "p", "p_gg", "p_gg_fdr",
               "sig_gg_fdr", "eps_gg", "eta2_p"]]

    if verbose:
        print(f"\nRM-ANOVA across {len(conditions)} conditions: "
              f"{conditions}\n")
        print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if save_csv and cfg is not None and window_name is not None:
        out_path = get_anova_csv_path(cfg, window_name)
        res.to_csv(out_path, index=False)
        if verbose:
            print(f"\n   saved: {out_path.name}")

    return res


def _greenhouse_geisser(data, F, df1, df2):
    """
    Greenhouse-Geisser epsilon and corrected p-value.

    `data` is (n_subjects, n_conditions) wide-format amplitude matrix.
    """
    n_subj, k = data.shape
    cov = np.cov(data, rowvar=False, ddof=1)

    # Mean of all elements
    mean_all = cov.mean()
    # Mean of diagonal
    mean_diag = np.diag(cov).mean()
    # Variance of row means
    row_means = cov.mean(axis=1)
    var_row_means = np.var(row_means, ddof=0)
    # Sum of squared elements
    sum_sq = (cov ** 2).sum()

    num = (k * (mean_diag - mean_all)) ** 2
    den = (k - 1) * (sum_sq - 2 * k * var_row_means * (k - 1)
                     + k ** 2 * mean_all ** 2)

    if den <= 0 or num <= 0:
        return 1.0, 1 - stats.f.cdf(F, df1, df2)

    eps = num / den
    eps = max(min(eps, 1.0), 1.0 / (k - 1))  # bounded

    df1_corr = df1 * eps
    df2_corr = df2 * eps
    p_gg = 1 - stats.f.cdf(F, df1_corr, df2_corr)
    return eps, p_gg


# ============================================================================
# Publication figure (now: arbitrary number of conditions)
# ============================================================================

def plot_results(cfg, window_name, rois, conditions, stats_df=None,
                 colors=None, ylim=None, xlim=None, figsize=None, ncols=None,
                 font="Arial", dpi=300, save_path=None,
                 alpha_fdr=0.05, sig_column="p_t_fdr", verbose=True):
    """
    Multi-panel publication figure: one panel per ROI, grand-average overlay
    of the listed conditions (any number) with SEM bands and shaded FDR-
    significant time windows.

    Parameters
    ----------
    cfg : SimpleNamespace
    window_name : str
    rois : dict {roi_name: [channel_list]}
    conditions : list of str
        Conditions to overlay (2+ supported).
    stats_df : pd.DataFrame or None
        From run_paired_tests or run_anova. Must have 'roi', 'tmin', 'tmax',
        and the column named in sig_column.
    colors : dict {cond_name: color} or None
        Defaults to a tab10-style palette.
    ylim : tuple or None
        (ymin, ymax) in µV. None = auto.
    xlim : tuple or None
        (xmin, xmax) in seconds. None = full epoch.
    figsize : tuple or None
    ncols : int or None
    font : str
    dpi : int
    save_path : Path or str or None
    alpha_fdr : float
    sig_column : str
        Which column in stats_df determines shading (e.g., 'p_t_fdr', 'p_gg_fdr').
    verbose : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from string import ascii_uppercase

    n_rois = len(rois)
    if ncols is None:
        ncols = min(n_rois, 3)
    nrows = int(np.ceil(n_rois / ncols))
    if figsize is None:
        figsize = (3.5 * ncols, 3.2 * nrows)

    if colors is None:
        # Default palette extends gracefully to N conditions
        default_palette = ["#3D5A80", "#E07A5F", "#81B29A", "#F2CC8F",
                           "#8E7DBE", "#3A506B"]
        colors = {c: default_palette[i % len(default_palette)]
                  for i, c in enumerate(conditions)}

    # Load per-subject evokeds
    subjects = find_subjects(cfg)
    evokeds_per_cond = {}
    for subject in subjects:
        evo_path = get_evoked_path(cfg, subject, window_name)
        if not evo_path.exists():
            continue
        for evo in mne.read_evokeds(evo_path, verbose="WARNING"):
            evokeds_per_cond.setdefault(evo.comment, []).append(evo)

    with plt.rc_context({"font.family": font, "font.size": 10,
                         "axes.spines.top": False,
                         "axes.spines.right": False,
                         "axes.linewidth": 1.0}):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
        axes_flat = axes.flatten()

        for i, (roi_name, channels) in enumerate(rois.items()):
            ax = axes_flat[i]

            for cond_name in conditions:
                evo_list = evokeds_per_cond.get(cond_name, [])
                if not evo_list:
                    continue
                valid_ch = [c for c in channels if c in evo_list[0].ch_names]
                subj_traces = np.array([
                    e.copy().pick(valid_ch).data.mean(axis=0) * 1e6
                    for e in evo_list
                ])
                times = evo_list[0].times
                mean_trace = subj_traces.mean(axis=0)
                sem = subj_traces.std(axis=0, ddof=1) / np.sqrt(len(subj_traces))

                ax.plot(times, mean_trace, color=colors[cond_name],
                        linewidth=1.6, label=cond_name)
                ax.fill_between(times, mean_trace - sem, mean_trace + sem,
                                color=colors[cond_name], alpha=0.18,
                                linewidth=0)

            # Shade FDR-significant time windows for this ROI
            if stats_df is not None and sig_column in stats_df.columns:
                sig_rows = stats_df[
                    (stats_df["roi"] == roi_name)
                    & (stats_df[sig_column] < alpha_fdr)
                ]
                # Avoid duplicate shading when multiple contrasts share a window
                shaded_ranges = set()
                for _, row in sig_rows.iterrows():
                    rng = (row["tmin"], row["tmax"])
                    if rng in shaded_ranges:
                        continue
                    shaded_ranges.add(rng)
                    ax.axvspan(rng[0], rng[1],
                               color="grey", alpha=0.18, linewidth=0)

            ax.axhline(0, color="black", linewidth=0.6)
            ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)

            ax.text(-0.12, 1.05, ascii_uppercase[i],
                    transform=ax.transAxes,
                    fontsize=14, fontweight="bold",
                    va="top", ha="left")

        for j in range(n_rois, len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right",
                   frameon=False, ncol=len(labels),
                   bbox_to_anchor=(0.98, 1.02))

        fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        png = save_path.with_suffix(".png")
        tif = save_path.with_suffix(".tiff")
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        fig.savefig(tif, dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
        if verbose:
            print(f"   saved: {png.name}")
            print(f"   saved: {tif.name}")

    return fig