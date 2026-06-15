"""EEG toolkit for preprocessing and analysis."""

from eeg_toolkit.config import load_config, save_config
from eeg_toolkit.io import (
    find_subjects,
    find_raw_files,
    get_raw_dir,
    get_analysis_dir,
    get_subject_dir,
    get_subject_path,
    load_status,
    update_status,
    is_step_done,
    get_excluded_subjects,
)
from eeg_toolkit.raw2fif import (
    convert_xdf_to_fif,
    convert_all_xdfs,
)
from eeg_toolkit.preprocessing import (
    preprocess_subject,
    preprocess_all,
    apply_filter,
    assign_eog_channels,
    apply_montage,
    resample_with_events,
)
from eeg_toolkit.bad_channels import (
    detect_bad_channels_auto,
    inspect_subject_bads,
    inspect_all_subjects,
)
from eeg_toolkit.event_codes import (
    derive_events_subject,
    derive_events_all,
)
from eeg_toolkit.epoching import (
    create_window,
    create_all_windows,
    epoch_all_subjects,
    get_epochs_path,
)
from eeg_toolkit.ica import (
    fit_ica_subject,
    fit_ica_all,
    label_ica_subject,
    label_ica_all,
    inspect_ica_subject,
    inspect_ica_all,
    apply_ica_subject,
    apply_ica_all,
    get_clean_epochs_path,
)
from eeg_toolkit.artifacts import (
    find_bad_epochs,
    inspect_trials_subject,
    inspect_all_trials,
    propagate_rejections_subject,
    propagate_rejections_all,
    apply_rejection_subject,
    apply_rejection_all,
    get_final_epochs_path,
)

from eeg_toolkit.evoked import (
    compute_evokeds_subject,
    compute_evokeds_all,
    compute_grand_averages,
    get_evoked_path,
    get_grand_average_path,
)

from eeg_toolkit.erp_explore import (
    load_grand_averages,
    plot_butterfly,
    plot_topomaps,
    plot_channel_overlay,
    plot_channel_overlay_with_ci,
    plot_roi_overlay,
)

from eeg_toolkit.erp_stats import (
    extract_mean_amplitudes,
    run_paired_tests,
    run_anova,
    plot_results,
    get_amplitudes_csv_path,
    get_stats_csv_path,
    get_anova_csv_path,
)

from eeg_toolkit.tfr import (
    get_tfr_path,
    get_grand_average_tfr_dir,
    get_grand_average_tfr_path,
    compute_tfr_subject,
    compute_tfr_all,
    load_subject_tfrs,
    compute_grand_average_tfr,
    load_grand_average_tfr,
    load_all_grand_averages,
    plot_tfr_heatmap,
    plot_tfr_conditions,
    plot_tfr_topomaps,
    plot_tfr_timecourse,
    extract_mean_band_power,
    run_tfr_paired_tests,
    run_tfr_cluster_test,
    plot_tfr_clusters
)

from eeg_toolkit.tfr_stats import (
    get_eeg_channels,
    build_adjacency,
    load_band_data,
    run_cluster_test,
    describe_clusters,
    plot_cluster_results,
)

from eeg_toolkit.mvpa import (
    decode_subject,
    decode_all,
    load_subject_scores,
    load_all_scores,
    get_decode_path,
    cross_decode_subject,
    cross_decode_all,
    load_all_cross_scores,
    get_cross_decode_path,
    align_epochs_behavior,
)

from eeg_toolkit.rsa import ( 
    compute_rdms_all,
    load_all_rdms,
)
__version__ = "0.1.0"