"""
Configuration loader for the EEG toolkit.

Loads YAML configuration files and returns a dot-accessible object,
so notebooks can use `cfg.preprocessing.filter.low_freq` instead of
`cfg['preprocessing']['filter']['low_freq']`.
"""

from pathlib import Path
import yaml
from types import SimpleNamespace


def _dict_to_namespace(d):
    """Recursively convert a dict to SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    else:
        return d


def _namespace_to_dict(ns):
    """Convert a SimpleNamespace back to a plain dict (useful for saving)."""
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()
                if not k.startswith("_")}
    elif isinstance(ns, list):
        return [_namespace_to_dict(item) for item in ns]
    else:
        return ns


def load_config(config_path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    cfg : SimpleNamespace
        Dot-accessible configuration object. Examples:
            cfg.paradigm
            cfg.paths.base_dir
            cfg.preprocessing.filter.low_freq
            cfg.event_dict.Spatial

    Examples
    --------
    >>> cfg = load_config('configs/eye_eeg_simul.yaml')
    >>> print(cfg.preprocessing.filter.low_freq)
    0.5
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = _dict_to_namespace(cfg_dict)
    # Keep metadata for reproducibility
    cfg._raw_dict = cfg_dict
    cfg._config_path = str(config_path.resolve())
    return cfg


def save_config(cfg, output_path):
    """
    Save a config namespace back to YAML.

    Useful to snapshot the exact config used in an analysis
    (e.g., save a copy alongside the results).

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object returned by `load_config`.
    output_path : str or Path
        Destination path for the YAML file.
    """
    cfg_dict = _namespace_to_dict(cfg)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, default_flow_style=False)