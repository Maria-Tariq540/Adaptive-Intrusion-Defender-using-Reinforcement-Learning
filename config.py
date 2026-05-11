"""config.py

Purpose:
    Central configuration loader for hyperparameters and runtime options.

Notes:
    - Reads a config file or defines defaults.
    - For this Phase 1 structure task, values are placeholders.

Later improvements:
    - Load from JSON/YAML
    - Support CLI overrides
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Hyperparameters and runtime configuration."""
    # RL hyperparameters (placeholders)

    learning_rate: float = 1e-3

    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997

    # Training loop parameters (placeholders)
    episodes: int = 5000
    max_steps_per_episode: int = 200

    # Misc
    seed: int = 42
    device: str = "cpu"

    # API auth / runtime
    api_token: str = "CHANGE_ME"  # set via hyperparameters.json for real use
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    predictions_csv_path: str = "predictions_log.csv"

    # --- Real dataset + ML detector (Phase 1) ---
    cic_ids_2017_csv_path: str = "data/CIC-IDS2017/CIC-IDS2017.csv"
    nsl_kdd_csv_path: str = "data/NSL-KDD.csv"

    ml_model_dir: str = "models"
    ml_detector_model_path: str = "models/ml_detector.joblib"

    ml_model_type: str = "rf"  # rf or xgb
    dataset_test_size: float = 0.2
    min_rows_required: int = 2000
    feature_columns_whitelist: str = ""  # optional comma-separated
    ml_enable: bool = True



def get_config() -> Config:

    """Return configuration instance.

    Priority:
      1) hyperparameters.json if present
      2) Config() defaults otherwise

    Returns:
        Config dataclass.
    """
    import json
    from pathlib import Path

    cfg = Config()
    json_path = Path(__file__).resolve().parent / "hyperparameters.json"
    if not json_path.exists():
        return cfg

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        # If JSON is malformed/unreadable, fall back to defaults.
        return cfg

    def get_val(name: str, cast=None):
        if name not in data:
            return getattr(cfg, name)
        v = data[name]
        if cast is not None:
            return cast(v)
        return v

    return Config(
        learning_rate=float(get_val("learning_rate")),
        discount_factor=float(get_val("discount_factor")),
        epsilon_start=float(get_val("epsilon_start")),
        epsilon_end=float(get_val("epsilon_end")),
        epsilon_decay=float(get_val("epsilon_decay")),
        # Force override for the accuracy task improvements (even if hyperparameters.json is older).
        episodes=5000,
        max_steps_per_episode=int(get_val("max_steps_per_episode")),
        seed=int(get_val("seed")),
        device=str(get_val("device")),
        api_token=str(get_val("api_token")),
        api_host=str(get_val("api_host")),
        api_port=int(get_val("api_port")),
        predictions_csv_path=str(get_val("predictions_csv_path")),
    )



