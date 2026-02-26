"""
dataviz/config.py

Loads config.json once at startup.
Provides a single CFG object with dot-access to all sections.
Never imports from other dataviz modules.

Usage:
    from dataviz.config import CFG
    fog = CFG.rendering.fog_start
"""

import json
from pathlib import Path
from typing import Any
from loguru import logger


# ---------------------------------------------------------------------------
# Defaults — used when a key is missing from the JSON.
# Every missing key emits a WARN so the user knows the file is incomplete.
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "playback": {
        "update_hz":          120,
        "default_play_speed": 0.5,
        "graph_window":       50,
    },
    "camera": {
        "default_yaw":        45.0,
        "default_pitch":      25.0,
        "default_dist":       3.5,
        "min_pitch":          5.0,
        "max_pitch":          85.0,
        "orbit_sensitivity":  0.3,
        "pan_sensitivity":    0.004,
        "zoom_sensitivity":   0.3,
    },
    "rendering": {
        "fog_start":              3.0,
        "fog_end":                18.0,
        "ground_size":            26.0,
        "ground_step":            0.5,
        "force_scale":            0.003,
        "force_min_magnitude":    1.0,
        "ghost_alpha_main_mode":  0.25,
        "ghost_alpha_graph_mode": 0.38,
        "trajectory_length":      200,
        "joint_limit_warn_tol":   0.05,
        "sphere_stacks":          10,
        "sphere_slices":          10,
    },
    "layout": {
        "window_width":       1480,
        "window_height":      860,
        "splitter_width":     6,
        "default_vp_frac":    0.54,
        "default_bot_frac":   0.22,
        "default_graph_h_frac": 0.72,
        "layout_save_file":   "viewer_layout.json",
    },
    "colors": {
        "leg_FR":       [0.30, 0.90, 0.30, 1.0],
        "leg_FL":       [0.30, 0.55, 1.00, 1.0],
        "leg_RR":       [1.00, 0.63, 0.16, 1.0],
        "leg_RL":       [0.90, 0.24, 0.90, 1.0],
        "background":   [0.10, 0.10, 0.13, 1.0],
        "ground_dark":  [0.32, 0.38, 0.44, 1.0],
        "ground_light": [0.50, 0.57, 0.64, 1.0],
        "robot_main":   [0.92, 0.92, 0.94, 1.0],
        "robot_ghost":  [0.75, 0.82, 1.00, 1.0],
        "graph_series": [
            [0.25, 0.55, 1.00],
            [0.20, 0.85, 0.40],
            [1.00, 0.70, 0.15],
            [0.85, 0.25, 0.85],
            [0.15, 0.85, 0.85],
            [1.00, 0.35, 0.35],
            [0.85, 0.85, 0.20],
            [0.60, 0.40, 1.00],
        ],
    },
    "ui": {
        "default_scale":    1.0,
        "invert_y_default": True,
    },
    "live_graphs": {
        "Joint Angles (deg)": {
            "series": ["q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11"],
            "y_min": -180.0, "y_max": 180.0,
        },
        "Body RPY (deg)": {
            "series": ["torso_roll","torso_pitch","torso_yaw"],
            "y_min": -40.0, "y_max": 40.0,
        },
        "Joint Torques (Nm)": {
            "series": ["tau0","tau1","tau2","tau3","tau4","tau5","tau6","tau7","tau8","tau9","tau10","tau11"],
            "y_min": -50.0, "y_max": 50.0,
        },
        "Contact Forces (N)": {
            "series": ["foot_force_0","foot_force_1","foot_force_2",
                       "foot_force_3","foot_force_4","foot_force_5",
                       "foot_force_6","foot_force_7","foot_force_8",
                       "foot_force_9","foot_force_10","foot_force_11"],
            "y_min": -50.0, "y_max": 500.0,
        },
        "Joint Velocities (rad/s)": {
            "series": ["dq0","dq1","dq2","dq3","dq4","dq5","dq6","dq7","dq8","dq9","dq10","dq11"],
            "y_min": -20.0, "y_max": 20.0,
        },
    },
    # graphs: viewer graph panel settings
    "graphs": {
        "y_axis_ticks": 5,   # number of numeric tick labels on the y-axis
    },
    # custom_channels: name -> filename in data dir. Empty by default.
    # Example: {"my_signal": "myrecordeddata.txt"}
    "custom_channels": {},
}


# ---------------------------------------------------------------------------
# Dot-access namespace
# ---------------------------------------------------------------------------
class _Namespace:
    """Recursively wraps a dict so values are accessible as attributes."""

    def __init__(self, d: dict, path: str = ""):
        self._path = path
        for k, v in d.items():
            if k.startswith("_"):          # skip comment keys
                continue
            if isinstance(v, dict):
                object.__setattr__(self, k, _Namespace(v, f"{path}.{k}" if path else k))
            else:
                object.__setattr__(self, k, v)

    def __repr__(self) -> str:
        keys = [k for k in self.__dict__ if not k.startswith("_")]
        return f"<Config {self._path} [{', '.join(keys)}]>"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Merge helper — fills missing keys with defaults and warns on each
# ---------------------------------------------------------------------------
def _merge_with_defaults(raw: dict, defaults: dict, section: str = "") -> dict:
    merged: dict = {}
    for key, default_val in defaults.items():
        section_key = f"{section}.{key}" if section else key
        if key not in raw:
            logger.warning(
                f"[config] Missing key '{section_key}' — using default: {default_val!r}"
            )
            merged[key] = default_val
        elif isinstance(default_val, dict) and isinstance(raw[key], dict):
            merged[key] = _merge_with_defaults(raw[key], default_val, section_key)
        else:
            merged[key] = raw[key]
    # Pass through any extra keys the user added (no warning, just forward)
    for key in raw:
        if key not in merged:
            merged[key] = raw[key]
    return merged


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path) -> _Namespace:
    """
    Load config.json from *config_path*.
    Falls back to defaults for any missing key (with WARN).
    Raises FileNotFoundError if the file does not exist at all.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"[config] Config file not found: {path.resolve()}\n"
            "Pass --config <path> or create a config.json next to test_modular.py"
        )

    try:
        with open(path, "r") as f:
            raw = json.load(f)
        logger.info(f"[config] Loaded config from {path.resolve()}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"[config] config.json is not valid JSON: {exc}") from exc

    merged = _merge_with_defaults(raw, _DEFAULTS)
    return _Namespace(merged)


# ---------------------------------------------------------------------------
# Module-level singleton — populated by viewer.py calling init_config()
# ---------------------------------------------------------------------------
CFG: _Namespace | None = None


def init_config(config_path: str | Path) -> _Namespace:
    """
    Call once at startup from test_modular.py / viewer.py.
    Sets the module-level CFG singleton and returns it.
    All other modules do: from dataviz.config import CFG
    """
    global CFG
    CFG = load_config(config_path)
    return CFG
