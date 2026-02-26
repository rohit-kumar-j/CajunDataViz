"""
dataviz/data.py

Owns:
  - Data file scanning and channel mapping
  - Channel map persistence (channel_map.json)
  - Data loading (txt → numpy)
  - URDF parsing (joints, limits, axes, origins)
  - Forward kinematics via yourdfpy

No GL, no imgui, no state objects.
Only imports: dataviz.config, standard library, numpy, yourdfpy, loguru.
"""

import json
import math
from pathlib import Path

import numpy as np
from loguru import logger

import dataviz.config as _cfg_mod


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------
REQUIRED_CHANNELS: list[str] = [
    "time",
    "torso_x", "torso_y", "torso_z",
    "torso_roll", "torso_pitch", "torso_yaw",
] + [f"q{i}" for i in range(12)]

_FOOT_NAMES = ["FR", "FL", "RR", "RL"]
_AXES       = ["x", "y", "z"]

OPTIONAL_CHANNELS: list[str] = [
    "desired_vel_x",
    "torso_vx", "torso_vy", "torso_vz",
    "contact_FR", "contact_FL", "contact_RR", "contact_RL",
    "qp_cost",
] + [f"dq{i}"                for i in range(12)] \
  + [f"tau{i}"               for i in range(12)] \
  + [f"foot_force_{i}"       for i in range(12)] \
  + [f"foot_pos_{i}"         for i in range(12)] \
  + [f"qp_desired_acc{i}"    for i in range(6)]  \
  + [f"qp_solved_acc{i}"     for i in range(6)]  \
  + [f"raibert_{f}_{a}"      for f in _FOOT_NAMES for a in _AXES] \
  + [f"raibert_corr_{f}_{a}" for f in _FOOT_NAMES for a in _AXES] \
  + [f"swing_foot_{f}_{a}"   for f in _FOOT_NAMES for a in _AXES]

ALL_CHANNELS: list[str] = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

CHANNEL_LABELS: dict[str, str] = {
    "time":         "Time (s)",
    "torso_x":      "Torso X (world)",
    "torso_y":      "Torso Y (world)",
    "torso_z":      "Torso Z (world)",
    "torso_roll":   "Torso Roll (rad)",
    "torso_pitch":  "Torso Pitch (rad)",
    "torso_yaw":    "Torso Yaw (rad)",
    "torso_vx":     "Torso Vel X (m/s)",
    "torso_vy":     "Torso Vel Y (m/s)",
    "torso_vz":     "Torso Vel Z (m/s)",
    "desired_vel_x": "Desired Vel X (m/s)",
    "contact_FR":   "Contact FR (0/1)",
    "contact_FL":   "Contact FL (0/1)",
    "contact_RR":   "Contact RR (0/1)",
    "contact_RL":   "Contact RL (0/1)",
    "qp_cost":      "QP Cost",
}
# QP acceleration components: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
_QP_ACC_LABELS = ["Lin X", "Lin Y", "Lin Z", "Ang X", "Ang Y", "Ang Z"]
for _i, _lbl in enumerate(_QP_ACC_LABELS):
    CHANNEL_LABELS[f"qp_desired_acc{_i}"] = f"QP Desired Acc {_lbl} (body)"
    CHANNEL_LABELS[f"qp_solved_acc{_i}"]  = f"QP Solved Acc {_lbl} (body)"
# Raibert landing positions and correction vectors
for _f in _FOOT_NAMES:
    for _a in _AXES:
        CHANNEL_LABELS[f"raibert_{_f}_{_a}"]      = f"Raibert Land {_f} {_a.upper()} (m)"
        CHANNEL_LABELS[f"raibert_corr_{_f}_{_a}"] = f"Raibert Corr {_f} {_a.upper()} (m)"
        CHANNEL_LABELS[f"swing_foot_{_f}_{_a}"]   = f"Swing Foot {_f} {_a.upper()} (m)"
for _i in range(12):
    CHANNEL_LABELS[f"q{_i}"]           = f"Joint Angle q{_i} (rad)"
    CHANNEL_LABELS[f"dq{_i}"]          = f"Joint Vel dq{_i} (rad/s)"
    CHANNEL_LABELS[f"tau{_i}"]         = f"Joint Torque tau{_i} (Nm)"
    CHANNEL_LABELS[f"foot_force_{_i}"] = f"Foot Force component {_i}"
    CHANNEL_LABELS[f"foot_pos_{_i}"]   = f"Foot Pos component {_i}"

# ---------------------------------------------------------------------------
# Auto-mapping patterns  (filename → channel name)
# ---------------------------------------------------------------------------
AUTO_PATTERNS: dict[str, str] = {
    "time.txt":          "time",
    "PosTorso0.txt":     "torso_x",
    "PosTorso4.txt":     "torso_y",
    "PosTorso5.txt":     "torso_z",
    "PosTorso2.txt":     "torso_roll",
    "PosTorso1.txt":     "torso_pitch",
    "PosTorso3.txt":     "torso_yaw",
    "PosTorso6.txt":     "torso_vx",
    "PosTorso7.txt":     "torso_vy",
    "PosTorso8.txt":     "torso_vz",
    "desired_vel_x.txt": "desired_vel_x",
    "contact_FR.txt":    "contact_FR",
    "contact_FL.txt":    "contact_FL",
    "contact_RR.txt":    "contact_RR",
    "contact_RL.txt":    "contact_RL",
    "qp_cost.txt":       "qp_cost",
}
for _i in range(12):
    AUTO_PATTERNS[f"q{_i}.txt"]                  = f"q{_i}"
    AUTO_PATTERNS[f"dq{_i}.txt"]                 = f"dq{_i}"
    AUTO_PATTERNS[f"tauM{_i}.txt"]               = f"tau{_i}"
    AUTO_PATTERNS[f"simforceFeetGlobal{_i}.txt"] = f"foot_force_{_i}"
    AUTO_PATTERNS[f"footPosFeetGlobal{_i}.txt"]  = f"foot_pos_{_i}"
for _i in range(6):
    AUTO_PATTERNS[f"qp_desired_acc{_i}.txt"] = f"qp_desired_acc{_i}"
    AUTO_PATTERNS[f"qp_solved_acc{_i}.txt"]  = f"qp_solved_acc{_i}"
for _f in _FOOT_NAMES:
    for _a in _AXES:
        AUTO_PATTERNS[f"raibert_{_f}_{_a}.txt"]      = f"raibert_{_f}_{_a}"
        AUTO_PATTERNS[f"raibert_corr_{_f}_{_a}.txt"] = f"raibert_corr_{_f}_{_a}"
        AUTO_PATTERNS[f"swing_foot_{_f}_{_a}.txt"]   = f"swing_foot_{_f}_{_a}"


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------
def scan_data_files(data_dir: str | Path) -> list[str]:
    """Return sorted list of .txt filenames in *data_dir*."""
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"[data] Data directory not found: {d.resolve()}")
    files = sorted(p.name for p in d.glob("*.txt"))
    logger.debug(f"[data] scan_data_files: {len(files)} .txt files in {d}")
    return files


def scan_unknown_files(files: list[str]) -> list[str]:
    """Return filenames NOT mapped by AUTO_PATTERNS (auto-detected unknowns).

    These show up in the picker under 'Auto-detected' so the user can see and
    optionally load them.  The channel key for each is extra_channel_name(fn).
    """
    return [fn for fn in files if fn not in AUTO_PATTERNS]


def extra_channel_name(filename: str) -> str:
    """Stable channel key for an unknown file.  'mySignal.txt' → 'extra:mySignal'."""
    return f"extra:{Path(filename).stem}"


def load_extra_channels(data_dir: str | Path, extra_map: dict[str, str], n: int) -> dict:
    """Load auto-detected (extra) channels from *data_dir*.

    *extra_map*: {channel_key: filename}  e.g. {'extra:mySignal': 'mySignal.txt'}
    Returns:     {channel_key: np.ndarray | None}  — arrays aligned to *n* frames.
    """
    d = Path(data_dir)
    result: dict = {}
    for key, filename in extra_map.items():
        if not filename:
            continue
        p = d / filename
        if not p.exists():
            logger.warning(f"[data] extra '{key}': '{filename}' not found — skipping")
            result[key] = None
            continue
        try:
            arr = np.loadtxt(str(p), dtype=np.float64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)
            elif len(arr) > n:
                arr = arr[:n]
            result[key] = arr
            logger.info(f"[data] extra '{key}': loaded '{filename}' ({n} frames)")
        except Exception as exc:
            logger.warning(f"[data] extra '{key}': failed '{filename}': {exc}")
            result[key] = None
    return result


def auto_map_channels(files: list[str]) -> dict[str, str | None]:
    """Map known filenames to channel names. Unmapped channels → None."""
    cmap: dict[str, str | None] = {ch: None for ch in ALL_CHANNELS}
    for fn in files:
        ch = AUTO_PATTERNS.get(fn)
        if ch and ch in cmap:
            cmap[ch] = fn
    mapped = sum(1 for v in cmap.values() if v is not None)
    logger.debug(f"[data] auto_map_channels: {mapped}/{len(ALL_CHANNELS)} channels mapped")
    return cmap


# ---------------------------------------------------------------------------
# Channel map persistence
# ---------------------------------------------------------------------------
def save_channel_map(data_dir: str | Path, cmap: dict) -> None:
    p = Path(data_dir) / "channel_map.json"
    try:
        with open(p, "w") as f:
            json.dump(cmap, f, indent=2)
        logger.info(f"[data] Channel map saved → {p}")
    except OSError as exc:
        logger.error(f"[data] Could not save channel map to {p}: {exc}")


def load_channel_map_file(data_dir: str | Path) -> dict | None:
    p = Path(data_dir) / "channel_map.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            cmap = json.load(f)
        logger.info(f"[data] Channel map loaded from {p}")
        return cmap
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[data] Could not read channel_map.json ({p}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _read_col(data_dir: Path, filename: str | None) -> np.ndarray | None:
    if filename is None:
        return None
    p = data_dir / filename
    if not p.exists():
        logger.warning(f"[data] File not found, skipping: {p}")
        return None
    try:
        arr = np.loadtxt(str(p), dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    except Exception as exc:
        logger.warning(f"[data] Could not load {filename}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Training config bounds extraction
# ---------------------------------------------------------------------------

def _unwrap_config_dict_fields(obj):
    """
    Recursively unwrap ml_collections ConfigDict internal representation.

    When ml_collections.ConfigDict.to_yaml() serialises a config, every
    ConfigDict node is dumped as its raw Python object with internal fields:

        _convert_dict: true
        _fields:
          key1: value1
          key2: value2
        _locked: true
        _type_safe: true

    The actual key-value data lives under '_fields'.  This function walks the
    parsed structure and replaces every such node with just the contents of
    '_fields', then recurses into the result.  Plain dicts and lists that do
    NOT have a '_fields' key are left unchanged (but their children are still
    recursed into).

    This makes the output identical to what you would get if the config had
    been saved with a plain dict serialiser — i.e. exactly what
    extract_bounds_from_config() already expects.
    """
    if isinstance(obj, dict):
        if "_fields" in obj and isinstance(obj["_fields"], dict):
            # This is a ConfigDict node — unwrap to its contents and recurse
            return _unwrap_config_dict_fields(obj["_fields"])
        else:
            # Plain dict — recurse into values
            return {k: _unwrap_config_dict_fields(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unwrap_config_dict_fields(item) for item in obj]
    else:
        # Scalar (str, int, float, bool, None, bytes, …) — return as-is
        return obj


def _make_config_loader():
    """
    Build a yaml Loader that can parse training config.yaml files without
    importing any src.* modules or instantiating any Python classes.

    The yaml file uses several special tags:
      !!python/object:ml_collections...ConfigDict  -> treat as plain dict
      !!python/name:src.envs.jump_env.JumpEnv      -> ignore (return None)
      !!python/object/apply:numpy...._reconstruct  -> decode binary to float list
      !!python/object/apply:torch..._rebuild_...   -> ignore (return None)
      !!python/tuple                               -> convert to list

    None of these require importing the actual classes.
    """
    import yaml
    import base64
    import struct

    loader_class = yaml.SafeLoader

    # ── ConfigDict → plain dict ───────────────────────────────────────────
    def _construct_config_dict(loader, tag_suffix, node):
        # ConfigDict is a mapping — just build a plain dict
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        logger.warning(f"[data] config.yaml: unexpected ConfigDict node type {type(node).__name__}, returning {{}}")
        return {}

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object:ml_collections",
        _construct_config_dict,
    )

    # ── python/name tags → None (class references we don't need) ─────────
    def _construct_python_name(loader, tag_suffix, node):
        val = loader.construct_scalar(node)
        logger.debug(f"[data] config.yaml: ignoring python/name tag: {val}")
        return None

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/name:",
        _construct_python_name,
    )

    # ── numpy binary reconstruction → decode to float list ───────────────
    def _construct_numpy_reconstruct(loader, tag_suffix, node):
        # ml_collections serialises numpy arrays via pickle's __reduce__:
        #
        #   !!python/object/apply:numpy.core.multiarray._reconstruct
        #     args: [<ndarray class>, (0,), b'b']      # b'b' is byte-order flag
        #     state: (1, (<shape>,), <dtype>, False, <BINARY DATA>)
        #
        # PyYAML represents this as a MappingNode with 'args' and 'state'.
        # The actual array bytes live in state[4].
        # Older/simpler serialisations may use a SequenceNode instead.
        #
        # We use the shape from state[1] to determine the dtype:
        #   If total_elements * 8 == byte_count → float64 (f8)
        #   If total_elements * 4 == byte_count → float32 (f4)
        # This avoids misinterpreting f4 data as f8 (which would give wrong
        # element count and garbage values).

        def _decode_bytes(raw_bytes, expected_elems=0):
            """Decode raw bytes into a list of floats.

            *expected_elems* is the product of the shape tuple (from state[1]).
            When available it lets us pick the right dtype unambiguously.
            """
            if not isinstance(raw_bytes, bytes) or len(raw_bytes) == 0:
                return []
            n_bytes = len(raw_bytes)

            # Build candidate list ordered by likelihood.
            # If we know expected_elems, put the matching dtype first.
            candidates = [('d', 8), ('f', 4)]
            if expected_elems > 0:
                for fmt_char, elem_size in candidates:
                    if expected_elems * elem_size == n_bytes:
                        # Exact match — move to front
                        candidates = [(fmt_char, elem_size)] + [
                            c for c in candidates if c[0] != fmt_char
                        ]
                        break

            for fmt_char, elem_size in candidates:
                n_elems = n_bytes // elem_size
                if n_elems > 0:
                    usable = n_elems * elem_size
                    try:
                        return list(struct.unpack(f"<{n_elems}{fmt_char}", raw_bytes[:usable]))
                    except struct.error:
                        continue
            return []

        def _shape_numel(shape):
            """Compute total number of elements from a shape tuple/list."""
            numel = 1
            if isinstance(shape, (list, tuple)):
                for s in shape:
                    if isinstance(s, int):
                        numel *= s
                    elif isinstance(s, (list, tuple)):
                        # Nested shape like ((6, 6),)
                        for ss in s:
                            if isinstance(ss, int):
                                numel *= ss
            return numel

        def _extract_from_state(state):
            """Extract float list from a state tuple: (1, shape, dtype, fortran, bytes)."""
            if isinstance(state, (list, tuple)) and len(state) >= 5:
                raw = state[4]
                if isinstance(raw, bytes):
                    shape = state[1]
                    expected = _shape_numel(shape)
                    result = _decode_bytes(raw, expected)
                    if result:
                        logger.debug(
                            f"[data] config.yaml: numpy reconstruct — "
                            f"decoded {len(result)} values from state (shape={shape})"
                        )
                        return result
            return None

        def _search_for_bytes(obj, depth=0):
            """Recursively search for binary data in nested lists/tuples."""
            if depth > 5:
                return None
            if isinstance(obj, bytes) and len(obj) >= 4:
                return _decode_bytes(obj)
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    result = _search_for_bytes(item, depth + 1)
                    if result:
                        return result
            return None

        try:
            # ── MappingNode: has 'args' and 'state' keys ──────────────────
            if isinstance(node, yaml.MappingNode):
                data = loader.construct_mapping(node, deep=True)
                state = data.get("state")
                if state is not None:
                    result = _extract_from_state(state)
                    if result:
                        return result
                # Fallback: search all values for bytes
                for v in data.values():
                    result = _search_for_bytes(v)
                    if result:
                        return result

            # ── SequenceNode: [class, args, state] ────────────────────────
            elif isinstance(node, yaml.SequenceNode):
                items = loader.construct_sequence(node, deep=True)
                # items[2] is typically the state tuple
                if len(items) >= 3:
                    result = _extract_from_state(items[2])
                    if result:
                        return result
                # Fallback: search all items for bytes
                for item in items:
                    result = _search_for_bytes(item)
                    if result:
                        return result

            logger.debug(f"[data] config.yaml: numpy reconstruct — returning []")
            return []
        except Exception as exc:
            logger.warning(f"[data] config.yaml: numpy reconstruct failed: {exc} — returning []")
            return []

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:numpy",
        _construct_numpy_reconstruct,
    )

    # ── numpy dtype → ignore ──────────────────────────────────────────────
    def _construct_numpy_dtype(loader, tag_suffix, node):
        return None

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:numpy.dtype",
        _construct_numpy_dtype,
    )

    # ── torch tensor reconstruction → None ───────────────────────────────
    def _construct_torch(loader, tag_suffix, node):
        logger.debug(f"[data] config.yaml: ignoring torch tag")
        return None

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:torch",
        _construct_torch,
    )
    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object:torch",
        _construct_torch,
    )

    # ── OrderedDict → plain dict ──────────────────────────────────────────
    def _construct_ordered_dict(loader, tag_suffix, node):
        if isinstance(node, yaml.SequenceNode):
            pairs = loader.construct_sequence(node, deep=True)
            return dict(p for p in pairs if isinstance(p, (list, tuple)) and len(p) == 2)
        return {}

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:collections",
        _construct_ordered_dict,
    )

    # ── python/tuple → list ───────────────────────────────────────────────
    def _construct_tuple(loader, node):
        return loader.construct_sequence(node, deep=True)

    loader_class.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        _construct_tuple,
    )

    # ── Catch-all for any remaining python/* tags → None ─────────────────
    def _construct_catchall(loader, tag_suffix, node):
        logger.debug(f"[data] config.yaml: unhandled tag '{tag_suffix}', returning None")
        return None

    loader_class.add_multi_constructor(
        "tag:yaml.org,2002:python/",
        _construct_catchall,
    )

    return loader_class


def _load_training_config(data_dir) -> dict:
    """
    Load config.yaml from *data_dir* as a plain nested dict.
    Does NOT import src.*, ml_collections, torch, or numpy.
    Uses a custom SafeLoader that decodes all special tags without
    instantiating any Python classes.

    The YAML is produced by ml_collections.ConfigDict.to_yaml(), which wraps
    every mapping in a ConfigDict object with internal keys (_fields, _locked,
    _convert_dict, _type_safe).  After parsing we call
    _unwrap_config_dict_fields() to strip that wrapper so the result is a
    plain nested dict that extract_bounds_from_config() can access normally.

    Raises:
        RuntimeError — if config.yaml is not found or cannot be parsed.
    """
    from pathlib import Path as _Path
    config_path = _Path(data_dir) / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"[data] config.yaml not found in {_Path(data_dir).resolve()}\n"
            f"The eval script must copy config.yaml into the data directory.\n"
            f"Expected: {config_path.resolve()}"
        )

    logger.info(f"[data] Loading training config: {config_path.resolve()}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            f"[data] PyYAML is required to load config.yaml: {exc}\n"
            "Install with: pip install pyyaml"
        ) from exc

    try:
        loader_cls = _make_config_loader()
        with open(config_path, "r") as f:
            raw = yaml.load(f, Loader=loader_cls)
    except Exception as exc:
        raise RuntimeError(
            f"[data] Failed to parse config.yaml at {config_path}: {exc}"
        )

    # Strip the ml_collections ConfigDict internal wrapper (_fields) from every
    # node so downstream code can access cfg["environment"]["robot"]["motors"]
    # directly instead of cfg["_fields"]["environment"]["_fields"]["robot"] etc.
    cfg = _unwrap_config_dict_fields(raw)

    # Sanity-check: top-level keys should now be the logical config keys
    if isinstance(cfg, dict):
        logger.debug(f"[data] config.yaml top-level keys after unwrap: {list(cfg.keys())}")
    else:
        raise RuntimeError(
            f"[data] config.yaml did not parse to a dict after unwrapping "
            f"(got {type(cfg).__name__}). The YAML structure may have changed."
        )

    logger.info("[data] config.yaml loaded and unwrapped — no src.* imports required")
    return cfg


def _cfg_get(d, *keys, default=None):
    """
    Safely walk a nested dict/list using a sequence of string keys or int indices.
    Returns default if any key is missing or the value is None.

    Example:
        _cfg_get(cfg, "environment", "robot", "motors")
        _cfg_get(cfg, "environment", "env", "velocity_lb")
    """
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        elif isinstance(cur, (list, tuple)) and isinstance(k, int):
            cur = cur[k] if 0 <= k < len(cur) else default
        else:
            return default
    return cur if cur is not None else default


def extract_bounds_from_config(train_cfg: dict) -> dict:
    """
    Extract per-channel bounds from a plain dict loaded from config.yaml.

    train_cfg is the result of _load_training_config() — a nested plain dict,
    NOT a ConfigDict. All access is via _cfg_get() (key-based, no dot access).

    Returns dict: channel_name -> (lo, hi) or None.
    None means no limit lines are drawn for that channel in the viewer.

    Raises:
        RuntimeError — if required sub-configs are completely absent.
    """
    logger.info("[data] extract_bounds_from_config: extracting channel bounds")
    bounds: dict = {}

    # ── Validate top-level structure ────────────────────────────────────────
    if not isinstance(train_cfg, dict):
        raise RuntimeError(
            f"[data] config.yaml did not parse to a dict (got {type(train_cfg).__name__}). "
            f"Check that the file was saved correctly by the training script."
        )

    env_section = _cfg_get(train_cfg, "environment")
    if env_section is None:
        raise RuntimeError(
            "[data] config.yaml missing top-level 'environment' key. "
            "Check the yaml was saved from the training run."
        )
    logger.debug("[data]   'environment' section found")

    # ── Motor specs → q/dq/tau bounds ──────────────────────────────────────
    motors = _cfg_get(env_section, "robot", "motors")
    if motors is None:
        raise RuntimeError(
            "[data] config.yaml missing environment -> robot -> motors. "
            "Make sure config.robot.motors is populated in g2_rotary.py."
        )
    if not isinstance(motors, (list, tuple)):
        raise RuntimeError(
            f"[data] environment.robot.motors is not a list (got {type(motors).__name__})"
        )

    logger.info(f"[data]   found {len(motors)} motors")
    if len(motors) != 12:
        logger.warning(
            f"[data]   expected 12 motors, got {len(motors)} — "
            f"bounds cover indices 0..{len(motors)-1} only"
        )

    for i, motor in enumerate(motors):
        if not isinstance(motor, dict):
            logger.error(f"[data]   motor[{i}] is not a dict (got {type(motor).__name__}) — skipping")
            bounds[f"q{i}"] = bounds[f"dq{i}"] = bounds[f"tau{i}"] = None
            continue

        motor_name = motor.get("name", f"motor_{i}")

        for ch_prefix, lo_key, hi_key, unit in [
            ("q",   "min_position", "max_position", "rad"),
            ("dq",  "min_velocity", "max_velocity", "rad/s"),
            ("tau", "min_torque",   "max_torque",   "Nm"),
        ]:
            ch = f"{ch_prefix}{i}"
            lo_raw = motor.get(lo_key)
            hi_raw = motor.get(hi_key)
            if lo_raw is None or hi_raw is None:
                logger.error(
                    f"[data]   {ch} ({motor_name}): key '{lo_key}' or '{hi_key}' missing"
                )
                bounds[ch] = None
                continue
            try:
                lo = float(lo_raw)
                hi = float(hi_raw)
                bounds[ch] = (lo, hi)
                logger.debug(f"[data]   {ch} ({motor_name}): [{lo:.4f}, {hi:.4f}] {unit}")
            except (TypeError, ValueError) as exc:
                logger.error(f"[data]   {ch} ({motor_name}): could not convert to float: {exc}")
                bounds[ch] = None

    # ── Velocity command → desired_vel_x ───────────────────────────────────
    vel_lo_raw = _cfg_get(env_section, "env", "velocity_lb")
    vel_hi_raw = _cfg_get(env_section, "env", "velocity_ub")
    if vel_lo_raw is None or vel_hi_raw is None:
        logger.error(
            "[data]   desired_vel_x: environment.env.velocity_lb/ub not found — no bound"
        )
        bounds["desired_vel_x"] = None
    else:
        try:
            vel_lo = float(vel_lo_raw)
            vel_hi = float(vel_hi_raw)
            bounds["desired_vel_x"] = (vel_lo, vel_hi)
            logger.info(f"[data]   desired_vel_x: [{vel_lo:.3f}, {vel_hi:.3f}] m/s")
        except (TypeError, ValueError) as exc:
            logger.error(f"[data]   desired_vel_x: could not convert velocity bounds: {exc}")
            bounds["desired_vel_x"] = None

    # ── Foot residual bounds from action_lb / action_ub ───────────────────
    # Action vector layout (21 elements, from config.yaml binary decode):
    #   [0]      alpha (Raibert gain)
    #   [1..8]   gait / misc params
    #   [9..20]  foot residuals: 3 per foot in order FR, FL, RR, RL
    #            each triplet is [delta_x, delta_y, delta_z] in body frame
    #
    # We store bounds as a (4, 3, 2) float32 array: [foot, axis, lo/hi]
    # This lets gl_core build a per-foot, per-axis asymmetric box.
    _FOOT_RESIDUAL_START = 9          # first index of foot residuals in action vector
    _N_FOOT_AXES         = 3          # x, y, z per foot
    _N_FEET              = 4          # FR, FL, RR, RL
    _EXPECTED_ACTION_LEN = _FOOT_RESIDUAL_START + _N_FEET * _N_FOOT_AXES  # 21

    action_lb_raw = _cfg_get(env_section, "env", "action_lb")
    action_ub_raw = _cfg_get(env_section, "env", "action_ub")

    if action_lb_raw is None or action_ub_raw is None:
        logger.warning(
            "[data]   raibert_bounds: environment.env.action_lb/ub not found — "
            "boxes will use default ±0.10 m extents"
        )
        bounds["raibert_bounds"] = None
    elif (not isinstance(action_lb_raw, (list, tuple)) or
          not isinstance(action_ub_raw, (list, tuple))):
        logger.warning(
            f"[data]   raibert_bounds: action_lb/ub are not lists "
            f"(got {type(action_lb_raw).__name__}) — using defaults"
        )
        bounds["raibert_bounds"] = None
    elif (len(action_lb_raw) < _EXPECTED_ACTION_LEN or
          len(action_ub_raw) < _EXPECTED_ACTION_LEN):
        logger.warning(
            f"[data]   raibert_bounds: action_lb/ub length {len(action_lb_raw)} < "
            f"{_EXPECTED_ACTION_LEN} expected — using defaults"
        )
        bounds["raibert_bounds"] = None
    else:
        try:
            rb = np.zeros((_N_FEET, _N_FOOT_AXES, 2), dtype=np.float32)
            for fi in range(_N_FEET):
                for ai in range(_N_FOOT_AXES):
                    idx = _FOOT_RESIDUAL_START + fi * _N_FOOT_AXES + ai
                    rb[fi, ai, 0] = float(action_lb_raw[idx])  # lo
                    rb[fi, ai, 1] = float(action_ub_raw[idx])  # hi
            bounds["raibert_bounds"] = rb
            logger.info(
                f"[data]   raibert_bounds extracted — "
                f"FR x:[{rb[0,0,0]:.3f},{rb[0,0,1]:.3f}] "
                f"y:[{rb[0,1,0]:.4f},{rb[0,1,1]:.4f}] "
                f"z:[{rb[0,2,0]:.3f},{rb[0,2,1]:.3f}]"
            )
        except (TypeError, ValueError, IndexError) as exc:
            logger.error(f"[data]   raibert_bounds: conversion failed: {exc} — using defaults")
            bounds["raibert_bounds"] = None

    # ── All other channels → None (no limit lines) ─────────────────────────
    no_bound = [
        "torso_x", "torso_y", "torso_z",
        "torso_roll", "torso_pitch", "torso_yaw",
        "contact_FR", "contact_FL", "contact_RR", "contact_RL",
    ] + [f"foot_force_{i}" for i in range(12)] \
      + [f"foot_pos_{i}"   for i in range(12)]

    for ch in no_bound:
        if ch not in bounds:
            bounds[ch] = None

    n_bounded   = sum(1 for v in bounds.values() if v is not None)
    n_unbounded = len(bounds) - n_bounded
    logger.info(
        f"[data] extract_bounds_from_config done: "
        f"{n_bounded} channels with bounds, {n_unbounded} without (limit lines suppressed)"
    )
    return bounds

def load_data(data_dir: str | Path, channel_map: dict) -> tuple[dict, int, dict]:
    """
    Load all channels from *data_dir* according to *channel_map*.
    Also loads config.yaml from *data_dir* to extract per-channel bounds.

    Returns (data_dict, n_frames, bounds_dict).
      bounds_dict: channel_name -> (lo, hi) or None.
                   None means no limit lines should be drawn for that channel.

    Raises:
        FileNotFoundError — if data_dir doesn't exist
        RuntimeError      — if 'time' channel is missing, or config.yaml is absent/unreadable
    """
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"[data] Data directory not found: {d.resolve()}")

    time_arr = _read_col(d, channel_map.get("time"))
    if time_arr is None:
        raise RuntimeError(
            f"[data] 'time' channel is missing or unreadable in {d}.\n"
            "Check your channel map — 'time' is a required channel."
        )
    n = len(time_arr)
    logger.info(f"[data] Loading data: {n} frames from {d}")

    data: dict = {"time": time_arr}

    # Scalar pose channels
    for ch in ["torso_x", "torso_y", "torso_z",
               "torso_roll", "torso_pitch", "torso_yaw",
               "torso_vx", "torso_vy", "torso_vz",
               "desired_vel_x", "qp_cost",
               "contact_FR", "contact_FL", "contact_RR", "contact_RL"]:
        data[ch] = _read_col(d, channel_map.get(ch))

    # QP acceleration channels (6 each: lin xyz + ang xyz, body frame)
    for _prefix, _key in [("qp_desired_acc", "qp_desired_acc"),
                           ("qp_solved_acc",  "qp_solved_acc")]:
        mat = np.zeros((n, 6), dtype=np.float64)
        any_loaded = False
        for _i in range(6):
            arr = _read_col(d, channel_map.get(f"{_prefix}{_i}"))
            if arr is not None:
                mat[:, _i] = arr[:n]
                any_loaded = True
        data[_key] = mat if any_loaded else None

    # Raibert landing positions, correction vectors, and swing foot positions (4 feet × 3 axes each)
    _FOOT_NAMES_LD = ["FR", "FL", "RR", "RL"]
    _AXES_LD       = ["x", "y", "z"]
    for _key_prefix, _ch_prefix in [("raibert_land",  "raibert_"),
                                     ("raibert_corr",  "raibert_corr_"),
                                     ("swing_foot",    "swing_foot_")]:
        mat = np.zeros((n, 4, 3), dtype=np.float64)
        any_loaded = False
        for _fi, _fn in enumerate(_FOOT_NAMES_LD):
            for _ai, _an in enumerate(_AXES_LD):
                ch = f"{_ch_prefix}{_fn}_{_an}"
                arr = _read_col(d, channel_map.get(ch))
                if arr is not None:
                    mat[:, _fi, _ai] = arr[:n]
                    any_loaded = True
        data[_key_prefix] = mat if any_loaded else None

    # Matrix channels: q, dq, tau
    for prefix, key in [("q", "q"), ("dq", "dq"), ("tau", "tau")]:
        mat = np.zeros((n, 12), dtype=np.float64)
        any_loaded = False
        for i in range(12):
            arr = _read_col(d, channel_map.get(f"{prefix}{i}"))
            if arr is not None:
                mat[:, i] = arr[:n]
                any_loaded = True
        data[key] = mat if any_loaded else None

    # Matrix channels: foot_forces, foot_pos
    for prefix, key in [("foot_force", "foot_forces"), ("foot_pos", "foot_pos")]:
        mat = np.zeros((n, 12), dtype=np.float64)
        any_loaded = False
        for i in range(12):
            arr = _read_col(d, channel_map.get(f"{prefix}_{i}"))
            if arr is not None:
                mat[:, i] = arr[:n]
                any_loaded = True
        data[key] = mat if any_loaded else None

    # Fill required channels with zeros if absent
    if data.get("q") is None:
        logger.warning("[data] No joint angle (q*) data found — using zeros")
        data["q"] = np.zeros((n, 12), dtype=np.float64)

    for ch in ["torso_x", "torso_y", "torso_z",
               "torso_roll", "torso_pitch", "torso_yaw"]:
        if data[ch] is None:
            logger.warning(f"[data] Channel '{ch}' missing — using zeros")
            data[ch] = np.zeros(n, dtype=np.float64)

    # Auxiliary data presence warnings
    _aux_checks = {
        "foot_forces": "foot force data (simforceFeetGlobal*) — force arrows will not render",
        "foot_pos":    "foot position data (footPosFeetGlobal*) — will fall back to FK-derived positions",
        "dq":          "joint velocity data (dq*) — dq graphs will show zeros",
        "tau":         "joint torque data (tauM*) — torque graphs will show zeros",
    }
    for key, desc in _aux_checks.items():
        if data.get(key) is None:
            logger.warning(f"[data] Missing auxiliary data: {desc}")

    _contact_keys = ["contact_FR", "contact_FL", "contact_RR", "contact_RL"]
    missing_contacts = [k for k in _contact_keys if data.get(k) is None]
    if missing_contacts:
        logger.warning(f"[data] Missing contact channels: {missing_contacts} — contact spheres will not colour")

    # ── Custom channels from config ───────────────────────────────────────
    data["custom"] = {}   # dict: channel_name -> np.ndarray or None
    import dataviz.config as _cfg_mod
    cfg = _cfg_mod.CFG
    if cfg is not None:
        raw_custom = getattr(cfg, "custom_channels", None)
        custom_dict = {}
        if raw_custom is not None:
            # custom_channels is a _Namespace or plain dict
            if hasattr(raw_custom, "__dict__"):
                custom_dict = {k: v for k, v in raw_custom.__dict__.items()
                               if not k.startswith("_")}
            elif isinstance(raw_custom, dict):
                custom_dict = raw_custom

        for ch_name, filename in custom_dict.items():
            if not filename:
                continue
            p = d / str(filename)
            if not p.exists():
                raise FileNotFoundError(
                    f"[data] custom_channels['{ch_name}']: file '{filename}' "
                    f"not found in {d}.\n"
                    f"Either add the file or remove the entry from config.json."
                )
            try:
                arr = np.loadtxt(str(p), dtype=np.float64)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                if len(arr) != n:
                    logger.warning(
                        f"[data] custom_channels['{ch_name}']: '{filename}' has "
                        f"{len(arr)} rows but data has {n} frames — truncating/padding"
                    )
                    if len(arr) < n:
                        arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)
                    else:
                        arr = arr[:n]
                data["custom"][ch_name] = arr
                logger.info(f"[data] custom_channels['{ch_name}']: loaded '{filename}' ({n} frames)")
            except Exception as exc:
                logger.error(f"[data] custom_channels['{ch_name}']: failed to load '{filename}': {exc}")
                data["custom"][ch_name] = None

    # ── Extra (auto-detected) channels from channel_map ───────────────────
    # Keys prefixed with 'extra:' are unknown files the user chose to load.
    extra_map = {k: v for k, v in channel_map.items()
                 if k.startswith("extra:") and v}
    data["extra"] = {}
    if extra_map:
        data["extra"] = load_extra_channels(d, extra_map, n)
        loaded_extra = [k for k, v in data["extra"].items() if v is not None]
        logger.info(f"[data] Loaded {len(loaded_extra)} extra channel(s): {loaded_extra}")

    # ── Load config.yaml and extract bounds ───────────────────────────────────
    train_cfg = _load_training_config(d)
    bounds = extract_bounds_from_config(train_cfg)

    dt = time_arr[1] - time_arr[0] if n > 1 else 0.0
    duration = time_arr[-1] - time_arr[0]
    logger.info(
        f"[data] load_data complete: {n} frames  dt={dt:.4f}s  duration={duration:.2f}s"
    )
    return data, n, bounds


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------
def _parse_vec3(s: str | None, default: str = "0 0 0") -> np.ndarray:
    parts = (s or default).split()
    return np.array([float(x) for x in parts], dtype=np.float64)


def parse_urdf(urdf_path: str | Path) -> tuple[
    list[str],                   # joint_order
    dict[str, tuple],            # joint_limits  {name: (lo, hi)}
    dict[str, np.ndarray],       # joint_axes    {name: vec3}
    dict[str, tuple],            # joint_origins {name: (xyz, rpy)}
    dict[str, str],              # link_parents  {child_link: parent_link}
]:
    """
    Parse a URDF file and return joint information.

    Raises:
        FileNotFoundError — if urdf_path does not exist
        ValueError        — if no actuated joints are found
    """
    import xml.etree.ElementTree as ET

    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"[data] URDF file not found: {path.resolve()}")

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        raise ValueError(f"[data] URDF is not valid XML: {path}: {exc}") from exc

    root = tree.getroot()
    joint_order: list[str] = []
    joint_limits:  dict = {}
    joint_axes:    dict = {}
    joint_origins: dict = {}
    link_parents:  dict = {}

    for joint in root.findall("joint"):
        jtype  = joint.get("type", "fixed")
        name   = joint.get("name")
        parent = joint.find("parent")
        child  = joint.find("child")

        if parent is not None and child is not None:
            link_parents[child.get("link")] = parent.get("link")

        if jtype not in ("revolute", "continuous", "prismatic"):
            continue

        origin   = joint.find("origin")
        xyz      = _parse_vec3(origin.get("xyz") if origin is not None else None)
        rpy      = _parse_vec3(origin.get("rpy") if origin is not None else None)
        joint_origins[name] = (xyz, rpy)

        axis_el = joint.find("axis")
        joint_axes[name] = _parse_vec3(
            axis_el.get("xyz") if axis_el is not None else None, default="1 0 0"
        )

        limit_el = joint.find("limit")
        if limit_el is not None:
            joint_limits[name] = (
                float(limit_el.get("lower", -1e9)),
                float(limit_el.get("upper",  1e9)),
            )
        else:
            logger.warning(f"[data] Joint '{name}' has no <limit> element — using ±1e9")
            joint_limits[name] = (-1e9, 1e9)

        joint_order.append(name)

    if not joint_order:
        raise ValueError(
            f"[data] No actuated joints (revolute/continuous/prismatic) found in {path}.\n"
            "Check the URDF file."
        )

    logger.info(f"[data] URDF parsed: {len(joint_order)} joints from {path.name}")
    return joint_order, joint_limits, joint_axes, joint_origins, link_parents


# ---------------------------------------------------------------------------
# Kinematics helpers (pure math, no GL)
# ---------------------------------------------------------------------------
def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    t    = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    return np.array([
        [t*x*x+c,   t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z, t*y*y+c,   t*y*z-s*x],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c  ],
    ], dtype=np.float64)


def euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        rot_axis_angle([0, 0, 1], yaw)
        @ rot_axis_angle([0, 1, 0], pitch)
        @ rot_axis_angle([1, 0, 0], roll)
    )


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    return euler_to_rot(rpy[0], rpy[1], rpy[2])


def is_near_limit(lo: float, hi: float, val: float, tol: float | None = None) -> bool:
    """Return True if *val* is within *tol* fraction of either joint limit."""
    if tol is None:
        tol = _cfg_mod.CFG.rendering.joint_limit_warn_tol if _cfg_mod.CFG is not None else 0.05
    span = abs(hi - lo)
    if span < 1e-9:
        return False
    return (val - lo) < tol * span or (hi - val) < tol * span


# ---------------------------------------------------------------------------
# Forward kinematics builder
# ---------------------------------------------------------------------------
def build_fk_fn(urdf_path: str | Path, joint_order: list[str]):
    """
    Build and return (fk_fn, robot) using yourdfpy.

    fk_fn(q_vec, torso_pos, torso_rpy) → dict[link_name, 4x4 float32 matrix]

    Raises:
        ImportError  — if yourdfpy is not installed
        RuntimeError — if yourdfpy fails to load the URDF
    """
    try:
        from yourdfpy import URDF as _URDF
    except ImportError as exc:
        raise ImportError(
            "[data] 'yourdfpy' is required for forward kinematics.\n"
            "Install with: pip install yourdfpy"
        ) from exc

    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"[data] URDF not found for FK: {path.resolve()}")

    try:
        robot = _URDF.load(str(path))
        logger.info(f"[data] yourdfpy loaded: {path.name}")
    except Exception as exc:
        raise RuntimeError(
            f"[data] yourdfpy failed to load {path}: {exc}"
        ) from exc

    ref = "trunk" if "trunk" in robot.link_map else robot.base_link
    logger.debug(f"[data] FK reference link: '{ref}'")

    def fk(q_vec: np.ndarray,
           torso_pos: np.ndarray,
           torso_rpy: tuple[float, float, float]) -> dict[str, np.ndarray]:
        cfg = {jname: float(q_vec[i]) for i, jname in enumerate(joint_order)}
        robot.update_cfg(cfg)

        R_world = euler_to_rot(*torso_rpy).astype(np.float32)
        T_world = np.eye(4, dtype=np.float32)
        T_world[:3, :3] = R_world
        T_world[:3, 3]  = torso_pos

        transforms: dict[str, np.ndarray] = {}

        # Primary path: scene graph geometry nodes
        try:
            scene = robot.scene
            for node_name in scene.graph.nodes_geometry:
                T_node, _ = scene.graph.get(node_name)
                if T_node is not None:
                    link_name = node_name.split("/")[0]
                    if link_name not in transforms:
                        transforms[link_name] = T_world @ T_node.astype(np.float32)
        except Exception as exc:
            logger.debug(f"[data] FK scene graph path failed: {exc}")

        # Fallback path: get_transform per link
        for link in robot.link_map:
            if link not in transforms:
                try:
                    T = robot.get_transform(link, ref).astype(np.float32)
                    transforms[link] = T_world @ T
                except Exception as exc:
                    logger.debug(f"[data] FK get_transform failed for link '{link}': {exc}")

        return transforms

    return fk, robot


# ---------------------------------------------------------------------------
# Robot/data directory scanner
# ---------------------------------------------------------------------------
def find_robots_dir() -> Path:
    """Walk up from cwd looking for a 'robots/' directory (max 4 levels)."""
    p = Path.cwd()
    for _ in range(4):
        candidate = p / "robots"
        if candidate.is_dir():
            logger.debug(f"[data] Found robots dir: {candidate}")
            return candidate
        p = p.parent
    logger.warning("[data] Could not find a 'robots/' directory — defaulting to ./robots")
    return Path("robots")


def _detect_legs(joint_order: list[str]) -> dict:
    """
    Group joint indices by leg label (e.g. 'FR', 'FL', 'RR', 'RL').
    Returns OrderedDict: {label: [joint_index, ...]}
    """
    from collections import OrderedDict
    legs: dict = OrderedDict()
    for i, name in enumerate(joint_order):
        parts = name.split("_")
        label = parts[1] if len(parts) >= 2 else parts[0]
        if label not in legs:
            legs[label] = []
        legs[label].append(i)
    return legs


def scan_robots(robots_dir: Path) -> list[dict]:
    """
    Scan *robots_dir* for robot bundles.
    Each bundle is a subdirectory with a URDF and data folders.

    Returns a list of dicts:
        {name, bundle (Path), urdf (Path), data_dirs (list[Path])}
    """
    robots: list[dict] = []

    if not robots_dir.is_dir():
        logger.warning(f"[data] Robots directory does not exist: {robots_dir.resolve()}")
        return robots

    for robot_bundle in sorted(robots_dir.iterdir()):
        if not robot_bundle.is_dir():
            continue

        # Find URDF (skip anything inside a 'data' subfolder)
        urdf_path = None
        for p in sorted(robot_bundle.rglob("*.urdf")):
            if "data" not in p.parts:
                urdf_path = p
                break

        if urdf_path is None:
            logger.debug(f"[data] No URDF found in {robot_bundle.name}, skipping")
            continue

        # Find data directories
        data_dirs: list[Path] = []
        direct_data = robot_bundle / "data"

        if direct_data.is_dir():
            sub_runs = [
                d for d in sorted(direct_data.iterdir())
                if d.is_dir() and (d / "time.txt").exists()
            ]
            if sub_runs:
                data_dirs.extend(sub_runs)
            elif (direct_data / "time.txt").exists():
                data_dirs.append(direct_data)

        for d in sorted(robot_bundle.iterdir()):
            if d.is_dir() and d.name != "data" and (d / "time.txt").exists():
                data_dirs.append(d)

        robots.append({
            "name":      robot_bundle.name,
            "bundle":    robot_bundle,
            "urdf":      urdf_path,
            "data_dirs": data_dirs,
        })
        logger.debug(
            f"[data] Robot '{robot_bundle.name}': urdf={urdf_path.name}, "
            f"{len(data_dirs)} data dir(s)"
        )

    logger.info(f"[data] scan_robots: found {len(robots)} robot(s) in {robots_dir}")
    return robots
