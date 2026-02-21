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

OPTIONAL_CHANNELS: list[str] = [
    "desired_vel_x",
    "contact_FR", "contact_FL", "contact_RR", "contact_RL",
] + [f"dq{i}"          for i in range(12)] \
  + [f"tau{i}"         for i in range(12)] \
  + [f"foot_force_{i}" for i in range(12)] \
  + [f"foot_pos_{i}"   for i in range(12)]

ALL_CHANNELS: list[str] = REQUIRED_CHANNELS + OPTIONAL_CHANNELS

CHANNEL_LABELS: dict[str, str] = {
    "time":         "Time (s)",
    "torso_x":      "Torso X (world)",
    "torso_y":      "Torso Y (world)",
    "torso_z":      "Torso Z (world)",
    "torso_roll":   "Torso Roll (rad)",
    "torso_pitch":  "Torso Pitch (rad)",
    "torso_yaw":    "Torso Yaw (rad)",
    "desired_vel_x": "Desired Vel X (m/s)",
    "contact_FR":   "Contact FR (0/1)",
    "contact_FL":   "Contact FL (0/1)",
    "contact_RR":   "Contact RR (0/1)",
    "contact_RL":   "Contact RL (0/1)",
}
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
    "desired_vel_x.txt": "desired_vel_x",
    "contact_FR.txt":    "contact_FR",
    "contact_FL.txt":    "contact_FL",
    "contact_RR.txt":    "contact_RR",
    "contact_RL.txt":    "contact_RL",
}
for _i in range(12):
    AUTO_PATTERNS[f"q{_i}.txt"]                  = f"q{_i}"
    AUTO_PATTERNS[f"dq{_i}.txt"]                 = f"dq{_i}"
    AUTO_PATTERNS[f"tauM{_i}.txt"]               = f"tau{_i}"
    AUTO_PATTERNS[f"simforceFeetGlobal{_i}.txt"] = f"foot_force_{_i}"
    AUTO_PATTERNS[f"footPosFeetGlobal{_i}.txt"]  = f"foot_pos_{_i}"


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


def load_data(data_dir: str | Path, channel_map: dict) -> tuple[dict, int]:
    """
    Load all channels from *data_dir* according to *channel_map*.
    Returns (data_dict, n_frames).

    Raises:
        FileNotFoundError  — if data_dir doesn't exist
        RuntimeError       — if the 'time' channel is missing/unloadable
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
               "desired_vel_x",
               "contact_FR", "contact_FL", "contact_RR", "contact_RL"]:
        data[ch] = _read_col(d, channel_map.get(ch))

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
                logger.error(
                    f"[data] custom_channels['{ch_name}']: file '{filename}' "
                    f"not found in {d} — channel will be unavailable"
                )
                data["custom"][ch_name] = None
                continue
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

    dt = time_arr[1] - time_arr[0] if n > 1 else 0.0
    duration = time_arr[-1] - time_arr[0]
    logger.info(
        f"[data] Loaded {n} frames  dt={dt:.4f}s  duration={duration:.2f}s"
    )
    return data, n


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
