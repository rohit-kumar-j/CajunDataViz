"""
dataviz/state.py

Owns:
  - AppState       (playback, camera, per-frame queries)
  - GraphScrubState (ghost robot, graph scrub window)
  - TabState       (per-tab container: data, robot, GL resources, UI fracs)

Pure Python — no GL, no imgui.
Only imports: dataviz.config, standard library, numpy, loguru.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

import dataviz.config as _cfg_mod


# ---------------------------------------------------------------------------
# AppState — playback + camera for one tab
# ---------------------------------------------------------------------------
class AppState:
    """
    Owns the main playback head (self.frame) and camera for one loaded tab.
    All frame navigation goes through set_frame() so side-effects stay consistent.
    """

    def __init__(self):
        # Playback
        self.data:        Optional[dict] = None
        self.n_frames:    int            = 0
        self.frame:       int            = 0
        self.playing:     bool           = False
        self.play_speed:  float          = _cfg_mod.CFG.playback.default_play_speed if _cfg_mod.CFG else 0.5
        self._last_t:     float          = 0.0
        self._frac_acc:   float          = 0.0
        self.loop_start:  int            = 0
        self.loop_end:    int            = 0

        # Camera
        self.cam_yaw:    float          = _cfg_mod.CFG.camera.default_yaw    if _cfg_mod.CFG else 45.0
        self.cam_pitch:  float          = _cfg_mod.CFG.camera.default_pitch  if _cfg_mod.CFG else 25.0
        self.cam_dist:   float          = _cfg_mod.CFG.camera.default_dist   if _cfg_mod.CFG else 3.5
        self.cam_target: np.ndarray     = np.array([0.0, 0.0, 0.3])
        self.cam_follow_mode: str       = "main"   # 'main' | 'ghost'

        # Overlays
        self.show_forces:       bool = True
        self.show_contacts:     bool = True
        self.show_limits:       bool = True
        self.show_grid:         bool = True
        self.show_trajectory:   bool = False
        self.show_joint_frames: bool = False

        # Rendering tweaks
        traj_len   = _cfg_mod.CFG.rendering.trajectory_length if _cfg_mod.CFG else 200
        fscale     = _cfg_mod.CFG.rendering.force_scale       if _cfg_mod.CFG else 0.003
        fog_s      = _cfg_mod.CFG.rendering.fog_start         if _cfg_mod.CFG else 3.0
        fog_e      = _cfg_mod.CFG.rendering.fog_end           if _cfg_mod.CFG else 18.0

        self.traj_length:  int   = traj_len
        self.force_scale:  float = fscale
        self.fog_start:    float = fog_s
        self.fog_end:      float = fog_e

        # Back-reference to GraphScrubState (set after construction)
        self.g_ref: Optional["GraphScrubState"] = None

    # ------------------------------------------------------------------
    # Frame navigation (always use these — keeps g_ref in sync)
    # ------------------------------------------------------------------
    def _hi(self) -> int:
        return self.loop_end if self.loop_end > 0 else max(0, self.n_frames - 1)

    def set_frame(self, f: int) -> None:
        """Jump to absolute frame. Clamps, syncs camera, syncs ghost."""
        if self.n_frames == 0:
            return
        self.frame = max(0, min(self.n_frames - 1, int(f)))
        self._sync_cam()

        G = self.g_ref
        if G is None:
            return

        if G.play_mode == "graph":
            G.backup_frame = self.frame
            gw = _cfg_mod.CFG.playback.graph_window if _cfg_mod.CFG else 50
            lo = max(0, self.frame - gw)
            hi = min(self.n_frames - 1, self.frame + gw)
            G.frame = max(lo, min(hi, G.frame))
        else:
            G.frame = self.frame

    def step(self, delta: int) -> None:
        self.set_frame(self.frame + delta)

    def toggle_play(self) -> None:
        self.playing   = not self.playing
        self._last_t   = time.time()
        self._frac_acc = 0.0

    # ------------------------------------------------------------------
    # Per-frame update (called at update_hz from viewer._update)
    # ------------------------------------------------------------------
    def advance(self) -> None:
        G = self.g_ref
        if G is not None and G.play_mode == "graph":
            return
        if not self.playing or self.data is None or self.n_frames < 2:
            return

        now      = time.time()
        dt_real  = now - self._last_t
        self._last_t = now
        dt_sim   = float(self.data["time"][1] - self.data["time"][0])
        self._frac_acc += dt_real * self.play_speed / max(dt_sim, 1e-9)
        steps    = int(self._frac_acc)
        self._frac_acc -= steps

        lo   = self.loop_start
        hi   = self._hi()
        span = max(hi - lo + 1, 2)
        self.frame = lo + (self.frame - lo + steps) % span
        self._sync_cam()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def update_cam(self) -> None:
        if self.data is None:
            return
        G = self.g_ref
        f = G.frame if (G is not None and self.cam_follow_mode == "ghost") else self.frame
        f = max(0, min(self.n_frames - 1, f))
        self.cam_target = np.array([
            float(self.data["torso_x"][f]),
            float(self.data["torso_y"][f]),
            float(self.data["torso_z"][f]),
        ])

    def _sync_cam(self) -> None:
        self.update_cam()

    def cam_eye(self) -> np.ndarray:
        min_p = _cfg_mod.CFG.camera.min_pitch if _cfg_mod.CFG else 5.0
        max_p = _cfg_mod.CFG.camera.max_pitch if _cfg_mod.CFG else 85.0
        yr = math.radians(self.cam_yaw)
        pr = math.radians(max(min_p, min(max_p, self.cam_pitch)))
        return self.cam_target + self.cam_dist * np.array([
            math.cos(pr) * math.cos(yr),
            math.cos(pr) * math.sin(yr),
            math.sin(pr),
        ])

    # ------------------------------------------------------------------
    # Per-frame data extraction (no copies — just views/scalars)
    # ------------------------------------------------------------------
    def frame_state(self, f: Optional[int] = None):
        """Return (q, torso_pos, torso_rpy) for frame *f* (defaults to self.frame)."""
        if f is None:
            f = self.frame
        d   = self.data
        q   = d["q"][f] if d.get("q") is not None else np.zeros(12)
        pos = np.array([
            float(d["torso_x"][f]),
            float(d["torso_y"][f]),
            float(d["torso_z"][f]),
        ])
        rpy = (
            float(d["torso_roll"][f]),
            float(d["torso_pitch"][f]),
            float(d["torso_yaw"][f]),
        )
        return q, pos, rpy


# ---------------------------------------------------------------------------
# GraphScrubState — ghost robot + graph scrub window
# ---------------------------------------------------------------------------
class GraphScrubState:
    """
    Controls the ghost robot position and the ±N frame graph scrub window.
    play_mode:
        'main'  — ghost follows main frame exactly (no independent motion)
        'graph' — ghost is independently playable within ±graph_window of backup_frame
    """

    def __init__(self):
        self.frame:         int   = 0
        self.frozen:        bool  = False
        self.play_mode:     str   = "main"   # 'main' | 'graph'
        self.backup_frame:  int   = 0
        self._playing:      bool  = False
        self._last_t:       float = 0.0
        self._frac_acc:     float = 0.0

    def _gw(self) -> int:
        return _cfg_mod.CFG.playback.graph_window if _cfg_mod.CFG else 50

    # ------------------------------------------------------------------
    def scrub(self, fi: int, n_frames: int) -> None:
        """Move ghost to *fi*, clamped to the current scrub window."""
        anchor = self.backup_frame if self.play_mode == "graph" else fi
        gw     = self._gw()
        lo     = max(0, anchor - gw)
        hi     = min(n_frames - 1, anchor + gw)
        self.frame  = max(lo, min(hi, fi))
        self.frozen = True

    def release(self) -> None:
        self.frozen = False

    def set_mode(self, mode: str, s_frame: int = 0) -> None:
        if mode == self.play_mode:
            return
        if mode == "graph":
            self.play_mode    = "graph"
            self._playing     = False
            self.backup_frame = s_frame
            self.frame        = s_frame
            logger.debug(f"[state] GraphScrubState → graph mode at frame {s_frame}")
        else:
            self.play_mode = "main"
            self._playing  = False
            self.frozen    = False
            logger.debug("[state] GraphScrubState → main mode")

    def toggle_graph_play(self, n_frames: int) -> None:
        if self.play_mode != "graph":
            return
        self._playing  = not self._playing
        self._last_t   = time.time()
        self._frac_acc = 0.0

    def advance_graph(self, n_frames: int, play_speed: float, dt_sim: float) -> None:
        if self.play_mode != "graph" or not self._playing:
            return
        gw   = self._gw()
        now  = time.time()
        dt   = now - self._last_t
        self._last_t = now
        self._frac_acc += dt * play_speed / max(dt_sim, 1e-9)
        steps = int(self._frac_acc)
        self._frac_acc -= steps

        lo   = max(0, self.backup_frame - gw)
        hi   = min(n_frames - 1, self.backup_frame + gw)
        span = max(hi - lo + 1, 2)
        self.frame = lo + (self.frame - lo + steps) % span


# ---------------------------------------------------------------------------
# TabState — per-tab container
# ---------------------------------------------------------------------------
class TabState:
    """
    One logical tab in the viewer. Holds all per-tab state:
      - Loaded data + robot description
      - AppState (S) and GraphScrubState (G)
      - GL resource handles (populated by gl_core after loading)
      - Graph configuration
      - Layout fractions
      - Picker state
    """

    _id_counter: int = 0

    def __init__(self, label: str = "New Tab"):
        TabState._id_counter += 1
        self.id:    int = TabState._id_counter
        self.label: str = label

        # Robot description (populated by do_load)
        self.urdf_path:    Optional[str]        = None
        self.joint_order:  list[str]            = []
        self.joint_limits: dict                 = {}
        self.joint_axes:   dict                 = {}
        self.joint_origins:dict                 = {}
        self.fk_fn                              = None
        self.urdf_robot                         = None
        self.channel_map:  dict                 = {}
        self.loaded:       bool                 = False

        # Per-tab playback state
        self.S = AppState()
        self.G = GraphScrubState()
        self.S.g_ref = self.G

        # Layout fractions (persist across tab switches, saved to layout file)
        vp_f  = _cfg_mod.CFG.layout.default_vp_frac       if _cfg_mod.CFG else 0.54
        bot_f = _cfg_mod.CFG.layout.default_bot_frac       if _cfg_mod.CFG else 0.22
        gh_f  = _cfg_mod.CFG.layout.default_graph_h_frac   if _cfg_mod.CFG else 0.72

        self.vp_frac:       float = vp_f
        self.bot_frac:      float = bot_f
        self.graph_h_frac:  float = gh_f

        # GL mesh resources (populated by gl_core.init_tab_gl)
        self.MESH_VAOS:    dict = {}
        self.LINK_TO_MESH: dict = {}
        self.legs:         dict = {}

        # FBO resources (populated by gl_core.ensure_fbo)
        self.fbo            = None
        self.fbo_tex        = None
        self.fbo_depth      = None
        self.fbo_w:   int   = 0
        self.fbo_h:   int   = 0

        # Graph configuration (populated by init_graphs)
        self.graphs:      list  = []
        self.GRAPH_PARAMS: dict = {}
        self.PARAM_KEYS:  list  = []
        self._graph_ph_drag: dict = {}

        # Per-tab picker state
        self.picker: dict = _make_picker_state()

        # Graph probes: list of dicts {param, graph_idx, series_idx, frame_idx, value, t}
        # Persistent until manually removed. Hidden when outside ±graph_window.
        self.probes: list = []
    # ------------------------------------------------------------------
    # Graph initialisation (called after do_load)
    # ------------------------------------------------------------------
    def init_graphs(self) -> None:
        """
        Build GRAPH_PARAMS lambdas and graph list from loaded data + _cfg_mod.CFG.live_graphs.

        Channel name → GRAPH_PARAMS key mapping:
          q0..q11        → 'q <joint_name>'   (index into joint_order)
          tau0..tau11    → 'tau <joint_name>'
          dq0..dq11      → 'dq <joint_name>'
          torso_roll/pitch/yaw/x/y/z → direct key
          contact_*      → direct key
          desired_vel_x  → direct key

        If a channel name from config cannot be resolved, logs an ERROR and skips it.
        """
        if self.S.data is None:
            logger.error(f"[state] Tab '{self.label}': init_graphs called before data loaded")
            return

        d  = self.S.data
        gp: dict = {}

        # ── Build the full GRAPH_PARAMS registry (same as before) ─────────
        gp["torso_x"]       = lambda data, f: float(data["torso_x"][f])
        gp["torso_y"]       = lambda data, f: float(data["torso_y"][f])
        gp["torso_z"]       = lambda data, f: float(data["torso_z"][f])
        gp["torso_roll"]    = lambda data, f: math.degrees(float(data["torso_roll"][f]))
        gp["torso_pitch"]   = lambda data, f: math.degrees(float(data["torso_pitch"][f]))
        gp["torso_yaw"]     = lambda data, f: math.degrees(float(data["torso_yaw"][f]))
        gp["desired_vel_x"] = lambda data, f: (
            float(data["desired_vel_x"][f]) if data.get("desired_vel_x") is not None else 0.0
        )
        for _name in ["contact_FL", "contact_FR", "contact_RL", "contact_RR"]:
            gp[_name] = (
                lambda n: lambda data, f: float(data[n][f]) if data.get(n) is not None else 0.0
            )(_name)
        for _ji, _jn in enumerate(self.joint_order):
            gp[f"q {_jn}"]   = (lambda j: lambda data, f: math.degrees(float(data["q"][f, j])))(_ji)
            gp[f"tau {_jn}"] = (
                lambda j: lambda data, f: float(data["tau"][f, j]) if data.get("tau") is not None else 0.0
            )(_ji)
            gp[f"dq {_jn}"]  = (
                lambda j: lambda data, f: float(data["dq"][f, j]) if data.get("dq") is not None else 0.0
            )(_ji)

        # Foot force components (12 scalars: 4 feet × 3 axes)
        _FOOT_LABELS = ["FR_x","FR_y","FR_z","FL_x","FL_y","FL_z",
                        "RR_x","RR_y","RR_z","RL_x","RL_y","RL_z"]
        for _fi in range(12):
            _lbl = _FOOT_LABELS[_fi]
            gp[f"foot_force {_lbl}"] = (
                lambda i: lambda data, f: float(data["foot_forces"][f, i])
                    if data.get("foot_forces") is not None else 0.0
            )(_fi)

        self.GRAPH_PARAMS = gp          # preliminary — will be rebuilt after custom channels
        self.PARAM_KEYS   = list(gp.keys())

        # ── Build a lookup: config channel name → GRAPH_PARAMS key ────────
        _channel_to_param: dict[str, str] = {}

        # Direct keys
        for direct in ["torso_x","torso_y","torso_z",
                        "torso_roll","torso_pitch","torso_yaw",
                        "desired_vel_x",
                        "contact_FL","contact_FR","contact_RL","contact_RR"]:
            _channel_to_param[direct] = direct

        # Indexed joint channels
        for prefix in ("q", "tau", "dq"):
            for idx, jn in enumerate(self.joint_order):
                _channel_to_param[f"{prefix}{idx}"] = f"{prefix} {jn}"

        # Foot force components: foot_force_0 .. foot_force_11
        for _fi in range(12):
            _channel_to_param[f"foot_force_{_fi}"] = f"foot_force {_FOOT_LABELS[_fi]}"

        # ── Custom channels from data["custom"] ───────────────────────────
        # These are arbitrary named scalar arrays loaded by data.py from config.
        custom_data = d.get("custom", {})
        for _ch_name, _arr in custom_data.items():
            if _arr is None:
                continue   # file failed to load — already logged as error
            # Add to GRAPH_PARAMS
            gp[f"custom {_ch_name}"] = (
                lambda nm: lambda data, f: float(data["custom"][nm][f])
                    if data.get("custom") and data["custom"].get(nm) is not None else 0.0
            )(_ch_name)
            # Add to channel_to_param so live_graphs config can reference them by name
            _channel_to_param[_ch_name] = f"custom {_ch_name}"

        # Rebuild PARAM_KEYS after custom channels are added
        self.GRAPH_PARAMS = gp
        self.PARAM_KEYS   = list(gp.keys())
        if custom_data:
            loaded = [k for k, v in custom_data.items() if v is not None]
            logger.info(f"[state] Custom channels registered for graphing: {loaded}")

        colors = (
            [tuple(c) for c in _cfg_mod.CFG.colors.graph_series]
            if _cfg_mod.CFG else [
                (0.25,0.55,1.00),(0.20,0.85,0.40),(1.00,0.70,0.15),(0.85,0.25,0.85),
                (0.15,0.85,0.85),(1.00,0.35,0.35),(0.85,0.85,0.20),(0.60,0.40,1.00),
            ]
        )

        def _ms(param: str, ci: int, lo: float = -30.0, hi: float = 30.0) -> dict:
            r, g, b = colors[ci % len(colors)][:3]
            return {
                "param": param, "enabled": True, "color": (r, g, b),
                "limit_lo": lo, "limit_hi": hi,
                "lim_lo_str": str(int(lo)), "lim_hi_str": str(int(hi)),
            }

        def _mg(title: str, series: list, lo: float, hi: float) -> dict:
            from dataviz.ui_panels import _graph_id_counter as _gic
            import dataviz.ui_panels as _uip
            _uip._graph_id_counter += 1
            return {
                "title": title, "series": series, "lo": lo, "hi": hi,
                "lo_str": str(lo), "hi_str": str(hi),
                "lo_edit": False, "hi_edit": False, "hovered": -1,
                "side_w": 200,
                "_gid": _uip._graph_id_counter,
            }

        # ── Read live_graphs from config and build graph list ─────────────
        self.graphs = []

        if _cfg_mod.CFG is None:
            logger.warning("[state] _cfg_mod.CFG not loaded — no live graphs will be created")
            logger.info(f"[state] Tab '{self.label}': graphs initialised (0 panels, no config)")
            return

        raw_graphs = getattr(_cfg_mod.CFG, "live_graphs", None)
        if raw_graphs is None:
            logger.warning(
                "[state] 'live_graphs' missing from config — no graphs will be created. "
                "Add a 'live_graphs' section to config.json."
            )
            logger.info(f"[state] Tab '{self.label}': graphs initialised (0 panels)")
            return

        # Iterate in config order
        try:
            graph_items = raw_graphs.__dict__.items() if hasattr(raw_graphs, "__dict__") else {}
        except Exception as exc:
            logger.error(f"[state] Could not iterate live_graphs config: {exc}")
            return

        for title, graph_cfg in graph_items:
            if title.startswith("_"):
                continue  # skip comment keys

            # graph_cfg is a _Namespace; extract fields
            try:
                series_names = list(graph_cfg.series)
                y_min = float(graph_cfg.y_min)
                y_max = float(graph_cfg.y_max)
            except AttributeError as exc:
                logger.error(
                    f"[state] live_graphs['{title}'] is malformed — "
                    f"expected {{series, y_min, y_max}}: {exc}"
                )
                continue

            # Resolve each channel name → param key
            series_list = []
            for ci, ch_name in enumerate(series_names):
                param_key = _channel_to_param.get(ch_name)
                if param_key is None:
                    logger.error(
                        f"[state] live_graphs['{title}']: channel '{ch_name}' cannot be resolved. "
                        f"Valid examples: q0..q11, tau0..tau11, torso_roll, desired_vel_x. "
                        f"Skipping this series."
                    )
                    continue
                if param_key not in gp:
                    logger.error(
                        f"[state] live_graphs['{title}']: resolved param '{param_key}' "
                        f"not found in GRAPH_PARAMS (joint count mismatch?). Skipping."
                    )
                    continue
                series_list.append(_ms(param_key, ci, y_min, y_max))

            if not series_list:
                logger.warning(
                    f"[state] live_graphs['{title}']: all series failed to resolve — "
                    f"graph will not be created."
                )
                continue

            self.graphs.append(_mg(title, series_list, y_min, y_max))
            logger.debug(
                f"[state] live_graphs['{title}']: {len(series_list)} series, "
                f"y=[{y_min}, {y_max}]"
            )

        logger.info(
            f"[state] Tab '{self.label}': graphs initialised "
            f"({len(self.graphs)} panels from config)"
        )


# ---------------------------------------------------------------------------
# Picker state factory
# ---------------------------------------------------------------------------
def _make_picker_state() -> dict:
    return {
        "open":      True,
        "page":      0,
        "robot_idx": 0,
        "data_idx":  0,
        "robots":    [],
        "files":     [],
        "cmap":      {},
        "combos":    {},
        "error":     "",
    }


# ---------------------------------------------------------------------------
# do_load — wire data + URDF into a TabState
# ---------------------------------------------------------------------------
def do_load(
    tab: TabState,
    urdf_path_str: str,
    data_dir_str: str,
    channel_map: Optional[dict] = None,
) -> None:
    """
    Load URDF and data into *tab*.
    Sets tab.loaded = True on success.

    Raises:
        FileNotFoundError — missing URDF or data dir
        RuntimeError      — data load failure
        ValueError        — URDF parse failure
    """
    # Lazy import to avoid circular at module level
    from dataviz.data import (
        parse_urdf, build_fk_fn, load_data,
        load_channel_map_file, scan_data_files, auto_map_channels,
    )

    urdf_path = Path(urdf_path_str)
    data_dir  = Path(data_dir_str)

    if not urdf_path.is_file():
        raise FileNotFoundError(f"[state] URDF not found: {urdf_path.resolve()}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"[state] Data directory not found: {data_dir.resolve()}")

    # Resolve channel map
    if channel_map is None:
        channel_map = load_channel_map_file(data_dir)
    if channel_map is None:
        files       = scan_data_files(data_dir)
        channel_map = auto_map_channels(files)

    logger.info(f"[state] Loading tab '{tab.label}': urdf={urdf_path.name}, data={data_dir.name}")

    joint_order, joint_limits, joint_axes, joint_origins, link_parents = \
        parse_urdf(urdf_path_str)

    fk_fn, urdf_robot = build_fk_fn(urdf_path_str, joint_order)
    data, n           = load_data(data_dir_str, channel_map)

    # Handle joint count mismatch
    if data["q"] is not None and data["q"].shape[1] != len(joint_order):
        m = min(data["q"].shape[1], len(joint_order))
        logger.warning(
            f"[state] Joint count mismatch: data has {data['q'].shape[1]} columns, "
            f"URDF has {len(joint_order)} joints — using first {m}"
        )
        joint_order = joint_order[:m]

    # Populate tab
    tab.urdf_path     = str(urdf_path)
    tab.joint_order   = joint_order
    tab.joint_limits  = joint_limits
    tab.joint_axes    = joint_axes
    tab.joint_origins = joint_origins
    tab.fk_fn         = fk_fn
    tab.urdf_robot    = urdf_robot
    tab.channel_map   = channel_map
    tab.label         = data_dir.name

    tab.S.data      = data
    tab.S.n_frames  = n
    tab.S.cam_target = np.array([
        float(data["torso_x"][0]),
        float(data["torso_y"][0]),
        float(data["torso_z"][0]),
    ])
    tab.S.cam_dist  = _cfg_mod.CFG.camera.default_dist   if _cfg_mod.CFG else 2.5
    tab.S.cam_pitch = _cfg_mod.CFG.camera.default_pitch  if _cfg_mod.CFG else 22.0
    tab.G.frame         = 0
    tab.G.backup_frame  = 0
    tab.loaded          = True
    tab.picker["open"]  = False

    logger.info(
        f"[state] Tab '{tab.label}' loaded: {n} frames, {len(joint_order)} joints"
    )
