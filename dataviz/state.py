"""
dataviz/state.py  — one change from original:
  AppState.__init__: added  self.show_raibert_boxes = True
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

import dataviz.config as _cfg_mod


class AppState:
    def __init__(self):
        self.data:        Optional[dict] = None
        self.n_frames:    int            = 0
        self.frame:       int            = 0
        self.playing:     bool           = False
        self.play_speed:  float          = _cfg_mod.CFG.playback.default_play_speed if _cfg_mod.CFG else 0.5
        self._last_t:     float          = 0.0
        self._frac_acc:   float          = 0.0
        self.loop_start:  int            = 0
        self.loop_end:    int            = 0

        self.cam_yaw:    float          = _cfg_mod.CFG.camera.default_yaw    if _cfg_mod.CFG else 45.0
        self.cam_pitch:  float          = _cfg_mod.CFG.camera.default_pitch  if _cfg_mod.CFG else 25.0
        self.cam_dist:   float          = _cfg_mod.CFG.camera.default_dist   if _cfg_mod.CFG else 3.5
        self.cam_target: np.ndarray     = np.array([0.0, 0.0, 0.3])
        self.cam_follow_mode: str       = "main"

        self.show_forces:       bool = True
        self.show_contacts:     bool = True
        self.show_limits:       bool = True
        self.show_grid:         bool = True
        self.show_trajectory:   bool = False
        self.show_joint_frames: bool = False
        self.show_raibert_boxes: bool = True   # ← NEW: toggle Raibert boxes/arrows

        self.realtime_mode:      bool  = False  # play at 1× wall-clock speed using time.txt
        self._realtime_start_wall: float = 0.0  # wall-clock time when realtime play began
        self._realtime_start_sim:  float = 0.0  # sim time at the frame where RT play began

        traj_len   = _cfg_mod.CFG.rendering.trajectory_length if _cfg_mod.CFG else 200
        fscale     = _cfg_mod.CFG.rendering.force_scale       if _cfg_mod.CFG else 0.003
        fog_s      = _cfg_mod.CFG.rendering.fog_start         if _cfg_mod.CFG else 3.0
        fog_e      = _cfg_mod.CFG.rendering.fog_end           if _cfg_mod.CFG else 18.0

        self.traj_length:  int   = traj_len
        self.force_scale:  float = fscale
        self.fog_start:    float = fog_s
        self.fog_end:      float = fog_e

        self.g_ref: Optional["GraphScrubState"] = None

    def _hi(self) -> int:
        return self.loop_end if self.loop_end > 0 else max(0, self.n_frames - 1)

    def set_frame(self, f: int) -> None:
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
        # In realtime mode, always force MAIN play mode so S.frame advances freely
        G = self.g_ref
        if getattr(self, "realtime_mode", False) and G is not None:
            if G.play_mode != "main":
                G.set_mode("main", self.frame)
            G._playing = False   # stop any graph-mode play
        self.playing   = not self.playing
        self._last_t   = time.time()
        self._frac_acc = 0.0
        if self.playing and self.realtime_mode and self.data is not None:
            self._realtime_start_wall = time.time()
            self._realtime_start_sim  = float(self.data["time"][self.frame])

    def start_realtime(self) -> None:
        """Re-anchor realtime playback. Forces MAIN mode and stops graph play."""
        G = self.g_ref
        if G is not None:
            if G.play_mode != "main":
                G.set_mode("main", self.frame)
            G._playing = False   # stop any active graph-mode playback
        if self.data is not None:
            self._realtime_start_wall = time.time()
            self._realtime_start_sim  = float(self.data["time"][self.frame])

    def advance(self) -> None:
        G = self.g_ref
        # In graph play mode S.frame is normally frozen, but realtime overrides this.
        if G is not None and G.play_mode == "graph" and not getattr(self, "realtime_mode", False):
            return
        if not self.playing or self.data is None or self.n_frames < 2:
            return

        if self.realtime_mode:
            # ── Realtime mode: seek to frame whose sim-time ≈ elapsed wall-clock ──
            # Use the FULL clip range (0 to n_frames-1), ignoring loop in/out,
            # so the entire recording plays through regardless of UI loop markers.
            elapsed    = time.time() - self._realtime_start_wall
            target_sim = self._realtime_start_sim + elapsed
            time_arr   = self.data["time"]
            total_end  = self.n_frames - 1
            end_t      = float(time_arr[total_end])
            start_t    = float(time_arr[0])
            if target_sim >= end_t:
                # Reached clip end — restart from frame 0
                self._realtime_start_wall = time.time()
                self._realtime_start_sim  = start_t
                self.set_frame(0)
            else:
                import numpy as _np
                idx = int(_np.searchsorted(time_arr, target_sim))
                idx = max(0, min(total_end, idx))
                # Pick the closer neighbour
                if idx > 0 and abs(float(time_arr[idx - 1]) - target_sim) < abs(float(time_arr[idx]) - target_sim):
                    idx -= 1
                self.set_frame(idx)
            return

        # ── Speed-multiplier mode (original behaviour) ────────────────────
        lo   = self.loop_start
        hi   = self._hi()
        now      = time.time()
        dt_real  = now - self._last_t
        self._last_t = now
        dt_sim   = float(self.data["time"][1] - self.data["time"][0])
        self._frac_acc += dt_real * self.play_speed / max(dt_sim, 1e-9)
        steps    = int(self._frac_acc)
        self._frac_acc -= steps

        span = max(hi - lo + 1, 2)
        self.frame = lo + (self.frame - lo + steps) % span
        self._sync_cam()

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

    def frame_state(self, f: Optional[int] = None):
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


class GraphScrubState:
    def __init__(self):
        self.frame:         int   = 0
        self.frozen:        bool  = False
        self.play_mode:     str   = "main"
        self.backup_frame:  int   = 0
        self._playing:      bool  = False
        self._last_t:       float = 0.0
        self._frac_acc:     float = 0.0

    def _gw(self) -> int:
        return _cfg_mod.CFG.playback.graph_window if _cfg_mod.CFG else 50

    def scrub(self, fi: int, n_frames: int) -> None:
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


class TabState:
    _id_counter: int = 0

    def __init__(self, label: str = "New Tab"):
        TabState._id_counter += 1
        self.id:    int = TabState._id_counter
        self.label: str = label

        self.urdf_path:    Optional[str]        = None
        self.joint_order:  list[str]            = []
        self.joint_limits: dict                 = {}
        self.joint_axes:   dict                 = {}
        self.joint_origins:dict                 = {}
        self.fk_fn                              = None
        self.urdf_robot                         = None
        self.channel_map:  dict                 = {}
        self.loaded:       bool                 = False

        self.S = AppState()
        self.G = GraphScrubState()
        self.S.g_ref = self.G

        vp_f  = _cfg_mod.CFG.layout.default_vp_frac       if _cfg_mod.CFG else 0.54
        bot_f = _cfg_mod.CFG.layout.default_bot_frac       if _cfg_mod.CFG else 0.22
        gh_f  = _cfg_mod.CFG.layout.default_graph_h_frac   if _cfg_mod.CFG else 0.72

        self.vp_frac:       float = vp_f
        self.bot_frac:      float = bot_f
        self.graph_h_frac:  float = gh_f

        self.MESH_VAOS:    dict = {}
        self.LINK_TO_MESH: dict = {}
        self.legs:         dict = {}

        self.fbo            = None
        self.fbo_tex        = None
        self.fbo_depth      = None
        self.fbo_w:   int   = 0
        self.fbo_h:   int   = 0

        self.graphs:      list  = []
        self.GRAPH_PARAMS: dict = {}
        self.PARAM_KEYS:  list  = []
        self._graph_ph_drag: dict = {}

        self.picker: dict = _make_picker_state()
        self.probes: list = []
        self.bounds: dict = {}
        self.raibert_bounds = None          # (4,3,2) float32 or None — from config.yaml action_lb/ub
        self.train_cfg_raw: dict | None = None  # raw parsed config.yaml dict for config panel
        self.show_config_panel: bool = False  # toggle: True=show config.yaml, False=show live graphs

    def init_graphs(self) -> None:
        if self.S.data is None:
            logger.error(f"[state] Tab '{self.label}': init_graphs called before data loaded")
            return

        d  = self.S.data
        gp: dict = {}

        gp["torso_x"]       = lambda data, f: float(data["torso_x"][f])
        gp["torso_y"]       = lambda data, f: float(data["torso_y"][f])
        gp["torso_z"]       = lambda data, f: float(data["torso_z"][f])
        gp["torso_roll"]    = lambda data, f: math.degrees(float(data["torso_roll"][f]))
        gp["torso_pitch"]   = lambda data, f: math.degrees(float(data["torso_pitch"][f]))
        gp["torso_yaw"]     = lambda data, f: math.degrees(float(data["torso_yaw"][f]))

        gp["torso_vx"] = lambda data, f: (
            float(data["torso_vx"][f]) if data.get("torso_vx") is not None else 0.0
        )
        gp["torso_vy"] = lambda data, f: (
            float(data["torso_vy"][f]) if data.get("torso_vy") is not None else 0.0
        )
        gp["torso_vz"] = lambda data, f: (
            float(data["torso_vz"][f]) if data.get("torso_vz") is not None else 0.0
        )

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

        _FOOT_LABELS = ["FR_x","FR_y","FR_z","FL_x","FL_y","FL_z",
                        "RR_x","RR_y","RR_z","RL_x","RL_y","RL_z"]
        for _fi in range(12):
            _lbl = _FOOT_LABELS[_fi]
            gp[f"foot_force {_lbl}"] = (
                lambda i: lambda data, f: float(data["foot_forces"][f, i])
                    if data.get("foot_forces") is not None else 0.0
            )(_fi)

        self.GRAPH_PARAMS = gp
        self.PARAM_KEYS   = list(gp.keys())

        _channel_to_param: dict[str, str] = {}

        for direct in ["torso_x","torso_y","torso_z",
                        "torso_roll","torso_pitch","torso_yaw",
                        "torso_vx","torso_vy","torso_vz",
                        "desired_vel_x",
                        "contact_FL","contact_FR","contact_RL","contact_RR"]:
            _channel_to_param[direct] = direct

        for prefix in ("q", "tau", "dq"):
            for idx, jn in enumerate(self.joint_order):
                _channel_to_param[f"{prefix}{idx}"] = f"{prefix} {jn}"

        for _fi in range(12):
            _channel_to_param[f"foot_force_{_fi}"] = f"foot_force {_FOOT_LABELS[_fi]}"

        custom_data = d.get("custom", {})
        for _ch_name, _arr in custom_data.items():
            if _arr is None:
                continue
            gp[f"custom {_ch_name}"] = (
                lambda nm: lambda data, f: float(data["custom"][nm][f])
                    if data.get("custom") and data["custom"].get(nm) is not None else 0.0
            )(_ch_name)
            _channel_to_param[_ch_name] = f"custom {_ch_name}"

        # ── Extra (auto-detected) channels — key = 'extra:stemname' ──────
        extra_data = d.get("extra", {})
        for _ex_key, _ex_arr in extra_data.items():
            if _ex_arr is None:
                continue
            _param_key = f"extra {_ex_key.removeprefix('extra:')}"
            gp[_param_key] = (
                lambda k: lambda data, f: float(data["extra"][k][f])
                    if data.get("extra") and data["extra"].get(k) is not None else 0.0
            )(_ex_key)
            _channel_to_param[_ex_key] = _param_key

        self.GRAPH_PARAMS = gp
        self.PARAM_KEYS   = list(gp.keys())
        if custom_data:
            loaded = [k for k, v in custom_data.items() if v is not None]
            logger.info(f"[state] Custom channels registered for graphing: {loaded}")
        if extra_data:
            loaded_ex = [k for k, v in extra_data.items() if v is not None]
            logger.info(f"[state] Extra (auto-detected) channels registered: {loaded_ex}")

        colors = (
            [tuple(c) for c in _cfg_mod.CFG.colors.graph_series]
            if _cfg_mod.CFG else [
                (0.25,0.55,1.00),(0.20,0.85,0.40),(1.00,0.70,0.15),(0.85,0.25,0.85),
                (0.15,0.85,0.85),(1.00,0.35,0.35),(0.85,0.85,0.20),(0.60,0.40,1.00),
            ]
        )

        def _ms(param: str, ci: int, lo: float | None = None, hi: float | None = None) -> dict:
            r, g, b = colors[ci % len(colors)][:3]
            _lo = lo if lo is not None else float("-inf")
            _hi = hi if hi is not None else float("inf")
            return {
                "param": param, "enabled": True, "color": (r, g, b),
                "limit_lo": _lo, "limit_hi": _hi,
                "lim_lo_str": f"{_lo:.3g}" if lo is not None else "",
                "lim_hi_str": f"{_hi:.3g}" if hi is not None else "",
                "has_bound":  lo is not None and hi is not None,
            }

        def _mg(title: str, series: list, lo: float = 0.0, hi: float = 0.0) -> dict:
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

        self.graphs = []

        if _cfg_mod.CFG is None:
            logger.warning("[state] _cfg_mod.CFG not loaded — no live graphs will be created")
            logger.info(f"[state] Tab '{self.label}': graphs initialised (0 panels, no config)")
            return

        raw_graphs = getattr(_cfg_mod.CFG, "live_graphs", None)
        if raw_graphs is None:
            logger.warning(
                "[state] 'live_graphs' missing from config — no graphs will be created."
            )
            logger.info(f"[state] Tab '{self.label}': graphs initialised (0 panels)")
            return

        try:
            graph_items = raw_graphs.__dict__.items() if hasattr(raw_graphs, "__dict__") else {}
        except Exception as exc:
            logger.error(f"[state] Could not iterate live_graphs config: {exc}")
            return

        for title, graph_cfg in graph_items:
            if title.startswith("_"):
                continue

            try:
                series_names = list(graph_cfg.series)
                y_min = float(graph_cfg.y_min)
                y_max = float(graph_cfg.y_max)
            except AttributeError as exc:
                logger.error(f"[state] live_graphs['{title}'] malformed: {exc}")
                continue

            series_list = []
            for ci, ch_name in enumerate(series_names):
                param_key = _channel_to_param.get(ch_name)
                if param_key is None:
                    logger.error(f"[state] live_graphs['{title}']: channel '{ch_name}' cannot be resolved. Skipping.")
                    continue
                if param_key not in gp:
                    logger.error(f"[state] live_graphs['{title}']: resolved param '{param_key}' not found. Skipping.")
                    continue
                ch_bound = self.bounds.get(ch_name)
                s_lo, s_hi = (ch_bound if ch_bound is not None else (None, None))
                series_list.append(_ms(param_key, ci, s_lo, s_hi))

            if not series_list:
                logger.warning(f"[state] live_graphs['{title}']: all series failed — graph skipped.")
                continue

            self.graphs.append(_mg(title, series_list, y_min, y_max))

        logger.info(f"[state] Tab '{self.label}': graphs initialised ({len(self.graphs)} panels)")


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


def do_load(
    tab: TabState,
    urdf_path_str: str,
    data_dir_str: str,
    channel_map: Optional[dict] = None,
) -> None:
    from dataviz.data import (
        parse_urdf, build_fk_fn, load_data,
        load_channel_map_file, scan_data_files, auto_map_channels,
        _load_training_config,
    )

    urdf_path = Path(urdf_path_str)
    data_dir  = Path(data_dir_str)

    if not urdf_path.is_file():
        raise FileNotFoundError(f"[state] URDF not found: {urdf_path.resolve()}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"[state] Data directory not found: {data_dir.resolve()}")

    if channel_map is None:
        files       = scan_data_files(data_dir)
        channel_map = auto_map_channels(files)
        saved       = load_channel_map_file(data_dir)
        if saved is not None:
            channel_map.update({k: v for k, v in saved.items() if v is not None})

    logger.info(f"[state] Loading tab '{tab.label}': urdf={urdf_path.name}, data={data_dir.name}")

    joint_order, joint_limits, joint_axes, joint_origins, link_parents = \
        parse_urdf(urdf_path_str)

    fk_fn, urdf_robot = build_fk_fn(urdf_path_str, joint_order)
    data, n, bounds   = load_data(data_dir_str, channel_map)

    if data["q"] is not None and data["q"].shape[1] != len(joint_order):
        m = min(data["q"].shape[1], len(joint_order))
        logger.warning(f"[state] Joint count mismatch — using first {m}")
        joint_order = joint_order[:m]

    tab.urdf_path     = str(urdf_path)
    tab.joint_order   = joint_order
    tab.joint_limits  = joint_limits
    tab.joint_axes    = joint_axes
    tab.joint_origins = joint_origins
    tab.fk_fn         = fk_fn
    tab.urdf_robot    = urdf_robot
    tab.channel_map   = channel_map
    tab.bounds        = bounds
    tab.raibert_bounds = bounds.get("raibert_bounds")  # (4,3,2) float32 or None

    # Store raw config dict for the config panel viewer
    try:
        tab.train_cfg_raw = _load_training_config(data_dir_str)
    except Exception as _cfg_exc:
        logger.warning(f"[state] Could not load train_cfg for config panel: {_cfg_exc}")
        tab.train_cfg_raw = None
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

    logger.info(f"[state] Tab '{tab.label}' loaded: {n} frames, {len(joint_order)} joints")
