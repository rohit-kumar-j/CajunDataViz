"""
dataviz/ui_panels.py  — original + changes:

CHANGES vs previous version:
  1. Import: added scan_unknown_files, extra_channel_name from dataviz.data
  2. draw_picker() page 0 "Next" button: calls scan_unknown_files(), pre-fills
       cmap with 'extra:<stem>' → filename entries so they auto-load
  3. draw_picker() page 2 optional panel: added groups
       Torso Velocity, QP Controller, Raibert Landing, Raibert Correction
     plus a new "Auto-detected" collapsing section with per-file checkboxes
  4. Bottom bar: shows "N auto-detected channel(s) will load" count

Everything else (render_graphs, draw_new_graph_modal, render_settings,
render_timeline, all _ui_* helpers) is IDENTICAL to the original.
"""

import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path as _Path

from loguru import logger

try:
    from imgui_bundle import imgui
    _IMGUI_BACKEND = "bundle"
except ImportError:
    try:
        import imgui
        _IMGUI_BACKEND = "classic"
    except ImportError:
        raise ImportError(
            "[ui_panels] Neither imgui-bundle nor imgui is installed.\n"
            "Install with: pip install imgui-bundle"
        )

import dataviz.config as _cfg_mod
from dataviz.data   import (
    ALL_CHANNELS, REQUIRED_CHANNELS, OPTIONAL_CHANNELS,
    CHANNEL_LABELS, scan_data_files, auto_map_channels,
    load_channel_map_file, save_channel_map, find_robots_dir, scan_robots,
    scan_unknown_files, extra_channel_name,
)
from dataviz.state  import TabState, do_load


# ---------------------------------------------------------------------------
# UIState — shared across all tabs (not per-tab)
# ---------------------------------------------------------------------------
@dataclass
class UIState:
    scale:    float = 1.0
    invert_y: bool  = False

    def sc(self, px: int | float) -> int:
        return max(1, int(px * self.scale))

    def set_font_scale(self) -> None:
        try:
            imgui.get_style().font_global_scale = self.scale
        except AttributeError:
            try:
                imgui.get_io().font_global_scale = self.scale
            except AttributeError:
                pass


def make_ui_state() -> UIState:
    scale    = _cfg_mod.CFG.ui.default_scale    if _cfg_mod.CFG else 1.0
    invert_y = _cfg_mod.CFG.ui.invert_y_default if _cfg_mod.CFG else False
    return UIState(scale=scale, invert_y=invert_y)


# ---------------------------------------------------------------------------
# ImGui compatibility wrappers
# ---------------------------------------------------------------------------
def _ui_button(label: str, w: float = 0, h: float = 0) -> bool:
    if _IMGUI_BACKEND == "bundle":
        return imgui.button(label, (w, h))
    return imgui.button(label, w, h)


def _ui_invisible_button(label: str, w: float, h: float) -> bool:
    if _IMGUI_BACKEND == "bundle":
        return imgui.invisible_button(label, (w, h))
    return imgui.invisible_button(label, w, h)


def _ui_begin_child(label: str, w: float = 0, h: float = 0, border: bool = False, flags: int = 0) -> bool:
    if _IMGUI_BACKEND == "bundle":
        return imgui.begin_child(label, (w, h), border, flags)
    return imgui.begin_child(label, w, h, border, flags)


def _ui_image(tex_id: int, w: float, h: float) -> None:
    if _IMGUI_BACKEND == "bundle":
        try:
            imgui.image(tex_id, (w, h), (0, 1), (1, 0))
        except TypeError:
            try:
                t_ref = imgui.ImTextureID(int(tex_id))
                imgui.image(t_ref, (w, h), (0, 1), (1, 0))
            except AttributeError:
                t_ref = imgui.ImTextureRef(int(tex_id))
                imgui.image(t_ref, (w, h), (0, 1), (1, 0))
    else:
        imgui.image(tex_id, w, h, uv0=(0, 1), uv1=(1, 0))


def _ui_get_mouse_pos() -> tuple[float, float]:
    pos = imgui.get_mouse_pos()
    try:    return pos.x, pos.y
    except AttributeError: return pos[0], pos[1]


def _ui_get_cursor_screen_pos() -> tuple[float, float]:
    pos = imgui.get_cursor_screen_pos()
    try:    return pos.x, pos.y
    except AttributeError: return pos[0], pos[1]


def _ui_get_mouse_delta() -> tuple[float, float]:
    d = imgui.get_io().mouse_delta
    try:    return d.x, d.y
    except AttributeError: return d[0], d[1]


def _ui_get_content_region_avail() -> tuple[float, float]:
    av = imgui.get_content_region_avail()
    try:    return av.x, av.y
    except AttributeError: return av[0], av[1]


def _ui_collapsing_header(label: str) -> bool:
    res = imgui.collapsing_header(label)
    return res[0] if isinstance(res, tuple) else res


def _ui_selectable(label: str) -> bool:
    if _IMGUI_BACKEND == "bundle":
        res = imgui.selectable(label, False)
    else:
        res = imgui.selectable(label)
    return res[0] if isinstance(res, tuple) else res


def _ui_begin_tab_item(label: str, flags: int = 0) -> bool:
    try:
        res = imgui.begin_tab_item(label, None, flags)
    except Exception:
        res = imgui.begin_tab_item(label, flags=flags)
    return res[0] if isinstance(res, tuple) else res


def _ui_begin_tab_item_closable(label: str, flags: int = 0) -> tuple[bool, bool]:
    if _IMGUI_BACKEND == "bundle":
        try:
            res = imgui.begin_tab_item(label, True, flags)
        except TypeError:
            res = imgui.begin_tab_item(label, p_open=True, flags=flags)
    else:
        res = imgui.begin_tab_item(label, opened=True, flags=flags)
    if isinstance(res, tuple):
        return res[0], res[1]
    return res, True


def _ui_input_text(label: str, value: str, buf_len: int, flags: int = 0) -> tuple[bool, str]:
    if _IMGUI_BACKEND == "bundle":
        return imgui.input_text(label, value, flags=flags)
    return imgui.input_text(label, value, buf_len, flags=flags)


def _ui_color_u32(r: float, g: float = 0.0, b: float = 0.0, a: float = 1.0) -> int:
    if _IMGUI_BACKEND == "bundle":
        return imgui.get_color_u32(imgui.ImVec4(r, g, b, a))
    return imgui.get_color_u32_rgba(r, g, b, a)


def _ui_add_rect(dl, p_min, p_max, col: int, rounding: float = 0.0, thickness: float = 1.0) -> None:
    if _IMGUI_BACKEND == "bundle":
        dl.add_rect(p_min, p_max, col, rounding, 0, thickness)
    else:
        dl.add_rect(p_min, p_max, col, rounding, thickness)


def _ui_push_clip_rect(dl, p_min, p_max, intersect: bool = True) -> None:
    if _IMGUI_BACKEND == "bundle":
        dl.push_clip_rect(p_min, p_max, intersect)
    else:
        dl.push_clip_rect(p_min[0], p_min[1], p_max[0], p_max[1], intersect)


def _ui_push_style_color(idx, r, g: float = 0.0, b: float = 0.0, a: float = 1.0) -> None:
    if isinstance(r, (tuple, list)):
        col = r
        if len(col) == 3:
            col = (col[0], col[1], col[2], 1.0)
    else:
        col = (r, g, b, a)
    if _IMGUI_BACKEND == "bundle":
        imgui.push_style_color(idx, imgui.ImVec4(*col))
    else:
        imgui.push_style_color(idx, col[0], col[1], col[2], col[3])


def _ui_begin(label: str, flags: int = 0):
    if _IMGUI_BACKEND == "bundle":
        return imgui.begin(label, None, flags)
    return imgui.begin(label, False, flags)


# ── Keyboard / Flag constants ───────────────────────────────────────────────
try:
    _UI_KEY_SPACE = imgui.Key.space
    _UI_KEY_LEFT  = imgui.Key.left_arrow
    _UI_KEY_RIGHT = imgui.Key.right_arrow
except AttributeError:
    try:
        _UI_KEY_SPACE = imgui.KEY_SPACE
        _UI_KEY_LEFT  = imgui.KEY_LEFT_ARROW
        _UI_KEY_RIGHT = imgui.KEY_RIGHT_ARROW
    except AttributeError:
        _UI_KEY_SPACE = 32; _UI_KEY_LEFT = 262; _UI_KEY_RIGHT = 263

try:
    _UI_INPUT_TEXT_ENTER = imgui.InputTextFlags_.enter_returns_true
except AttributeError:
    try:    _UI_INPUT_TEXT_ENTER = imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
    except AttributeError: _UI_INPUT_TEXT_ENTER = 32

try:
    _UI_TAB_SET_SELECTED = imgui.TabItemFlags_.set_selected
except AttributeError:
    try:    _UI_TAB_SET_SELECTED = imgui.TAB_ITEM_SET_SELECTED
    except AttributeError: _UI_TAB_SET_SELECTED = 8


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_tl_drag: dict = {"active": None}
_graph_id_counter: int = 0
_new_graph_modal: dict = {
    "open":       False,
    "tab_id":     -1,
    "search":     "",
    "checked":    {},
    "title":      "",
    "y_min_str":  "-30",
    "y_max_str":  "30",
}


# ---------------------------------------------------------------------------
# Helper: short series label
# ---------------------------------------------------------------------------
def _short_label(param: str) -> str:
    lbl = param.split(" ", 1)[-1] if " " in param else param
    lbl = lbl.replace("_hip_joint",   "_h") \
              .replace("_thigh_joint", "_t") \
              .replace("_calf_joint",  "_c")
    return lbl.replace("_joint", "").replace("_", "-")


# ---------------------------------------------------------------------------
# Optional channel groups
# ---------------------------------------------------------------------------
_FOOT_NAMES = ["FR", "FL", "RR", "RL"]
_AXES       = ["x", "y", "z"]

# Each entry: (group_name, [channel_keys...])
_OPTIONAL_GROUPS: list[tuple[str, list[str]]] = [
    ("Vel / Contacts", [
        "torso_vx", "torso_vy", "torso_vz",
        "desired_vel_x",
        "contact_FR", "contact_FL", "contact_RR", "contact_RL",
    ]),
    ("Joint Vel dq",  [f"dq{i}"          for i in range(12)]),
    ("Torque tau",    [f"tau{i}"          for i in range(12)]),
    ("Foot Forces",   [f"foot_force_{i}"  for i in range(12)]),
    ("Foot Pos",      [f"foot_pos_{i}"    for i in range(12)]),
    ("QP Controller", (
        ["qp_cost"]
        + [f"qp_desired_acc{i}" for i in range(6)]
        + [f"qp_solved_acc{i}"  for i in range(6)]
    )),
    ("Raibert Landing", [
        f"raibert_{f}_{a}" for f in _FOOT_NAMES for a in _AXES
    ]),
    ("Raibert Correction", [
        f"raibert_corr_{f}_{a}" for f in _FOOT_NAMES for a in _AXES
    ]),
]


def _draw_opt_group(grp_name, grp_chs, cmap, combos, combo_items, ow):
    """Render one optional channel group as a collapsing header + combos."""
    n_mapped = sum(1 for ch in grp_chs if cmap.get(ch))
    hdr = f"{grp_name}  ({n_mapped}/{len(grp_chs)})"
    opened = _ui_collapsing_header(hdr)
    if not opened:
        return
    imgui.push_item_width(ow - 180)
    for ch in grp_chs:
        lbl    = CHANNEL_LABELS.get(ch, ch)
        mapped = bool(cmap.get(ch))
        col    = (0.3, 1.0, 0.3, 1) if mapped else (0.5, 0.5, 0.5, 1)
        imgui.text_colored(col, f"{lbl[:26]:<26}")
        imgui.same_line()
        cur  = combos.get(ch, 0)
        ch2, ni = imgui.combo(f"##{ch}p", cur, combo_items)
        if ch2:
            combos[ch] = ni
            cmap[ch]   = combo_items[ni] if ni > 0 else None
    imgui.pop_item_width()


# ---------------------------------------------------------------------------
# Picker panel
# ---------------------------------------------------------------------------
def draw_picker(tab: TabState, avail_w: float, avail_h: float) -> None:
    """Render the robot / data-folder picker inside the current ImGui window."""
    ps     = tab.picker
    robots = ps["robots"]

    if not robots:
        rdir         = find_robots_dir()
        ps["robots"] = scan_robots(rdir)
        robots       = ps["robots"]

    _ui_push_style_color(imgui.Col_.child_bg, 0.08, 0.08, 0.11, 1.0)

    if not robots:
        imgui.text_colored((1, 0.4, 0.4, 1), "No robots found in robots/ directory.")
        imgui.spacing()
        imgui.text_disabled("Expected: robots/<name>/urdf/*.urdf  and  robots/<name>/data/")
        imgui.pop_style_color()
        return

    if ps["page"] == 0:
        # ── Page 1: Pick robot + data dir ────────────────────────────────
        imgui.text_colored((0.65, 0.85, 1.0, 1), "Select Robot")
        imgui.separator()
        imgui.push_item_width(avail_w - 20)
        _, ps["robot_idx"] = imgui.list_box(
            "##rb", ps["robot_idx"],
            [r["name"] for r in robots],
            min(5, len(robots)),
        )
        imgui.pop_item_width()

        robot = robots[ps["robot_idx"]]
        imgui.spacing()
        imgui.text_colored((0.5, 0.7, 0.5, 1), "URDF:")
        imgui.same_line()
        imgui.text_disabled(str(robot["urdf"]))
        imgui.spacing()
        imgui.text_colored((0.65, 0.85, 1.0, 1), "Select Data Folder")
        imgui.separator()

        data_dirs = robot["data_dirs"]
        if not data_dirs:
            imgui.text_colored((1, 0.5, 0.2, 1), "  No data folders found.")
        else:
            imgui.push_item_width(avail_w - 20)
            _, ps["data_idx"] = imgui.list_box(
                "##dd", ps["data_idx"],
                [d.name for d in data_dirs],
                min(5, len(data_dirs)),
            )
            imgui.pop_item_width()

        imgui.spacing(); imgui.separator(); imgui.spacing()
        if ps.get("error"):
            imgui.text_colored((1, 0.3, 0.3, 1), ps["error"])
            imgui.spacing()

        can_next = bool(data_dirs)
        if _ui_button("Next: Map Channels  →", 200, 30) and can_next:
            dd    = str(data_dirs[ps["data_idx"]])
            files = scan_data_files(dd)
            saved = load_channel_map_file(dd)
            cmap  = saved if saved else auto_map_channels(files)

            # ── Auto-detect unknown .txt files ────────────────────────────
            unknown = scan_unknown_files(files)
            for fn in unknown:
                key = extra_channel_name(fn)   # e.g. 'extra:raibert_FR_x'
                if key not in cmap:
                    cmap[key] = fn             # default: auto-load

            ps["files"]   = files
            ps["unknown"] = unknown            # raw filenames not in AUTO_PATTERNS
            ps["cmap"]    = cmap
            # Build combos for ALL known channels + extra keys already in cmap
            all_keys = list(ALL_CHANNELS) + [k for k in cmap if k.startswith("extra:")]
            ps["combos"] = {
                ch: (([None] + files).index(cmap.get(ch))
                     if cmap.get(ch) in ([None] + files) else 0)
                for ch in all_keys
            }
            ps["page"] = 1

    else:
        # ── Page 2: Channel mapping ───────────────────────────────────────
        files       = ps["files"]
        cmap        = ps["cmap"]
        combos      = ps["combos"]
        unknown     = ps.get("unknown", [])
        combo_items = ["--- none ---"] + files
        half_w      = (avail_w - 12) // 2

        # Left pane: required channels
        _ui_begin_child("##preq", half_w, avail_h - 90, True)
        imgui.text_colored((1.0, 0.85, 0.3, 1), "REQUIRED CHANNELS")
        imgui.separator()
        imgui.push_item_width(half_w - 180)
        for ch in REQUIRED_CHANNELS:
            lbl    = CHANNEL_LABELS.get(ch, ch)
            mapped = bool(cmap.get(ch))
            col    = (0.3, 1.0, 0.3, 1) if mapped else (1.0, 0.35, 0.35, 1)
            imgui.text_colored(col, f"{lbl[:24]:<24}")
            imgui.same_line()
            cur  = combos.get(ch, 0)
            ch2, ni = imgui.combo(f"##{ch}p", cur, combo_items)
            if ch2:
                combos[ch] = ni
                cmap[ch]   = combo_items[ni] if ni > 0 else None
        imgui.pop_item_width()
        imgui.end_child()

        imgui.same_line()

        # Right pane: optional channels (all groups + auto-detected)
        _ui_begin_child("##popt", 0, avail_h - 90, True)
        imgui.text_colored((0.55, 0.75, 1.0, 1), "OPTIONAL CHANNELS")
        imgui.separator()
        ow = (avail_w - 12) // 2 - 20

        # Known groups
        for grp_name, grp_chs in _OPTIONAL_GROUPS:
            _draw_opt_group(grp_name, grp_chs, cmap, combos, combo_items, ow)

        # ── Auto-detected section ─────────────────────────────────────────
        if unknown:
            imgui.spacing()
            imgui.separator()
            n_auto_on = sum(1 for fn in unknown if cmap.get(extra_channel_name(fn)))
            hdr = f"Auto-detected  ({n_auto_on}/{len(unknown)} loading)"
            if _ui_collapsing_header(hdr):
                imgui.text_colored(
                    (0.75, 0.85, 0.55, 1),
                    "Files found but not in known patterns — loading as extra:<name>"
                )
                imgui.spacing()
                for fn in unknown:
                    key      = extra_channel_name(fn)   # 'extra:stemname'
                    cur_fn   = cmap.get(key)
                    is_on    = cur_fn is not None
                    chk_ch, new_on = imgui.checkbox(f"##auto_{fn}", is_on)
                    if chk_ch:
                        cmap[key]   = fn if new_on else None
                        combos[key] = (files.index(fn) + 1) if (new_on and fn in files) else 0

                    col = (0.35, 1.0, 0.55, 1) if is_on else (0.45, 0.45, 0.45, 1)
                    imgui.same_line()
                    stem = _Path(fn).stem
                    imgui.text_colored(col, f"{stem[:32]:<32}")
                    imgui.same_line()
                    imgui.text_disabled(fn)

        imgui.end_child()

        # Bottom bar
        imgui.spacing(); imgui.separator(); imgui.spacing()
        if _ui_button("← Back", 90, 28):
            ps["page"] = 0
        imgui.same_line()
        if _ui_button("Load", 90, 28):
            robot = robots[ps["robot_idx"]]
            dd    = str(robot["data_dirs"][ps["data_idx"]])
            save_channel_map(dd, cmap)
            try:
                do_load(tab, str(robot["urdf"]), dd, cmap)
                ps["error"] = ""
            except Exception as exc:
                ps["error"] = str(exc)
                ps["page"]  = 0
                logger.error(f"[ui_panels] Load failed: {exc}")

        missing = [ch for ch in REQUIRED_CHANNELS if not cmap.get(ch)]
        if missing:
            imgui.same_line()
            imgui.text_colored((1.0, 0.65, 0.1, 1), f"⚠ {len(missing)} required unmapped")

        if unknown:
            n_extra = sum(1 for fn in unknown if cmap.get(extra_channel_name(fn)))
            if n_extra:
                imgui.same_line()
                imgui.text_colored(
                    (0.35, 1.0, 0.55, 1),
                    f"  + {n_extra} auto-detected channel(s) will load"
                )

        if ps.get("error"):
            imgui.text_colored((1, 0.3, 0.3, 1), ps["error"])

    imgui.pop_style_color()


# ---------------------------------------------------------------------------
# Graph panel  — UNCHANGED from original
# ---------------------------------------------------------------------------
def render_graphs(tab: TabState, avail_w: float, ui: UIState) -> None:
    S = tab.S
    G = tab.G
    d = S.data
    if d is None:
        return

    sc             = ui.sc
    _graphs        = tab.graphs
    _GRAPH_PARAMS  = tab.GRAPH_PARAMS
    _PARAM_KEYS    = tab.PARAM_KEYS
    _graph_ph_drag = tab._graph_ph_drag
    gw             = _cfg_mod.CFG.playback.graph_window if _cfg_mod.CFG else 50

    def _get_val(data, frame, param):
        fn = _GRAPH_PARAMS.get(param)
        if fn is None:
            return 0.0
        try:
            return fn(data, frame)
        except Exception:
            return 0.0

    def _mk_series(param: str, color_idx: int, lo: float | None = None, hi: float | None = None) -> dict:
        colors = (
            [tuple(c) for c in _cfg_mod.CFG.colors.graph_series]
            if _cfg_mod.CFG else [(0.25,0.55,1.00),(0.20,0.85,0.40),(1.00,0.70,0.15),(0.85,0.25,0.85)]
        )
        r, g2, b = colors[color_idx % len(colors)][:3]
        _lo = lo if lo is not None else float("-inf")
        _hi = hi if hi is not None else float("inf")
        return {
            "param": param, "enabled": True, "color": (r, g2, b),
            "limit_lo": _lo, "limit_hi": _hi,
            "lim_lo_str": f"{_lo:.3g}" if lo is not None else "",
            "lim_hi_str": f"{_hi:.3g}" if hi is not None else "",
            "has_bound":  lo is not None and hi is not None,
        }

    def _mk_graph(title: str, series: list, lo: float, hi: float) -> dict:
        global _graph_id_counter
        _graph_id_counter += 1
        return {
            "title": title, "series": series, "lo": lo, "hi": hi,
            "lo_str": str(lo), "hi_str": str(hi),
            "lo_edit": False, "hi_edit": False, "hovered": -1,
            "side_w": 200,
            "_gid": _graph_id_counter,
        }

    pw     = avail_w
    gh     = sc(160)
    SIDE_W_MIN = sc(120)
    SIDE_W_MAX = int(pw * 0.55)

    gi = 0
    while gi < len(_graphs):
        graph  = _graphs[gi]
        gid    = graph.get("_gid", gi)
        lo, hi = graph["lo"], graph["hi"]
        series = graph["series"]
        span   = hi - lo if hi != lo else 1.0

        imgui.text_colored((0.65, 0.75, 0.88, 1), graph["title"][:30])
        imgui.same_line(); imgui.text_disabled("["); imgui.same_line()
        imgui.push_item_width(54)
        if graph["lo_edit"]:
            ch2, ns = _ui_input_text(f"##lo{gid}", graph["lo_str"], 14, _UI_INPUT_TEXT_ENTER)
            if ch2:
                try:
                    graph["lo"] = float(ns); graph["lo_str"] = ns
                except ValueError as exc:
                    logger.warning(f"[ui_panels] Invalid graph y-min value '{ns}': {exc}")
                graph["lo_edit"] = False
            if not imgui.is_item_active():
                graph["lo_edit"] = False
        else:
            any_lo = any(_get_val(d, G.frame, s["param"]) < lo for s in series if s["enabled"])
            col = (1.0, 0.25, 0.25, 1) if any_lo else (0.5, 0.5, 0.5, 1)
            imgui.text_colored(col, f"{lo:.3g}")
            if imgui.is_item_clicked():
                graph["lo_edit"] = True
        imgui.pop_item_width()
        imgui.same_line(); imgui.text_disabled(","); imgui.same_line()
        imgui.push_item_width(54)
        if graph["hi_edit"]:
            ch2, ns = _ui_input_text(f"##hi{gid}", graph["hi_str"], 14, _UI_INPUT_TEXT_ENTER)
            if ch2:
                try:
                    graph["hi"] = float(ns); graph["hi_str"] = ns
                except ValueError as exc:
                    logger.warning(f"[ui_panels] Invalid graph y-max value '{ns}': {exc}")
                graph["hi_edit"] = False
            if not imgui.is_item_active():
                graph["hi_edit"] = False
        else:
            any_hi = any(_get_val(d, S.frame, s["param"]) > hi for s in series if s["enabled"])
            col = (1.0, 0.25, 0.25, 1) if any_hi else (0.5, 0.5, 0.5, 1)
            imgui.text_colored(col, f"{hi:.3g}")
            if imgui.is_item_clicked():
                graph["hi_edit"] = True
        imgui.pop_item_width()
        imgui.same_line(); imgui.text_disabled("]"); imgui.same_line()
        if _ui_button(f"X##xg{gid}", 20, 0):
            _graphs.pop(gi); continue

        gi += 1
        if gi > len(_graphs):
            break
        graph  = _graphs[gi - 1]
        lo, hi = graph["lo"], graph["hi"]
        series = graph["series"]
        span   = hi - lo if hi != lo else 1.0
        gi2    = gi - 1
        body_start_x, body_start_y = _ui_get_cursor_screen_pos()

        SIDE_W = max(SIDE_W_MIN, min(SIDE_W_MAX, sc(graph.get("side_w", 200))))
        _ui_push_style_color(imgui.Col_.child_bg, 0.07, 0.07, 0.10, 1.0)
        _ui_begin_child(f"##side{gid}", SIDE_W, gh, True)

        if _ui_button(f"+##as{gid}", 22, 0):
            imgui.open_popup(f"##addpop{gid}")
        if imgui.begin_popup(f"##addpop{gid}"):
            imgui.text("Add series:"); imgui.separator()
            for pk in _PARAM_KEYS:
                lbl = _short_label(pk)[:28]
                if _ui_selectable(f"{lbl}##{gi}_{pk}"):
                    series.append(_mk_series(pk, len(series)))
                    imgui.close_current_popup()
            imgui.end_popup()

        imgui.separator()
        for si in range(len(series) - 1, -1, -1):
            s = series[si]; r2, g2, b2 = s["color"]
            try:
                _ui_push_style_color(imgui.Col_.check_mark, r2, g2, b2, 1.0)
                _, s["enabled"] = imgui.checkbox(f"##{gid}_{si}c", s["enabled"])
                imgui.pop_style_color()
            except Exception:
                _, s["enabled"] = imgui.checkbox(f"##{gid}_{si}c", s["enabled"])
            imgui.same_line()
            lbl = _short_label(s["param"])[:18]
            imgui.text_colored((r2, g2, b2, 1), lbl)
            if imgui.is_item_clicked():
                s["_edit"] = True
            if s.get("_edit", False):
                imgui.push_item_width(SIDE_W - 6)
                cur_idx = _PARAM_KEYS.index(s["param"]) if s["param"] in _PARAM_KEYS else 0
                try:
                    ch2, ni = imgui.combo(f"##{gid}_{si}p", cur_idx, _PARAM_KEYS)
                    if ch2:
                        s["param"] = _PARAM_KEYS[ni]; s["_edit"] = False
                except AttributeError as exc:
                    logger.warning(f"[ui_panels] imgui.combo error for series param edit: {exc}")
                imgui.pop_item_width()
                if not imgui.is_item_active() and not imgui.is_item_focused():
                    s["_edit"] = False
            imgui.same_line()
            if s.get("has_bound", False):
                imgui.push_item_width(38)
                ch_lo, new_lo_s = _ui_input_text(
                    f"##lo{gid}_{si}", s.get("lim_lo_str", ""), 8, _UI_INPUT_TEXT_ENTER
                )
                if ch_lo:
                    try:
                        s["limit_lo"] = float(new_lo_s); s["lim_lo_str"] = new_lo_s; s["has_bound"] = True
                    except ValueError as exc:
                        logger.warning(f"[ui_panels] Invalid series limit_lo '{new_lo_s}': {exc}")
                imgui.pop_item_width(); imgui.same_line()
                imgui.push_item_width(38)
                ch_hi, new_hi_s = _ui_input_text(
                    f"##hi{gid}_{si}", s.get("lim_hi_str", ""), 8, _UI_INPUT_TEXT_ENTER
                )
                if ch_hi:
                    try:
                        s["limit_hi"] = float(new_hi_s); s["lim_hi_str"] = new_hi_s; s["has_bound"] = True
                    except ValueError as exc:
                        logger.warning(f"[ui_panels] Invalid series limit_hi '{new_hi_s}': {exc}")
                imgui.pop_item_width(); imgui.same_line()
            else:
                imgui.text_disabled("--  --"); imgui.same_line()
            if _ui_button(f"-##{gi}_{si}r", 20, 0):
                series.pop(si); continue
            cv  = _get_val(d, G.frame, s["param"])
            has_b_side = s.get("has_bound", False)
            oob = has_b_side and (cv < s["limit_lo"] or cv > s["limit_hi"])
            col = (1.0, 0.2, 0.2, 1) if oob else (r2 * 0.85, g2 * 0.85, b2 * 0.85, 1)
            imgui.text_colored(col, f"  {cv:+.3f}")

        imgui.end_child(); imgui.pop_style_color()

        imgui.same_line()
        sp_x, sp_y = _ui_get_cursor_screen_pos()
        sp_w = sc(6); sp_h = gh
        mx_s, my_s = _ui_get_mouse_pos()
        sp_hovered = (sp_x <= mx_s <= sp_x + sp_w) and (sp_y <= my_s <= sp_y + sp_h)

        drag_key = f"_split_{gid}"
        sp_dragging = _graph_ph_drag.get(drag_key, False)

        if sp_dragging and imgui.is_mouse_down(0):
            dx, _ = _ui_get_mouse_delta()
            raw = graph.get("side_w", 200) + dx / ui.scale
            graph["side_w"] = max(SIDE_W_MIN / ui.scale, min(SIDE_W_MAX / ui.scale, raw))
        elif not imgui.is_mouse_down(0):
            _graph_ph_drag[drag_key] = False
            sp_dragging = False

        if sp_hovered and imgui.is_mouse_clicked(0):
            _graph_ph_drag[drag_key] = True
            sp_dragging = True

        if sp_hovered or sp_dragging:
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            dl_sp = imgui.get_window_draw_list()
            dl_sp.add_rect_filled((sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h),
                                  _ui_color_u32(0.45, 0.65, 1.0, 0.85))
        else:
            dl_sp = imgui.get_window_draw_list()
            dl_sp.add_rect_filled((sp_x, sp_y), (sp_x + sp_w, sp_y + sp_h),
                                  _ui_color_u32(0.28, 0.28, 0.32, 0.60))

        imgui.dummy((sp_w, sp_h))

        imgui.same_line()
        y_tick_w = sc(38)
        canvas_w = max(sc(60), pw - SIDE_W - sp_w - sc(4) - y_tick_w)
        canvas_origin_x_raw, canvas_origin_y = _ui_get_cursor_screen_pos()
        canvas_origin_x = canvas_origin_x_raw + y_tick_w
        dl   = imgui.get_window_draw_list()
        sk_h = sc(14)

        sk_x  = canvas_origin_x; sk_y = canvas_origin_y
        _rt_mode = getattr(S, "realtime_mode", False) and S.playing
        if _rt_mode:
            sk_start = 0; sk_end = max(1, S.frame)
        else:
            sk_start = max(0, S.frame - gw); sk_end = min(S.n_frames, S.frame + gw)
        sk_span  = max(sk_end - sk_start, 1)
        gf       = G.frame

        dl.add_rect_filled((sk_x, sk_y), (sk_x + canvas_w, sk_y + sk_h), _ui_color_u32(0.07, 0.07, 0.10))
        for ti in range(0, sk_span + 1, max(1, sk_span // 8)):
            tx   = sk_x + canvas_w * ti / sk_span
            fi_t = max(0, min(S.n_frames - 1, sk_start + ti))
            dl.add_line((tx, sk_y + sk_h - sc(5)), (tx, sk_y + sk_h), _ui_color_u32(0.3, 0.3, 0.35, 0.6), 1)
            if d.get("time") is not None:
                try:    tl = f"{float(d['time'][fi_t]):.2f}s"
                except (IndexError, ValueError): tl = f"{fi_t - S.frame:+d}"
            else:
                tl = f"{fi_t - S.frame:+d}"
            if ti < sk_span - sk_span // 10:
                dl.add_text((tx + 2, sk_y + 1), _ui_color_u32(0.32, 0.32, 0.38), tl)

        ph_sk_x = sk_x + canvas_w * (S.frame - sk_start) / sk_span
        dl.add_line((ph_sk_x, sk_y), (ph_sk_x, sk_y + sk_h), _ui_color_u32(0.8, 0.8, 1.0, 0.9), 2)
        dl.add_triangle_filled(
            (ph_sk_x - sc(4), sk_y), (ph_sk_x + sc(4), sk_y), (ph_sk_x, sk_y + sc(7)),
            _ui_color_u32(0.8, 0.8, 1.0, 1.0),
        )
        if sk_start <= gf <= sk_end:
            gf_sk_x = sk_x + canvas_w * (gf - sk_start) / sk_span
            dl.add_line((gf_sk_x, sk_y), (gf_sk_x, sk_y + sk_h), _ui_color_u32(1, 0.92, 0.2, 0.8), 1.5)
            dl.add_triangle_filled(
                (gf_sk_x - sc(4), sk_y + sk_h), (gf_sk_x + sc(4), sk_y + sk_h),
                (gf_sk_x, sk_y + sk_h - 7), _ui_color_u32(1, 0.92, 0.2, 0.9),
            )
        _sk_mx, _sk_my = _ui_get_mouse_pos()
        _sk_hit = (sk_x <= _sk_mx <= sk_x + canvas_w) and (sk_y <= _sk_my <= sk_y + sk_h)
        if _sk_hit and imgui.is_mouse_down(0):
            new_gf = sk_start + int((_sk_mx - sk_x) / canvas_w * sk_span)
            G.scrub(new_gf, S.n_frames)
        imgui.dummy((canvas_w, sk_h))

        gx  = canvas_origin_x
        gy2 = canvas_origin_y + sk_h
        dl.add_rect_filled((gx, gy2), (gx + canvas_w, gy2 + gh), _ui_color_u32(0.05, 0.05, 0.07))
        _ui_add_rect(dl, (gx, gy2), (gx + canvas_w, gy2 + gh), _ui_color_u32(0.18, 0.18, 0.22), 0.0, 1.0)

        if _rt_mode:
            start_auto = 0; end_auto = max(1, S.frame)
        else:
            start_auto = max(0, S.frame - gw)
            end_auto   = min(S.n_frames, S.frame + gw)
        data_vals  = []
        for _s in series:
            if not _s.get("enabled", True): continue
            _fn = _GRAPH_PARAMS.get(_s["param"])
            if _fn is None: continue
            for _fi in range(start_auto, end_auto, max(1, (end_auto - start_auto) // 200)):
                try:
                    data_vals.append(_fn(d, _fi))
                except Exception:
                    pass
        if data_vals:
            d_lo = min(data_vals); d_hi = max(data_vals)
            pad  = max(abs(d_hi - d_lo) * 0.05, 0.01)
            ax_lo = d_lo - pad; ax_hi = d_hi + pad
        else:
            ax_lo = -1.0; ax_hi = 1.0
        ax_span = ax_hi - ax_lo if ax_hi != ax_lo else 1.0

        def vy(v):
            return gy2 + gh - (v - ax_lo) / ax_span * gh

        n_ticks = int(getattr(getattr(_cfg_mod.CFG, "graphs", None), "y_axis_ticks", 5))
        _ui_push_clip_rect(dl, (canvas_origin_x_raw, gy2), (canvas_origin_x, gy2 + gh), True)
        for _ti in range(n_ticks):
            _tv  = ax_lo + (ax_hi - ax_lo) * _ti / max(n_ticks - 1, 1)
            _ty  = vy(_tv)
            dl.add_line((canvas_origin_x - sc(4), _ty), (canvas_origin_x, _ty),
                        _ui_color_u32(0.35, 0.35, 0.38, 0.7), 1)
            _tlbl = f"{_tv:.3g}"
            dl.add_text((canvas_origin_x_raw, _ty - sc(6)),
                        _ui_color_u32(0.42, 0.42, 0.46), _tlbl)
        dl.pop_clip_rect()

        _ui_push_clip_rect(dl, (gx, gy2), (gx + canvas_w, gy2 + gh), True)

        if ax_lo < 0 < ax_hi:
            dl.add_line((gx, vy(0)), (gx + canvas_w, vy(0)), _ui_color_u32(0.32, 0.32, 0.32, 0.3), 1)

        if _rt_mode:
            start = 0; end = max(2, S.frame + 1)
        else:
            start = max(0, S.frame - gw); end = min(S.n_frames, S.frame + gw)
        nw    = max(end - start, 2)
        g_center   = max(start, min(end - 1, G.frame))
        ph_x       = gx + canvas_w * (g_center - start) / max(nw - 1, 1)
        main_ph_x  = gx + canvas_w * (S.frame  - start) / max(nw - 1, 1)
        dl.add_rect_filled((ph_x, gy2), (gx + canvas_w, gy2 + gh), _ui_color_u32(1, 1, 1, 0.025))

        mx2, my2 = _ui_get_mouse_pos()
        mouse_in = gx <= mx2 <= gx + canvas_w and gy2 <= my2 <= gy2 + gh
        hov_si   = -1
        if mouse_in:
            bd = 18.0
            for si2, s in enumerate(series):
                if not s["enabled"]: continue
                fn = _GRAPH_PARAMS.get(s["param"])
                if fn is None: continue
                fi_m = start + int((mx2 - gx) / canvas_w * (nw - 1))
                fi_m = max(start, min(end - 1, fi_m))
                try:
                    pd = abs(vy(fn(d, fi_m)) - my2)
                    if pd < bd:
                        bd = pd; hov_si = si2
                except Exception as exc:
                    logger.debug(f"[ui_panels] Hover distance check failed for param '{s['param']}': {exc}")
        graph["hovered"] = hov_si

        for si2, s in enumerate(series):
            if not s["enabled"]: continue
            fn = _GRAPH_PARAMS.get(s["param"])
            if fn is None: continue
            r2, g2, b2 = s["color"]
            s_lo = s["limit_lo"]; s_hi = s["limit_hi"]
            is_hov    = hov_si == si2
            is_dimmed = hov_si >= 0 and not is_hov
            base_a    = 0.20 if is_dimmed else (1.0 if is_hov else 0.85)
            lw        = 2.5  if is_hov    else 1.4

            if is_hov and mouse_in and s.get("has_bound", False):
                dl.add_line((gx, vy(s_lo)), (gx + canvas_w, vy(s_lo)), _ui_color_u32(r2, g2, b2, 0.6), 1)
                dl.add_line((gx, vy(s_hi)), (gx + canvas_w, vy(s_hi)), _ui_color_u32(r2, g2, b2, 0.6), 1)
                dl.add_text((gx + canvas_w - 44, vy(s_hi) - 12), _ui_color_u32(r2, g2, b2, 0.85), f"{s_hi:.4g}")
                dl.add_text((gx + canvas_w - 44, vy(s_lo) + 2),  _ui_color_u32(r2, g2, b2, 0.85), f"{s_lo:.4g}")

            pp3 = ppy3 = pv2 = None
            has_b = s.get("has_bound", False)
            c_in  = _ui_color_u32(r2, g2, b2, base_a)
            c_out = _ui_color_u32(1.0, 0.15, 0.15, base_a) if has_b else c_in
            for k in range(nw):
                fi2 = start + k
                try:
                    v2 = fn(d, fi2)
                except Exception as exc:
                    logger.debug(f"[ui_panels] Graph value eval failed at frame {fi2}: {exc}")
                    v2 = 0.0
                px4 = gx + canvas_w * k / (nw - 1)
                py4 = vy(v2)
                if pp3 is not None:
                    fut = fi2 > G.frame
                    a2  = base_a * (0.35 if fut else 1.0)
                    c_in2  = _ui_color_u32(r2, g2, b2, a2)
                    c_out2 = _ui_color_u32(1.0, 0.15, 0.15, a2)
                    ts = [0.0, 1.0]; dv = v2 - pv2
                    if has_b and abs(dv) > 1e-9:
                        t_lo = (s_lo - pv2) / dv
                        if 0.0 < t_lo < 1.0: ts.append(t_lo)
                        t_hi = (s_hi - pv2) / dv
                        if 0.0 < t_hi < 1.0: ts.append(t_hi)
                    ts.sort()
                    for ti in range(len(ts) - 1):
                        ta, tb = ts[ti], ts[ti + 1]
                        xa = pp3  + (px4 - pp3)  * ta; ya = ppy3 + (py4 - ppy3) * ta
                        xb = pp3  + (px4 - pp3)  * tb; yb = ppy3 + (py4 - ppy3) * tb
                        vm = pv2  + dv * (ta + tb) * 0.5
                        col2 = (c_out2 if (vm < s_lo or vm > s_hi) else c_in2) if has_b else c_in2
                        dl.add_line((xa, ya), (xb, yb), col2, lw)
                pp3, ppy3, pv2 = px4, py4, v2
        dl.pop_clip_rect()

        is_ph_drag = _graph_ph_drag.get(gi2, False)

        if mouse_in and not is_ph_drag:
            if imgui.is_mouse_clicked(0) and hov_si >= 0:
                s_place = series[hov_si]
                fn_p    = _GRAPH_PARAMS.get(s_place["param"])
                fi_p    = start + int((mx2 - gx) / canvas_w * (nw - 1))
                fi_p    = max(start, min(end - 1, fi_p))
                if fn_p is not None:
                    try:
                        val_p = fn_p(d, fi_p)
                        t_p   = float(d["time"][fi_p]) if d.get("time") is not None else fi_p
                        tab.probes.append({
                            "param":      s_place["param"],
                            "graph_idx":  gi2,
                            "series_idx": hov_si,
                            "frame_idx":  fi_p,
                            "value":      val_p,
                            "t":          t_p,
                            "color":      s_place["color"],
                        })
                    except Exception:
                        pass

            if imgui.is_mouse_clicked(1):
                REMOVE_RADIUS_SQ = sc(16) ** 2
                best_idx  = -1
                best_dist = REMOVE_RADIUS_SQ
                for pi, probe in enumerate(tab.probes):
                    if probe["graph_idx"] != gi2:
                        continue
                    pfi = probe["frame_idx"]
                    if not (start <= pfi <= end - 1):
                        continue
                    fn_chk = _GRAPH_PARAMS.get(probe["param"])
                    if fn_chk is None:
                        continue
                    try:
                        pv  = fn_chk(d, pfi)
                        px_p = gx + canvas_w * (pfi - start) / max(nw - 1, 1)
                        py_p = vy(pv)
                        dist = (mx2 - px_p)**2 + (my2 - py_p)**2
                        if dist < best_dist:
                            best_dist = dist; best_idx = pi
                    except Exception:
                        pass
                if best_idx >= 0:
                    tab.probes.pop(best_idx)

        _ui_push_clip_rect(dl, (gx, gy2), (gx + canvas_w, gy2 + gh), True)
        for probe in tab.probes:
            if probe["graph_idx"] != gi2:
                continue
            pfi = probe["frame_idx"]
            if not (start <= pfi <= end - 1):
                continue
            fn_p = _GRAPH_PARAMS.get(probe["param"])
            if fn_p is None:
                continue
            try:
                pv   = fn_p(d, pfi)
                r2p, g2p, b2p = probe["color"]
                px_p = gx + canvas_w * (pfi - start) / max(nw - 1, 1)
                py_p = vy(pv)

                dash_h = sc(4); gap_h = sc(3)
                y_cur  = gy2
                while y_cur < gy2 + gh:
                    y_end = min(y_cur + dash_h, gy2 + gh)
                    dl.add_line((px_p, y_cur), (px_p, y_end), _ui_color_u32(r2p, g2p, b2p, 0.55), 1)
                    y_cur += dash_h + gap_h

                x_cur = gx
                while x_cur < gx + canvas_w:
                    x_end = min(x_cur + dash_h, gx + canvas_w)
                    dl.add_line((x_cur, py_p), (x_end, py_p), _ui_color_u32(r2p, g2p, b2p, 0.35), 1)
                    x_cur += dash_h + gap_h

                dl.add_circle_filled((px_p, py_p), sc(4), _ui_color_u32(r2p, g2p, b2p, 1.0))
                dl.add_circle((px_p, py_p), sc(5), _ui_color_u32(1, 1, 1, 0.8), 0, 1.0)

                t_str  = f"t={probe['t']:.3f}s" if d.get("time") is not None else f"f={pfi}"
                v_str  = f"{pv:+.3f}"
                lbl_w  = sc(80)
                lbl_h  = sc(28)
                lx     = px_p + sc(6)
                ly     = py_p - lbl_h - sc(4)
                if lx + lbl_w > gx + canvas_w: lx = px_p - lbl_w - sc(6)
                if ly < gy2:                    ly = py_p + sc(6)

                dl.add_rect_filled((lx - 2, ly - 2), (lx + lbl_w, ly + lbl_h),
                                   _ui_color_u32(0.05, 0.05, 0.08, 0.85))
                _ui_add_rect(dl, (lx - 2, ly - 2), (lx + lbl_w, ly + lbl_h),
                             _ui_color_u32(r2p, g2p, b2p, 0.6), 0.0, 1.0)
                dl.add_text((lx, ly),           _ui_color_u32(0.75, 0.75, 0.75, 1.0), t_str)
                dl.add_text((lx, ly + sc(13)),  _ui_color_u32(r2p,  g2p,  b2p,  1.0), v_str)
            except Exception:
                pass
        dl.pop_clip_rect()

        dl.add_line((main_ph_x, gy2), (main_ph_x, gy2 + gh), _ui_color_u32(0.8, 0.8, 1.0, 0.55), 1.5)
        dl.add_line((ph_x,      gy2), (ph_x,      gy2 + gh), _ui_color_u32(1, 0.92, 0.2, 0.9),   1.5)
        hw  = sc(8); th2 = sc(12)
        dl.add_triangle_filled((ph_x - hw, gy2), (ph_x + hw, gy2), (ph_x, gy2 + th2), _ui_color_u32(1, 0.92, 0.2, 1.0))
        dl.add_triangle_filled((ph_x - hw, gy2 + gh), (ph_x + hw, gy2 + gh), (ph_x, gy2 + gh - th2), _ui_color_u32(1, 0.92, 0.2, 0.75))

        near_ph    = abs(mx2 - ph_x) < hw + 6
        if imgui.is_mouse_down(0):
            if is_ph_drag or (mouse_in and near_ph):
                _graph_ph_drag[gi2] = True
                fi_new = start + int((mx2 - gx) / canvas_w * (nw - 1))
                G.scrub(fi_new, S.n_frames)
        else:
            if is_ph_drag:
                _graph_ph_drag[gi2] = False

        imgui.dummy((canvas_w, gh))

        axis_h = sc(16); ax_y = gy2 + gh
        dl.add_rect_filled((gx, ax_y), (gx + canvas_w, ax_y + axis_h), _ui_color_u32(0.04, 0.04, 0.06))
        for ti in range(7):
            frac = ti / 6; fi_t = start + int(frac * (nw - 1)); fi_t = max(0, min(S.n_frames - 1, fi_t))
            tx = gx + canvas_w * frac
            dl.add_line((tx, ax_y), (tx, ax_y + 4), _ui_color_u32(0.35, 0.35, 0.38, 0.8), 1)
            rel = fi_t - S.frame
            if d.get("time") is not None:
                try:    lbl = f"{float(d['time'][fi_t]):.2f}s"
                except (IndexError, ValueError): lbl = f"{rel:+d}"
            else:
                lbl = f"{rel:+d}"
            if ti < 6:
                dl.add_text((tx + 2, ax_y + 4), _ui_color_u32(0.38, 0.38, 0.4), lbl)

        imgui.set_cursor_screen_pos((body_start_x, body_start_y + sk_h + gh + axis_h + sc(6)))
        imgui.spacing()


# ---------------------------------------------------------------------------
# New-graph modal  — UNCHANGED
# ---------------------------------------------------------------------------
def draw_new_graph_modal(tab, ui: UIState) -> None:
    global _graph_id_counter
    if not _new_graph_modal["open"] or _new_graph_modal["tab_id"] != tab.id:
        return

    sc = ui.sc
    io = imgui.get_io()

    if _new_graph_modal.get("_needs_open", False):
        imgui.set_next_window_size((sc(560), sc(580)))
        imgui.set_next_window_pos(
            (io.display_size.x * 0.5, io.display_size.y * 0.5),
            imgui.Cond_.always,
            (0.5, 0.5),
        )
        _new_graph_modal["_needs_open"] = False

    flags = (
        imgui.WindowFlags_.no_collapse        |
        imgui.WindowFlags_.no_scrollbar       |
        imgui.WindowFlags_.no_scroll_with_mouse
    )

    _ui_begin("New Graph##ng_win", flags)

    _PARAM_KEYS = tab.PARAM_KEYS
    checked     = _new_graph_modal["checked"]
    for pk in _PARAM_KEYS:
        if pk not in checked:
            checked[pk] = False

    imgui.text_colored((0.65, 0.85, 1.0, 1.0), "Add New Graph")
    imgui.same_line()
    avail_x = _ui_get_content_region_avail()[0]
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail_x - sc(24))
    if _ui_button("X##ngclose", sc(24), sc(20)):
        _new_graph_modal["open"] = False
        imgui.end()
        return
    imgui.separator()
    imgui.push_item_width(sc(240))
    ch, val = _ui_input_text("Title##ngtitle", _new_graph_modal["title"], 64)
    if ch:
        _new_graph_modal["title"] = val
    imgui.pop_item_width()
    imgui.same_line()
    imgui.push_item_width(sc(72))
    ch, val = _ui_input_text("Min##ngymin", _new_graph_modal["y_min_str"], 12)
    if ch:
        _new_graph_modal["y_min_str"] = val
    imgui.pop_item_width()
    imgui.same_line()
    imgui.push_item_width(sc(72))
    ch, val = _ui_input_text("Max##ngymx", _new_graph_modal["y_max_str"], 12)
    if ch:
        _new_graph_modal["y_max_str"] = val
    imgui.pop_item_width()
    imgui.spacing()

    imgui.push_item_width(sc(220))
    ch, val = _ui_input_text("##ngsearch", _new_graph_modal["search"], 64)
    if ch:
        _new_graph_modal["search"] = val
    imgui.pop_item_width()
    imgui.same_line()
    imgui.text_disabled("Search")
    imgui.same_line()

    filt        = _new_graph_modal["search"].lower()
    visible_keys = [pk for pk in _PARAM_KEYS
                    if filt in pk.lower() or filt in _short_label(pk).lower()]
    n_checked   = sum(1 for v in checked.values() if v)
    col         = (0.4, 0.9, 0.4, 1) if n_checked else (0.5, 0.5, 0.5, 1)
    imgui.text_colored(col, f"{n_checked} selected")

    imgui.same_line()
    if _ui_button("All##ngall", sc(36), 0):
        for pk in visible_keys:
            checked[pk] = True
    imgui.same_line()
    if _ui_button("None##ngnone", sc(44), 0):
        for pk in visible_keys:
            checked[pk] = False
    imgui.separator()

    avail_w, avail_h = _ui_get_content_region_avail()
    list_h = max(sc(200), avail_h - sc(52))

    colors_list = (
        [tuple(c) for c in _cfg_mod.CFG.colors.graph_series]
        if _cfg_mod.CFG else [
            (0.25,0.55,1.00),(0.20,0.85,0.40),(1.00,0.70,0.15),(0.85,0.25,0.85),
            (0.15,0.85,0.85),(1.00,0.35,0.35),(0.85,0.85,0.20),(0.60,0.40,1.00),
        ]
    )

    _ui_push_style_color(imgui.Col_.child_bg, 0.06, 0.06, 0.09, 1.0)
    _ui_begin_child("##nglist", 0, list_h, False)

    for ci, pk in enumerate(visible_keys):
        r, g, b = colors_list[ci % len(colors_list)][:3]
        lbl = _short_label(pk)

        try:
            _ui_push_style_color(imgui.Col_.check_mark,      r,       g,       b,       1.0)
            _ui_push_style_color(imgui.Col_.frame_bg_active,  r * 0.3, g * 0.3, b * 0.3, 0.6)
            _, checked[pk] = imgui.checkbox(f"##{pk}_ngcb", checked[pk])
            imgui.pop_style_color(2)
        except AttributeError as exc:
            logger.warning(f"[ui_panels] Checkbox style push failed for '{pk}': {exc}")
            _, checked[pk] = imgui.checkbox(f"##{pk}_ngcb", checked[pk])

        imgui.same_line()
        imgui.text_colored((r, g, b, 1.0), f"{lbl:<26}")
        imgui.same_line()
        imgui.text_disabled(pk)

    imgui.end_child()
    imgui.pop_style_color()

    imgui.separator()
    can_add = n_checked > 0
    if not can_add:
        _ui_push_style_color(imgui.Col_.button,         0.22, 0.22, 0.25, 0.5)
        _ui_push_style_color(imgui.Col_.button_hovered, 0.22, 0.22, 0.25, 0.5)
        _ui_push_style_color(imgui.Col_.button_active,  0.22, 0.22, 0.25, 0.5)

    if _ui_button("Add Graph##ngadd", sc(110), sc(28)) and can_add:
        try:
            y_min = float(_new_graph_modal["y_min_str"])
        except ValueError:
            y_min = -30.0
        try:
            y_max = float(_new_graph_modal["y_max_str"])
        except ValueError:
            y_max = 30.0

        title = _new_graph_modal["title"].strip() or "New Graph"
        series_list = []
        for ci2, pk in enumerate(pk2 for pk2 in _PARAM_KEYS if checked.get(pk2, False)):
            r2, g2, b2 = colors_list[ci2 % len(colors_list)][:3]
            series_list.append({
                "param": pk, "enabled": True, "color": (r2, g2, b2),
                "limit_lo": y_min, "limit_hi": y_max,
                "lim_lo_str": str(y_min), "lim_hi_str": str(y_max),
            })

        global _graph_id_counter
        _graph_id_counter += 1
        tab.graphs.append({
            "title": title, "series": series_list,
            "lo": y_min, "hi": y_max,
            "lo_str": str(y_min), "hi_str": str(y_max),
            "lo_edit": False, "hi_edit": False, "hovered": -1,
            "side_w": 200,
            "_gid": _graph_id_counter,
        })
        logger.info(f"[ui_panels] New graph '{title}' added: {len(series_list)} series")
        _new_graph_modal["open"] = False

    if not can_add:
        imgui.pop_style_color(3)

    imgui.same_line()
    if _ui_button("Cancel##ngcancel", sc(80), sc(28)):
        _new_graph_modal["open"] = False

    if not can_add:
        imgui.same_line()
        imgui.text_colored((0.55, 0.55, 0.55, 1), "← select at least one series")

    imgui.end()


# ---------------------------------------------------------------------------
# Config panel — renders config.yaml as a styled, readable panel
# ---------------------------------------------------------------------------

_CFG_SKIP_KEYS = {"_locked", "_type_safe", "_convert_dict"}

# Preferred section display order
_CFG_SECTION_ORDER = [
    "gait", "raibert", "qp", "env", "obs", "rewards", "robot", "training",
]

# Per-section (accent_rgb, bg_rgb) — accent for header bar, bg for row tinting
_CFG_SECTION_STYLE: dict[str, tuple] = {
    "gait":     ((0.30, 0.82, 0.52), (0.08, 0.18, 0.12)),
    "raibert":  ((0.95, 0.68, 0.18), (0.18, 0.13, 0.04)),
    "qp":       ((0.75, 0.35, 1.00), (0.14, 0.08, 0.20)),
    "obs":      ((0.28, 0.80, 0.88), (0.06, 0.16, 0.20)),
    "env":      ((1.00, 0.50, 0.28), (0.20, 0.10, 0.05)),
    "rewards":  ((1.00, 0.82, 0.22), (0.18, 0.15, 0.04)),
    "robot":    ((0.35, 0.68, 1.00), (0.06, 0.12, 0.22)),
    "training": ((0.60, 0.62, 0.68), (0.10, 0.10, 0.13)),
}
_CFG_DEFAULT_STYLE = ((0.62, 0.68, 0.72), (0.10, 0.10, 0.13))


def _cfg_scalar_str(v) -> str | None:
    """Short display string for a scalar. Returns None for complex types."""
    if isinstance(v, (bytes, bytearray)): return f"<bytes {len(v)}>"
    if v is None:   return "(none)"
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float): return f"{v:.6g}"
    if isinstance(v, int):   return str(v)
    if isinstance(v, str):   return f'"{v}"' if len(v) < 48 else f'"{v[:45]}..."'
    # numpy arrays — show shape and compact values
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            if v.ndim == 0:
                return f"{float(v):.6g}"
            elif v.size <= 6:
                vals = ", ".join(f"{x:.4g}" for x in v.flat)
                return f"[{vals}]"
            else:
                vals = ", ".join(f"{x:.4g}" for x in list(v.flat)[:4])
                return f"[{vals} ...  shape={list(v.shape)}]"
    except ImportError:
        pass
    return None



def _cfg_list_str(lst) -> str:
    """Compact one-liner for a flat list of scalars."""
    parts = []
    for x in lst[:10]:
        s = _cfg_scalar_str(x)
        parts.append(s if s else "?")
    suffix = f"  +{len(lst)-10} more" if len(lst) > 10 else ""
    return "[" + ",  ".join(parts) + suffix + "]"


def _render_kv_row(key: str, value, accent_col: tuple, row_idx: int, dl, sc) -> None:
    """Render one key=value row with alternating background tinting."""
    # Alternating row background
    if row_idx % 2 == 0:
        rx, ry = _ui_get_cursor_screen_pos()
        rw, _ = _ui_get_content_region_avail()
        dl.add_rect_filled(
            (rx - 4, ry - 1),
            (rx + rw + 4, ry + sc(16)),
            _ui_color_u32(0.12, 0.12, 0.16, 0.55),
        )

    imgui.push_item_width(-1)

    # Key column (fixed ~40% width using indent + same_line trick)
    key_w = 160
    imgui.text_colored(accent_col, f"  {key}")
    imgui.same_line(key_w)
    imgui.text_colored((0.45, 0.45, 0.50, 1.0), "|")
    imgui.same_line()

    val_str = _cfg_scalar_str(value)
    if val_str is not None:
        # Colour booleans distinctively
        if isinstance(value, bool):
            col = (0.35, 0.90, 0.50, 1.0) if value else (0.90, 0.38, 0.35, 1.0)
        elif isinstance(value, (int, float)):
            col = (0.88, 0.88, 0.60, 1.0)
        elif isinstance(value, str):
            col = (0.70, 0.88, 0.70, 1.0)
        else:
            col = (0.80, 0.80, 0.80, 1.0)
        imgui.text_colored(col, val_str)
    elif isinstance(value, (list, tuple)) and len(value) > 0 and not isinstance(value[0], dict):
        if isinstance(value[0], (list, tuple)):
            imgui.text_colored((0.60, 0.70, 0.88, 1.0),
                               f"matrix {len(value)}x{len(value[0])}")
        else:
            imgui.text_colored((0.60, 0.82, 0.92, 1.0), _cfg_list_str(value))
    else:
        imgui.text_disabled("...")

    imgui.pop_item_width()


def _render_cfg_section_body(data: dict, accent_col: tuple, sc, dl) -> None:
    """Render all scalar/list key-value pairs in a dict, then motor sub-cards."""
    # Separate scalars/lists from nested dicts and motor-lists
    scalars  = {k: v for k, v in data.items()
                if k not in _CFG_SKIP_KEYS
                and not (isinstance(v, dict))
                and not (isinstance(v, (list, tuple)) and len(v) > 0
                         and isinstance(v[0], dict))}
    subdicts = {k: v for k, v in data.items()
                if k not in _CFG_SKIP_KEYS and isinstance(v, dict)}
    sublists = {k: v for k, v in data.items()
                if k not in _CFG_SKIP_KEYS
                and isinstance(v, (list, tuple)) and len(v) > 0
                and isinstance(v[0], dict)}

    # ── Scalar/list rows ─────────────────────────────────────────────────
    for i, (k, v) in enumerate(sorted(scalars.items())):
        _render_kv_row(k, v, accent_col, i, dl, sc)

    # ── Nested dicts (sub-sections like motors > sub-config) ─────────────
    for k, v in sorted(subdicts.items()):
        imgui.spacing()
        imgui.text_colored((0.55, 0.55, 0.62, 1.0), f"  >> {k}")
        imgui.separator()
        for i2, (k2, v2) in enumerate(sorted(v.items())):
            if k2 not in _CFG_SKIP_KEYS:
                _render_kv_row(k2, v2, accent_col, i2, dl, sc)

    # ── List-of-dicts (e.g. motors) as collapsing mini-cards ─────────────
    for k, lst in sorted(sublists.items()):
        imgui.spacing()
        hdr_open = _ui_collapsing_header(f"  {k}  [{len(lst)} entries]##sub_{k}")
        if hdr_open:
            for mi, item in enumerate(lst):
                if not isinstance(item, dict):
                    continue
                name_lbl = item.get("name", f"[{mi}]")
                card_open = _ui_collapsing_header(f"    {name_lbl}##card_{k}_{mi}")
                if card_open:
                    for i3, (k3, v3) in enumerate(sorted(item.items())):
                        if k3 != "name":
                            _render_kv_row(k3, v3, accent_col, i3, dl, sc)


def render_config_panel(tab: TabState, avail_w: float, avail_h: float, ui: UIState) -> None:
    """Render the config.yaml browser as a styled, table-like panel."""
    sc = ui.sc
    dl = imgui.get_window_draw_list()

    if tab.train_cfg_raw is None:
        imgui.spacing()
        imgui.text_colored((1.0, 0.55, 0.2, 1.0), "  No config.yaml found for this run.")
        imgui.spacing()
        imgui.text_disabled("  Expected: config.yaml in the data directory.")
        imgui.text_disabled("  The eval script copies config.yaml into the output folder.")
        return

    cfg = tab.train_cfg_raw

    ordered = [k for k in _CFG_SECTION_ORDER if k in cfg]
    rest    = sorted(k for k in cfg if k not in _CFG_SECTION_ORDER and k not in _CFG_SKIP_KEYS)

    for section_key in ordered + rest:
        if section_key in _CFG_SKIP_KEYS:
            continue
        value  = cfg[section_key]
        accent_rgb, bg_rgb = _CFG_SECTION_STYLE.get(section_key, _CFG_DEFAULT_STYLE)
        accent_col = (*accent_rgb, 1.0)

        imgui.push_id(f"cfs_{section_key}")
        imgui.spacing()

        # ── Section header bar ────────────────────────────────────────────
        # Draw a filled rect behind the collapsing header for colour
        hdr_x, hdr_y = _ui_get_cursor_screen_pos()
        hdr_w, _ = _ui_get_content_region_avail()
        hdr_h = sc(20)
        dl.add_rect_filled(
            (hdr_x, hdr_y),
            (hdr_x + hdr_w, hdr_y + hdr_h),
            _ui_color_u32(*bg_rgb, 1.0),
        )
        dl.add_line(
            (hdr_x, hdr_y), (hdr_x + hdr_w, hdr_y),
            _ui_color_u32(*accent_rgb, 0.80), 1.5,
        )
        dl.add_line(
            (hdr_x, hdr_y + hdr_h), (hdr_x + hdr_w, hdr_y + hdr_h),
            _ui_color_u32(*accent_rgb, 0.25), 1.0,
        )

        _ui_push_style_color(imgui.Col_.header,         *accent_rgb, 0.0)   # transparent fill (we drew it)
        _ui_push_style_color(imgui.Col_.header_hovered, *accent_rgb, 0.18)
        _ui_push_style_color(imgui.Col_.header_active,  *accent_rgb, 0.32)
        _ui_push_style_color(imgui.Col_.text,           *accent_rgb, 1.0)

        if isinstance(value, dict):
            section_open = _ui_collapsing_header(f"  {section_key.upper()}")
        else:
            # Scalar top-level key — just show inline
            section_open = False
            imgui.text_colored(accent_col, f"  {section_key}")
            imgui.same_line()
            vs = _cfg_scalar_str(value)
            imgui.text_colored((0.88, 0.88, 0.60, 1.0), vs or "...")

        imgui.pop_style_color(4)

        if section_open and isinstance(value, dict):
            imgui.indent(sc(8))
            _render_cfg_section_body(value, accent_col, sc, dl)
            imgui.unindent(sc(8))

        imgui.pop_id()

    imgui.spacing()



# ---------------------------------------------------------------------------
# Settings panel  — UNCHANGED, + show_raibert_boxes checkbox added
# ---------------------------------------------------------------------------
def render_settings(tab: TabState, ui: UIState) -> None:
    S  = tab.S
    G  = tab.G
    sc = ui.sc

    imgui.text_colored((0.65, 0.75, 0.88, 1), "CAMERA")
    imgui.separator()

    if imgui.radio_button("Follow Main##fm",  S.cam_follow_mode == "main"):
        S.cam_follow_mode = "main";  S.update_cam()
    imgui.same_line()
    if imgui.radio_button("Follow Ghost##fg", S.cam_follow_mode == "ghost"):
        S.cam_follow_mode = "ghost"; S.update_cam()

    imgui.push_item_width(sc(80))
    cv, vv = imgui.drag_float("Yaw##cy",   S.cam_yaw,   0.5, -180, 180, "%.0f°")
    if cv: S.cam_yaw   = vv
    imgui.same_line()
    cv, vv = imgui.drag_float("Pitch##cp", S.cam_pitch, 0.3, 5, 85, "%.0f°")
    if cv: S.cam_pitch = vv
    imgui.same_line()
    cv, vv = imgui.drag_float("Dist##cd",  S.cam_dist,  0.05, 0.3, 15, "%.1fm")
    if cv: S.cam_dist  = vv
    imgui.pop_item_width()
    imgui.same_line()
    if _ui_button("Reset##rc"):
        S.cam_yaw   = _cfg_mod.CFG.camera.default_yaw   if _cfg_mod.CFG else 45.0
        S.cam_pitch = _cfg_mod.CFG.camera.default_pitch  if _cfg_mod.CFG else 25.0
        S.cam_dist  = _cfg_mod.CFG.camera.default_dist   if _cfg_mod.CFG else 3.5

    imgui.spacing()
    imgui.text_colored((0.65, 0.75, 0.88, 1), "OVERLAYS")
    imgui.separator()
    _, S.show_forces       = imgui.checkbox("Forces##sf",   S.show_forces);       imgui.same_line()
    _, S.show_contacts     = imgui.checkbox("Contacts##sc", S.show_contacts);     imgui.same_line()
    _, S.show_limits       = imgui.checkbox("Limits##sl",   S.show_limits);       imgui.same_line()
    _, S.show_grid         = imgui.checkbox("Grid##sg",     S.show_grid);         imgui.same_line()
    _, S.show_trajectory   = imgui.checkbox("Traj##st",     S.show_trajectory);   imgui.same_line()
    _, S.show_joint_frames = imgui.checkbox("JFrames##sjf", S.show_joint_frames); imgui.same_line()
    # show_raibert_boxes was added in a previous session — guard against old state objects
    if hasattr(S, "show_raibert_boxes"):
        _, S.show_raibert_boxes = imgui.checkbox("Raibert##srb", S.show_raibert_boxes)

    imgui.spacing()
    imgui.text_colored((0.65, 0.75, 0.88, 1), "FOG")
    imgui.separator()
    imgui.push_item_width(sc(120))
    cv, vv = imgui.drag_float("Start##fgs", S.fog_start, 0.2, 0.5, 50, "%.1f")
    if cv: S.fog_start = vv
    imgui.same_line()
    cv, vv = imgui.drag_float("End##fge", S.fog_end, 0.4, 1, 200, "%.1f")
    if cv: S.fog_end = max(S.fog_start + 1, vv)
    imgui.pop_item_width()

    imgui.spacing()
    imgui.text_colored((0.65, 0.75, 0.88, 1), "PREFERENCES")
    imgui.separator()
    ch, v = imgui.checkbox("Invert Y-Axis##invy", ui.invert_y)
    if ch:
        ui.invert_y = v
    imgui.same_line()
    imgui.push_item_width(sc(140))
    cv, vv = imgui.drag_float("Scale##uisc", ui.scale, 0.05, 0.5, 3.0, "%.2fx")
    if cv:
        ui.scale = max(0.5, min(3.0, round(vv, 2)))
        ui.set_font_scale()
    imgui.pop_item_width()


# ---------------------------------------------------------------------------
# Timeline panel  — UNCHANGED
# ---------------------------------------------------------------------------
def _timeline_bar(
    dl, px: float, py: float, bw: float, bh: float,
    n_fr: int, lo: int, hi: int, cur: int,
    sc,
) -> tuple[int, int, int, bool]:
    MINI_H = sc(16)
    MAIN_H = max(20, bh - MINI_H * 2 - 2)
    in_y   = py
    out_y  = py + MINI_H + 1
    main_y = py + MINI_H * 2 + 2

    xlo = px + bw * lo  / max(n_fr - 1, 1)
    xhi = px + bw * hi  / max(n_fr - 1, 1)
    xfr = px + bw * cur / max(n_fr - 1, 1)

    dl.add_rect_filled((px, in_y), (px + bw, in_y + MINI_H), _ui_color_u32(0.07, 0.11, 0.07))
    dl.add_rect_filled((xlo, in_y), (px + bw, in_y + MINI_H), _ui_color_u32(0.08, 0.28, 0.08, 0.55))
    dl.add_line((xlo, in_y), (xlo, in_y + MINI_H), _ui_color_u32(0.2, 0.9, 0.2, 0.9), 2)
    dl.add_triangle_filled((xlo - 4, in_y), (xlo + 4, in_y), (xlo, in_y + MINI_H), _ui_color_u32(0.2, 0.9, 0.2, 1.0))
    dl.add_line((xfr, in_y), (xfr, in_y + MINI_H), _ui_color_u32(1, 0.92, 0.2, 0.4), 1)
    dl.add_text((px + 2, in_y), _ui_color_u32(0.25, 0.75, 0.25, 0.9), f"IN {lo}")

    dl.add_rect_filled((px, out_y), (px + bw, out_y + MINI_H), _ui_color_u32(0.11, 0.07, 0.07))
    dl.add_rect_filled((px, out_y), (xhi, out_y + MINI_H), _ui_color_u32(0.28, 0.08, 0.08, 0.55))
    dl.add_line((xhi, out_y), (xhi, out_y + MINI_H), _ui_color_u32(0.9, 0.2, 0.2, 0.9), 2)
    dl.add_triangle_filled((xhi - 4, out_y), (xhi + 4, out_y), (xhi, out_y + MINI_H), _ui_color_u32(0.9, 0.2, 0.2, 1.0))
    dl.add_line((xfr, out_y), (xfr, out_y + MINI_H), _ui_color_u32(1, 0.92, 0.2, 0.4), 1)
    dl.add_text((px + bw - 52, out_y), _ui_color_u32(0.75, 0.25, 0.25, 0.9), f"OUT {hi}")

    dl.add_rect_filled((px, main_y), (px + bw, main_y + MAIN_H), _ui_color_u32(0.09, 0.09, 0.11))
    dl.add_rect_filled((xlo, main_y), (xhi, main_y + MAIN_H), _ui_color_u32(0.12, 0.40, 0.12, 0.38))
    _ui_add_rect(dl, (xlo, main_y), (xhi, main_y + MAIN_H), _ui_color_u32(0.2, 0.65, 0.2, 0.5), 0.0, 1.0)

    gw    = _cfg_mod.CFG.playback.graph_window if _cfg_mod.CFG else 50
    w50_lo = max(0,      cur - gw); w50_hi = min(n_fr - 1, cur + gw)
    xwl   = px + bw * w50_lo / max(n_fr - 1, 1)
    xwh   = px + bw * w50_hi / max(n_fr - 1, 1)
    mid   = main_y + MAIN_H // 2
    dl.add_rect_filled((xwl, mid - 2), (xwh, mid + 2), _ui_color_u32(0.4, 0.6, 1.0, 0.35))

    for i in range(0, 11):
        tx = px + bw * i / 10
        dl.add_line((tx, main_y + MAIN_H - 5), (tx, main_y + MAIN_H), _ui_color_u32(0.35, 0.35, 0.35, 0.6), 1)
        dl.add_text((tx + 2, main_y + MAIN_H - 14), _ui_color_u32(0.42, 0.42, 0.42), str(int(n_fr * i / 10)))

    dl.add_line((xfr, main_y), (xfr, main_y + MAIN_H), _ui_color_u32(1, 0.92, 0.2, 1), 2)
    hw = sc(8); th = min(sc(12), MAIN_H // 2)
    dl.add_triangle_filled((xfr - hw, main_y), (xfr + hw, main_y), (xfr, main_y + th), _ui_color_u32(1, 0.92, 0.2, 1))
    dl.add_triangle_filled((xfr - hw, main_y + MAIN_H), (xfr + hw, main_y + MAIN_H), (xfr, main_y + MAIN_H - th), _ui_color_u32(1, 0.92, 0.2, 0.8))

    mx, my = _ui_get_mouse_pos()
    in_widget = (px <= mx <= px + bw) and (py <= my <= py + bh)
    new_lo, new_hi, new_cur = lo, hi, cur
    changed = False

    if imgui.is_mouse_down(0):
        if _tl_drag["active"] is None and in_widget:
            if   my <= in_y + MINI_H:   _tl_drag["active"] = "in"
            elif my <= out_y + MINI_H:  _tl_drag["active"] = "out"
            elif abs(mx - xlo) < 10:    _tl_drag["active"] = "in"
            elif abs(mx - xhi) < 10:    _tl_drag["active"] = "out"
            else:                       _tl_drag["active"] = "seek"

        a = _tl_drag["active"]
        if a == "in":
            new_lo  = int((mx - px) / bw * (n_fr - 1)); new_lo  = max(0, min(hi - 1, new_lo));   changed = True
        elif a == "out":
            new_hi  = int((mx - px) / bw * (n_fr - 1)); new_hi  = max(lo + 1, min(n_fr - 1, new_hi)); changed = True
        elif a == "seek":
            new_cur = int((mx - px) / bw * (n_fr - 1)); new_cur = max(lo, min(hi, new_cur));      changed = True
    else:
        _tl_drag["active"] = None

    return new_lo, new_hi, new_cur, changed


def render_timeline(tab: TabState, avail_w: float, avail_h: float, ui: UIState) -> None:
    S  = tab.S
    G  = tab.G
    sc = ui.sc

    if S.data is None:
        imgui.text_disabled("No data loaded.")
        return

    n_fr = S.n_frames

    # ── Playback mode radio ───────────────────────────────────────────────
    imgui.text_colored((0.55, 0.55, 0.60, 1.0), "Playback:")
    imgui.same_line()
    was_rt = getattr(S, "realtime_mode", False)
    if imgui.radio_button("Speed##rtoff", not was_rt):
        if was_rt:
            S.realtime_mode = False
    imgui.same_line()
    if imgui.radio_button("Realtime (1x)##rton", was_rt):
        if not was_rt:
            S.realtime_mode = True
            S.start_realtime()   # anchor clock to current frame
    imgui.same_line()
    # Speed multiplier only relevant in speed mode
    if not was_rt:
        imgui.push_item_width(sc(90))
        cv, vv = imgui.drag_float("##sprt", S.play_speed, 0.02, 0.1, 5.0, "%.2fx")
        if cv:
            S.play_speed = max(0.05, vv)
        imgui.pop_item_width()
        imgui.same_line()
    else:
        imgui.text_colored((0.35, 0.90, 0.55, 1.0), "  wall-clock sync")
        imgui.same_line()
    imgui.separator()


    imgui.same_line()
    if _ui_button("<<##b",   sc(28), 0): S.step(-10)
    imgui.same_line()
    if _ui_button("< ##c",   sc(28), 0): S.step(-1)
    imgui.same_line()

    mode_lbl = "GRAPH" if G.play_mode == "graph" else "MAIN "
    mc       = (0.2, 0.72, 0.25, 1.0) if G.play_mode == "graph" else (0.22, 0.42, 0.82, 1.0)
    _ui_push_style_color(imgui.Col_.button, *mc)
    if _ui_button(f"{mode_lbl}##pm", sc(46), 0):
        if G.play_mode == "main":
            G.set_mode("graph", S.frame)
        else:
            G.set_mode("main",  S.frame)
    imgui.pop_style_color()
    imgui.same_line()

    if G.play_mode == "main" or getattr(S, "realtime_mode", False):
        play_lbl = "Pause" if S.playing else "Play "
        if _ui_button(f"{play_lbl}##d", sc(46), 0): S.toggle_play()
    else:
        play_lbl = "Pause" if G._playing else "Play "
        if _ui_button(f"{play_lbl}##d", sc(46), 0): G.toggle_graph_play(S.n_frames)

    imgui.same_line()
    if _ui_button("> ##e",   sc(28), 0): S.step(1)
    imgui.same_line()
    if _ui_button(">>##f",   sc(28), 0): S.step(10)
    imgui.same_line()
    if _ui_button(">|##g",   sc(28), 0): S.set_frame(S._hi())

    imgui.same_line()
    imgui.push_item_width(sc(120))
    ch, v = imgui.drag_int("##fr", S.frame, 1, 0, n_fr - 1, "Frame %d")
    if ch:
        S.set_frame(v)
    imgui.pop_item_width()
    imgui.same_line()
    vx_str = (
        f"  vx={float(S.data['desired_vel_x'][S.frame]):.2f}"
        if S.data.get("desired_vel_x") is not None else ""
    )
    imgui.text_disabled(f"t={float(S.data['time'][S.frame]):.2f}s{vx_str}")

    imgui.spacing()
    lo2  = S.loop_start
    hi2  = S._hi()
    dl2  = imgui.get_window_draw_list()
    bx, by = _ui_get_cursor_screen_pos()
    btn_area = sc(150)
    bw2  = max(100, avail_w - btn_area - 20)
    bh2  = max(sc(30), avail_h - sc(60))

    nlo2, nhi2, nfr2, ch2 = _timeline_bar(dl2, bx, by, bw2, bh2, n_fr, lo2, hi2, S.frame, sc)
    if ch2:
        S.loop_start = nlo2
        S.loop_end   = nhi2
        S.set_frame(nfr2)

    _ui_invisible_button("##tl2", bw2, bh2)
    imgui.set_cursor_screen_pos((bx + bw2 + 6, by))
    if _ui_button("[In]##si",  sc(44), 0): S.loop_start = S.frame
    imgui.same_line()
    if _ui_button("[Out]##so", sc(44), 0): S.loop_end   = S.frame
    if _ui_button("Reset##lr", sc(44), 0):
        S.loop_start = 0; S.loop_end = 0
    imgui.same_line()
    imgui.text_disabled(f"{lo2}-{hi2}")
