"""
dataviz/viewer.py

Owns:
  - Window creation (pyglet)
  - ImGui context + renderer setup
  - GLResources construction
  - Tab bar + draw_frame loop
  - _update() scheduled at update_hz
  - pyglet event handlers
  - Layout save/load
  - run_viewer() — public entry point

Imports everything, contains as little logic as possible.
All rendering delegates to gl_core. All UI delegates to ui_panels.
"""

import ctypes
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import pyglet
    from pyglet.gl import (
        glBindFramebuffer, GL_FRAMEBUFFER,
        glClearColor, glClear,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        glDisable, GL_DEPTH_TEST, GL_BLEND,
        glViewport,
    )
except ImportError as exc:
    raise ImportError(
        f"[viewer] pyglet not available: {exc}\n"
        "Install with: pip install pyglet"
    ) from exc

try:
    from imgui_bundle import imgui
    _IMGUI_BACKEND = "bundle"
except ImportError:
    try:
        import imgui
        _IMGUI_BACKEND = "classic"
    except ImportError:
        raise ImportError(
            "[viewer] Neither imgui-bundle nor imgui is installed.\n"
            "Install with: pip install imgui-bundle"
        )

from dataviz.config    import CFG, init_config
from dataviz.state     import TabState, do_load
from dataviz.data      import find_robots_dir, scan_robots
from dataviz.gl_core   import (
    GLResources, build_gl_resources,
    ensure_fbo, delete_fbo, render_3d, init_tab_gl,
)
from dataviz.ui_panels import (
    UIState, make_ui_state,
    draw_picker, render_graphs, render_settings, render_timeline,
    render_config_panel,
    draw_new_graph_modal,
    _ui_begin, _ui_begin_child, _ui_button, _ui_image,
    _ui_get_cursor_screen_pos, _ui_get_content_region_avail,
    _ui_get_mouse_delta, _ui_get_mouse_pos,
    _ui_push_style_color, _ui_begin_tab_item, _ui_begin_tab_item_closable,
    _UI_KEY_SPACE, _UI_KEY_LEFT, _UI_KEY_RIGHT,
    _UI_TAB_SET_SELECTED,
)


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
def _detect_platform() -> str:
    p = platform.system().lower()
    if p == "darwin":  return "macos"
    if p == "windows": return "windows"
    return "linux"

_PLATFORM = _detect_platform()


# ---------------------------------------------------------------------------
# Layout persistence helpers
# ---------------------------------------------------------------------------
def _layout_file() -> str:
    return CFG.layout.layout_save_file if CFG else "viewer_layout.json"


def save_layout(tab: TabState, ui: UIState, win) -> None:
    d = {
        "vp_frac":  tab.vp_frac,
        "bot_frac": tab.bot_frac,
        "ui_scale": ui.scale,
        "invert_y": ui.invert_y,
        "win_w":    win.width,
        "win_h":    win.height,
    }
    try:
        with open(_layout_file(), "w") as f:
            json.dump(d, f, indent=2)
        logger.info(f"[viewer] Layout saved → {_layout_file()}")
    except Exception as exc:
        logger.warning(f"[viewer] Could not save layout: {exc}")


def load_layout(tab: TabState, ui: UIState, win) -> None:
    try:
        with open(_layout_file(), "r") as f:
            d = json.load(f)
        tab.vp_frac   = float(d.get("vp_frac",  CFG.layout.default_vp_frac  if CFG else 0.54))
        tab.bot_frac  = float(d.get("bot_frac",  CFG.layout.default_bot_frac if CFG else 0.22))
        ui.scale      = float(d.get("ui_scale",  1.0))
        ui.invert_y   = bool( d.get("invert_y",  False))
        ui.set_font_scale()
        win.set_size(int(d.get("win_w", win.width)), int(d.get("win_h", win.height)))
        logger.info(f"[viewer] Layout loaded from {_layout_file()}")
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"[viewer] Could not load layout: {exc}")


# ---------------------------------------------------------------------------
# run_viewer — public entry point
# ---------------------------------------------------------------------------
def run_viewer(
    config_path: str = "config.json",
    initial_urdf: str | None = None,
    initial_data: str | None = None,
) -> None:
    """
    Main entry point.
    Called from test_modular.py after CLI args are parsed.
    """
    # ── Config ────────────────────────────────────────────────────────────
    logger.info(f"[viewer] Platform: {_PLATFORM}  imgui backend: {_IMGUI_BACKEND}")
    # CFG should already be initialised by test_modular.py via init_config().
    # Only re-load if somehow called directly without prior init.
    if CFG is None:
        logger.warning("[viewer] CFG not initialised — calling init_config. "
                       "This should not happen when launched via test_modular.py.")
        init_config(config_path)

    W = CFG.layout.window_width  if CFG else 1480
    H = CFG.layout.window_height if CFG else 860

    # ── Window ────────────────────────────────────────────────────────────
    cfg = pyglet.gl.Config(
        double_buffer=True, depth_size=24,
        major_version=3, minor_version=3,
        forward_compatible=True,
        sample_buffers=1, samples=4,
    )
    try:
        win = pyglet.window.Window(
            width=W, height=H, caption="Go2 Viewer",
            resizable=True, config=cfg,
        )
    except Exception as exc:
        raise RuntimeError(f"[viewer] Could not create pyglet window: {exc}") from exc
    logger.info(f"[viewer] Window created: {W}×{H}")

    # ── ImGui ─────────────────────────────────────────────────────────────
    imgui.create_context()
    _io = imgui.get_io()
    try:
        _io.ini_file_name = b""
    except AttributeError:
        try:
            _io.ini_file_name = ""
        except AttributeError:
            pass

    if _IMGUI_BACKEND == "bundle":
        try:
            from imgui_bundle.python_backends.pyglet_backend import PygletProgrammablePipelineRenderer as _Rend
            _renderer = _Rend(win)
        except Exception as exc:
            logger.warning(f"[viewer] imgui-bundle pyglet backend failed ({exc}), falling back")
            from imgui.integrations.opengl import ProgrammablePipelineRenderer as _Rend
            _renderer = _Rend()
    else:
        from imgui.integrations.opengl import ProgrammablePipelineRenderer as _Rend
        _renderer = _Rend()


    def _set_display_size(w: int, h: int) -> None:
        try:
            try:
                fbw, fbh = win.get_framebuffer_size()
            except Exception:
                fbw, fbh = w, h
            scale_x = fbw / w if w > 0 else 1.0
            scale_y = fbh / h if h > 0 else 1.0
            _io.display_size = (w, h)
            _io.display_framebuffer_scale = (scale_x, scale_y)
        except Exception as exc:
            logger.warning(f"[viewer] _set_display_size failed: {exc}")
            try:
                _io.display_size = (w, h)
            except Exception:
                pass


    try:
        _fbw, _fbh = win.get_framebuffer_size()
    except Exception:
        _fbw, _fbh = W, H
    _set_display_size(_fbw, _fbh)

    # ── Shared UI state ───────────────────────────────────────────────────
    ui = make_ui_state()

    # ── GL resources (shaders + shared VAOs) ─────────────────────────────
    try:
        gl: GLResources = build_gl_resources()
    except RuntimeError as exc:
        logger.critical(f"[viewer] GL init failed: {exc}")
        sys.exit(1)

    # ── Tab registry ──────────────────────────────────────────────────────
    _tabs               = []
    _active_tab         = 0
    _switch_to_tab      = 0
    _ignore_plus_frames = 0
    _fbo_delete_queue   = []

    SPLITTER_W = CFG.layout.splitter_width if CFG else 6
    WIN_FLAGS  = (
        imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
        imgui.WindowFlags_.no_move      | imgui.WindowFlags_.no_collapse |
        imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
    )

    # ── Per-tab content renderer ──────────────────────────────────────────
    def draw_tab_content(tab: TabState) -> None:
        S  = tab.S
        G  = tab.G
        sc = ui.sc
        total_w, total_h = _ui_get_content_region_avail()

        if not tab.loaded:
            draw_picker(tab, total_w, total_h)

            # After picker calls do_load, init GL + graphs here (needs GL context)
            if tab.loaded:
                _finish_tab_load(tab)
                # Switch focus to this tab now that it's loaded
                if tab in _tabs:
                    _switch_to_tab = _tabs.index(tab)
            return

        # Layout
        bot_h   = max(sc(100), int(total_h * tab.bot_frac))
        top_h   = max(sc(100), total_h - bot_h - SPLITTER_W)
        vp_w    = max(sc(200), int(total_w * tab.vp_frac))
        right_w = max(sc(200), total_w - vp_w - SPLITTER_W)

        # ── 3D Renderer ───────────────────────────────────────────────────
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0.0, 0.0))
        _ui_push_style_color(imgui.Col_.child_bg, 0.0, 0.0, 0.0, 1.0)
        _ui_begin_child(f"##renderer{tab.id}", vp_w, top_h, False, imgui.WindowFlags_.no_scrollbar)

        rw_a, rh_a = _ui_get_content_region_avail()
        rw = max(1, int(rw_a)); rh = max(1, int(rh_a))
        ensure_fbo(tab, rw, rh)

        glBindFramebuffer(GL_FRAMEBUFFER, tab.fbo)
        render_3d(tab, gl, rw, rh)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND)
        try:
            _vp_w, _vp_h = win.get_framebuffer_size()
        except Exception:
            _vp_w, _vp_h = win.width, win.height
        glViewport(0, 0, _vp_w, _vp_h)

        tex_id = int(tab.fbo_tex.value)
        _ui_image(tex_id, rw, rh)
        img_hovered = imgui.is_item_hovered()

        # Camera input
        if img_hovered:
            orb  = CFG.camera.orbit_sensitivity if CFG else 0.3
            pan  = CFG.camera.pan_sensitivity   if CFG else 0.004
            zoom = CFG.camera.zoom_sensitivity  if CFG else 0.3

            if imgui.is_mouse_dragging(0):
                dx, dy = _ui_get_mouse_delta()
                S.cam_yaw  -= dx * orb
                psign       = -1.0 if ui.invert_y else 1.0
                S.cam_pitch = max(
                    CFG.camera.min_pitch if CFG else 5.0,
                    min(CFG.camera.max_pitch if CFG else 85.0,
                        S.cam_pitch - dy * orb * psign)
                )
            if imgui.is_mouse_dragging(1):
                dx, dy = _ui_get_mouse_delta()
                yr = math.radians(S.cam_yaw)
                r  = np.array([-math.sin(yr), math.cos(yr), 0.0])
                S.cam_target += r * (-dx * pan) + np.array([0, 0, 1]) * (dy * pan)
            scroll = imgui.get_io().mouse_wheel
            if abs(scroll) > 0.01:
                S.cam_dist = max(0.3, S.cam_dist - scroll * zoom)

            # Keyboard
            if imgui.is_key_pressed(_UI_KEY_SPACE, False):
                if G.play_mode == "graph": G.toggle_graph_play(S.n_frames)
                else:                      S.toggle_play()
            if imgui.is_key_pressed(_UI_KEY_LEFT, True):
                if G.play_mode == "graph": G.frame = max(0, G.frame - 1)
                else:                      S.step(-1)
            if imgui.is_key_pressed(_UI_KEY_RIGHT, True):
                if G.play_mode == "graph": G.frame = min(S.n_frames - 1, G.frame + 1)
                else:                      S.step(1)

        imgui.end_child()
        imgui.pop_style_color()
        imgui.pop_style_var()

        # Vertical splitter
        imgui.same_line()
        _ui_push_style_color(imgui.Col_.button,         0.22, 0.22, 0.26, 0.7)
        _ui_push_style_color(imgui.Col_.button_hovered, 0.45, 0.65, 1.0,  0.8)
        _ui_push_style_color(imgui.Col_.button_active,  0.45, 0.65, 1.0,  1.0)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 0.0))
        _ui_button(f"##vsplit{tab.id}", SPLITTER_W, top_h)
        if imgui.is_item_active():
            dx, _ = _ui_get_mouse_delta()
            tab.vp_frac = max(0.2, min(0.8, (vp_w + dx) / max(total_w, 1)))
        if imgui.is_item_hovered() or imgui.is_item_active():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        imgui.pop_style_var(); imgui.pop_style_color(3)

        # Right panel
        imgui.same_line()
        _ui_begin_child(f"##rightpanel{tab.id}", right_w, top_h, False)

        graph_h    = max(sc(80),   int(top_h * tab.graph_h_frac))
        settings_h = max(sc(60),   top_h - graph_h - SPLITTER_W)
        # Reserve space for the button bar below the scrollable graphs/config
        btn_bar_h  = sc(30)
        scroll_h   = max(sc(60), graph_h - btn_bar_h)

        _ui_begin_child(f"##graphs{tab.id}", 0, scroll_h, False,
                        imgui.WindowFlags_.horizontal_scrollbar)

        # ── Toggle button: "Config" ↔ "Live Graphs" ──────────────────────
        # When showing graphs → button label is "Config"   (click to switch)
        # When showing config → button label is "Live Graphs" (click to switch)
        show_cfg = getattr(tab, "show_config_panel", False)
        toggle_lbl = "Config##cfgtog" if not show_cfg else "Live Graphs##cfgtog"
        _ui_push_style_color(imgui.Col_.button,
                             *(0.28, 0.48, 0.80, 0.80) if not show_cfg
                              else (0.22, 0.58, 0.38, 0.80))
        _ui_push_style_color(imgui.Col_.button_hovered,
                             *(0.40, 0.60, 1.00, 1.00) if not show_cfg
                              else (0.28, 0.75, 0.50, 1.00))
        _ui_push_style_color(imgui.Col_.button_active,
                             *(0.50, 0.70, 1.00, 1.00) if not show_cfg
                              else (0.35, 0.85, 0.60, 1.00))
        if _ui_button(toggle_lbl, sc(110), sc(22)):
            tab.show_config_panel = not show_cfg
            show_cfg = tab.show_config_panel
        imgui.pop_style_color(3)

        imgui.same_line()
        if _ui_button("Save Layout##sl"): save_layout(tab, ui, win)
        imgui.same_line()
        if _ui_button("Load Layout##ll"): load_layout(tab, ui, win)
        imgui.separator()

        if not show_cfg:
            render_graphs(tab, right_w - 8, ui)
        else:
            render_config_panel(tab, right_w - 8, scroll_h - sc(40), ui)

        imgui.end_child()

        # + New Graph button only visible when showing graphs
        imgui.separator()
        if not show_cfg:
            if _ui_button("+ New Graph##ng", sc(130), sc(24)):
                from dataviz.ui_panels import _new_graph_modal
                _new_graph_modal["open"]        = True
                _new_graph_modal["_needs_open"] = True
                _new_graph_modal["tab_id"]      = tab.id
                _new_graph_modal["search"]      = ""
                _new_graph_modal["checked"]     = {pk: False for pk in tab.PARAM_KEYS}
                _new_graph_modal["title"]       = "New Graph"
                _new_graph_modal["y_min_str"]   = "-30"
                _new_graph_modal["y_max_str"]   = "30"

        # Splitter inside right panel
        _ui_push_style_color(imgui.Col_.button,         0.22, 0.22, 0.26, 0.7)
        _ui_push_style_color(imgui.Col_.button_hovered, 0.45, 0.65, 1.0,  0.8)
        _ui_push_style_color(imgui.Col_.button_active,  0.45, 0.65, 1.0,  1.0)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 0.0))
        _ui_button(f"##hsplit_r{tab.id}", right_w - 8, SPLITTER_W)
        if imgui.is_item_active():
            _, dy = _ui_get_mouse_delta()
            new_gh = graph_h + dy
            tab.graph_h_frac = max(0.3, min(0.9, new_gh / max(top_h, 1)))
        if imgui.is_item_hovered() or imgui.is_item_active():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
        imgui.pop_style_var(); imgui.pop_style_color(3)

        _ui_begin_child(f"##settings{tab.id}", 0, settings_h, False)
        render_settings(tab, ui)
        imgui.end_child()

        imgui.end_child()  # right panel

        # Bottom splitter (top ↔ timeline)
        _ui_push_style_color(imgui.Col_.button,         0.22, 0.22, 0.26, 0.7)
        _ui_push_style_color(imgui.Col_.button_hovered, 0.45, 0.65, 1.0,  0.8)
        _ui_push_style_color(imgui.Col_.button_active,  0.45, 0.65, 1.0,  1.0)
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 0.0))
        _ui_button(f"##hsplit{tab.id}", total_w, SPLITTER_W)
        if imgui.is_item_active():
            _, dy = _ui_get_mouse_delta()
            tab.bot_frac = max(0.1, min(0.5, tab.bot_frac - dy / max(total_h, 1)))
        if imgui.is_item_hovered() or imgui.is_item_active():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
        imgui.pop_style_var(); imgui.pop_style_color(3)

        # Timeline
        _ui_begin_child(f"##timeline{tab.id}", total_w, bot_h, False)
        render_timeline(tab, total_w, bot_h, ui)
        imgui.end_child()

    # ── Post-load GL + graph init ─────────────────────────────────────────
    def _finish_tab_load(tab: TabState) -> None:
        """Called after do_load succeeds — initialise GL meshes and graphs."""
        try:
            init_tab_gl(tab, gl, str(Path(tab.urdf_path).parent))
            tab.init_graphs()
            logger.info(f"[viewer] Tab '{tab.label}' fully initialised")
        except Exception as exc:
            logger.error(f"[viewer] Post-load init failed for '{tab.label}': {exc}")
            tab.loaded = False
            tab.picker["error"] = str(exc)
            tab.picker["open"]  = True

    # ── Main draw frame ───────────────────────────────────────────────────
    def draw_frame() -> None:
        nonlocal _ignore_plus_frames, _active_tab, _switch_to_tab

        while _fbo_delete_queue:
            _delete_fbo_safe(_fbo_delete_queue.pop(0))

        try:
            _fbw, _fbh = win.get_framebuffer_size()
        except Exception:
            _fbw, _fbh = win.width, win.height
        _set_display_size(_fbw, _fbh)

        ui.set_font_scale()
        imgui.new_frame()

        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((win.width, win.height))
        imgui.push_style_var(imgui.StyleVar_.window_padding,    (0.0, 0.0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)

        bg = CFG.colors.background if CFG else [0.10, 0.10, 0.13, 1.0]
        _ui_push_style_color(imgui.Col_.window_bg, *bg)
        _ui_begin("##main", WIN_FLAGS)
        imgui.pop_style_color()
        imgui.pop_style_var(2)

        tabs_to_remove: list[TabState] = []

        if imgui.begin_tab_bar("##tabs"):
            for ti, t in enumerate(_tabs):
                flags = imgui.TabItemFlags_.none
                if ti == _switch_to_tab:
                    flags = _UI_TAB_SET_SELECTED

                opened, keep_open = _ui_begin_tab_item_closable(f"{t.label}##tab{t.id}", flags)

                if not keep_open:
                    tabs_to_remove.append(t)

                if imgui.is_item_activated():
                    _active_tab = ti

                if opened:
                    draw_tab_content(t)
                    imgui.end_tab_item()

            _switch_to_tab = -1

            # '+' new-tab button
            if _ignore_plus_frames > 0:
                _ignore_plus_frames -= 1
            else:
                opened = _ui_begin_tab_item("+##newtab")
                if opened:
                    imgui.end_tab_item()
                    new_tab = TabState("New Tab")
                    new_tab.picker["robots"] = scan_robots(find_robots_dir())
                    _tabs.append(new_tab)
                    _switch_to_tab       = len(_tabs) - 1
                    _active_tab          = _switch_to_tab
                    _ignore_plus_frames  = 2

            imgui.end_tab_bar()

        # New-graph picker window — rendered outside the tab bar so it floats above everything
        if _tabs and 0 <= _active_tab < len(_tabs):
            draw_new_graph_modal(_tabs[_active_tab], ui)

        imgui.end()
        imgui.render()

        # Process closures
        for t in tabs_to_remove:
            _fbo_delete_queue.append(t)
            if t in _tabs:
                _tabs.remove(t)

        # Ensure at least one tab
        if not _tabs:
            new_tab = TabState("New Tab")
            new_tab.picker["robots"] = scan_robots(find_robots_dir())
            _tabs.append(new_tab)
            _switch_to_tab = 0
            _active_tab    = 0

        if _active_tab >= len(_tabs):
            _active_tab    = max(0, len(_tabs) - 1)
            _switch_to_tab = _active_tab

        try:
            _renderer.render(imgui.get_draw_data())
        except Exception as exc:
            logger.warning(f"[viewer] imgui render error: {exc}")

    def _delete_fbo_safe(t: TabState) -> None:
        try:
            delete_fbo(t)
        except Exception as exc:
            logger.warning(f"[viewer] FBO delete failed for '{t.label}': {exc}")

    # ── Pyglet events ─────────────────────────────────────────────────────
    if _IMGUI_BACKEND != "bundle":
        _mb  = [False, False, False]

        @win.event
        def on_mouse_motion(x, y, dx, dy):
            try:
                _io.mouse_pos = (x, win.height - y)
            except AttributeError as exc:
                logger.warning(f"[viewer] on_mouse_motion: could not set mouse_pos: {exc}")

        @win.event
        def on_mouse_press(x, y, btn, mods):
            from pyglet.window import mouse as pm
            if btn == pm.LEFT:     _mb[0] = True
            elif btn == pm.RIGHT:  _mb[2] = True
            elif btn == pm.MIDDLE: _mb[1] = True
            try:
                _io.mouse_pos     = (x, win.height - y)
                _io.mouse_down[0] = _mb[0]
                _io.mouse_down[1] = _mb[1]
                _io.mouse_down[2] = _mb[2]
            except AttributeError as exc:
                logger.warning(f"[viewer] on_mouse_press: imgui IO error: {exc}")

        @win.event
        def on_mouse_release(x, y, btn, mods):
            from pyglet.window import mouse as pm
            if btn == pm.LEFT:     _mb[0] = False
            elif btn == pm.RIGHT:  _mb[2] = False
            elif btn == pm.MIDDLE: _mb[1] = False
            try:
                _io.mouse_down[0] = _mb[0]
                _io.mouse_down[1] = _mb[1]
                _io.mouse_down[2] = _mb[2]
            except AttributeError as exc:
                logger.warning(f"[viewer] on_mouse_release: imgui IO error: {exc}")

        @win.event
        def on_mouse_drag(x, y, dx, dy, btn, mods):
            try:
                _io.mouse_pos = (x, win.height - y)
            except AttributeError as exc:
                logger.warning(f"[viewer] on_mouse_drag: could not set mouse_pos: {exc}")

        @win.event
        def on_mouse_scroll(x, y, sx, sy):
            try:
                _io.mouse_wheel = float(sy)
            except AttributeError as exc:
                logger.warning(f"[viewer] on_mouse_scroll: imgui IO error: {exc}")

        @win.event
        def on_text(text):
            try:
                for ch in text:
                    _io.add_input_character(ord(ch))
            except AttributeError as exc:
                logger.warning(f"[viewer] on_text: imgui IO error: {exc}")

    @win.event
    def on_key_press(sym, mods):
        if sym == pyglet.window.key.ESCAPE:
            logger.info("[viewer] ESC pressed — closing")
            win.close()


    @win.event
    def on_resize(width, height):
        try:
            fbw, fbh = win.get_framebuffer_size()
        except Exception:
            fbw, fbh = width, height
        _set_display_size(fbw, fbh)

    @win.event
    def on_draw():
        glClearColor(*([0.10, 0.10, 0.13, 1.0]))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_frame()

    # ── Scheduled update ──────────────────────────────────────────────────
    def _update(dt: float) -> None:
        for t in _tabs:
            t.S.advance()
            if t.S.data is not None and len(t.S.data["time"]) > 1:
                dt_sim = float(t.S.data["time"][1] - t.S.data["time"][0])
                t.G.advance_graph(t.S.n_frames, t.S.play_speed, dt_sim)
            # In realtime mode always follow S.frame; otherwise only in main mode
            if getattr(t.S, "realtime_mode", False) and t.S.playing:
                t.G.frame = t.S.frame
            elif t.G.play_mode == "main" and not t.G.frozen:
                t.G.frame = t.S.frame
            t.S.update_cam()

    hz = CFG.playback.update_hz if CFG else 120
    pyglet.clock.schedule_interval(_update, 1.0 / hz)

    # ── Initial tab ───────────────────────────────────────────────────────
    if initial_urdf and initial_data:
        first_tab = TabState("loading...")
        _tabs.append(first_tab)
        try:
            do_load(first_tab, initial_urdf, initial_data)
            _finish_tab_load(first_tab)
        except Exception as exc:
            logger.error(f"[viewer] Failed to load initial tab: {exc}")
            first_tab.picker["open"]  = True
            first_tab.picker["error"] = str(exc)
            first_tab.loaded = False
    else:
        first_tab = TabState("New Tab")
        first_tab.picker["robots"] = scan_robots(find_robots_dir())
        _tabs.append(first_tab)

    _active_tab    = 0
    _switch_to_tab = 0

    logger.info("[viewer] Starting — ESC to quit")
    pyglet.app.run()

    # ── Cleanup ───────────────────────────────────────────────────────────
    for t in _tabs:
        _delete_fbo_safe(t)
    try:
        _renderer.shutdown()
    except Exception:
        pass
    logger.info("[viewer] Shutdown complete")
