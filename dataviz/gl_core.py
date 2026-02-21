"""
dataviz/gl_core.py

Owns:
  - GLResources dataclass (single container for all shared GL handles)
  - Shader compilation and linking
  - VAO builders (mesh, lines)
  - Primitive mesh generators (sphere, checker ground)
  - FBO creation / deletion / resize
  - All draw_* functions
  - render_3d(tab, gl, w, h)   ← main per-frame 3D render entry point
  - init_tab_gl(tab, gl)       ← load STL meshes into a TabState

Imports: dataviz.config, dataviz.state, dataviz.data, standard library, numpy, pyglet.gl, loguru.
No imgui.
"""

import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

try:
    from pyglet.gl import (
        glCreateShader, glShaderSource, glCompileShader,
        glGetShaderiv, glGetShaderInfoLog,
        glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
        glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
        glGenFramebuffers, glBindFramebuffer,
        glFramebufferTexture2D, glFramebufferRenderbuffer,
        glCheckFramebufferStatus,
        glDeleteFramebuffers, glDeleteTextures, glDeleteRenderbuffers,
        glUseProgram, glGetUniformLocation,
        glUniform1f, glUniform3f, glUniform4f,
        glUniformMatrix4fv, glUniformMatrix3fv,
        glDrawArrays, glDrawElements,
        glViewport, glClearColor, glClear,
        glEnable, glDisable, glDepthFunc,
        glBlendFunc,
        GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS, GL_LINK_STATUS, GL_INFO_LOG_LENGTH,
        GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
        GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
        GL_FLOAT, GL_UNSIGNED_INT, GL_UNSIGNED_BYTE, GL_TRUE,
        GL_TRIANGLES, GL_LINES,
        GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL,
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_TEXTURE_2D, GL_RGB, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
        GL_LINEAR, GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_COMPLETE,
    )
except ImportError as _e:
    raise ImportError(
        f"[gl_core] pyglet.gl not available: {_e}\n"
        "Install pyglet: pip install pyglet"
    ) from _e

import dataviz.config as _cfg_mod
from dataviz.data   import euler_to_rot, is_near_limit


# ---------------------------------------------------------------------------
# GLSL Shaders
# ---------------------------------------------------------------------------
_VERT_LIT = b"""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP; uniform mat4 uModel; uniform mat3 uNormalMat;
out vec3 vNormal; out vec3 vFragPos;
void main(){
    vFragPos = (uModel*vec4(aPos,1)).xyz;
    vNormal  = normalize(uNormalMat*aNormal);
    gl_Position = uMVP*vec4(aPos,1);
}"""

_FRAG_LIT = b"""
#version 330 core
in vec3 vNormal; in vec3 vFragPos;
uniform vec3 uColor; uniform vec3 uLightPos; uniform vec3 uCamPos;
uniform float uAlpha; uniform float uFogStart; uniform float uFogEnd;
out vec4 FragColor;
void main(){
    vec3 n=normalize(vNormal); vec3 l=normalize(uLightPos-vFragPos);
    vec3 v=normalize(uCamPos-vFragPos); vec3 h=normalize(l+v);
    float amb=0.32; float dif=max(dot(n,l),0.0)*0.68;
    float spc=pow(max(dot(n,h),0.0),48.0)*0.30;
    vec3 col=uColor*(amb+dif)+vec3(spc);
    float d=length(vFragPos-uCamPos);
    float fog=clamp((d-uFogStart)/(uFogEnd-uFogStart),0.0,1.0);
    col=mix(col,vec3(0.52,0.60,0.68),fog);
    FragColor=vec4(col,uAlpha);
}"""

_VERT_FLAT = b"""
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP; uniform vec4 uColor;
out vec4 vColor;
void main(){ vColor=uColor; gl_Position=uMVP*vec4(aPos,1); }"""

_FRAG_FLAT = b"""
#version 330 core
in vec4 vColor; out vec4 FragColor;
void main(){ FragColor=vColor; }"""


# ---------------------------------------------------------------------------
# GLResources — single container for shared GL handles
# ---------------------------------------------------------------------------
@dataclass
class GLResources:
    """
    Created once by build_gl_resources() and passed to every render/draw call.
    Tab-specific resources (mesh VAOs, FBOs) live in TabState, not here.
    """
    prog_lit:          int = 0
    prog_flat:         int = 0
    vao_sphere:        Tuple = field(default_factory=tuple)   # (vao, n_idx)
    vao_ground_dark:   Tuple = field(default_factory=tuple)
    vao_ground_light:  Tuple = field(default_factory=tuple)
    vao_lines:         int   = 0    # VAO handle
    vbo_lines:         int   = 0    # VBO handle (for dynamic upload)
    max_line_pts:      int   = 4096
    vao_arrow_shaft:   Tuple = field(default_factory=tuple)   # (vao, n_idx) unit cylinder
    vao_arrow_cone:    Tuple = field(default_factory=tuple)   # (vao, n_idx) unit cone


# ---------------------------------------------------------------------------
# Shader compilation
# ---------------------------------------------------------------------------
def _compile_shader(src: bytes, stype: int) -> int:
    sid = glCreateShader(stype)
    buf = ctypes.create_string_buffer(src)
    ptr = ctypes.cast(
        ctypes.pointer(ctypes.pointer(buf)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
    )
    glShaderSource(sid, 1, ptr, None)
    glCompileShader(sid)
    ok = ctypes.c_int(0)
    glGetShaderiv(sid, GL_COMPILE_STATUS, ctypes.byref(ok))
    if not ok.value:
        n2 = ctypes.c_int(0)
        glGetShaderiv(sid, GL_INFO_LOG_LENGTH, ctypes.byref(n2))
        log = (ctypes.c_char * n2.value)()
        glGetShaderInfoLog(sid, n2.value, None, log)
        raise RuntimeError(f"[gl_core] Shader compile error:\n{log.value.decode()}")
    return sid


def _link_program(vsrc: bytes, fsrc: bytes) -> int:
    vs = _compile_shader(vsrc, GL_VERTEX_SHADER)
    fs = _compile_shader(fsrc, GL_FRAGMENT_SHADER)
    p  = glCreateProgram()
    glAttachShader(p, vs)
    glAttachShader(p, fs)
    glLinkProgram(p)
    ok = ctypes.c_int(0)
    glGetProgramiv(p, GL_LINK_STATUS, ctypes.byref(ok))
    if not ok.value:
        n2 = ctypes.c_int(0)
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, ctypes.byref(n2))
        log = (ctypes.c_char * n2.value)()
        glGetProgramInfoLog(p, n2.value, None, log)
        raise RuntimeError(f"[gl_core] Shader link error:\n{log.value.decode()}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return p


# ---------------------------------------------------------------------------
# Uniform setters
# ---------------------------------------------------------------------------
def _u4(prog: int, name: str, M: np.ndarray) -> None:
    loc  = glGetUniformLocation(prog, name.encode())
    flat = np.ascontiguousarray(M, dtype=np.float32).flatten()
    glUniformMatrix4fv(loc, 1, GL_TRUE, (ctypes.c_float * 16)(*flat))


def _u3f(prog: int, name: str, M: np.ndarray) -> None:
    loc  = glGetUniformLocation(prog, name.encode())
    flat = np.ascontiguousarray(M, dtype=np.float32).flatten()
    glUniformMatrix3fv(loc, 1, GL_TRUE, (ctypes.c_float * 9)(*flat))


def _uf3(prog: int, n: str, x: float, y: float, z: float) -> None:
    glUniform3f(glGetUniformLocation(prog, n.encode()), x, y, z)


def _uf4(prog: int, n: str, x: float, y: float, z: float, w: float) -> None:
    glUniform4f(glGetUniformLocation(prog, n.encode()), x, y, z, w)


def _uf1(prog: int, n: str, v: float) -> None:
    glUniform1f(glGetUniformLocation(prog, n.encode()), v)


# ---------------------------------------------------------------------------
# VAO builders
# ---------------------------------------------------------------------------
def build_vao_mesh(
    verts: np.ndarray,
    norms: np.ndarray,
    faces: np.ndarray,
) -> Tuple[ctypes.c_uint, int]:
    """
    Upload interleaved (position, normal) mesh to GPU.
    Returns (vao_handle, n_indices).
    """
    data2 = np.hstack([verts, norms]).astype(np.float32).flatten()
    idx   = faces.flatten().astype(np.uint32)

    vao = ctypes.c_uint(0); glGenVertexArrays(1, ctypes.byref(vao))
    vbo = ctypes.c_uint(0); glGenBuffers(1, ctypes.byref(vbo))
    ebo = ctypes.c_uint(0); glGenBuffers(1, ctypes.byref(ebo))

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER, data2.nbytes,
        data2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        GL_STATIC_DRAW,
    )
    stride = 6 * ctypes.sizeof(ctypes.c_float)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(3 * 4))
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, idx.nbytes,
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        GL_STATIC_DRAW,
    )
    glBindVertexArray(ctypes.c_uint(0))
    return vao, len(idx)


def build_vao_lines(max_pts: int = 4096) -> Tuple[ctypes.c_uint, ctypes.c_uint]:
    """
    Build a dynamic-draw line VAO.
    Returns (vao_handle, vbo_handle).
    """
    buf = np.zeros((max_pts, 3), dtype=np.float32)
    vao = ctypes.c_uint(0); glGenVertexArrays(1, ctypes.byref(vao))
    vbo = ctypes.c_uint(0); glGenBuffers(1, ctypes.byref(vbo))
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER, buf.nbytes,
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        GL_DYNAMIC_DRAW,
    )
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 12, ctypes.c_void_p(0))
    glBindVertexArray(ctypes.c_uint(0))
    return vao, vbo


def upload_lines(vbo: ctypes.c_uint, pts: list | np.ndarray) -> None:
    arr = np.array(pts, dtype=np.float32).flatten()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER, arr.nbytes,
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        GL_DYNAMIC_DRAW,
    )


# ---------------------------------------------------------------------------
# Primitive mesh generators
# ---------------------------------------------------------------------------
def _sphere_mesh(r: float = 1.0, stacks: int = 10, slices: int = 10):
    vv, nn, ff = [], [], []
    for i in range(stacks + 1):
        lat = math.pi * (-0.5 + i / stacks)
        for j in range(slices + 1):
            lon = 2 * math.pi * j / slices
            x   = math.cos(lat) * math.cos(lon)
            y   = math.cos(lat) * math.sin(lon)
            z   = math.sin(lat)
            vv.append([x * r, y * r, z * r])
            nn.append([x, y, z])
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            ff += [[a, a + 1, a + slices + 1], [a + 1, a + slices + 2, a + slices + 1]]
    return np.array(vv, np.float32), np.array(nn, np.float32), np.array(ff, np.uint32)


def _cylinder_mesh(radius: float = 1.0, height: float = 1.0, slices: int = 12):
    """Unit cylinder along +Z axis, centred at origin base."""
    vv, nn, ff = [], [], []
    for i in range(slices):
        a0 = 2 * math.pi * i / slices
        a1 = 2 * math.pi * (i + 1) / slices
        x0, y0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, y1 = math.cos(a1) * radius, math.sin(a1) * radius
        b = len(vv)
        # Side quad: two triangles
        for x, y, z, nx, ny in [(x0,y0,0,x0/radius,y0/radius),(x1,y1,0,x1/radius,y1/radius),
                                  (x1,y1,height,x1/radius,y1/radius),(x0,y0,height,x0/radius,y0/radius)]:
            vv.append([x, y, z]); nn.append([nx, ny, 0.0])
        ff += [[b, b+1, b+2], [b, b+2, b+3]]
    return np.array(vv, np.float32), np.array(nn, np.float32), np.array(ff, np.uint32)


def _cone_mesh(radius: float = 1.0, height: float = 1.0, slices: int = 12):
    """Cone with base at origin, tip at +Z=height."""
    vv, nn, ff = [], [], []
    tip = [0.0, 0.0, height]
    sl  = math.sqrt(radius**2 + height**2)  # slant length for normals
    nz  = radius / sl
    nr  = height / sl
    for i in range(slices):
        a0 = 2 * math.pi * i / slices
        a1 = 2 * math.pi * (i + 1) / slices
        x0, y0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, y1 = math.cos(a1) * radius, math.sin(a1) * radius
        b = len(vv)
        vv += [[x0,y0,0], [x1,y1,0], tip]
        nn += [[math.cos(a0)*nr, math.sin(a0)*nr, nz],
               [math.cos(a1)*nr, math.sin(a1)*nr, nz],
               [0, 0, 1]]
        ff.append([b, b+1, b+2])
    return np.array(vv, np.float32), np.array(nn, np.float32), np.array(ff, np.uint32)


def _ground_mesh_checker(size: float = 26.0, step: float = 0.5):
    def build(parity):
        vv, nn, ff = [], [], []
        rng = range(int(-size / step), int(size / step) + 1)
        for xi in rng:
            for yi in rng:
                if (xi + yi) % 2 != parity:
                    continue
                x0, y0 = xi * step, yi * step
                b = len(vv)
                for x, y in [(x0, y0), (x0 + step, y0), (x0 + step, y0 + step), (x0, y0 + step)]:
                    vv.append([x, y, 0.0])
                    nn.append([0, 0, 1])
                ff += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        return np.array(vv, np.float32), np.array(nn, np.float32), np.array(ff, np.uint32)

    return build(0), build(1)


# ---------------------------------------------------------------------------
# FBO management
# ---------------------------------------------------------------------------
def create_fbo(w: int, h: int) -> Tuple[ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]:
    """
    Create FBO + color texture + depth renderbuffer.
    Returns (fbo, tex, rbo).
    """
    tex = ctypes.c_uint(0); glGenTextures(1, ctypes.byref(tex))
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)

    rbo = ctypes.c_uint(0); glGenRenderbuffers(1, ctypes.byref(rbo))
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    fbo = ctypes.c_uint(0); glGenFramebuffers(1, ctypes.byref(fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        logger.warning(f"[gl_core] FBO incomplete (status={status})")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, tex, rbo


def delete_fbo(tab) -> None:
    """Delete FBO resources stored in *tab* and clear the handles."""
    if tab.fbo is not None:
        glDeleteFramebuffers(1,   ctypes.byref(tab.fbo))
        glDeleteTextures(1,       ctypes.byref(tab.fbo_tex))
        glDeleteRenderbuffers(1,  ctypes.byref(tab.fbo_depth))
        tab.fbo = tab.fbo_tex = tab.fbo_depth = None
        tab.fbo_w = tab.fbo_h = 0


def ensure_fbo(tab, w: int, h: int) -> None:
    """Create or resize the tab's FBO if dimensions have changed."""
    w = max(1, int(w))
    h = max(1, int(h))
    if tab.fbo is None or tab.fbo_w != w or tab.fbo_h != h:
        delete_fbo(tab)
        fbo, tex, depth = create_fbo(w, h)
        tab.fbo       = fbo
        tab.fbo_tex   = tex
        tab.fbo_depth = depth
        tab.fbo_w     = w
        tab.fbo_h     = h
        logger.debug(f"[gl_core] FBO resized → {w}×{h} for tab '{tab.label}'")


# ---------------------------------------------------------------------------
# Tab GL resource init (called after do_load)
# ---------------------------------------------------------------------------
def init_tab_gl(tab, gl: GLResources, urdf_dir: str) -> None:
    """
    Load STL meshes for *tab* from the meshes/ directory next to the URDF.
    Populates tab.MESH_VAOS and tab.LINK_TO_MESH.
    """
    import trimesh

    URDF_DIR = Path(urdf_dir)
    MESH_DIR = URDF_DIR.parent / "meshes"

    MESH_KEYS = ["base", "hip", "thigh", "thigh_mirror", "calf", "calf_mirror"]
    for key in MESH_KEYS:
        p = MESH_DIR / f"{key}.stl"
        if p.exists():
            try:
                m = trimesh.load(str(p), force="mesh")
                tab.MESH_VAOS[key] = build_vao_mesh(
                    np.array(m.vertices, dtype=np.float32),
                    np.array(m.vertex_normals, dtype=np.float32),
                    np.array(m.faces, dtype=np.uint32),
                )
                logger.info(f"[gl_core] [{tab.label}] mesh '{key}': {len(m.faces)} tris")
            except Exception as exc:
                logger.warning(f"[gl_core] [{tab.label}] mesh '{key}' failed: {exc}")
        else:
            logger.debug(f"[gl_core] [{tab.label}] mesh '{key}' not found at {p}")

    tab.LINK_TO_MESH = {
        "trunk":          "base",
        "1_FR_hip":       "hip",         "2_FL_hip":   "hip",
        "3_RR_hip":       "hip",         "4_RL_hip":   "hip",
        "1_FR_thigh":     "thigh_mirror", "2_FL_thigh": "thigh",
        "3_RR_thigh":     "thigh_mirror", "4_RL_thigh": "thigh",
        "1_FR_calf":      "calf_mirror",  "2_FL_calf":  "calf",
        "3_RR_calf":      "calf_mirror",  "4_RL_calf":  "calf",
    }

    from dataviz.data import _detect_legs
    tab.legs = _detect_legs(tab.joint_order)
    logger.info(f"[gl_core] [{tab.label}] GL resources initialised")


# ---------------------------------------------------------------------------
# Build shared GLResources (called once at startup)
# ---------------------------------------------------------------------------
def build_gl_resources() -> GLResources:
    """
    Compile shaders, build shared primitive VAOs.
    Must be called after a GL context exists.
    """
    logger.info("[gl_core] Compiling shaders...")
    prog_lit  = _link_program(_VERT_LIT,  _FRAG_LIT)
    prog_flat = _link_program(_VERT_FLAT, _FRAG_FLAT)
    logger.info("[gl_core] Shaders compiled OK")

    # Sphere
    st = _cfg_mod.CFG.rendering.sphere_stacks if _cfg_mod.CFG else 10
    sl = _cfg_mod.CFG.rendering.sphere_slices if _cfg_mod.CFG else 10
    vao_sphere = build_vao_mesh(*_sphere_mesh(1.0, st, sl))

    # Ground
    gs   = _cfg_mod.CFG.rendering.ground_size if _cfg_mod.CFG else 26.0
    gst  = _cfg_mod.CFG.rendering.ground_step if _cfg_mod.CFG else 0.5
    (gv0, gn0, gf0), (gv1, gn1, gf1) = _ground_mesh_checker(gs, gst)
    vao_ground_dark  = build_vao_mesh(gv0, gn0, gf0)
    vao_ground_light = build_vao_mesh(gv1, gn1, gf1)

    # Lines
    vao_lines, vbo_lines = build_vao_lines(4096)

    # Arrow primitives (unit size, oriented along +Z — transformed at draw time)
    vao_arrow_shaft = build_vao_mesh(*_cylinder_mesh(0.012, 1.0, 12))
    vao_arrow_cone  = build_vao_mesh(*_cone_mesh(0.030, 0.18, 12))

    gl = GLResources(
        prog_lit         = prog_lit,
        prog_flat        = prog_flat,
        vao_sphere       = vao_sphere,
        vao_ground_dark  = vao_ground_dark,
        vao_ground_light = vao_ground_light,
        vao_lines        = vao_lines,
        vbo_lines        = vbo_lines,
        max_line_pts     = 4096,
        vao_arrow_shaft  = vao_arrow_shaft,
        vao_arrow_cone   = vao_arrow_cone,
    )
    logger.info("[gl_core] GLResources built")
    return gl


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
def _persp(fov: float, asp: float, near: float, far: float) -> np.ndarray:
    f  = 1.0 / math.tan(math.radians(fov) / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f / asp, 0,  0,                  0             ],
        [0,       f,  0,                  0             ],
        [0,       0,  (far + near) * nf,  2*far*near*nf ],
        [0,       0, -1,                  0             ],
    ], dtype=np.float32)


def _lookat(eye: np.ndarray, tgt: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.array(eye, dtype=np.float64)
    f   = np.array(tgt, dtype=np.float64) - eye
    f  /= np.linalg.norm(f)
    u2  = np.array(up, dtype=np.float64)
    if abs(np.dot(f, u2)) > 0.99:
        u2 = np.array([0., 1., 0.])
    r   = np.cross(f, u2); r  /= np.linalg.norm(r)
    u2  = np.cross(r, f)
    M   = np.eye(4, dtype=np.float32)
    M[0, :3] = r;   M[0, 3] = float(-np.dot(r,  eye))
    M[1, :3] = u2;  M[1, 3] = float(-np.dot(u2, eye))
    M[2, :3] = -f;  M[2, 3] = float( np.dot(f,  eye))
    return M


# ---------------------------------------------------------------------------
# Draw primitives
# ---------------------------------------------------------------------------
def draw_mesh(
    vao: ctypes.c_uint,
    n_idx: int,
    prog: int,
    model: np.ndarray,
    color3: tuple,
    proj: np.ndarray,
    view: np.ndarray,
    eye: np.ndarray,
    alpha: float,
    fog_start: float,
    fog_end: float,
) -> None:
    glUseProgram(prog)
    mvp = proj @ view @ model
    _u4(prog, "uMVP",       mvp)
    _u4(prog, "uModel",     model)
    nm  = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
    _u3f(prog, "uNormalMat", nm)
    _uf3(prog, "uColor",    *color3)
    _uf3(prog, "uLightPos", *(eye + np.array([0, 0, 3.0])))
    _uf3(prog, "uCamPos",   *eye)
    _uf1(prog, "uAlpha",    float(alpha))
    _uf1(prog, "uFogStart", fog_start)
    _uf1(prog, "uFogEnd",   fog_end)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, n_idx, GL_UNSIGNED_INT, ctypes.c_void_p(0))
    glBindVertexArray(ctypes.c_uint(0))


def draw_sphere(
    gl: GLResources,
    pos: np.ndarray,
    r: float,
    col3: tuple,
    proj: np.ndarray,
    view: np.ndarray,
    eye: np.ndarray,
    fog_start: float,
    fog_end: float,
) -> None:
    s = np.diag([r, r, r, 1.0]).astype(np.float32)
    T = np.eye(4, dtype=np.float32); T[:3, 3] = pos
    vao, n_idx = gl.vao_sphere
    draw_mesh(vao, n_idx, gl.prog_lit, T @ s, col3, proj, view, eye, 1.0, fog_start, fog_end)


def draw_line_seg(
    gl: GLResources,
    p1: np.ndarray | list,
    p2: np.ndarray | list,
    col4: tuple,
    mvp: np.ndarray,
) -> None:
    upload_lines(gl.vbo_lines, [p1, p2])
    glUseProgram(gl.prog_flat)
    _u4(gl.prog_flat,  "uMVP",   mvp)
    _uf4(gl.prog_flat, "uColor", *col4)
    glBindVertexArray(gl.vao_lines)
    glDrawArrays(GL_LINES, 0, 2)
    glBindVertexArray(ctypes.c_uint(0))


def draw_force_arrow(
    gl: GLResources,
    origin: np.ndarray,
    force_vec: np.ndarray,
    col3: tuple,
    proj: np.ndarray,
    view: np.ndarray,
    eye: np.ndarray,
    fog_start: float,
    fog_end: float,
    force_scale: float,
) -> None:
    """
    Draw a 3D force arrow: cylinder shaft + cone tip, oriented along force_vec.
    Arrow length is proportional to force magnitude * force_scale.
    Total arrow = shaft (80% of length) + cone tip (20%, fixed proportion).
    """
    fv  = np.array(force_vec, dtype=np.float64)
    mag = float(np.linalg.norm(fv))
    if mag < 1e-6:
        return

    direction = fv / mag
    arrow_len = mag * force_scale

    # Fixed cone height, shaft fills the rest
    cone_h  = min(0.18, arrow_len * 0.30)
    shaft_l = max(0.0, arrow_len - cone_h)

    # Build rotation matrix: +Z → direction
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, z)) > 0.9999:
        # Near-parallel: use a different up vector
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = z

    x_axis = np.cross(direction, up);  x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction, x_axis)

    # 4x4 rotation+translation matrix
    def _make_T(pos, scale_z, rot_axes):
        R = np.eye(4, dtype=np.float32)
        R[0, :3] = rot_axes[0]
        R[1, :3] = rot_axes[1]
        R[2, :3] = rot_axes[2]
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        S = np.eye(4, dtype=np.float32)
        S[2, 2] = scale_z
        return T @ R.T @ S

    rot = (x_axis, y_axis, direction)  # columns of rotation

    # Draw shaft
    if shaft_l > 1e-4:
        model_shaft = _make_T(origin, shaft_l, rot)
        vao, n_idx  = gl.vao_arrow_shaft
        draw_mesh(vao, n_idx, gl.prog_lit, model_shaft, col3, proj, view, eye, 1.0, fog_start, fog_end)

    # Draw cone at tip of shaft
    cone_origin = origin + direction * shaft_l
    model_cone  = _make_T(cone_origin, cone_h / 0.18, rot)  # 0.18 is unit cone height
    vao, n_idx  = gl.vao_arrow_cone
    draw_mesh(vao, n_idx, gl.prog_lit, model_cone, col3, proj, view, eye, 1.0, fog_start, fog_end)


def draw_ground(
    gl: GLResources,
    proj: np.ndarray,
    view: np.ndarray,
    eye: np.ndarray,
    cam_target: np.ndarray,
    fog_start: float,
    fog_end: float,
) -> None:
    step = _cfg_mod.CFG.rendering.ground_step if _cfg_mod.CFG else 0.5
    snap = 2 * step
    rx, ry = cam_target[0], cam_target[1]
    ox = math.floor(rx / snap) * snap
    oy = math.floor(ry / snap) * snap
    T  = np.eye(4, dtype=np.float32)
    T[0, 3] = ox
    T[1, 3] = oy

    # Ground colors — stored as RGBA in config, we use only RGB for the lit shader
    _dark_fallback  = (0.32, 0.38, 0.44)
    _light_fallback = (0.50, 0.57, 0.64)
    if _cfg_mod.CFG is not None:
        raw_dark  = getattr(_cfg_mod.CFG.colors, "ground_dark",  None)
        raw_light = getattr(_cfg_mod.CFG.colors, "ground_light", None)
        if raw_dark is None:
            logger.warning("[gl_core] config missing colors.ground_dark — using default")
            dark = _dark_fallback
        else:
            dark = tuple(raw_dark[:3])
        if raw_light is None:
            logger.warning("[gl_core] config missing colors.ground_light — using default")
            light = _light_fallback
        else:
            light = tuple(raw_light[:3])
    else:
        dark  = _dark_fallback
        light = _light_fallback

    vd, nd = gl.vao_ground_dark
    vl, nl = gl.vao_ground_light
    draw_mesh(vd, nd, gl.prog_lit, T, dark,  proj, view, eye, 1.0, fog_start, fog_end)
    draw_mesh(vl, nl, gl.prog_lit, T, light, proj, view, eye, 1.0, fog_start, fog_end)


# ---------------------------------------------------------------------------
# Leg colour helper
# ---------------------------------------------------------------------------
def get_leg_color(label: str) -> tuple:
    if _cfg_mod.CFG is not None:
        key = f"leg_{label}"
        c   = getattr(_cfg_mod.CFG.colors, key, None)
        if c is not None:
            return tuple(c)
    fallback = {
        "FR": (0.30, 0.90, 0.30, 1.0),
        "FL": (0.30, 0.55, 1.00, 1.0),
        "RR": (1.00, 0.63, 0.16, 1.0),
        "RL": (0.90, 0.24, 0.90, 1.0),
    }
    return fallback.get(label, (0.7, 0.7, 0.7, 1.0))


# ---------------------------------------------------------------------------
# Main 3D render (called with FBO bound)
# ---------------------------------------------------------------------------
def render_3d(tab, gl: GLResources, vp_w: int, vp_h: int) -> None:
    """
    Render the full 3D scene for *tab* into the currently bound FBO.
    *vp_w* / *vp_h* are the FBO dimensions.
    """
    from dataviz.state import TabState  # local import to avoid circular at top

    S = tab.S
    G = tab.G

    if not tab.loaded:
        return

    glViewport(0, 0, vp_w, vp_h)

    bg = _cfg_mod.CFG.colors.background if _cfg_mod.CFG else [0.52, 0.60, 0.68, 1.0]
    glClearColor(*bg)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    eye  = S.cam_eye()
    proj = _persp(60.0, vp_w / max(1, vp_h), 0.01, 500.0)
    view = _lookat(eye, S.cam_target, [0, 0, 1])
    mvp  = proj @ view

    fog_s = S.fog_start
    fog_e = S.fog_end

    # Ground
    if S.show_grid:
        draw_ground(gl, proj, view, eye, S.cam_target, fog_s, fog_e)

    # World axes
    L = 0.4
    for p2, c in [
        ([L, 0, 0], (1, 0, 0, 1)),
        ([0, L, 0], (0, 1, 0, 1)),
        ([0, 0, L], (0, 0, 1, 1)),
    ]:
        draw_line_seg(gl, [0, 0, 0], p2, c, mvp)

    if S.data is None:
        return

    q, torso_pos, torso_rpy = S.frame_state(S.frame)
    link_T = tab.fk_fn(q, torso_pos, torso_rpy)

    # ── Trajectory ────────────────────────────────────────────────────────
    if S.show_trajectory:
        tl    = S.traj_length
        start = max(0, S.frame - tl)
        for i in range(start, S.frame - 1):
            t = (i - start) / max(S.frame - start, 1)
            draw_line_seg(
                gl,
                [S.data["torso_x"][i],     S.data["torso_y"][i],     S.data["torso_z"][i]],
                [S.data["torso_x"][i + 1], S.data["torso_y"][i + 1], S.data["torso_z"][i + 1]],
                (0.4 + 0.5 * t, 0.4 + 0.5 * t, 0.1, 0.4 + 0.5 * t),
                mvp,
            )

    # ── Main robot ────────────────────────────────────────────────────────
    robot_col = tuple(_cfg_mod.CFG.colors.robot_main[:3]) if _cfg_mod.CFG else (0.92, 0.92, 0.94)
    for link_name, mesh_key in tab.LINK_TO_MESH.items():
        if link_name not in link_T or mesh_key not in tab.MESH_VAOS:
            continue
        model        = link_T[link_name]
        vao, n_idx   = tab.MESH_VAOS[mesh_key]
        draw_mesh(vao, n_idx, gl.prog_lit, model, robot_col, proj, view, eye, 1.0, fog_s, fog_e)

    # ── Ghost robot ───────────────────────────────────────────────────────
    ghost_alpha = (
        _cfg_mod.CFG.rendering.ghost_alpha_graph_mode if (_cfg_mod.CFG and G.play_mode == "graph")
        else (_cfg_mod.CFG.rendering.ghost_alpha_main_mode if _cfg_mod.CFG else 0.25)
    )
    if ghost_alpha > 0.01 and G.frame != S.frame:
        gq, gpos, grpy = S.frame_state(G.frame)
        ghost_T        = tab.fk_fn(gq, gpos, grpy)
        ghost_col      = tuple(_cfg_mod.CFG.colors.robot_ghost[:3]) if _cfg_mod.CFG else (0.75, 0.82, 1.0)
        for link_name, mesh_key in tab.LINK_TO_MESH.items():
            if link_name not in ghost_T or mesh_key not in tab.MESH_VAOS:
                continue
            vao, n_idx = tab.MESH_VAOS[mesh_key]
            draw_mesh(
                vao, n_idx, gl.prog_lit, ghost_T[link_name],
                ghost_col, proj, view, eye, ghost_alpha, fog_s, fog_e,
            )

    # ── Foot contacts + forces + limit warnings ───────────────────────────
    leg_keys    = list(tab.legs.keys())
    has_foot_pos = (
        S.data.get("foot_pos") is not None
        and np.any(S.data["foot_pos"] != 0)
    )
    force_min = _cfg_mod.CFG.rendering.force_min_magnitude if _cfg_mod.CFG else 1.0

    for lbl, ji in tab.legs.items():
        fi2 = leg_keys.index(lbl) if lbl in leg_keys else 0

        # Foot position
        if has_foot_pos:
            foot_pos = S.data["foot_pos"][S.frame, fi2 * 3:(fi2 + 1) * 3].astype(np.float32)
        else:
            foot_link = None
            for ln in link_T:
                low = ln.lower()
                if lbl.lower() in low and "foot" in low:
                    foot_link = ln; break
            foot_pos = link_T[foot_link][:3, 3] if (foot_link and foot_link in link_T) else None

        if foot_pos is None:
            continue

        # Contact sphere
        cv    = S.data.get(f"contact_{lbl}")
        in_c  = cv is not None and cv[S.frame] > 0.5
        fc    = (0.1, 1.0, 0.4) if (S.show_contacts and in_c) else (0.15, 0.15, 0.15)
        if S.show_contacts:
            draw_sphere(gl, foot_pos, 0.025, fc, proj, view, eye, fog_s, fog_e)

        # Force arrow
        if S.show_forces and S.data.get("foot_forces") is not None:
            fv  = S.data["foot_forces"][S.frame, fi2 * 3:(fi2 + 1) * 3]
            mag = np.linalg.norm(fv)
            if mag > force_min:
                leg_col = get_leg_color(lbl)
                draw_force_arrow(
                    gl, foot_pos, fv, leg_col[:3],
                    proj, view, eye, fog_s, fog_e, S.force_scale,
                )

        # Limit warnings
        if S.show_limits:
            for i in ji:
                jname = tab.joint_order[i]
                lo, hi = tab.joint_limits[jname]
                if is_near_limit(lo, hi, float(q[i])):
                    for ln in link_T:
                        if jname.replace("_joint", "") in ln:
                            pos = link_T[ln][:3, 3]
                            draw_sphere(gl, pos, 0.028, (1, 0.1, 0.1), proj, view, eye, fog_s, fog_e)
                            break

    # ── Joint frames ──────────────────────────────────────────────────────
    if S.show_joint_frames:
        Lf = max(0.04, S.cam_dist * 0.025)
        for jname in tab.joint_order:
            child = jname.replace("_joint", "")
            T = None
            for ln in link_T:
                if child in ln and "visual" not in ln.lower():
                    T = link_T[ln]; break
            if T is None:
                for ln in link_T:
                    if child in ln:
                        T = link_T[ln]; break
            if T is None:
                continue
            origin = T[:3, 3]; Rw = T[:3, :3]
            draw_line_seg(gl, origin, origin + Rw[:, 0] * Lf, (1, 0, 0, 0.9), mvp)
            draw_line_seg(gl, origin, origin + Rw[:, 1] * Lf, (0, 1, 0, 0.9), mvp)
            draw_line_seg(gl, origin, origin + Rw[:, 2] * Lf, (0, 0, 1, 0.9), mvp)
            draw_sphere(gl, origin, Lf * 0.12, (0.9, 0.9, 0.2), proj, view, eye, fog_s, fog_e)
