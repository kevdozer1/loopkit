"""rack_cell: the Menagerie Panda plus generated MJCF for the rack, rails, patch panel, ports, LEDs, label strip and tray.

``make_scene(seed)`` samples the per-seed randomization (``sample_config``), writes the
MJCF and compiles it. The same seed gives the same XML string and the same compiled
model, bit for bit; a test checks this.

Randomization per seed (docs/02_task_suite.md): rack pose +-2 cm in x and y and +-3 deg
yaw; 2 to 4 lights with intensity and direction jitter; one LED state per port; 0 to 6
ports already holding a plug; a target port (always free) and a target slot; tray part
poses +-5 mm and +-10 deg yaw (each part rests on its own small holder, which moves with it).
"""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..paths import panda_dir
from . import geometry as G

SCENE_ID = "rack_cell_v1"
TIMESTEP = 0.001
GRIPPER_KP = 4000.0          # N per m of tendon length (each finger sees a quarter of it as joint stiffness)
GRIPPER_KV = 1200.0         # heavy damping: about 0.1 m/s finger speed at 30 N, like the real hand
GRIPPER_FORCE_DEFAULT = 70.0  # N per finger, the Franka Hand's continuous grasp force
ROBOT_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
                "left_finger", "right_finger")
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class LightSpec:
    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    diffuse: float
    castshadow: bool


@dataclass(frozen=True)
class SceneConfig:
    seed: int
    rack_pose: tuple[float, float, float]          # world x, y, yaw (rad) of the rack frame
    led_states: tuple[str, ...]                    # per port 1..12
    occupied: tuple[int, ...]                      # ports that already hold a plug
    target_port: int                               # T3, always free
    target_slot: int                               # T2
    box_pose: tuple[float, float, float]           # world x, y, yaw of the box (and its riser)
    plug_pose: tuple[float, float, float]          # world x, y, yaw of the plug body (and its holder)
    lights: tuple[LightSpec, ...] = field(default=())

    def to_dict(self) -> dict:
        return asdict(self)


def sample_config(seed: int) -> SceneConfig:
    """The per-seed randomization. A pure function of ``seed``."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4C4B]))  # "LK"
    rx, ry, ryaw = G.RACK_NOMINAL
    rack = (
        rx + rng.uniform(-G.RACK_JITTER_XY, G.RACK_JITTER_XY),
        ry + rng.uniform(-G.RACK_JITTER_XY, G.RACK_JITTER_XY),
        ryaw + rng.uniform(-G.RACK_JITTER_YAW, G.RACK_JITTER_YAW),
    )
    leds = tuple(str(rng.choice(G.LED_STATES)) for _ in range(G.N_PORTS))
    n_occ = int(rng.integers(0, 7))
    occupied = tuple(sorted(int(p) for p in rng.choice(np.arange(1, G.N_PORTS + 1), size=n_occ, replace=False)))
    free = [p for p in range(1, G.N_PORTS + 1) if p not in occupied]
    target_port = int(rng.choice(free))
    target_slot = int(rng.integers(1, G.N_SLOTS + 1))

    def part(nominal):
        x, y, yaw = nominal
        return (
            x + rng.uniform(-G.PART_JITTER_XY, G.PART_JITTER_XY),
            y + rng.uniform(-G.PART_JITTER_XY, G.PART_JITTER_XY),
            yaw + rng.uniform(-G.PART_JITTER_YAW, G.PART_JITTER_YAW),
        )

    box = part(G.BOX_NOMINAL)
    plug = part(G.PLUG_NOMINAL)
    center = np.array([0.52, -0.08, 0.25])
    lights = []
    for i in range(int(rng.integers(2, 5))):
        az = rng.uniform(-math.pi, math.pi)
        el = rng.uniform(math.radians(40), math.radians(80))
        dist = rng.uniform(1.6, 2.6)
        pos = center + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        d = center + rng.uniform(-0.15, 0.15, size=3) - pos
        d /= np.linalg.norm(d)
        # No shadows: the shadow map made a few pixels differ by one level between identical renders on
        # this GPU, which broke bit-for-bit determinism and the router's image-hash cache.
        lights.append(LightSpec(tuple(round(float(v), 6) for v in pos), tuple(round(float(v), 6) for v in d),
                                round(float(rng.uniform(0.25, 0.6)), 4), False))
    r6 = lambda t: tuple(round(float(v), 9) for v in t)  # noqa: E731
    return SceneConfig(int(seed), r6(rack), leds, occupied, target_port, target_slot, r6(box), r6(plug), tuple(lights))


# ----------------------------------------------------------------------------- textures
@lru_cache(maxsize=1)
def label_texture_png() -> bytes:
    """Port numbers 1..12, black on white, one cell per port pitch. 10 px per mm."""
    ppm = 10
    w = int(round(G.N_PORTS * G.PORT_PITCH * 1000 * ppm))
    h = int(round(G.LABEL_H * 1000 * ppm))
    img = Image.new("RGB", (w, h), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=int(h * 0.82))
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    cell = w / G.N_PORTS
    for i in range(G.N_PORTS):
        text = str(i + 1)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = cell * i + (cell - tw) / 2 - box[0]
        y = (h - th) / 2 - box[1]
        draw.text((x, y), text, fill=(10, 10, 10), font=font)
        if i:
            draw.line([(cell * i, 0), (cell * i, h)], fill=(170, 170, 170), width=2)
    # MuJoCo maps a 2D texture mirrored onto the -x face of a box (the face the robot sees), so mirror it first
    img = ImageOps.mirror(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------- mjcf helpers
def _f(*vals) -> str:
    return " ".join(f"{float(v):.6g}" for v in vals)


def _quat_z(yaw: float) -> str:
    return _f(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))


def _box(parent: ET.Element, name: str, lo, hi, **attrs) -> ET.Element:
    """An axis-aligned box geom from its min and max corners (in the parent frame)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    if np.any(half <= 0):
        raise ValueError(f"degenerate box {name}: {lo} {hi}")
    return ET.SubElement(parent, "geom", name=name, type="box", pos=_f(*center), size=_f(*half), **attrs)


def _sloped(parent: ET.Element, name: str, center, half, axis: str, angle: float, **attrs) -> ET.Element:
    """A box geom rotated by ``angle`` about the parent's ``axis`` ("y" or "z")."""
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    quat = (c, 0, s, 0) if axis == "y" else (c, 0, 0, s)
    return ET.SubElement(parent, "geom", name=name, type="box", pos=_f(*center), size=_f(*half), quat=_f(*quat), **attrs)


def _look_at(eye, target, up=(0, 0, 1)) -> str:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return _f(*x, *y)


# ----------------------------------------------------------------------------- robot
def _robot_tree() -> ET.Element:
    root = ET.parse(panda_dir() / "panda.xml").getroot()
    root.set("model", SCENE_ID)
    comp = root.find("compiler")
    comp.set("meshdir", str((panda_dir() / "assets").resolve()).replace("\\", "/"))
    opt = root.find("option")
    opt.set("timestep", str(TIMESTEP))
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")
    opt.set("noslip_iterations", "3")
    for key in root.findall("keyframe"):
        root.remove(key)
    world = root.find("worldbody")
    for light in world.findall("light"):
        world.remove(light)
    # fingertip pads: high friction, torsional friction for yaw slip
    for i in range(1, 6):
        g = root.find(f".//default[@class='fingertip_pad_collision_{i}']/geom")
        g.set("friction", "1.5 0.005 0.0001")
        g.set("condim", "4")
        g.set("solref", "0.002 1")
    for body in root.iter("body"):
        if body.get("name") in ROBOT_BODIES:
            body.set("gravcomp", "1")
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tcp", pos=_f(0, 0, G.TCP_OFFSET), size="0.003", rgba="1 0 0 1", group="5")
    # wrist camera: on the hand's -x side, looking along the approach axis (hand +z), tilted towards the
    # fingertips. MuJoCo cameras look along their -z with +y up. Image up is hand -y, which is world up
    # in the canonical tool orientation (finger closing axis vertical, tool y pointing down).
    tilt = math.radians(14.0)
    x_cam = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    y_cam = np.array([0.0, -1.0, 0.0])
    ET.SubElement(hand, "camera", name="wrist", pos=_f(-0.055, 0.0, 0.035), xyaxes=_f(*x_cam, *y_cam), fovy="60")
    # stronger, force-limited gripper: ctrl is the half-width target (m) of the finger tendon. Closing
    # targets -0.05 (past closed) so the servo saturates at the commanded force whatever the object width.
    act = root.find(".//actuator/general[@name='actuator8']")
    act.set("ctrlrange", "-0.05 0.04")
    act.set("gainprm", _f(GRIPPER_KP, 0, 0))
    act.set("biasprm", _f(0, -GRIPPER_KP, -GRIPPER_KV))
    act.set("forcerange", _f(-2 * GRIPPER_FORCE_DEFAULT, 2 * GRIPPER_FORCE_DEFAULT))
    return root


# ----------------------------------------------------------------------------- scene parts
def _materials(asset: ET.Element) -> None:
    mats = {
        "floor_m": ("0.80 0.81 0.82 1", "0.1"),
        "rack_frame": ("0.18 0.19 0.21 1", "0.3"),
        "rail_m": ("0.55 0.57 0.60 1", "0.5"),
        "panel_m": (_f(*G.PANEL_RGBA), "0.3"),
        "port_m": ("0.08 0.08 0.09 1", "0.1"),
        "tray_m": ("0.70 0.72 0.70 1", "0.1"),
        "stand_m": ("0.35 0.36 0.38 1", "0.1"),
        "box_m": ("0.20 0.34 0.62 1", "0.3"),
        "bezel_m": ("0.85 0.15 0.75 1", "0.2"),   # magenta: a hue nothing else in the cell uses
        "plug_m": ("0.95 0.78 0.10 1", "0.3"),
        "boot_m": ("0.12 0.38 0.20 1", "0.1"),    # dark green: never confused with a dark port opening
        "holder_m": ("0.45 0.47 0.50 1", "0.1"),
    }
    for name, (rgba, spec) in mats.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular=spec, shininess="0.3")
    for state, rgba in G.LED_RGBA.items():
        # emission 0.55 keeps amber's green channel near 0.6 instead of saturating to yellow
        ET.SubElement(asset, "material", name=f"led_{state}", rgba=_f(*rgba), emission="0" if state == "off" else "0.55",
                      specular="0", shininess="0")
    ET.SubElement(asset, "texture", name="labels_tex", type="2d", file="labels.png")
    ET.SubElement(asset, "material", name="labels_m", texture="labels_tex", rgba="1 1 1 1", specular="0.1")
    ET.SubElement(asset, "texture", name="skybox", type="skybox", builtin="gradient", rgb1="0.92 0.93 0.94",
                  rgb2="0.75 0.77 0.80", width="256", height="1536")


STATIC = {"contype": "1", "conaffinity": "1"}
VISUAL = {"contype": "0", "conaffinity": "0", "group": "1"}
METAL = {"friction": "0.3 0.005 0.0001", "solref": "0.003 1", "solimp": "0.95 0.99 0.001"}


def _plug_geoms(parent: ET.Element, prefix: str, center=(0.0, 0.0, 0.0), dynamic: bool = False) -> None:
    """Plug body, key and boot as boxes, in the parent frame with the body centre at ``center``."""
    cx, cy, cz = center
    L, W, H = G.PLUG_BODY
    kl, kw, kh = G.PLUG_KEY
    bl, bw, bh = G.PLUG_BOOT
    common = {"friction": "0.5 0.005 0.0001", "solref": "0.003 1", "solimp": "0.95 0.99 0.001"}
    mass_b = {"mass": f"{G.PLUG_MASS_BODY:.6g}"} if dynamic else {}
    mass_k = {"mass": "0.0002"} if dynamic else {}
    mass_t = {"mass": f"{G.PLUG_MASS_BOOT:.6g}"} if dynamic else {}
    _box(parent, f"{prefix}_body", (cx - L / 2, cy - W / 2, cz - H / 2), (cx + L / 2, cy + W / 2, cz + H / 2),
         material="plug_m", **common, **mass_b)
    _box(parent, f"{prefix}_key", (cx + L / 2 - 0.004 - kl, cy - kw / 2, cz + H / 2), (cx + L / 2 - 0.004, cy + kw / 2, cz + H / 2 + kh),
         material="plug_m", **common, **mass_k)
    _box(parent, f"{prefix}_boot", (cx - L / 2 - bl, cy - bw / 2, cz - bh / 2), (cx - L / 2, cy + bw / 2, cz + bh / 2),
         material="boot_m", **common, **mass_t)


def _rack(world: ET.Element, cfg: SceneConfig) -> None:
    x, y, yaw = cfg.rack_pose
    rack = ET.SubElement(world, "body", name="rack", pos=_f(x, y, 0.0), quat=_quat_z(yaw))
    top = G.PANEL_TOP + 0.02
    # frame: four posts, top and bottom bars
    px, py = G.POST_SIZE
    for side, sy in (("l", 1), ("r", -1)):
        for end, x0 in (("f", 0.0), ("b", G.RACK_DEPTH - px)):
            _box(rack, f"post_{end}{side}", (x0, sy * G.POST_Y - py / 2, 0.0), (x0 + px, sy * G.POST_Y + py / 2, top),
                 material="rack_frame", **STATIC)
        _box(rack, f"top_bar_{side}", (0.0, sy * G.POST_Y - py / 2, top - 0.02), (G.RACK_DEPTH, sy * G.POST_Y + py / 2, top),
             material="rack_frame", **STATIC)
        _box(rack, f"base_bar_{side}", (0.0, sy * G.POST_Y - py / 2, 0.0), (G.RACK_DEPTH, sy * G.POST_Y + py / 2, 0.03),
             material="rack_frame", **STATIC)
    _box(rack, "top_cross", (G.RACK_DEPTH - px, -G.POST_Y, top - 0.02), (G.RACK_DEPTH, G.POST_Y, top), material="rack_frame", **STATIC)

    # shelf slots: L rails with entry flare and lead-in, back stop, cross supports
    hw = G.BOX_SIZE[1] / 2 + G.RAIL_LATERAL_CLEARANCE
    for k in range(1, G.N_SLOTS + 1):
        zt = G.slot_rail_top(k)
        for side, sy in (("l", 1), ("r", -1)):
            ylo, yhi = sorted((sy * (hw - G.RAIL_FLANGE_WIDTH), sy * (hw + G.RAIL_GUIDE_THICK)))
            _box(rack, f"rail{k}{side}_flange", (G.RAIL_X0, ylo, zt - G.RAIL_THICK), (G.RAIL_X1, yhi, zt),
                 material="rail_m", **STATIC, **METAL)
            glo, ghi = sorted((sy * hw, sy * (hw + G.RAIL_GUIDE_THICK)))
            _box(rack, f"rail{k}{side}_guide", (G.RAIL_X0 + G.RAIL_FLARE_LEN, glo, zt), (G.RAIL_X1, ghi, zt + G.RAIL_GUIDE_HEIGHT),
                 material="rail_m", **STATIC, **METAL)
            # flare: a guide plate angled outwards over the first RAIL_FLARE_LEN
            ang = math.atan2(G.RAIL_FLARE_OPEN, G.RAIL_FLARE_LEN)
            flen = math.hypot(G.RAIL_FLARE_OPEN, G.RAIL_FLARE_LEN)
            cxf = G.RAIL_X0 + G.RAIL_FLARE_LEN / 2
            cyf = sy * (hw + G.RAIL_FLARE_OPEN / 2) + sy * (G.RAIL_GUIDE_THICK / 2) / math.cos(ang)
            _sloped(rack, f"rail{k}{side}_flare", (cxf, cyf, zt + G.RAIL_GUIDE_HEIGHT / 2),
                    (flen / 2, G.RAIL_GUIDE_THICK / 2, G.RAIL_GUIDE_HEIGHT / 2), "z", sy * ang,
                    material="rail_m", **STATIC, **METAL)
            # lead-in ramp in front of the flange: 3 mm rise over 10 mm
            rang = math.atan2(G.RAIL_THICK, 0.010)
            rlen = math.hypot(G.RAIL_THICK, 0.010)
            _sloped(rack, f"rail{k}{side}_ramp", (G.RAIL_X0 - 0.005, (ylo + yhi) / 2, zt - G.RAIL_THICK / 2 - 0.0008),
                    (rlen / 2, (yhi - ylo) / 2, 0.0008), "y", -rang, material="rail_m", **STATIC, **METAL)
            # supports from the rail to the posts (outside the box footprint, behind the front plane)
            slo, shi = sorted((sy * (hw + G.RAIL_GUIDE_THICK), sy * (G.POST_Y - py / 2)))
            for j, (sx0, sx1) in enumerate(((0.01, 0.025), (0.10, 0.115))):
                _box(rack, f"rail{k}{side}_support{j}", (sx0, slo, zt - G.RAIL_THICK - 0.004), (sx1, shi, zt - G.RAIL_THICK),
                     material="rack_frame", **STATIC)
        _box(rack, f"stop{k}", (G.STOP_X, -0.09, zt), (G.STOP_X + 0.003, 0.09, zt + 0.015), material="rail_m", **STATIC, **METAL)

    # patch panel faceplate with 12 chamfered openings
    t = G.PANEL_THICK
    ch = G.PORT_CHAMFER
    zp = G.PORT_Z
    zlo, zhi = zp - G.PORT_H / 2 - ch, zp + G.PORT_H / 2 + ch
    _box(rack, "panel_top", (-t, -G.PANEL_HALF_WIDTH, zhi), (0.0, G.PANEL_HALF_WIDTH, G.PANEL_TOP), material="panel_m", **STATIC, **METAL)
    _box(rack, "panel_bottom", (-t, -G.PANEL_HALF_WIDTH, G.PANEL_BOTTOM), (0.0, G.PANEL_HALF_WIDTH, zlo), material="panel_m", **STATIC, **METAL)
    edges = []
    for n in range(1, G.N_PORTS + 1):
        yc = G.port_y(n)
        edges.append((yc - G.PORT_W / 2 - ch, yc + G.PORT_W / 2 + ch))
    # bars between widened openings, and the two outer sections
    spans = [(-G.PANEL_HALF_WIDTH, min(e[0] for e in edges))]
    ordered = sorted(edges)
    for (a0, a1), (b0, b1) in zip(ordered, ordered[1:]):
        spans.append((a1, b0))
    spans.append((max(e[1] for e in edges), G.PANEL_HALF_WIDTH))
    for j, (y0, y1) in enumerate(spans):
        _box(rack, f"panel_bar{j}", (-t, y0, zlo), (0.0, y1, zhi), material="panel_m", **STATIC, **METAL)
    occ = set(cfg.occupied)
    ang = math.radians(45.0)
    cl = ch * math.sqrt(2) / 2   # half length of a chamfer face
    ct = 0.0005                  # chamfer plate half thickness
    xc = -t + ch / 2             # chamfer spans x in [-t, -t + ch]
    xb0, xb1 = -t + ch, -t + ch + G.PORT_DEPTH  # straight bore
    for n in range(1, G.N_PORTS + 1):
        yc = G.port_y(n)
        w2, h2 = G.PORT_W / 2, G.PORT_H / 2
        wall = G.PORT_WALL
        kw2 = G.KEY_NOTCH_W / 2
        # chamfer faces (top split around the key notch), normals pointing into the opening
        off = ct / math.sqrt(2)
        _sloped(rack, f"port{n}_chb", (xc + off, yc, zp - h2 - ch / 2 - off), (cl, w2 + ch, ct), "y", -ang, material="port_m", **STATIC, **METAL)
        for side, s in (("l", 1), ("r", -1)):
            # left/right faces: rotate about z
            _sloped(rack, f"port{n}_ch{side}", (xc + off, yc + s * (w2 + ch / 2 + off), zp), (cl, ct, h2 + ch), "z", -s * ang,
                    material="port_m", **STATIC, **METAL)
            # top face, one piece each side of the notch
            ylo, yhi = sorted((yc + s * kw2, yc + s * (w2 + ch)))
            _sloped(rack, f"port{n}_cht{side}", (xc + off, (ylo + yhi) / 2, zp + h2 + ch / 2 + off), (cl, (yhi - ylo) / 2, ct), "y", ang,
                    material="port_m", **STATIC, **METAL)
        # straight bore walls, the key notch in the top wall, and the stop face
        _box(rack, f"port{n}_wb", (xb0, yc - w2 - wall, zp - h2 - wall), (xb1, yc + w2 + wall, zp - h2), material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_wl", (xb0, yc + w2, zp - h2), (xb1, yc + w2 + wall, zp + h2 + G.KEY_NOTCH_H + wall), material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_wr", (xb0, yc - w2 - wall, zp - h2), (xb1, yc - w2, zp + h2 + G.KEY_NOTCH_H + wall), material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_wtl", (xb0, yc + kw2, zp + h2), (xb1, yc + w2, zp + h2 + G.KEY_NOTCH_H + wall), material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_wtr", (xb0, yc - w2, zp + h2), (xb1, yc - kw2, zp + h2 + G.KEY_NOTCH_H + wall), material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_wtn", (xb0, yc - kw2, zp + h2 + G.KEY_NOTCH_H), (xb1, yc + kw2, zp + h2 + G.KEY_NOTCH_H + wall),
             material="port_m", **STATIC, **METAL)
        _box(rack, f"port{n}_stop", (xb1, yc - w2 - wall, zp - h2 - wall), (xb1 + 0.002, yc + w2 + wall, zp + h2 + G.KEY_NOTCH_H + wall),
             material="port_m", **STATIC, **METAL)
        # LED above the port
        lx, ly, lz = G.LED_SIZE
        _box(rack, f"led{n}", (-t - lx, yc - ly / 2, zp + G.LED_DZ - lz / 2), (-t, yc + ly / 2, zp + G.LED_DZ + lz / 2),
             material=f"led_{cfg.led_states[n - 1]}", **VISUAL)
        if n in occ:
            _plug_geoms(rack, f"occ{n}", tuple(G.plug_seated_rack(n)))
    # label strip below the ports (a thin textured box proud of the faceplate)
    y0 = G.port_y(1) + G.PORT_PITCH / 2
    y1 = G.port_y(G.N_PORTS) - G.PORT_PITCH / 2
    zl = zp + G.LABEL_DZ
    ET.SubElement(rack, "geom", name="labels", type="box", pos=_f(-t - 0.0002, (y0 + y1) / 2, zl),
                  size=_f(0.0002, abs(y0 - y1) / 2, G.LABEL_H / 2), material="labels_m", **VISUAL)
    # rack-frame sites used by checkers and oracles (privileged)
    for n in range(1, G.N_PORTS + 1):
        ET.SubElement(rack, "site", name=f"port{n}_mouth", pos=_f(*G.port_mouth_rack(n)), size="0.001", group="5")
    for k in range(1, G.N_SLOTS + 1):
        ET.SubElement(rack, "site", name=f"slot{k}_seat", pos=_f(*G.box_seated_rack(k)), size="0.002", group="5")


def _tray(world: ET.Element, cfg: SceneConfig) -> None:
    cx, cy = G.TRAY_CENTER
    sx, sy = G.TRAY_SIZE
    z = G.TRAY_TOP_Z
    _box(world, "tray_stand", (cx - sx / 2 + 0.02, cy - sy / 2 + 0.02, 0.0), (cx + sx / 2 - 0.02, cy + sy / 2 - 0.02, z - 0.006),
         material="stand_m", **STATIC)
    _box(world, "tray_plate", (cx - sx / 2, cy - sy / 2, z - 0.006), (cx + sx / 2, cy + sy / 2, z), material="tray_m", **STATIC)
    for name, lo, hi in (
        ("tray_rim_n", (cx - sx / 2, cy - sy / 2, z), (cx - sx / 2 + 0.006, cy + sy / 2, z + 0.008)),
        ("tray_rim_f", (cx + sx / 2 - 0.006, cy - sy / 2, z), (cx + sx / 2, cy + sy / 2, z + 0.008)),
        ("tray_rim_l", (cx - sx / 2, cy + sy / 2 - 0.006, z), (cx + sx / 2, cy + sy / 2, z + 0.008)),
        ("tray_rim_r", (cx - sx / 2, cy - sy / 2, z), (cx + sx / 2, cy - sy / 2 + 0.006, z + 0.008)),
    ):
        _box(world, name, lo, hi, material="tray_m", **STATIC)
    # box riser and box
    bx, by, byaw = cfg.box_pose
    riser = ET.SubElement(world, "body", name="box_riser", pos=_f(bx, by, z), quat=_quat_z(byaw))
    rx, ry, rz = G.BOX_RISER_SIZE
    _box(riser, "box_riser_geom", (G.BOX_RISER_DX - rx / 2, -ry / 2, 0.0), (G.BOX_RISER_DX + rx / 2, ry / 2, rz),
         material="holder_m", **STATIC, friction="0.6 0.005 0.0001")
    box = ET.SubElement(world, "body", name="box", pos=_f(bx, by, G.BOX_REST_Z), quat=_quat_z(byaw))
    ET.SubElement(box, "freejoint", name="box_joint")
    d, w, h = G.BOX_SIZE
    _box(box, "box_geom", (-d / 2, -w / 2, -h / 2), (d / 2, w / 2, h / 2), material="box_m", mass=f"{G.BOX_MASS:.6g}",
         friction="0.3 0.005 0.0001", solref="0.003 1", solimp="0.95 0.99 0.001")
    # front bezel stripe (visual only) so the front edge stands out from above and in front
    _box(box, "box_bezel_top", (-d / 2, -w / 2 + 0.004, h / 2), (-d / 2 + G.BOX_BEZEL_W, w / 2 - 0.004, h / 2 + 0.0005), material="bezel_m", **VISUAL)
    _box(box, "box_bezel_front", (-d / 2 - 0.0005, -w / 2 + 0.004, -h / 2 + 0.004), (-d / 2, w / 2 - 0.004, h / 2 - 0.004),
         material="bezel_m", **VISUAL)
    ET.SubElement(box, "site", name="box_center", pos="0 0 0", size="0.002", group="5")
    # plug holder and plug
    px_, py_, pyaw = cfg.plug_pose
    holder = ET.SubElement(world, "body", name="plug_holder", pos=_f(px_, py_, z), quat=_quat_z(pyaw))
    hx, hy, hz = G.PLUG_HOLDER_SIZE
    _box(holder, "plug_holder_geom", (G.PLUG_HOLDER_DX - hx / 2, -hy / 2, 0.0), (G.PLUG_HOLDER_DX + hx / 2, hy / 2, hz),
         material="holder_m", **STATIC, friction="0.6 0.005 0.0001")
    plug = ET.SubElement(world, "body", name="plug", pos=_f(px_, py_, G.PLUG_REST_Z), quat=_quat_z(pyaw))
    ET.SubElement(plug, "freejoint", name="plug_joint")
    _plug_geoms(plug, "plug", dynamic=True)
    ET.SubElement(plug, "site", name="plug_tip", pos=_f(G.PLUG_BODY[0] / 2, 0, 0), size="0.001", group="5")


def rack_nominal_to_world(p) -> np.ndarray:
    """A rack frame point at the NOMINAL rack pose, in world (cameras are fixed to the world)."""
    rx, ry, ryaw = G.RACK_NOMINAL
    return np.array([rx, ry, 0.0]) + G.rot_z(ryaw) @ np.asarray(p, float)


def _cameras_lights(world: ET.Element, cfg: SceneConfig) -> None:
    # overhead: straight down over the tray and the rack front, image up = world +x
    # centred over the tray so both parts are seen nearly square-on; the rack front edge stays in view
    eye = (0.50, -0.25, 1.15)
    ET.SubElement(world, "camera", name="overhead", pos=_f(*eye),
                  xyaxes=_look_at(eye, (0.50, -0.25, 0.0), up=(1, 0, 0)), fovy="40")
    # front: about 0.5 m from the panel along its normal, raised 0.25 m, looking down about 30 deg
    panel_c = rack_nominal_to_world((-G.PANEL_THICK, 0.0, G.PORT_Z - 0.004))
    eye = rack_nominal_to_world((-G.PANEL_THICK - 0.42, 0.0, G.PORT_Z + 0.25))
    ET.SubElement(world, "camera", name="front", pos=_f(*eye), xyaxes=_look_at(eye, panel_c), fovy="30")
    # scene: a third-person view for videos
    eye = (0.08, -0.92, 0.88)
    ET.SubElement(world, "camera", name="scene", pos=_f(*eye), xyaxes=_look_at(eye, (0.52, -0.02, 0.24)), fovy="50")
    # close-up of the panel and slots for evidence images (privileged view, never given to skills)
    target = rack_nominal_to_world((0.0, 0.0, G.PORT_Z - 0.05))
    eye = rack_nominal_to_world((-0.30, -0.06, G.PORT_Z + 0.10))
    ET.SubElement(world, "camera", name="closeup", pos=_f(*eye), xyaxes=_look_at(eye, target), fovy="40")
    for i, light in enumerate(cfg.lights):
        d = light.diffuse
        ET.SubElement(world, "light", name=f"light{i}", pos=_f(*light.pos), dir=_f(*light.dir), directional="false",
                      diffuse=_f(d, d, d), specular="0.1 0.1 0.1", castshadow="true" if light.castshadow else "false",
                      attenuation="1 0 0", cutoff="70", exponent="1")


def build_mjcf(cfg: SceneConfig) -> tuple[str, dict[str, bytes]]:
    """The scene XML for ``cfg`` and its in-memory assets."""
    root = _robot_tree()
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "headlight", diffuse="0.25 0.25 0.25", ambient="0.30 0.30 0.30", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "quality", shadowsize="2048")
    asset = root.find("asset")
    _materials(asset)
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.05", material="floor_m", **STATIC)
    _rack(world, cfg)
    _tray(world, cfg)
    _cameras_lights(world, cfg)
    xml = ET.tostring(root, encoding="unicode")
    return xml, {"labels.png": label_texture_png()}


# ----------------------------------------------------------------------------- scene
@dataclass
class Scene:
    """A compiled rack_cell for one seed. ``model`` and ``data`` are privileged: only the
    SimBackend (as a private attribute), checkers and oracles touch them."""

    cfg: SceneConfig
    xml: str
    model: mujoco.MjModel
    data: mujoco.MjData

    def body_pose(self, name: str) -> np.ndarray:
        b = self.model.body(name).id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[b].reshape(3, 3)
        T[:3, 3] = self.data.xpos[b]
        return T

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()


def compile_scene(cfg: SceneConfig) -> Scene:
    xml, assets = build_mjcf(cfg)
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)
    return Scene(cfg, xml, model, data)


def make_scene(seed: int) -> Scene:
    """Sample the per-seed randomization and compile the scene (robot not yet posed)."""
    return compile_scene(sample_config(seed))
