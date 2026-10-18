"""Classical-CV perception for the rack_cell skills.

Inputs are camera images with their intrinsics and extrinsics (from an Observation)
and the TaskSpec's nominal geometry. Nothing else: no simulator state.

* ``estimate_rack_pose``: the 12 port openings in the front camera (a free port is a
  dark opening, an occupied one shows a yellow plug), then a 3-DOF fit of the rack pose
  (x, y, yaw; the rack stands on the floor) by robust least squares on reprojection error.
* ``locate_box``: the magenta bezel strip on the box's top front edge in the overhead
  camera, back-projected onto the known top-face plane.
* ``locate_plug``: the yellow plug body in the overhead camera, back-projected onto its
  known top plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..schemas import CameraImage, Observation


class PerceptionError(RuntimeError):
    """The target could not be found or the fit is not trustworthy."""


# ------------------------------------------------------------------------- geometry
def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def frame_to_world(pose: tuple[float, float, float], pts) -> np.ndarray:
    """Points in a floor frame (x, y, yaw) to world."""
    x, y, yaw = pose
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ rot_z(yaw).T + np.array([x, y, 0.0])


def project(cam: CameraImage, pts_world) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) and depths of world points."""
    T = cam.extrinsics.base_to_camera()
    pts = np.atleast_2d(np.asarray(pts_world, float))
    pc = pts @ T[:3, :3].T + T[:3, 3]
    uvw = pc @ cam.intrinsics.matrix().T
    return uvw[:, :2] / uvw[:, 2:3], pc[:, 2]


def backproject_to_plane(cam: CameraImage, uv, z: float) -> np.ndarray:
    """World points where the pixel rays meet the horizontal plane ``z``."""
    uv = np.atleast_2d(np.asarray(uv, float))
    Kinv = np.linalg.inv(cam.intrinsics.matrix())
    Tcb = cam.extrinsics.camera_to_base()
    rays_c = np.c_[uv, np.ones(len(uv))] @ Kinv.T
    rays_w = rays_c @ Tcb[:3, :3].T
    origin = Tcb[:3, 3]
    t = (z - origin[2]) / rays_w[:, 2]
    return origin + rays_w * t[:, None]


def hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


@dataclass
class Blob:
    area: int
    centroid: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    pixels: np.ndarray = field(repr=False)  # (N, 2) as (u, v)

    @property
    def aspect(self) -> float:
        return self.bbox[2] / max(1, self.bbox[3])


def blobs(mask: np.ndarray, min_area: int = 1, max_area: int = 10**9) -> list[Blob]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        ys, xs = np.nonzero(labels == i)
        out.append(Blob(area, np.array(cents[i]), tuple(int(v) for v in stats[i, :4]), np.c_[xs, ys]))
    return out


def roi_mask(shape, uv_points, margin_px: float) -> np.ndarray:
    """A mask of the axis-aligned box around ``uv_points`` grown by ``margin_px``."""
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    uv = np.asarray(uv_points, float)
    u0, v0 = np.floor(uv.min(axis=0) - margin_px).astype(int)
    u1, v1 = np.ceil(uv.max(axis=0) + margin_px).astype(int)
    m[max(0, v0):min(h, v1), max(0, u0):min(w, u1)] = True
    return m


# ------------------------------------------------------------------------- colours
def mask_dark(img: np.ndarray, v_max: int = 50) -> np.ndarray:
    return hsv(img)[..., 2] <= v_max


def mask_yellow(img: np.ndarray) -> np.ndarray:
    h = hsv(img)
    return (h[..., 0] >= 17) & (h[..., 0] <= 34) & (h[..., 1] >= 110) & (h[..., 2] >= 110)


def mask_magenta(img: np.ndarray) -> np.ndarray:
    h = hsv(img)
    return (h[..., 0] >= 140) & (h[..., 0] <= 172) & (h[..., 1] >= 90) & (h[..., 2] >= 70)


def mask_boot_green(img: np.ndarray) -> np.ndarray:
    h = hsv(img)
    return (h[..., 0] >= 50) & (h[..., 0] <= 80) & (h[..., 1] >= 60) & (h[..., 2] >= 35)


def mask_blue(img: np.ndarray) -> np.ndarray:
    h = hsv(img)
    return (h[..., 0] >= 100) & (h[..., 0] <= 125) & (h[..., 1] >= 90) & (h[..., 2] >= 60)


# ------------------------------------------------------------------------- rack
@dataclass
class RackEstimate:
    x: float
    y: float
    yaw: float                    # rad
    residual_px: float
    n_ports: int
    port_px: dict[int, list[float]]
    occupied_seen: list[int]

    @property
    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def to_world(self, p_rack) -> np.ndarray:
        return frame_to_world(self.pose, p_rack)[0]

    def summary(self) -> dict[str, Any]:
        return {"x": round(self.x, 5), "y": round(self.y, 5), "yaw_deg": round(math.degrees(self.yaw), 3),
                "residual_px": round(self.residual_px, 3), "n_ports": self.n_ports}


# Where a feature's image centroid sits on the rack, measured against simulator truth over 40 seeds
# (119 occupied ports, RMS 0.43 px; 96 free ports): a free port's dark opening centres on the mouth
# (0.2 mm high), an occupied port's visible yellow plug centres 4.5 mm proud of the rack front plane.
FREE_DZ = 0.0002
OCCUPIED_X = -0.0045
OCCUPIED_DZ = 0.0003


def _port_points_rack(geom: dict) -> dict[int, np.ndarray]:
    ports = geom["ports"]
    fx = geom["rack"]["faceplate_front_x"]
    return {int(n): np.array([fx, float(y), float(ports["center_z"])]) for n, y in ports["center_y"].items()}


def _fit_rack(cam: CameraImage, uv: np.ndarray, pts_rack: np.ndarray, x0: tuple[float, float, float]):
    def resid(p):
        pred, _ = project(cam, frame_to_world((p[0], p[1], p[2]), pts_rack))
        return (pred - uv).ravel()

    sol = least_squares(resid, np.asarray(x0, float), loss="soft_l1", f_scale=1.0, x_scale=[0.01, 0.01, 0.02])
    r = sol.fun.reshape(-1, 2)
    return sol.x, float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


def estimate_rack_pose(obs: Observation, geom: dict, camera: str = "front") -> RackEstimate:
    """Rack pose from the row of port openings (free: dark; occupied: yellow plug)."""
    cam = obs.cameras[camera]
    img = obs.image(camera)
    nom = geom["rack"]["nominal_pose"]
    x0 = (float(nom["x"]), float(nom["y"]), math.radians(float(nom["yaw_deg"])))
    pts = _port_points_rack(geom)
    n_ports = len(pts)
    order = sorted(pts)
    uv_pred, depth = project(cam, frame_to_world(x0, np.array([pts[n] for n in order])))
    if np.any(depth <= 0):
        raise PerceptionError(f"the port row is behind the {camera} camera")
    pitch_px = float(np.median(np.linalg.norm(np.diff(uv_pred, axis=0), axis=1)))
    tol = geom["rack"].get("pose_tolerance", {"xy": 0.02, "yaw_deg": 3.0})
    margin = pitch_px * (tol["xy"] / float(geom["ports"]["pitch"]) + 1.5)
    roi = roi_mask(img.shape, uv_pred, margin)
    port_area = pitch_px * pitch_px * 0.75 * 0.6
    dark = [b for b in blobs(mask_dark(img) & roi, int(port_area * 0.35), int(port_area * 2.5)) if 1.0 <= b.aspect <= 2.6]
    # an occupied port's plug is about as big as an opening; amber LEDs are a sixth of that
    yellow = [b for b in blobs(mask_yellow(img) & roi, int(port_area * 0.45), int(port_area * 3.0))]
    feats = [(b.centroid, "free") for b in dark] + [(b.centroid, "occupied") for b in yellow]
    if len(feats) < 4:
        raise PerceptionError(f"found {len(feats)} port-like features in the {camera} image; need at least 4")
    # keep the features that lie on the port row (a line in the image)
    cents = np.array([f[0] for f in feats])
    line_v = np.polyval(np.polyfit(uv_pred[:, 0], uv_pred[:, 1], 1), cents[:, 0])
    keep = np.abs(cents[:, 1] - line_v - np.median(cents[:, 1] - line_v)) < 0.4 * pitch_px
    feats = [f for f, k in zip(feats, keep) if k]
    feats.sort(key=lambda f: f[0][0])
    # relative indices from the gaps, then the absolute offset with the best fit
    rel = [0]
    for a, b in zip(feats, feats[1:]):
        rel.append(rel[-1] + max(1, int(round((b[0][0] - a[0][0]) / pitch_px))))
    if rel[-1] >= n_ports:
        raise PerceptionError(f"the port row spans {rel[-1] + 1} pitches in the {camera} image; expected at most {n_ports}")

    def feature_point(n: int, kind: str) -> np.ndarray:
        x, y, z = pts[n]
        return np.array([OCCUPIED_X, y, z + OCCUPIED_DZ]) if kind == "occupied" else np.array([x, y, z + FREE_DZ])

    uv = np.array([f[0] for f in feats])
    best = None
    for k in range(0, n_ports - rel[-1]):
        idx = [order[r + k] for r in rel]
        P = np.array([feature_point(n, kind) for n, (_, kind) in zip(idx, feats)])
        sol, res = _fit_rack(cam, uv, P, x0)
        if best is None or res < best[1]:
            best = (sol, res, idx)
    sol, res, idx = best
    if res > 3.0:
        raise PerceptionError(f"rack fit residual {res:.1f} px is too large")
    occupied = [n for n, f in zip(idx, feats) if f[1] == "occupied"]
    return RackEstimate(float(sol[0]), float(sol[1]), float(sol[2]), res, len(feats),
                        {n: [round(float(f[0][0]), 2), round(float(f[0][1]), 2)] for n, f in zip(idx, feats)}, occupied)


def backproject_to_plane_n(cam: CameraImage, uv, point, normal) -> np.ndarray:
    """World points where pixel rays meet the plane through ``point`` with ``normal``."""
    uv = np.atleast_2d(np.asarray(uv, float))
    Kinv = np.linalg.inv(cam.intrinsics.matrix())
    Tcb = cam.extrinsics.camera_to_base()
    rays = (np.c_[uv, np.ones(len(uv))] @ Kinv.T) @ Tcb[:3, :3].T
    o = Tcb[:3, 3]
    n = np.asarray(normal, float)
    t = ((np.asarray(point, float) - o) @ n) / (rays @ n)
    return o + rays * t[:, None]


# The wrist camera sees a port opening obliquely, so the dark region's centroid sits a little off the mouth
# centre. Measured against simulator truth (see reports/S3.md); applied in the faceplate plane, in metres.
WRIST_PORT_BIAS = (0.0, 0.0)  # (along the panel's +y, up)


def locate_port_in_wrist(obs: Observation, mouth_est, yaw: float, geom: dict, max_px: float = 30.0):
    """The port opening nearest the predicted mouth in the wrist image.

    Returns (pixel centroid, mouth estimate on the faceplate plane, blob area) or raises PerceptionError.
    """
    cam = obs.cameras["wrist"]
    img = obs.image("wrist")
    mouth_est = np.asarray(mouth_est, float)
    uv0, depth = project(cam, mouth_est)
    u0, v0 = uv0[0]
    if depth[0] <= 0:
        raise PerceptionError("the port is behind the wrist camera")
    # expected size of the opening at this range
    f = cam.intrinsics.fx
    w_px = f * float(geom["ports"]["opening_w"]) / depth[0]
    h_px = f * float(geom["ports"]["opening_h"]) / depth[0]
    area = w_px * h_px
    cand = [b for b in blobs(mask_dark(img), int(area * 0.25), int(area * 3.5))
            if 0.6 <= b.aspect <= 3.0 and np.hypot(b.centroid[0] - u0, b.centroid[1] - v0) <= max_px]
    if not cand:
        raise PerceptionError(f"no port opening within {max_px:.0f} px of the prediction in the wrist image")
    best = min(cand, key=lambda b: np.hypot(b.centroid[0] - u0, b.centroid[1] - v0))
    d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    p = backproject_to_plane_n(cam, best.centroid, mouth_est, d)[0]
    left = np.array([-d[1], d[0], 0.0])
    p = p - WRIST_PORT_BIAS[0] * left - WRIST_PORT_BIAS[1] * np.array([0.0, 0.0, 1.0])
    return best.centroid, p, best.area


# ------------------------------------------------------------------------- parts
@dataclass
class PartEstimate:
    name: str
    x: float
    y: float
    z: float
    yaw: float           # rad, the part's x axis (away from the robot)
    pixels: int
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "x": round(self.x, 5), "y": round(self.y, 5), "z": round(self.z, 5),
                "yaw_deg": round(math.degrees(self.yaw), 3), "pixels": self.pixels, **self.extra}


def _principal_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    return c, v[:, int(np.argmax(w))]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _tray_roi(cam: CameraImage, img: np.ndarray, center_xy, z: float, half: float) -> np.ndarray:
    cx, cy = center_xy
    corners = [(cx + dx, cy + dy, z) for dx in (-half, half) for dy in (-half, half)]
    uv, _ = project(cam, corners)
    return roi_mask(img.shape, uv, 4)


def locate_box(obs: Observation, geom: dict, camera: str = "overhead") -> PartEstimate:
    """The box from its magenta top-front bezel strip (and the blue top to tell front from back)."""
    cam = obs.cameras[camera]
    img = obs.image(camera)
    app = geom["appearance"]["box"]
    size = geom["box"]["size"]
    nom = geom["tray"]["box_nominal"]
    z_top = float(app["top_z"])
    roi = _tray_roi(cam, img, (nom["x"], nom["y"]), z_top, 0.20)
    cand = blobs(mask_magenta(img) & roi, 30)
    if not cand:
        raise PerceptionError("no magenta bezel in the overhead image near the box stand")
    strip = max(cand, key=lambda b: b.area)
    P = backproject_to_plane(cam, strip.pixels, z_top + 0.0005)[:, :2]
    c, axis = _principal_axis(P)
    length = float(np.ptp(P @ axis))
    normal = np.array([-axis[1], axis[0]])
    # the box body (blue top) lies behind the strip, away from the robot
    blue = mask_blue(img) & roi
    ys, xs = np.nonzero(blue)
    if len(xs) > 50:
        B = backproject_to_plane(cam, np.c_[xs, ys], z_top)[:, :2]
        if (B.mean(axis=0) - c) @ normal < 0:
            normal = -normal
    else:
        nominal_dir = np.array([math.cos(math.radians(nom["yaw_deg"])), math.sin(math.radians(nom["yaw_deg"]))])
        if normal @ nominal_dir < 0:
            normal = -normal
    yaw = math.atan2(normal[1], normal[0])
    bezel_w = float(app["bezel_width"])
    center = c + normal * (float(size["depth_x"]) / 2 - bezel_w / 2)
    # The strip's ends are a pixel or two each, so its centre along the box width is only good to about
    # 1 mm. When the whole top face (blue plus the strip) is in view, take the width position from the
    # face centroid instead: thousands of pixels, about 0.1 mm.
    face_refined = False
    faces = blobs((mask_blue(img) | mask_magenta(img)) & roi, 200)
    if faces:
        F = backproject_to_plane(cam, max(faces, key=lambda b: b.area).pixels, z_top)[:, :2]
        span_w, span_d = float(np.ptp(F @ axis)), float(np.ptp(F @ normal))
        if (abs(span_w - float(size["width_y"])) < 0.008 and abs(span_d - float(size["depth_x"])) < 0.008):
            center = center + ((F.mean(axis=0) - center) @ axis) * axis
            face_refined = True
    z_c = z_top - float(size["height_z"]) / 2
    return PartEstimate("box", float(center[0]), float(center[1]), z_c, yaw, strip.area,
                        {"strip_length_mm": round(length * 1000, 1), "face_refined": face_refined})


# The fitted yaw reads 0.36 deg high on average: the 3 mm boot is about 2 px wide from above, so its visible
# pixels are biased. Calibrated once against simulator truth on dev seeds 0-99 (overhead camera and plug
# stand as now placed) and checked on seeds 100-199.
PLUG_YAW_BIAS_DEG = 0.36


def locate_plug(obs: Observation, geom: dict, camera: str = "overhead") -> PartEstimate:
    """The plug from its yellow body in the overhead camera."""
    cam = obs.cameras[camera]
    img = obs.image(camera)
    app = geom["appearance"]["plug"]
    nom = geom["tray"]["plug_nominal"]
    z_top = float(app["top_z"])
    roi = _tray_roi(cam, img, (nom["x"], nom["y"]), z_top, 0.08)
    cand = blobs(mask_yellow(img) & roi, 8)
    if not cand:
        raise PerceptionError("no yellow plug body in the overhead image near the plug holder")
    body = max(cand, key=lambda b: b.area)
    P = backproject_to_plane(cam, body.pixels, z_top)[:, :2]
    c, axis = _principal_axis(P)
    nominal_dir = np.array([math.cos(math.radians(nom["yaw_deg"])), math.sin(math.radians(nom["yaw_deg"]))])
    if axis @ nominal_dir < 0:
        axis = -axis
    # The green boot extends the plug to about 45 mm: refit the axis on body plus boot for a longer baseline.
    # Boot pixels are green pixels within the boot's footprint behind the body (on the boot's own top plane).
    plug = geom["plug"]
    body_len, boot_len = float(plug["body"]["length_x"]), float(plug["boot"]["length_x"])
    z_boot_top = z_top - (float(plug["body"]["height_z"]) - float(plug["boot"]["height_z"])) / 2
    boot_mask = mask_boot_green(img) & roi
    ys, xs = np.nonzero(boot_mask)
    boot_pts = np.empty((0, 2))
    if len(xs):
        D = backproject_to_plane(cam, np.c_[xs, ys], z_boot_top)[:, :2]
        rel = D - c
        along, across = rel @ axis, rel @ np.array([-axis[1], axis[0]])
        sel = (along < -body_len / 2 + 0.002) & (along > -body_len / 2 - boot_len - 0.004) & (np.abs(across) < 0.006)
        boot_pts = D[sel]
    n_boot = len(boot_pts)
    if n_boot >= 6:
        d = boot_pts.mean(axis=0) - P.mean(axis=0)
        axis2 = -d / np.linalg.norm(d)  # from the boot centroid to the body centroid = the plug's +x
        if axis2 @ axis > 0.9:  # agree with the body's own axis to within 25 deg
            axis = axis2
    yaw = math.atan2(axis[1], axis[0])
    z_c = z_top - float(plug["body"]["height_z"]) / 2
    # Render and compare: the body's and boot's visible side faces shift a plain centroid towards the camera's
    # nadir, so fit the projected 3D boxes to the yellow and green silhouettes.
    x, y, yaw, score = _fit_plug_silhouette(cam, mask_yellow(img) & roi, mask_boot_green(img) & roi, geom,
                                            (float(c[0]), float(c[1]), yaw), z_c)
    yaw -= math.radians(PLUG_YAW_BIAS_DEG)
    return PartEstimate("plug", x, y, z_c, yaw, body.area, {"boot_pixels": n_boot, "fit_rms_px": round(score, 3)})


def _box_corners(center, half, yaw: float) -> np.ndarray:
    hx, hy, hz = half
    pts = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return pts @ rot_z(yaw).T + np.asarray(center, float)


def _hull_centroid(cam: CameraImage, corners: np.ndarray) -> np.ndarray:
    """Area centroid (pixels) of the convex hull of projected 3D points: continuous in the pose."""
    uv, _ = project(cam, corners)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    x, y = hull[:, 0], hull[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return uv.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * a)


def _fit_plug_silhouette(cam: CameraImage, yellow: np.ndarray, green: np.ndarray, geom: dict,
                         x0: tuple[float, float, float], z_c: float) -> tuple[float, float, float, float]:
    """Pose (x, y, yaw) of the plug body centre whose projected body (with key) and boot match the observed
    yellow and green region centroids. Moment matching is smooth and sub-pixel, unlike a pixel IoU."""
    plug = geom["plug"]
    bl, bw, bh = (float(plug["body"][k]) for k in ("length_x", "width_y", "height_z"))
    kl, kw, kh = (float(plug["key"][k]) for k in ("length_x", "width_y", "height_z"))
    tl, tw, th = (float(plug["boot"][k]) for k in ("length_x", "width_y", "height_z"))
    yb = blobs(yellow, 8)
    gb = blobs(green, 4)
    if not yb:
        return x0[0], x0[1], x0[2], 0.0
    y_obs = max(yb, key=lambda b: b.area).centroid
    g_obs = None
    if gb:
        # the boot blob nearest the yellow body
        g_obs = min(gb, key=lambda b: np.linalg.norm(b.centroid - y_obs)).centroid

    def predicted(p):
        x, y, yaw = p
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        body = _box_corners((x, y, z_c), (bl / 2, bw / 2, bh / 2), yaw)
        key_c = np.array([x, y, z_c + bh / 2 + kh / 2]) + d * (bl / 2 - 0.004 - kl / 2)
        key = _box_corners(key_c, (kl / 2, kw / 2, kh / 2), yaw)
        boot_c = np.array([x, y, z_c]) - d * (bl / 2 + tl / 2)
        boot = _box_corners(boot_c, (tl / 2, tw / 2, th / 2), yaw)
        return _hull_centroid(cam, np.vstack([body, key])), _hull_centroid(cam, boot)

    def resid(p):
        cy, cg = predicted(p)
        r = list(cy - y_obs)
        if g_obs is not None:
            r += list(0.7 * (cg - g_obs))
        return np.array(r)

    sol = least_squares(resid, np.array(x0, float), x_scale=[0.001, 0.001, 0.02])
    x, y, yaw = sol.x
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return float(x), float(y), float(yaw), rms
