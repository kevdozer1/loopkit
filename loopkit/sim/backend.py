"""SimBackend: the Backend protocol on the rack_cell scene.

Skills talk to this class through ``observe()`` and ``act()`` only. The MuJoCo model
and data live in the private ``_scene`` attribute. Checkers and oracles, which may read
privileged state, use ``backend.privileged`` (a :class:`PrivilegedView`); skills never do,
and a test enforces it.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from ..runtime.backend import ActResult, Command, Grip, MoveJoints, MoveTo, Pose, Wait
from ..schemas import AABB, CameraExtrinsics, CameraImage, CameraIntrinsics, Observation, Safety
from . import geometry as G
from .rack_cell import TIMESTEP, Scene, make_scene
from .robot import Q_REST, Panda, rotation_error, tool_rotation

CAMERAS = ("overhead", "front", "wrist")
CTRL_STEPS = 2                     # physics steps per control tick (2 ms)
OPEN_WIDTH = 0.08
DEFAULT_SAFETY = Safety(
    workspace_bounds=AABB(min=G.WORKSPACE_MIN, max=G.WORKSPACE_MAX),
    max_speed=G.MAX_SPEED,
    max_force=G.MAX_FORCE,
)
COMPLIANT_GAIN = 0.15              # joint stiffness scale in compliant moves


def _min_jerk(s: float) -> float:
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def _yaw_of(R: np.ndarray) -> float:
    """Approach yaw (deg) of a tool frame: the heading of its z axis."""
    return math.degrees(math.atan2(R[1, 2], R[0, 2]))


class VideoRecorder:
    """Collects frames at a fixed sim-time rate; ``save`` writes an mp4 (H.264, yuv420p)."""

    LAYOUTS = {
        # main view (640 x 480) on the left, two 320 x 240 views stacked on the right
        "manip": ("scene", "wrist", "closeup"),
        "inspect": ("front", "overhead", "scene"),
    }

    def __init__(self, fps: int = 25, layout: str = "manip"):
        self.fps = fps
        self.layout = layout
        self.frames: list[np.ndarray] = []
        self.next_t = 0.0

    def save(self, path: Path) -> Path:
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=7,
                                    pixelformat="yuv420p", macro_block_size=16)
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return path


class PrivilegedView:
    """Simulator truth for checkers and oracles. Skills must never use this."""

    def __init__(self, backend: "SimBackend"):
        self._b = backend

    @property
    def scene(self) -> Scene:
        return self._b._scene

    @property
    def cfg(self):
        return self._b._scene.cfg

    @property
    def model(self) -> mujoco.MjModel:
        return self._b._scene.model

    @property
    def data(self) -> mujoco.MjData:
        return self._b._scene.data

    def body_pose(self, name: str) -> np.ndarray:
        return self._b._scene.body_pose(name)

    def rack_T(self) -> np.ndarray:
        return self.body_pose("rack")

    def rack_to_world(self, p) -> np.ndarray:
        T = self.rack_T()
        return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]

    def world_to_rack(self, p) -> np.ndarray:
        T = self.rack_T()
        return T[:3, :3].T @ (np.asarray(p, float) - T[:3, 3])

    def geom_label(self, gid: int) -> str:
        """A geom's name, or ``<body>/<index>`` for the unnamed Menagerie geoms."""
        m = self.model
        name = m.geom(gid).name
        return name or f"{m.body(int(m.geom_bodyid[gid])).name}/geom{gid}"

    def contacts(self) -> list[tuple[int, int, float]]:
        """(geom1 id, geom2 id, normal force N) for every active contact."""
        m, d = self.model, self.data
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            out.append((int(c.geom1), int(c.geom2), float(abs(f6[0]))))
        return out

    def body_contacts(self, body: str) -> list[tuple[str, str, float]]:
        """Contacts involving any geom of ``body``: (own geom, other geom, force), as labels."""
        m = self.model
        bid = m.body(body).id
        out = []
        for g1, g2, f in self.contacts():
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == bid:
                out.append((self.geom_label(g1), self.geom_label(g2), f))
            elif b2 == bid:
                out.append((self.geom_label(g2), self.geom_label(g1), f))
        return out

    def robot_contacts(self) -> list[tuple[str, str, float]]:
        """Contacts between a robot body and anything else (robot self-contacts excluded)."""
        from .rack_cell import ROBOT_BODIES

        m = self.model
        robot = {m.body(n).id for n in ROBOT_BODIES}
        out = []
        for g1, g2, f in self.contacts():
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 in robot) != (b2 in robot):
                r, o = (g1, g2) if b1 in robot else (g2, g1)
                out.append((self.geom_label(r), self.geom_label(o), f))
        return out

    def render(self, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
        return self._b._render(camera, width, height)

    def gripper_width(self) -> float:
        return self._b._arm.gripper_width()


class SimBackend:
    """rack_cell behind the Backend protocol."""

    def __init__(self, safety: Safety | None = None, render: bool = True, cameras: tuple[str, ...] = CAMERAS):
        self.safety = safety or DEFAULT_SAFETY
        self.render_enabled = render
        self.cameras = tuple(cameras)
        self._scene: Scene | None = None
        self._arm: Panda | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._q_cmd = Q_REST.copy()
        self._video: VideoRecorder | None = None
        self.privileged = PrivilegedView(self)
        self.stats = {"max_force": 0.0, "safety_violation": None}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int) -> Observation:
        self.close()
        self._scene = make_scene(seed)
        m, d = self._scene.model, self._scene.data
        self._arm = Panda(m, d)
        self._kp0 = m.actuator_gainprm[self._arm.act, 0].copy()
        self._kv0 = -m.actuator_biasprm[self._arm.act, 2].copy()
        self._grip_force_range = m.actuator_forcerange[self._arm.grip_act].copy()
        self._arm.set_q(Q_REST)
        self._q_cmd = Q_REST.copy()
        q_f = OPEN_WIDTH / 2
        d.qpos[self._arm.fqadr] = q_f
        d.ctrl[self._arm.grip_act] = q_f
        mujoco.mj_forward(m, d)
        self._steps(int(0.3 / TIMESTEP))
        self._t0 = float(d.time)
        self.stats = {"max_force": 0.0, "safety_violation": None}
        return self.observe()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def stop(self) -> None:
        """Hold the current measured pose."""
        if self._arm is None:
            return
        self._q_cmd = self._arm.q()
        self._scene.data.ctrl[self._arm.act] = self._q_cmd
        self._steps(int(0.05 / TIMESTEP))

    @property
    def sim_time(self) -> float:
        return float(self._scene.data.time) - self._t0

    # ------------------------------------------------------------------ video
    def start_video(self, fps: int = 25, layout: str = "manip") -> VideoRecorder:
        self._video = VideoRecorder(fps=fps, layout=layout)
        self._video.next_t = self.sim_time
        self._video_frame()
        return self._video

    def stop_video(self) -> VideoRecorder | None:
        video, self._video = self._video, None
        return video

    def _video_frame(self) -> None:
        a, b, c = VideoRecorder.LAYOUTS[self._video.layout]
        main = self._render(a, 640, 480)
        right = np.concatenate([self._render(b, 320, 240), self._render(c, 320, 240)], axis=0)
        self._video.frames.append(np.concatenate([main, right], axis=1))

    # ------------------------------------------------------------------ physics
    def _steps(self, n: int) -> None:
        m, d = self._scene.model, self._scene.data
        for _ in range(n):
            mujoco.mj_step(m, d)
            if self._video is not None and self.sim_time + 1e-9 >= self._video.next_t:
                self._video.next_t += 1.0 / self._video.fps
                self._video_frame()
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("physics diverged")

    def _render(self, camera: str, width: int, height: int) -> np.ndarray:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self._scene.model, height, width)
        r = self._renderers[key]
        r.update_scene(self._scene.data, camera=camera)
        return r.render().copy()

    def _set_compliance(self, scale: float) -> None:
        m = self._scene.model
        a = self._arm.act
        m.actuator_gainprm[a, 0] = self._kp0 * scale
        m.actuator_biasprm[a, 1] = -self._kp0 * scale
        m.actuator_biasprm[a, 2] = -self._kv0 * math.sqrt(scale)

    # ------------------------------------------------------------------ observe
    def _camera_image(self, name: str) -> CameraImage:
        m, d = self._scene.model, self._scene.data
        cid = m.camera(name).id
        fovy = float(m.cam_fovy[cid])
        W, H = G.CAMERA_W, G.CAMERA_H
        f = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)
        intr = CameraIntrinsics(width=W, height=H, fx=f, fy=f, cx=(W - 1) / 2.0, cy=(H - 1) / 2.0, fovy_deg=fovy)
        R_mj = d.cam_xmat[cid].reshape(3, 3)
        R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])
        extr = CameraExtrinsics(position=tuple(float(v) for v in d.cam_xpos[cid]),
                                rotation=tuple(tuple(float(v) for v in row) for row in R_cv))
        rgb = self._render(name, W, H) if self.render_enabled else None
        return CameraImage(name=name, intrinsics=intr, extrinsics=extr, rgb=rgb)

    def observe(self) -> Observation:
        arm = self._arm
        p, R = arm.tcp_pose()
        return Observation(
            cameras={name: self._camera_image(name) for name in self.cameras},
            joint_positions=[float(v) for v in arm.q()],
            joint_velocities=[float(v) for v in arm.qd()],
            gripper_width=arm.gripper_width(),
            gripper_force=arm.gripper_force(),
            tcp_position=tuple(float(v) for v in p),
            tcp_rotation=tuple(tuple(float(v) for v in row) for row in R),
            wrench_estimate=[float(v) for v in arm.wrench_estimate()],
            sim_time=self.sim_time,
        )

    # ------------------------------------------------------------------ act
    def _result(self, ok: bool, reason: str = "", t0: float = 0.0, max_force: float = 0.0,
                stopped: bool = False, violation: str | None = None) -> ActResult:
        p, R = self._arm.tcp_pose()
        if violation:
            self.stats["safety_violation"] = violation
        return ActResult(ok=ok, reason=reason, sim_seconds=round(self.sim_time - t0, 4),
                         tcp_position=tuple(float(v) for v in p), tcp_yaw_deg=_yaw_of(R),
                         max_force=round(max_force, 3), stopped_on_force=stopped,
                         gripper_width=self._arm.gripper_width(), gripper_force=self._arm.gripper_force(),
                         safety_violation=violation)

    def act(self, command: Command) -> ActResult:
        if isinstance(command, MoveTo):
            return self._move_to(command)
        if isinstance(command, MoveJoints):
            return self._move_joints(command)
        if isinstance(command, Grip):
            return self._grip(command)
        if isinstance(command, Wait):
            t0 = self.sim_time
            self._steps(int(command.seconds / TIMESTEP))
            return self._result(True, "waited", t0)
        raise TypeError(f"unknown command {type(command).__name__}")

    def _force_now(self, F0: np.ndarray) -> float:
        return float(np.linalg.norm(self._arm.wrench_estimate()[:3] - F0))

    def _move_to(self, cmd: MoveTo) -> ActResult:
        arm = self._arm
        t0 = self.sim_time
        target_p = np.asarray(cmd.pose.position, float)
        target_R = tool_rotation(math.radians(cmd.pose.yaw_deg), math.radians(cmd.pose.pitch_deg))
        bounds = self.safety.workspace_bounds
        if not bounds.contains(target_p):
            v = f"target {np.round(target_p, 3).tolist()} is outside the workspace bounds"
            return self._result(False, "safety: " + v, t0, violation=v)
        speed = min(cmd.speed, self.safety.max_speed)
        p0, R0 = arm.fk(self._q_cmd)
        dist = float(np.linalg.norm(target_p - p0))
        rot = rotation_error(target_R, R0)
        ang = float(np.linalg.norm(rot))
        # min-jerk peaks at 1.875 x the mean speed
        T = max(1.875 * dist / speed, ang / 0.8, 0.05)
        n = max(1, int(math.ceil(T / (CTRL_STEPS * TIMESTEP))))
        F0 = arm.wrench_estimate()[:3]
        max_f = 0.0
        if cmd.compliant:
            self._set_compliance(COMPLIANT_GAIN)
        try:
            q = self._q_cmd.copy()
            for i in range(1, n + 1):
                s = _min_jerk(i / n)
                p = p0 + s * (target_p - p0)
                Rm = _rotvec_to_R(s * rot) @ R0
                q_new, ok, res = arm.ik_local(p, Rm, q, iters=30, tol_pos=1e-5, tol_rot=1e-4)
                if not ok and res > 1e-3:
                    self._q_cmd = q
                    return self._result(False, f"unreachable: IK residual {res * 1000:.1f} mm at {np.round(p, 3).tolist()}", t0, max_f)
                if np.max(np.abs(q_new - q)) > 0.03:
                    self._q_cmd = q
                    return self._result(False, "unreachable: the path passes too close to a singularity", t0, max_f)
                q = q_new
                self._q_cmd = q
                self._scene.data.ctrl[arm.act] = q
                self._steps(CTRL_STEPS)
                if i % 5 == 0 or i == n:
                    f = self._force_now(F0)
                    max_f = max(max_f, f)
                    self.stats["max_force"] = max(self.stats["max_force"], f)
                    tcp, _ = arm.tcp_pose()
                    if not bounds.contains(tcp, margin=0.01):
                        self.stop()
                        v = f"TCP {np.round(tcp, 3).tolist()} left the workspace bounds"
                        return self._result(False, "safety: " + v, t0, max_f, violation=v)
                    if f > self.safety.max_force:
                        self.stop()
                        v = f"contact force {f:.1f} N above the limit {self.safety.max_force:.0f} N"
                        return self._result(False, "safety: " + v, t0, max_f, violation=v)
                    if cmd.guard_force is not None and f > cmd.guard_force:
                        self.stop()
                        return self._result(True, f"stopped on contact ({f:.1f} N)", t0, max_f, stopped=True)
                if cmd.timeout_s is not None and self.sim_time - t0 > cmd.timeout_s:
                    self.stop()
                    return self._result(False, "timeout", t0, max_f)
            # settle; outside contact, remove the static error (payload droop) with a slow integral loop
            target_cmd = target_p.copy()
            for _ in range(60):
                self._steps(5)
                tcp, _ = arm.tcp_pose()
                err = target_p - tcp
                f = self._force_now(F0)
                max_f = max(max_f, f)
                if f > self.safety.max_force:
                    self.stop()
                    v = f"contact force {f:.1f} N above the limit {self.safety.max_force:.0f} N"
                    return self._result(False, "safety: " + v, t0, max_f, violation=v)
                if cmd.guard_force is not None and f > cmd.guard_force:
                    self.stop()
                    return self._result(True, f"stopped on contact ({f:.1f} N)", t0, max_f, stopped=True)
                if np.linalg.norm(err) < 1e-4 and np.max(np.abs(arm.qd())) < 0.01:
                    break
                if not cmd.compliant and np.linalg.norm(err) < 0.01:
                    target_cmd = target_cmd + 0.5 * err
                    q_new, ok, _ = arm.ik_local(target_cmd, target_R, self._q_cmd, iters=20)
                    if ok:
                        self._q_cmd = q_new
                        self._scene.data.ctrl[arm.act] = q_new
            return self._result(True, "reached", t0, max_f)
        finally:
            if cmd.compliant:
                self._set_compliance(1.0)

    def _move_joints(self, cmd: MoveJoints) -> ActResult:
        arm = self._arm
        t0 = self.sim_time
        q1 = np.clip(np.asarray(cmd.q, float), arm.lo + 1e-3, arm.hi - 1e-3)
        q0 = self._q_cmd.copy()
        T = max(1.875 * float(np.max(np.abs(q1 - q0))) / cmd.speed, 0.05)
        n = max(1, int(math.ceil(T / (CTRL_STEPS * TIMESTEP))))
        F0 = arm.wrench_estimate()[:3]
        max_f = 0.0
        bounds = self.safety.workspace_bounds
        for i in range(1, n + 1):
            q = q0 + _min_jerk(i / n) * (q1 - q0)
            self._q_cmd = q
            self._scene.data.ctrl[arm.act] = q
            self._steps(CTRL_STEPS)
            if i % 5 == 0:
                f = self._force_now(F0)
                max_f = max(max_f, f)
                tcp, _ = arm.tcp_pose()
                if not bounds.contains(tcp, margin=0.01) or f > self.safety.max_force:
                    self.stop()
                    v = (f"TCP {np.round(tcp, 3).tolist()} left the workspace bounds" if not bounds.contains(tcp, margin=0.01)
                         else f"contact force {f:.1f} N above the limit {self.safety.max_force:.0f} N")
                    return self._result(False, "safety: " + v, t0, max_f, violation=v)
        self._steps(int(0.1 / TIMESTEP))
        return self._result(True, "reached", t0, max_f)

    def _grip(self, cmd: Grip) -> ActResult:
        arm = self._arm
        m, d = self._scene.model, self._scene.data
        t0 = self.sim_time
        if cmd.width is None:
            # each finger gets half the tendon force; the target past closed makes the servo saturate
            m.actuator_forcerange[arm.grip_act] = (-2 * cmd.force, 2 * cmd.force)
            d.ctrl[arm.grip_act] = -0.05
        else:
            m.actuator_forcerange[arm.grip_act] = self._grip_force_range
            d.ctrl[arm.grip_act] = float(np.clip(cmd.width / 2, 0.0, 0.04))
        still = 0
        for _ in range(100):  # up to 1 s
            self._steps(10)
            if np.max(np.abs(d.qvel[arm.fvadr])) < 2e-3:
                still += 1
                if still >= 5:
                    break
            else:
                still = 0
        width = arm.gripper_width()
        if cmd.width is None:
            holding = width > 0.002
            return self._result(True, "closed on an object" if holding else "closed empty", t0)
        return self._result(True, "opened", t0)


def _rotvec_to_R(v: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


__all__ = ["CAMERAS", "DEFAULT_SAFETY", "PrivilegedView", "SimBackend", "VideoRecorder", "Pose"]
