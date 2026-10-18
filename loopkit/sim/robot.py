"""Panda kinematics and low-level control on a rack_cell scene.

Everything here uses the robot's own model and joint encoders (proprioception):
forward kinematics, the Jacobian, damped least-squares IK with a posture bias for
the redundant joint, and the external wrench estimated from joint torques.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import geometry as G
from .rack_cell import ARM_JOINTS, FINGER_JOINTS

# Canonical tool orientation for every rack_cell task at yaw 0: approach (tool z) along world +x,
# finger closing axis (tool y) pointing down, tool x to the robot's right. Columns are the tool x,
# y, z axes in world. Of the two rolls about the approach axis this one keeps every cell pose on
# the joint-5-positive branch with the larger joint margins (see reports/S2.md, layout).
TOOL_FORWARD = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
# The ready posture: TCP at READY_TCP with tool yaw READY_YAW, on the joint-5-positive branch.
READY_TCP = (0.42, -0.08, 0.30)
READY_YAW = math.radians(25.0)
Q_REST = np.array([0.78, -1.15, -1.33, -2.80, 1.92, 2.83, -0.77])
# Rolling the tool 180 deg about its approach axis gives the same grasp (symmetric gripper).
FLIP_ABOUT_APPROACH = np.diag([-1.0, -1.0, 1.0])
# Panda joint limits (Menagerie panda.xml) and a fixed, seeded set of diverse IK seeds.
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
DIVERSE_SEEDS = list(np.random.default_rng(20261017).uniform(JOINT_LO + 0.2, JOINT_HI - 0.2, size=(24, 7)))


def tool_rotation(yaw: float, pitch: float = 0.0) -> np.ndarray:
    """Forward tool orientation yawed by ``yaw`` about world z (and pitched about the tool x)."""
    c, s = math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return G.rot_z(yaw) @ TOOL_FORWARD @ rx


def rotation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) that rotates R_cur onto R_des."""
    R = R_des @ R_cur.T
    cos = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(cos)
    if ang < 1e-9:
        return np.zeros(3)
    if math.pi - ang < 1e-6:
        # 180 deg: pick the axis from the symmetric part
        w, v = np.linalg.eigh((R + np.eye(3)) / 2)
        axis = v[:, int(np.argmax(w))]
        return axis * ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return axis * ang


class Panda:
    """Indices and kinematics for the Panda inside a compiled rack_cell model."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m, self.d = model, data
        self.jids = [model.joint(n).id for n in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = model.jnt_range[self.jids, 0].copy()
        self.hi = model.jnt_range[self.jids, 1].copy()
        self.act = np.array([model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.grip_act = model.actuator("actuator8").id
        self.fqadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in FINGER_JOINTS])
        self.fvadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in FINGER_JOINTS])
        self.tcp = model.site("tcp").id
        self.hand = model.body("hand").id
        self._kd = mujoco.MjData(model)  # scratch data for kinematics, never stepped
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    # ------------------------------------------------------------- state
    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def qd(self) -> np.ndarray:
        return self.d.qvel[self.vadr].copy()

    def set_q(self, q: np.ndarray) -> None:
        """Teleport the arm (reset only) and hold it there."""
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.vadr] = 0.0
        self.d.ctrl[self.act] = q

    def gripper_width(self) -> float:
        """Gap between the fingertip contact pads (encoder based): the two finger joints summed."""
        return float(self.d.qpos[self.fqadr].sum())

    def gripper_force(self) -> float:
        """Grip force per finger estimated from the gripper actuator (N)."""
        return abs(float(self.d.actuator_force[self.grip_act])) / 2.0

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    # ---------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kd = self._kd
        kd.qpos[:] = self.d.qpos
        kd.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, kd)
        mujoco.mj_comPos(self.m, kd)
        return kd.site_xpos[self.tcp].copy(), kd.site_xmat[self.tcp].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray | None = None) -> np.ndarray:
        """6 x 7 TCP Jacobian (linear rows first) at ``q`` (default: the measured joints)."""
        kd = self._kd
        kd.qpos[:] = self.d.qpos
        if q is not None:
            kd.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, kd)
        mujoco.mj_comPos(self.m, kd)
        mujoco.mj_jacSite(self.m, kd, self._jp, self._jr, self.tcp)
        return np.vstack([self._jp[:, self.vadr], self._jr[:, self.vadr]])

    def ik_local(self, pos, R, q0: np.ndarray, iters: int = 300, tol_pos: float = 2e-5,
                 tol_rot: float = 2e-4, posture: np.ndarray | None = None,
                 posture_gain: float = 0.05) -> tuple[np.ndarray, bool, float]:
        """Damped least-squares IK from one seed. Returns (q, converged, position residual m).

        The nullspace term pulls towards ``posture`` (default: the seed) and away from
        joint limits, so a solution stays on the seed's branch.
        """
        q = np.asarray(q0, float).copy()
        target_q = q.copy() if posture is None else np.asarray(posture, float)
        pos = np.asarray(pos, float)
        R = np.asarray(R, float)
        lam2 = 1e-4
        span = self.hi - self.lo
        mid = (self.hi + self.lo) / 2
        err_prev = np.inf
        ep = np.zeros(3)
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = rotation_error(R, Rc)
            err = float(np.linalg.norm(ep) + 0.1 * np.linalg.norm(er))
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(er) < tol_rot:
                return q, True, float(np.linalg.norm(ep))
            lam2 = max(1e-6, lam2 * 0.5) if err < err_prev else min(1e-1, lam2 * 4.0)
            err_prev = err
            J = self.jacobian(q)
            e = np.concatenate([ep, er])
            JJt = J @ J.T + lam2 * np.eye(6)
            Jpinv = J.T @ np.linalg.inv(JJt)
            dq = Jpinv @ e
            # nullspace: stay near the posture target, and away from joint limits
            limit_push = -((q - mid) / (span / 2)) ** 3
            dq += (np.eye(7) - Jpinv @ J) @ (posture_gain * (target_q - q) + 0.05 * limit_push)
            step = np.max(np.abs(dq))
            if step > 0.25:
                dq *= 0.25 / step
            q = np.clip(q + dq, self.lo + 1e-4, self.hi - 1e-4)
        return q, False, float(np.linalg.norm(ep))

    def ik(self, pos, R, q0: np.ndarray | None = None, *, allow_flip: bool = True, iters: int = 300,
           min_margin: float = 0.05) -> tuple[np.ndarray, bool, float]:
        """Robust IK: the seed first, then the rest posture and a fixed set of diverse seeds.

        With ``allow_flip`` the tool may also be rolled 180 deg about its approach axis (the
        gripper is symmetric). Among converged solutions it returns the one closest to the
        seed, preferring solutions at least ``min_margin`` rad inside the joint limits.
        Returns (q, converged, position residual m).
        """
        q0 = self.q() if q0 is None else np.asarray(q0, float)
        targets = [np.asarray(R, float)]
        if allow_flip:
            targets.append(np.asarray(R, float) @ FLIP_ABOUT_APPROACH)
        best = None
        best_res = np.inf
        best_fail = q0
        for i, seed in enumerate([q0, Q_REST, *DIVERSE_SEEDS]):
            for Rt in targets:
                q, ok, res = self.ik_local(pos, Rt, seed, iters=iters, posture=q0 if i == 0 else seed)
                if not ok:
                    if res < best_res:
                        best_res, best_fail = res, q
                    continue
                margin = float(np.min(np.minimum(q - self.lo, self.hi - q)))
                cost = float(np.linalg.norm(q - q0)) + (10.0 if margin < min_margin else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, q, margin)
            if best is not None and i == 0 and best[2] >= min_margin:
                break  # the seed's own branch works: keep it
        if best is None:
            return best_fail, False, float(best_res)
        return best[1], True, 0.0

    # ----------------------------------------------------------- sensing
    def wrench_estimate(self) -> np.ndarray:
        """External force (N) and torque (Nm) at the TCP, from the arm's external joint torques.

        A torque-sensing arm estimates external joint torques as measured torque minus its
        dynamics model (M qdd + C + g); the Panda reports this as tau_ext. Here that estimate is
        the solver's constraint torque on the 7 arm joints, which is what a perfect estimator
        would read, so arm acceleration does not show up as force. F = pinv(J^T) tau_ext.
        """
        J = self.jacobian()
        tau_ext = self.d.qfrc_constraint[self.vadr].copy()
        return np.linalg.pinv(J.T, rcond=1e-3) @ tau_ext

    def contact_force(self) -> float:
        return float(np.linalg.norm(self.wrench_estimate()[:3]))
