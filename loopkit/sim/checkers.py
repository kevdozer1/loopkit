"""Privileged checkers for T1, T2 and T3.

A checker reads simulator truth through ``backend.privileged`` (body poses, contacts)
and returns a Verdict with metrics, a failure class and one evidence image. Failure
classes follow docs/02_task_suite.md: a wrong port or slot is ``perception``, a dropped
part is ``grasp``, a lateral error or jam is ``alignment``, a part left in the hand or a
program that never ends is ``stopping``, a budget hit is ``timeout``, a bounds or force
breach is ``safety``. A program that never touched the part gets ``other``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..runtime.backend import Wait
from ..schemas import Evidence, TaskSpec, Verdict
from . import geometry as G

PASS_POS_M = 0.003
PASS_YAW_DEG = 2.0
PASS_DEPTH_M = 0.008
PASS_LATERAL_M = 0.0005
FINGER_PREFIXES = ("left_finger/", "right_finger/", "hand/")


@dataclass
class RunInfo:
    """How the program run ended, as seen by the interpreter (or an oracle runner)."""

    sim_seconds: float = 0.0
    timed_out: bool = False
    safety_violation: str | None = None
    failing_node: str | None = None
    reached_terminal: bool = True


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _yaw_deg(R: np.ndarray) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def _evidence(backend, task: TaskSpec, camera: str, evidence_dir: Path | None) -> list[str]:
    if evidence_dir is None:
        return []
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{task.id}_seed{task.seed}_{camera}.png"
    Image.fromarray(backend.privileged.render(camera)).save(path)
    return [str(path)]


def _common_failure(run: RunInfo | None) -> str | None:
    if run is None:
        return None
    if run.safety_violation:
        return "safety"
    if run.timed_out:
        return "timeout"
    return None


# ------------------------------------------------------------------------------- T1
def truth_report(backend) -> dict[str, dict[str, Any]]:
    cfg = backend.privileged.cfg
    occ = set(cfg.occupied)
    return {str(n): {"occupied": n in occ, "led": cfg.led_states[n - 1]} for n in range(1, G.N_PORTS + 1)}


def check_t1(task: TaskSpec, backend, outputs: dict | None, run: RunInfo | None = None,
             evidence_dir: Path | None = None) -> Verdict:
    truth = truth_report(backend)
    report = (outputs or {}).get("report")
    right, wrong = 0, []
    if isinstance(report, dict):
        for n, fields in truth.items():
            got = report.get(n) if isinstance(report.get(n), dict) else report.get(int(n)) if isinstance(report.get(int(n), None), dict) else {}
            for key in ("occupied", "led"):
                if isinstance(got, dict) and got.get(key) == fields[key]:
                    right += 1
                else:
                    wrong.append(f"{n}.{key}")
    score = right / (2 * G.N_PORTS)
    passed = right == 2 * G.N_PORTS
    failure = None if passed else (_common_failure(run) or ("perception" if isinstance(report, dict) else "other"))
    metrics = {"fields_right": right, "fields_total": 2 * G.N_PORTS, "wrong_fields": wrong[:24],
               "report_present": isinstance(report, dict)}
    return Verdict(passed=passed, score=1.0 if passed else min(score, 0.999), failure_class=failure,
                   evidence=Evidence(metrics=metrics, image_paths=_evidence(backend, task, "front", evidence_dir)),
                   task_id=task.id, seed=task.seed, checker_id=task.checker_id,
                   failing_node=None if passed else (run.failing_node if run else None),
                   summary=f"{right}/24 fields right")


# ------------------------------------------------------------------------------- T2
def box_metrics(backend, slot: int) -> dict[str, Any]:
    pv = backend.privileged
    T = pv.body_pose("box")
    Tr = pv.rack_T()
    p_rack = Tr[:3, :3].T @ (T[:3, 3] - Tr[:3, 3])
    yaw_rel = _wrap_deg(_yaw_deg(T[:3, :3]) - _yaw_deg(Tr[:3, :3]))
    seated = G.box_seated_rack(slot)
    err = p_rack - seated
    other = G.box_seated_rack(3 - slot) if G.N_SLOTS == 2 else None
    contacts = pv.body_contacts("box")
    on_l = any(o == f"rail{slot}l_flange" for _, o, f in contacts)
    on_r = any(o == f"rail{slot}r_flange" for _, o, f in contacts)
    fingers = any(o.startswith(FINGER_PREFIXES) for _, o, f in contacts)
    bx, by, byaw = pv.cfg.box_pose
    start = np.array([bx, by, G.BOX_REST_Z])
    return {
        "pos_err_mm": round(float(np.linalg.norm(err)) * 1000, 3),
        "depth_err_mm": round(float(err[0]) * 1000, 3),
        "lateral_err_mm": round(float(err[1]) * 1000, 3),
        "height_err_mm": round(float(err[2]) * 1000, 3),
        "yaw_err_deg": round(abs(yaw_rel), 3),
        "on_left_rail": on_l,
        "on_right_rail": on_r,
        "touching_gripper": fingers,
        "gripper_width_mm": round(pv.gripper_width() * 1000, 2),
        "dist_other_slot_mm": round(float(np.linalg.norm(p_rack - other)) * 1000, 2) if other is not None else None,
        "moved_from_stand_mm": round(float(np.linalg.norm(T[:3, 3] - start)) * 1000, 2),
        "box_world": [round(float(v), 4) for v in T[:3, 3]],
        "box_rack": [round(float(v), 4) for v in p_rack],
    }


def check_t2(task: TaskSpec, backend, outputs: dict | None = None, run: RunInfo | None = None,
             evidence_dir: Path | None = None, settle_s: float = 0.5) -> Verdict:
    slot = int(task.params["target_slot"])
    backend.act(Wait(seconds=settle_s))
    m = box_metrics(backend, slot)
    pos_ok = m["pos_err_mm"] <= PASS_POS_M * 1000
    yaw_ok = m["yaw_err_deg"] <= PASS_YAW_DEG
    rails_ok = m["on_left_rail"] and m["on_right_rail"]
    released = not m["touching_gripper"]
    passed = pos_ok and yaw_ok and rails_ok and released and _common_failure(run) is None
    if passed:
        score, failure = 1.0, None
    else:
        near = m["pos_err_mm"] <= 60.0
        score = 0.0
        if near:
            score = (0.4 * _clamp01(1 - m["pos_err_mm"] / 60.0) + 0.2 * _clamp01(1 - m["yaw_err_deg"] / 10.0)
                     + 0.1 * m["on_left_rail"] + 0.1 * m["on_right_rail"] + 0.2 * released)
        score = min(score, 0.9)
        failure = _common_failure(run)
        if failure is None:
            if m["dist_other_slot_mm"] is not None and m["dist_other_slot_mm"] <= 40.0:
                failure = "perception"
            elif m["moved_from_stand_mm"] < 5.0:
                failure = "other"
            elif near and pos_ok and yaw_ok and rails_ok and not released:
                failure = "stopping"
            elif near:
                failure = "alignment"
            elif not m["touching_gripper"]:
                failure = "grasp"
            elif run is not None and not run.reached_terminal:
                failure = "stopping"
            else:
                failure = "alignment"
    m["passed_checks"] = {"position": pos_ok, "yaw": yaw_ok, "on_both_rails": rails_ok, "released": released}
    return Verdict(passed=passed, score=score, failure_class=failure,
                   evidence=Evidence(metrics=m, image_paths=_evidence(backend, task, "closeup", evidence_dir)),
                   task_id=task.id, seed=task.seed, checker_id=task.checker_id,
                   failing_node=None if passed else (run.failing_node if run else None),
                   summary=f"pos {m['pos_err_mm']:.1f} mm, yaw {m['yaw_err_deg']:.2f} deg, rails {int(m['on_left_rail'])}{int(m['on_right_rail'])}, "
                           f"released {int(released)}")


# ------------------------------------------------------------------------------- T3
def plug_metrics(backend, port: int) -> dict[str, Any]:
    pv = backend.privileged
    T = pv.body_pose("plug")
    Tr = pv.rack_T()
    tip_w = T[:3, 3] + T[:3, :3] @ np.array([G.PLUG_BODY[0] / 2, 0.0, 0.0])
    tip = Tr[:3, :3].T @ (tip_w - Tr[:3, 3])
    rear = Tr[:3, :3].T @ (T[:3, 3] - T[:3, :3] @ np.array([G.PLUG_BODY[0] / 2, 0.0, 0.0]) - Tr[:3, 3])
    mouth_x = -G.PANEL_THICK
    depth = float(tip[0] - mouth_x)
    lat_tip = float(math.hypot(tip[1] - G.port_y(port), tip[2] - G.PORT_Z))
    in_port = None
    for n in range(1, G.N_PORTS + 1):
        if tip[0] - mouth_x > 0.002 and math.hypot(tip[1] - G.port_y(n), tip[2] - G.PORT_Z) < 0.004:
            in_port = n
    contacts = pv.body_contacts("plug")
    fingers = any(o.startswith(FINGER_PREFIXES) for _, o, f in contacts)
    px, py, pyaw = pv.cfg.plug_pose
    start = np.array([px, py, G.PLUG_REST_Z])
    return {
        "depth_mm": round(depth * 1000, 3),
        "lateral_err_mm": round(lat_tip * 1000, 3),
        "rear_lateral_err_mm": round(float(math.hypot(rear[1] - G.port_y(port), rear[2] - G.PORT_Z)) * 1000, 3),
        "in_port": in_port,
        "touching_gripper": fingers,
        "gripper_width_mm": round(pv.gripper_width() * 1000, 2),
        "moved_from_holder_mm": round(float(np.linalg.norm(T[:3, 3] - start)) * 1000, 2),
        "plug_world": [round(float(v), 4) for v in T[:3, 3]],
        "tip_rack": [round(float(v), 4) for v in tip],
    }


def check_t3(task: TaskSpec, backend, outputs: dict | None = None, run: RunInfo | None = None,
             evidence_dir: Path | None = None, settle_s: float = 1.0) -> Verdict:
    port = int(task.params["target_port"])
    backend.act(Wait(seconds=settle_s))  # "still seated after 1 s"
    m = plug_metrics(backend, port)
    depth_ok = m["depth_mm"] >= PASS_DEPTH_M * 1000
    lat_ok = m["lateral_err_mm"] <= PASS_LATERAL_M * 1000
    released = not m["touching_gripper"]
    passed = depth_ok and lat_ok and released and m["in_port"] == port and _common_failure(run) is None
    if passed:
        score, failure = 1.0, None
    else:
        near = m["lateral_err_mm"] <= 10.0 and m["depth_mm"] > -5.0
        score = 0.0
        if near:
            score = 0.5 * _clamp01(m["depth_mm"] / 8.0) + 0.3 * _clamp01(1 - m["lateral_err_mm"] / 2.0) + 0.2 * released
        score = min(score, 0.9)
        failure = _common_failure(run)
        if failure is None:
            if m["in_port"] is not None and m["in_port"] != port:
                failure = "perception"
            elif m["moved_from_holder_mm"] < 3.0:
                failure = "other"
            elif near and depth_ok and lat_ok and not released:
                failure = "stopping"
            elif near:
                failure = "alignment"
            elif not m["touching_gripper"]:
                failure = "grasp"
            elif run is not None and not run.reached_terminal:
                failure = "stopping"
            else:
                failure = "alignment"
    m["passed_checks"] = {"depth": depth_ok, "lateral": lat_ok, "released": released, "right_port": m["in_port"] == port}
    return Verdict(passed=passed, score=score, failure_class=failure,
                   evidence=Evidence(metrics=m, image_paths=_evidence(backend, task, "closeup", evidence_dir)),
                   task_id=task.id, seed=task.seed, checker_id=task.checker_id,
                   failing_node=None if passed else (run.failing_node if run else None),
                   summary=f"depth {m['depth_mm']:.1f} mm, lateral {m['lateral_err_mm']:.2f} mm, port {m['in_port']}, "
                           f"released {int(released)}")


CHECKERS = {"T1_inspect": check_t1, "T2_rack_slot": check_t2, "T3_plug_insert": check_t3}


def check(task: TaskSpec, backend, outputs: dict | None = None, run: RunInfo | None = None,
          evidence_dir: Path | None = None) -> Verdict:
    return CHECKERS[task.checker_id](task, backend, outputs, run, evidence_dir)
