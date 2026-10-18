"""Oracle programs for T1 to T3 (privileged), the do-nothing program, and a runner.

The oracles read simulator truth through ``backend.privileged``, including the part's
pose in the hand after the grasp, so they show the tasks are solvable with the scene's
contact settings. They are never shown to a writer. The do-nothing program must score 0.

``uv run python -m loopkit.sim.oracles --task T2 --seeds 0-49`` prints a pass rate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..paths import runs_dir, videos_dir
from ..runtime.backend import Grip, MoveTo, Pose, Wait
from . import geometry as G
from .backend import SimBackend
from .checkers import RunInfo, check, truth_report
from .tasks import task_spec

TRANSIT_Z = 0.42  # high enough that the wrist clears the parts on the tray
STAGE_BACK = 0.12  # staging point this far in front of the pre-insert pose, clear of the panel and its plugs


class OracleFailed(RuntimeError):
    pass


def _yaw(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _tool_yaw(R_tcp: np.ndarray) -> float:
    """Approach yaw of the tool frame (heading of its z axis)."""
    return math.atan2(R_tcp[1, 2], R_tcp[0, 2])


def _do(backend: SimBackend, cmd, label: str, log: list) -> None:
    r = backend.act(cmd)
    log.append({"step": label, "ok": r.ok, "reason": r.reason, "sim_s": r.sim_seconds, "max_force": r.max_force})
    if not r.ok:
        raise OracleFailed(f"{label}: {r.reason}")


def _move(backend, p, yaw, label, log, speed=0.12, **kw) -> None:
    _do(backend, MoveTo(pose=Pose(position=tuple(float(v) for v in p), yaw_deg=math.degrees(yaw)), speed=speed, **kw),
        label, log)


def _tcp(backend) -> tuple[np.ndarray, np.ndarray]:
    d = backend.privileged.data
    sid = backend.privileged.model.site("tcp").id
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def _pick(backend, body: str, grasp_offset: float, width_open: float, force: float, log: list,
          pre: float = 0.07) -> None:
    """Grasp ``body`` with the TCP ``grasp_offset`` along its x axis from its centre, lift and back off."""
    pv = backend.privileged
    T = pv.body_pose(body)
    yaw = _yaw(T[:3, :3])
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    grasp = T[:3, 3] + grasp_offset * fwd
    _do(backend, Grip(width=width_open), "open", log)
    _move(backend, grasp - pre * fwd, yaw, "pre_grasp", log, speed=0.15)
    _move(backend, grasp, yaw, "approach", log, speed=0.05)
    _do(backend, Grip(force=force), "close", log)
    _move(backend, grasp + [0, 0, 0.03], yaw, "lift", log, speed=0.05)
    _move(backend, grasp + [0, 0, 0.03] - 0.06 * fwd, yaw, "back_off", log, speed=0.08)
    _move(backend, grasp + [0, 0, TRANSIT_Z - grasp[2]] - 0.06 * fwd, yaw, "rise", log, speed=0.10)


def _held_offset(backend, body: str, point_in_body) -> tuple[np.ndarray, float]:
    """A body point's position in the tool frame, and the body's yaw minus the tool's yaw."""
    pv = backend.privileged
    T = pv.body_pose(body)
    p_tcp, R_tcp = _tcp(backend)
    pt_w = T[:3, 3] + T[:3, :3] @ np.asarray(point_in_body, float)
    return R_tcp.T @ (pt_w - p_tcp), _yaw(T[:3, :3]) - _tool_yaw(R_tcp)


def _tool_R(yaw: float) -> np.ndarray:
    from .robot import tool_rotation

    return tool_rotation(yaw)


def oracle_t2(backend: SimBackend, task) -> dict:
    log: list = []
    slot = int(task.params["target_slot"])
    _pick(backend, "box", -G.BOX_SIZE[0] / 2 + 0.012, 0.075, 70.0, log)
    offset, dyaw = _held_offset(backend, "box", (0.0, 0.0, 0.0))
    pv = backend.privileged
    Tr = pv.rack_T()
    ryaw = _yaw(Tr[:3, :3])
    rfwd = Tr[:3, 0]
    tool_yaw = ryaw - dyaw
    R = _tool_R(tool_yaw)
    seat = pv.rack_to_world(G.box_seated_rack(slot))

    def tcp_for(box_center):
        return box_center - R @ offset

    up = np.array([0.0, 0.0, 0.004])
    pre = tcp_for(seat - 0.16 * rfwd + up)
    stage = pre - STAGE_BACK * rfwd
    _move(backend, [stage[0], stage[1], TRANSIT_Z], tool_yaw, "transit", log, speed=0.15)
    _move(backend, stage, tool_yaw, "descend", log, speed=0.08)
    _move(backend, pre, tool_yaw, "to_pre_insert", log, speed=0.06)
    _move(backend, tcp_for(seat - 0.085 * rfwd + up), tool_yaw, "slide_onto_rails", log, speed=0.04)
    _move(backend, tcp_for(seat - 0.085 * rfwd), tool_yaw, "lower_onto_rails", log, speed=0.02)
    _move(backend, tcp_for(seat + 0.002 * rfwd), tool_yaw, "push_to_stop", log, speed=0.03, compliant=True, guard_force=30.0)
    _do(backend, Grip(width=0.075), "release", log)
    p, _ = _tcp(backend)
    _move(backend, p - 0.08 * rfwd, tool_yaw, "retreat", log, speed=0.06)
    return {"log": log}


def oracle_t3(backend: SimBackend, task) -> dict:
    log: list = []
    port = int(task.params["target_port"])
    boot_center = -G.PLUG_BODY[0] / 2 - G.PLUG_BOOT[0] / 2
    _pick(backend, "plug", boot_center, 0.03, 30.0, log, pre=0.06)
    tip_tool, dyaw = _held_offset(backend, "plug", (G.PLUG_BODY[0] / 2, 0.0, 0.0))
    pv = backend.privileged
    Tr = pv.rack_T()
    ryaw = _yaw(Tr[:3, :3])
    rfwd = Tr[:3, 0]
    tool_yaw = ryaw - dyaw
    R = _tool_R(tool_yaw)
    mouth = pv.rack_to_world(G.port_mouth_rack(port))

    def tcp_for(tip):
        return tip - R @ tip_tool

    pre = tcp_for(mouth - 0.025 * rfwd)
    stage = pre - STAGE_BACK * rfwd
    _move(backend, [stage[0], stage[1], TRANSIT_Z], tool_yaw, "transit", log, speed=0.15)
    _move(backend, stage, tool_yaw, "descend", log, speed=0.08)
    _move(backend, pre, tool_yaw, "to_pre_insert", log, speed=0.06)
    _move(backend, tcp_for(mouth - 0.004 * rfwd), tool_yaw, "to_mouth", log, speed=0.03)
    _move(backend, tcp_for(mouth + 0.0125 * rfwd), tool_yaw, "insert", log, speed=0.01, compliant=True, guard_force=20.0)
    _do(backend, Grip(width=0.03), "release", log)
    p, _ = _tcp(backend)
    _move(backend, p - 0.06 * rfwd, tool_yaw, "retreat", log, speed=0.05)
    return {"log": log}


def oracle_t1(backend: SimBackend, task) -> dict:
    backend.act(Wait(seconds=2.0))  # hold still so a recording shows the panel
    return {"report": truth_report(backend)}


def do_nothing(backend: SimBackend, task) -> dict:
    return {}


ORACLES: dict[str, Callable] = {"T1": oracle_t1, "T2": oracle_t2, "T3": oracle_t3}


def run_program(task_id: str, seed: int, program: Callable, video: Path | None = None,
                evidence_dir: Path | None = None) -> dict:
    """Run ``program(backend, task)`` on one seed and score it. Returns a JSON-safe record."""
    task = task_spec(task_id, seed)
    backend = SimBackend(safety=task.safety, render=False)
    wall = time.perf_counter()
    backend.reset(seed)
    if video is not None:
        backend.start_video(layout="inspect" if task_id == "T1" else "manip")
    t0 = backend.sim_time
    outputs, error = {}, None
    try:
        outputs = program(backend, task) or {}
    except OracleFailed as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    sim_s = backend.sim_time - t0
    run = RunInfo(sim_seconds=sim_s, timed_out=sim_s > task.budgets.max_seconds,
                  safety_violation=backend.stats.get("safety_violation"))
    verdict = check(task, backend, outputs, run, evidence_dir)
    record = {"task": task_id, "seed": seed, "passed": verdict.passed, "score": round(verdict.score, 4),
              "failure_class": verdict.failure_class, "summary": verdict.summary, "sim_seconds": round(sim_s, 3),
              "wall_seconds": round(time.perf_counter() - wall, 2), "error": error,
              "metrics": verdict.evidence.metrics, "image_paths": verdict.evidence.image_paths,
              "steps": outputs.get("log") if isinstance(outputs, dict) else None}
    if video is not None:
        rec = backend.stop_video()
        if rec is not None and rec.frames:
            record["video"] = str(rec.save(video))
    backend.close()
    return record


def _run_one(args) -> dict:
    task_id, seed, which, video_dir, evidence_dir = args
    program = do_nothing if which == "nothing" else ORACLES[task_id]
    video = Path(video_dir) / f"{task_id}_{which}_seed{seed}.mp4" if video_dir else None
    return run_program(task_id, seed, program, video, Path(evidence_dir) if evidence_dir else None)


def run_many(task_id: str, seeds: list[int], which: str = "oracle", workers: int = 12,
             video_dir: Path | None = None, evidence_dir: Path | None = None) -> list[dict]:
    jobs = [(task_id, s, which, str(video_dir) if video_dir else None, str(evidence_dir) if evidence_dir else None)
            for s in seeds]
    if workers <= 1:
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run oracle or do-nothing programs on rack_cell seeds")
    ap.add_argument("--task", default="T2")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--program", choices=["oracle", "nothing"], default="oracle")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--video", action="store_true", help="write an mp4 per seed to LOOPKIT_DATA/videos")
    ap.add_argument("--out", default=None, help="JSON results file (default under LOOPKIT_DATA/runs/oracles)")
    args = ap.parse_args()
    seeds = _parse_seeds(args.seeds)
    t = time.perf_counter()
    results = run_many(args.task, seeds, args.program, args.workers, videos_dir() if args.video else None,
                       runs_dir() / "oracles" / "evidence")
    passed = sum(r["passed"] for r in results)
    print(f"{args.task} {args.program}: {passed}/{len(results)} passed, mean score "
          f"{np.mean([r['score'] for r in results]):.3f}, wall {time.perf_counter() - t:.1f}s", flush=True)
    for r in results:
        if not r["passed"] or args.program == "nothing":
            print(f"  seed {r['seed']}: {r['failure_class']} score {r['score']} | {r['summary']} | {r['error'] or ''}"[:300], flush=True)
    out = Path(args.out) if args.out else runs_dir() / "oracles" / f"{args.task}_{args.program}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("results:", out, flush=True)


if __name__ == "__main__":
    main()
