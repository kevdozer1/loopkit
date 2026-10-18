"""One graded episode: run a graph on a fresh SimBackend, score it with the checker.

This is harness code, like the checkers. It reads simulator truth twice: the checker grades the
run, and ``scene_features`` records how the scene was randomized so a failure summary can say what
differed between failing and passing seeds (docs/01_architecture.md, rehearse-and-patch step 2).
Skills never see either. The program package only receives the plain-data record returned here.
"""

from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Any

from ..router.errors import SpendCapExceeded
from ..schemas import ProgramGraph
from . import geometry as G
from .backend import SimBackend
from .checkers import RunInfo, check
from .tasks import task_spec


def scene_features(cfg) -> dict[str, Any]:
    """Scene facts for failure summaries."""
    rx, ry, ryaw = G.RACK_NOMINAL
    x, y, yaw = cfg.rack_pose
    return {
        "rack_dx_mm": round((x - rx) * 1000, 1), "rack_dy_mm": round((y - ry) * 1000, 1),
        "rack_dyaw_deg": round(math.degrees(yaw - ryaw), 2),
        "box_dyaw_deg": round(math.degrees(cfg.box_pose[2] - G.BOX_NOMINAL[2]), 1),
        "plug_dyaw_deg": round(math.degrees(cfg.plug_pose[2] - G.PLUG_NOMINAL[2]), 1),
        "occupied_ports": len(cfg.occupied), "target_slot": cfg.target_slot, "target_port": cfg.target_port,
        "lights": len(cfg.lights),
    }


def run_episode(task_id: str, seed: int, graph_json: dict, trace_root: str | None = None, run_id: str | None = None,
                video: str | None = None, reader_model: str | None = None, chat_kwargs: dict | None = None,
                evidence_dir: str | None = None) -> dict:
    """Run one graph on one seed and score it. Process safe: everything in and out is plain data."""
    from ..runtime.interpreter import run_graph
    from ..skills import SkillContext

    t0 = time.perf_counter()
    task = task_spec(task_id, seed)
    graph = ProgramGraph.model_validate(graph_json)
    trace = None
    if trace_root and run_id:
        from ..traces import TraceWriter

        trace = TraceWriter(run_id, root=Path(trace_root),
                            meta={"task_id": task_id, "seed": seed, "graph_hash": graph.graph_hash(), "kind": "episode"})
    backend = SimBackend(safety=task.safety)
    try:
        backend.reset(seed)
        if video:
            backend.start_video(layout="inspect" if task_id == "T1" else "manip")
        ctx = SkillContext(backend=backend, task=task, trace=trace, reader_model=reader_model,
                           chat_kwargs=dict(chat_kwargs or {}))
        if reader_model:
            from ..router import chat

            ctx.chat = chat
            ctx.chat_kwargs.setdefault("trace", trace)
        res = run_graph(graph, backend, task, trace=trace, ctx=ctx)
        outputs = {}
        if task_id == "T1":
            for node_out in res.outputs.values():
                if isinstance(node_out, dict) and "report" in node_out:
                    outputs = {"report": node_out["report"]}
        run = RunInfo(sim_seconds=res.sim_seconds, timed_out=res.status == "timeout",
                      safety_violation=backend.stats.get("safety_violation"), failing_node=res.failing_node,
                      reached_terminal=res.reached_terminal)
        verdict = check(task, backend, outputs, run, Path(evidence_dir) if evidence_dir else None)
        if trace is not None:
            trace.verdict(verdict, task_id=task_id, seed=seed)
        if video:
            rec = backend.stop_video()
            if rec is not None and rec.frames:
                rec.save(Path(video))
        hist = [{"node": h["node"], "skill": h["skill"], "ok": h["ok"], "error": h["error"],
                 "outputs": {k: v for k, v in (h["outputs"] or {}).items() if k != "report"}} for h in res.history]
        return {
            "task": task_id, "seed": seed, "passed": verdict.passed, "score": round(verdict.score, 4),
            "failure_class": verdict.failure_class, "failing_node": verdict.failing_node or res.failing_node,
            "summary": verdict.summary, "metrics": verdict.evidence.metrics, "status": res.status,
            "last_error": res.last_error if not verdict.passed else None,
            "sim_seconds": res.sim_seconds, "wall_seconds": round(time.perf_counter() - t0, 2),
            "history": hist[-12:], "model_calls": res.model_calls, "model_usd": round(res.model_usd, 6),
            "scene": scene_features(backend.privileged.cfg), "graph_hash": graph.graph_hash(), "error": None,
        }
    except SpendCapExceeded:
        raise  # crosses the process pool to the bake-off runner, which records a budget_stop
    except Exception as exc:  # noqa: BLE001
        return {"task": task_id, "seed": seed, "passed": False, "score": 0.0, "failure_class": "other",
                "failing_node": None, "summary": "episode crashed", "metrics": {}, "status": "error",
                "last_error": f"{type(exc).__name__}: {exc}", "sim_seconds": 0.0,
                "wall_seconds": round(time.perf_counter() - t0, 2), "history": [], "model_calls": 0, "model_usd": 0.0,
                "scene": {}, "graph_hash": None, "error": traceback.format_exc(limit=5)}
    finally:
        backend.close()
        if trace is not None:
            trace.close()


def first_observation_images(task_id: str, seed: int) -> tuple[list, tuple[str, ...]]:
    """The 3 camera images a writer sees: the first Observation of the given seed."""
    b = SimBackend(safety=task_spec(task_id, seed).safety)
    obs = b.reset(seed)
    names = ("overhead", "front", "wrist")
    imgs = [obs.image(n) for n in names]
    b.close()
    return imgs, names
