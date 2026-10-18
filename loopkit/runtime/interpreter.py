"""The graph interpreter: runs a ProgramGraph on a Backend with no model in the loop.

Semantics (the contract in loopkit/schemas/graph.py):

* A node runs its skill with params resolved from literals, ``{"ref": "<node>.<field>"}``
  (an earlier node's outputs; outputs exist only after that node succeeds) and
  ``{"ref": "task.<param>"}``.
* A failed node is re-run in place up to ``max_retries`` times, then its failure edge is taken.
* A node visited more than ``max_visits`` times fails without running and takes its failure edge.
* ``done`` ends the run as finished, ``abort`` as given up. A safety stop ends the run at once.
* Per-node ``timeout_s`` and the task's ``max_seconds`` are sim-time budgets; a step limit backs
  everything up.

Every node emits ``node_start`` and ``node_end`` trace events.
"""

from __future__ import annotations

import copy
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from ..router.errors import SpendCapExceeded
from ..schemas import ProgramGraph, TaskSpec
from ..skills import REGISTRY, SkillContext, SkillFailure, call_skill

MAX_STEPS = 200


@dataclass
class RunResult:
    status: str                          # done | abort | safety | timeout | step_limit | error
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0
    failing_node: str | None = None
    last_error: str | None = None
    failure_class_hint: str | None = None
    model_calls: int = 0
    model_usd: float = 0.0

    @property
    def reached_terminal(self) -> bool:
        return self.status in ("done", "abort")

    @property
    def finished(self) -> bool:
        return self.status == "done"


class UnresolvedRef(SkillFailure):
    pass


def _lookup(container: Any, path: list[str], ref: str) -> Any:
    cur = container
    for part in path:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise UnresolvedRef(f"reference {ref!r} has no value (missing '{part}')")
    return cur


def resolve_params(value: Any, blackboard: dict[str, dict], task: TaskSpec) -> Any:
    """Replace every {"ref": ...} inside ``value``."""
    if isinstance(value, dict) and set(value) == {"ref"} and isinstance(value["ref"], str):
        ref = value["ref"]
        node, _, rest = ref.partition(".")
        path = rest.split(".") if rest else []
        if node == "task":
            return copy.deepcopy(_lookup(task.params, path, ref))
        if node not in blackboard:
            raise UnresolvedRef(f"reference {ref!r}: node '{node}' has not succeeded yet")
        return copy.deepcopy(_lookup(blackboard[node], path, ref))
    if isinstance(value, dict):
        return {k: resolve_params(v, blackboard, task) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_params(v, blackboard, task) for v in value]
    return value


def _sim_time(backend) -> float:
    t = getattr(backend, "sim_time", None)
    if t is not None:
        return float(t)
    return float(backend.observe().sim_time)


def run_graph(graph: ProgramGraph, backend, task: TaskSpec, *, trace=None, ctx: SkillContext | None = None,
              max_steps: int = MAX_STEPS) -> RunResult:
    """Run ``graph`` on ``backend`` for ``task``. The graph should already be validated."""
    ctx = ctx or SkillContext(backend=backend, task=task, trace=trace)
    t_wall = time.perf_counter()
    t0 = _sim_time(backend)
    result = RunResult(status="error")
    blackboard: dict[str, dict] = {}
    visits: dict[str, int] = {}
    node_id = graph.entry
    steps = 0

    def emit(kind: str, payload: dict) -> None:
        if trace is not None:
            trace.emit(kind, payload)

    while True:
        if steps >= max_steps:
            result.status, result.failing_node = "step_limit", node_id
            result.last_error = f"stopped after {max_steps} node runs"
            break
        steps += 1
        node = graph.node(node_id)
        spec = REGISTRY[node.skill].spec
        visits[node_id] = visits.get(node_id, 0) + 1
        elapsed = _sim_time(backend) - t0
        if elapsed > task.budgets.max_seconds:
            result.status, result.failing_node = "timeout", node_id
            result.last_error = f"sim time {elapsed:.1f} s passed the task budget {task.budgets.max_seconds:.0f} s"
            break
        ok, outputs, error, hint = False, {}, None, None
        over_visits = node.max_visits is not None and visits[node_id] > node.max_visits
        attempts = 1 + graph.retries_for(node_id)
        safety = False
        for attempt in range(attempts):
            if over_visits:
                error = f"visited more than max_visits={node.max_visits} times; not run"
                emit("node_start", {"node_id": node_id, "skill": node.skill, "params": {}, "attempt": attempt,
                                    "skipped": True})
                emit("node_end", {"node_id": node_id, "skill": node.skill, "ok": False, "outputs": {},
                                  "error": error, "sim_seconds": 0.0})
                break
            n_t0 = _sim_time(backend)
            try:
                params = resolve_params(node.params, blackboard, task)
            except SkillFailure as exc:
                params, error = {}, str(exc)
                emit("node_start", {"node_id": node_id, "skill": node.skill, "params": node.params, "attempt": attempt})
                emit("node_end", {"node_id": node_id, "skill": node.skill, "ok": False, "outputs": {}, "error": error,
                                  "sim_seconds": 0.0})
                break  # a missing input will not appear on a retry
            emit("node_start", {"node_id": node_id, "skill": node.skill, "params": params, "attempt": attempt})
            try:
                outputs = call_skill(node.skill, ctx, params)
                ok, error = True, None
            except SkillFailure as exc:
                ok, outputs, error, hint = False, exc.outputs, exc.reason, exc.failure_class
                safety = exc.failure_class == "safety"
            except SpendCapExceeded:
                backend.stop()
                raise  # a spend cap ends the run and the unit; it is not a skill failure
            except Exception as exc:  # noqa: BLE001  (a skill bug must not kill the run silently)
                ok, outputs, error = False, {}, f"{type(exc).__name__}: {exc}"
                hint = "other"
                result.last_error = traceback.format_exc(limit=4)
            dt = _sim_time(backend) - n_t0
            if ok and node.timeout_s is not None and dt > node.timeout_s:
                ok, error = False, f"took {dt:.1f} s of sim time, over the node timeout {node.timeout_s:.1f} s"
                hint = "timeout"
            emit("node_end", {"node_id": node_id, "skill": node.skill, "ok": ok, "outputs": outputs,
                              "error": error, "sim_seconds": round(dt, 4), "attempt": attempt})
            result.history.append({"node": node_id, "skill": node.skill, "ok": ok, "error": error,
                                   "attempt": attempt, "sim_seconds": round(dt, 3), "outputs": outputs})
            if ok or safety:
                break
        if safety:
            result.status, result.failing_node, result.last_error = "safety", node_id, error
            result.failure_class_hint = "safety"
            backend.stop()
            break
        if ok:
            blackboard[node_id] = outputs
            if spec.terminal == "done":
                result.status = "done"
                break
            if spec.terminal == "abort":
                result.status, result.failing_node = "abort", result.failing_node or node_id
                result.last_error = result.last_error or outputs.get("reason") or "aborted"
                break
            node_id = graph.next_node(node_id, "success")
        else:
            result.failing_node, result.last_error = node_id, error
            result.failure_class_hint = hint or result.failure_class_hint
            nxt = graph.next_node(node_id, "failure")
            if nxt is None:
                result.status = "error"
                break
            node_id = nxt
        if node_id is None:
            result.status, result.last_error = "error", "an edge leads nowhere"
            break
    result.outputs = blackboard
    result.sim_seconds = round(_sim_time(backend) - t0, 3)
    result.wall_seconds = round(time.perf_counter() - t_wall, 3)
    result.model_calls, result.model_usd = ctx.model_calls, ctx.model_usd
    return result
