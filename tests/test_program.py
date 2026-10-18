"""S3: interpreter semantics, the writer's validate-and-repair loop, and rehearse-and-patch, with the FakeModel."""

from __future__ import annotations

import json
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field

from loopkit import paths
from loopkit.program.fake_writer import reference_writer
from loopkit.program.reference import reference_graph
from loopkit.program.rehearse import rehearse_and_patch, scene_differences, summarize_failures
from loopkit.program.writer import write_graph
from loopkit.router.adapters.fake import fake_script
from loopkit.runtime.interpreter import run_graph
from loopkit.schemas import ProgramGraph
from loopkit.sim.tasks import task_spec
from loopkit.skills import SkillFailure, catalog
from loopkit.skills.base import skill


@pytest.fixture(autouse=True)
def _isolated_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))


# ------------------------------------------------------------------------- test-only skills
class Count(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int


_STATE = {"flaky": 0, "calls": []}


@skill("other")
def t_counter(ctx, start: Annotated[int, Field(ge=0, le=100)] = 0) -> Count:
    """Return start + 1."""
    _STATE["calls"].append(("t_counter", start))
    return Count(n=start + 1)


@skill("other")
def t_flaky(ctx, fail_times: Annotated[int, Field(ge=0, le=10)] = 2) -> Count:
    """Fail the first fail_times calls, then succeed."""
    _STATE["flaky"] += 1
    if _STATE["flaky"] <= fail_times:
        raise SkillFailure(f"flaky failure {_STATE['flaky']}")
    return Count(n=_STATE["flaky"])


@skill("other")
def t_never(ctx) -> Count:
    """Always fail."""
    raise SkillFailure("never works")


@skill("other")
def t_unsafe(ctx) -> Count:
    """A safety stop."""
    raise SkillFailure("safety stop: contact force too high", failure_class="safety")


class _Backend:
    sim_time = 0.0

    def stop(self):
        _STATE["calls"].append(("stop",))


def _graph(nodes, edges, entry, terminals, max_retries=0):
    return ProgramGraph.model_validate({"nodes": nodes, "edges": edges, "entry": entry, "max_retries": max_retries,
                                       "terminal_nodes": terminals})


def _reset():
    _STATE.update(flaky=0, calls=[])


def test_interpreter_refs_flow_between_nodes():
    _reset()
    task = task_spec("T2", 3)
    g = _graph([{"id": "a", "skill": "t_counter", "params": {"start": {"ref": "task.target_slot"}}},
                {"id": "b", "skill": "t_counter", "params": {"start": {"ref": "a.n"}}},
                {"id": "ok", "skill": "done", "params": {}}, {"id": "no", "skill": "abort", "params": {}}],
               [{"src": "a", "dst": "b", "on": "success"}, {"src": "a", "dst": "no", "on": "failure"},
                {"src": "b", "dst": "ok", "on": "success"}, {"src": "b", "dst": "no", "on": "failure"}],
               "a", ["ok", "no"])
    res = run_graph(g, _Backend(), task)
    assert res.status == "done"
    assert res.outputs["a"]["n"] == task.params["target_slot"] + 1
    assert res.outputs["b"]["n"] == task.params["target_slot"] + 2


def test_interpreter_retries_in_place():
    _reset()
    task = task_spec("T2", 3)
    g = _graph([{"id": "f", "skill": "t_flaky", "params": {"fail_times": 2}, "max_retries": 2},
                {"id": "ok", "skill": "done", "params": {}}, {"id": "no", "skill": "abort", "params": {}}],
               [{"src": "f", "dst": "ok", "on": "success"}, {"src": "f", "dst": "no", "on": "failure"}],
               "f", ["ok", "no"])
    res = run_graph(g, _Backend(), task)
    assert res.status == "done" and _STATE["flaky"] == 3
    _reset()
    g2 = g.model_copy(update={"nodes": [g.nodes[0].model_copy(update={"max_retries": 1}), *g.nodes[1:]]})
    res = run_graph(g2, _Backend(), task)
    assert res.status == "abort" and res.failing_node == "f"


def test_interpreter_max_visits_ends_a_loop():
    _reset()
    task = task_spec("T2", 3)
    g = _graph([{"id": "try", "skill": "t_never", "params": {}, "max_visits": 3},
                {"id": "again", "skill": "t_counter", "params": {}},
                {"id": "ok", "skill": "done", "params": {}}, {"id": "no", "skill": "abort", "params": {}}],
               [{"src": "try", "dst": "ok", "on": "success"}, {"src": "try", "dst": "again", "on": "failure"},
                {"src": "again", "dst": "try", "on": "success"}, {"src": "again", "dst": "no", "on": "failure"}],
               "try", ["ok", "no"])
    # the loop is bounded only if the bounded node's failure edge can leave it; this one cannot
    assert any(e.code == "unbounded_cycle" for e in g.validate(catalog(), None))
    res = run_graph(g, _Backend(), task, max_steps=50)
    assert res.status == "step_limit"  # the backstop still ends it
    g_ok = _graph([{"id": "try", "skill": "t_never", "params": {}, "max_visits": 3},
                   {"id": "again", "skill": "t_counter", "params": {}},
                   {"id": "ok", "skill": "done", "params": {}}, {"id": "no", "skill": "abort", "params": {}}],
                  [{"src": "try", "dst": "again", "on": "success"}, {"src": "try", "dst": "no", "on": "failure"},
                   {"src": "again", "dst": "try", "on": "success"}, {"src": "again", "dst": "no", "on": "failure"}],
                  "try", ["ok", "no"])
    res = run_graph(g_ok, _Backend(), task)
    assert res.status == "abort" and res.failing_node == "try"


def test_interpreter_safety_stop_ends_the_run():
    _reset()
    task = task_spec("T2", 3)
    g = _graph([{"id": "u", "skill": "t_unsafe", "params": {}, "max_retries": 3},
                {"id": "ok", "skill": "done", "params": {}}, {"id": "no", "skill": "abort", "params": {}}],
               [{"src": "u", "dst": "ok", "on": "success"}, {"src": "u", "dst": "no", "on": "failure"}],
               "u", ["ok", "no"])
    res = run_graph(g, _Backend(), task)
    assert res.status == "safety" and res.failing_node == "u"
    assert ("stop",) in _STATE["calls"]
    assert sum(1 for h in res.history if h["node"] == "u") == 1  # no retries after a safety stop


# ------------------------------------------------------------------------- writer
def test_writer_repairs_until_valid():
    task = task_spec("T2")
    good = reference_graph("T2")
    bad_skill = json.loads(json.dumps(good))
    bad_skill["nodes"][1]["skill"] = "grab"
    with fake_script(["this is not json", bad_skill, good]):
        res = write_graph("fake", task, catalog(), None)
    assert res.ok and res.attempts == 3 and len(res.calls) == 3
    assert res.graph.graph_hash() == ProgramGraph.model_validate(good).graph_hash()


def test_writer_sends_validator_errors_back():
    task = task_spec("T2")
    seen = []

    def script(req):
        seen.append(req.messages[-1]["content"])
        return "{" if len(seen) == 1 else reference_graph("T2")

    with fake_script(script):
        res = write_graph("fake", task, catalog(), None)
    assert res.ok and res.attempts == 2
    assert "does not validate" in seen[1] and "bad_json" in seen[1]


def test_writer_gives_up_after_three_repairs():
    task = task_spec("T2")
    with fake_script(["nope"] * 4):
        res = write_graph("fake", task, catalog(), None)
    assert not res.ok and res.attempts == 4 and res.errors


# ------------------------------------------------------------------------- rehearse and patch
def _stub_episode(task_id, seed, graph_json, *args):
    """Passes when the graph slides the box in; fails in the grasp otherwise. No simulator."""
    skills = {n["skill"] for n in graph_json["nodes"]}
    ok = "slide_into" in skills
    return {"task": task_id, "seed": seed, "passed": ok, "score": 1.0 if ok else 0.0,
            "failure_class": None if ok else "grasp", "failing_node": None if ok else "pick_box",
            "summary": "stub", "metrics": {}, "status": "done" if ok else "abort", "last_error": None if ok else "missed",
            "sim_seconds": 1.0, "wall_seconds": 0.0, "history": [], "model_calls": 0, "model_usd": 0.0,
            "scene": {"target_slot": 1 if seed % 2 else 2, "rack_dyaw_deg": float(seed % 3)}, "graph_hash": None,
            "error": None}


def _no_slide_graph():
    g = reference_graph("T2")
    g["nodes"] = [n for n in g["nodes"] if n["id"] != "slide"]
    g["edges"] = [e for e in g["edges"] if "slide" not in (e["src"], e["dst"])]
    g["edges"] += [{"src": "find_slot", "dst": "let_go", "on": "success"}]
    return g


def test_patch_loop_keeps_the_best_graph_and_logs_rounds():
    task = task_spec("T2")
    prompts = []

    def script(req):
        prompts.append(req.messages[-1]["content"])
        return _no_slide_graph() if len(prompts) == 1 else reference_graph("T2")

    class Trace:
        def __init__(self):
            self.events = []

        def emit(self, kind, payload):
            self.events.append((kind, payload))

    tr = Trace()
    with fake_script(script):
        res = rehearse_and_patch("fake", task, catalog(), [1000, 1001, 1002], trace=tr, episode_fn=_stub_episode,
                                 images=[], image_names=())
    assert res.best_round == 1 and res.best_pass_rate == 1.0
    assert res.stopped_because == "all training runs passed"
    patches = [p for k, p in tr.events if k == "patch"]
    assert [p["round"] for p in patches] == [0, 1]
    assert "failing node" in patches[1]["failure_summary"].lower()
    assert "HOW IT DID IN REHEARSAL" in prompts[1]


def test_patch_loop_stops_without_improvement():
    task = task_spec("T2")
    with fake_script(lambda req: _no_slide_graph()):
        res = rehearse_and_patch("fake", task, catalog(), [1000, 1001], episode_fn=_stub_episode, images=[],
                                 image_names=(), max_rounds=5, stop_after_no_improvement=2)
    assert res.best_pass_rate == 0.0
    assert res.stopped_because == "no improvement for 2 rounds"
    assert len(res.rounds) == 3


def test_failure_summary_mentions_nodes_classes_and_scene():
    results = [_stub_episode("T2", s, _no_slide_graph()) for s in (1, 3, 5)] + \
              [_stub_episode("T2", s, reference_graph("T2")) for s in (2, 4, 6)]
    text = summarize_failures(results)
    assert "3 passed, 3 failed" in text and "grasp 3" in text and "pick_box" in text
    assert any("target slot" in d for d in scene_differences([r for r in results if r["passed"]],
                                                             [r for r in results if not r["passed"]]))


@pytest.mark.skipif(not (paths.panda_dir() / "panda.xml").is_file(), reason="Menagerie Panda not cloned")
def test_rehearsal_with_the_simulator_and_reference_writer():
    task = task_spec("T2")
    with fake_script(reference_writer):
        res = rehearse_and_patch("fake", task, catalog(), [100, 101], workers=2, max_rounds=1)
    assert res.best_pass_rate == 1.0 and res.writer_calls == 1
