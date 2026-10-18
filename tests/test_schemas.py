"""Adversarial tests for loopkit's schemas (docs/01_architecture.md, Schemas section).

What is covered:

* JSON round trips for TaskSpec, SkillSpec, ProgramGraph, Observation (pixels
  stay out of JSON), Verdict and TraceEvent of every kind.
* JSON Schema export, the Verdict consistency rules, TraceEvent payload checks
  and the Observation privilege boundary (cameras and proprioception only).
* ProgramGraph.validate(catalog, task) against a small realistic catalog: one
  test per bad-graph kind in the spec, the other error codes, edge cases, a
  brute-force cross-check on random graphs and a seeded fuzz.

Near the bottom, "fixed defects" holds the tests that were strict xfails until the
S1 schema fixes (each docstring names the defect), and "S1 schema decisions" has
the tests for each numbered decision of that fix, which is the contract the S3
interpreter implements: loop bounds, refs only to nodes that have succeeded,
closed params, error text, strict graph numbers and the JSON Schema export.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import random
import re
import time
from typing import Any, Literal, get_args

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from loopkit.schemas import (
    AABB,
    EVENT_KINDS,
    FAILURE_CLASSES,
    GRAPH_FORMAT_HINT,
    PAYLOAD_MODELS,
    Budgets,
    CameraExtrinsics,
    CameraImage,
    CameraIntrinsics,
    Edge,
    Evidence,
    GraphError,
    ModelCallPayload,
    Node,
    NodeEndPayload,
    Observation,
    PatchPayload,
    ProgramGraph,
    Safety,
    SkillSpec,
    TaskSpec,
    TraceEvent,
    Verdict,
    VerdictPayload,
    as_catalog,
    catalog_text,
    describe,
    format_errors,
    is_ref,
    iter_refs,
    normalize_params,
    parse_graph,
    validate_graph,
)
from loopkit.schemas.export import JSON_DIR, MODELS, export_json_schemas
from loopkit.schemas.jsonschema_lite import check as check_schema

# ====================================================================== catalog
VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    """A JSON Schema object the way skill params and returns are generated."""
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


CATALOG = [
    SkillSpec(
        name="locate", category="perception",
        docstring="Find a part or a fixture feature in one camera image and return its pose in the base frame.",
        params=obj({
            "target": {"type": "string", "enum": ["box", "plug", "slot", "port"]},
            "index": {"type": "integer", "minimum": 0, "maximum": 11, "description": "slot or port index"},
            "camera": {"type": "string", "enum": ["overhead", "front", "wrist"], "default": "front"},
        }, ("target",)),
        returns=obj({
            "position": VEC3,
            "yaw": {"type": "number"},
            "width_mm": {"type": "integer", "minimum": 0},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "found": {"type": "boolean"},
        }, ("position", "yaw", "width_mm", "found")),
        postconditions=["position and yaw are in the base frame"],
    ),
    SkillSpec(
        name="approach", category="motion",
        docstring="Move the tool centre point to a standoff above a pose, then hold.",
        params=obj({
            "position": VEC3,
            "yaw": {"type": "number", "default": 0.0},
            "standoff": {"type": "number", "minimum": 0.0, "maximum": 0.2, "default": 0.05},
            "speed": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 0.25, "default": 0.1},
        }, ("position",)),
        returns=obj({"tcp_position": VEC3, "reached": {"type": "boolean"}}, ("tcp_position", "reached")),
    ),
    SkillSpec(
        name="grasp", category="manipulation", docstring="Close the gripper on the part under the hand.",
        params=obj({
            "width_mm": {"type": "number", "minimum": 0, "maximum": 80},
            "force": {"type": "number", "minimum": 0, "maximum": 70, "default": 20},
        }, ("width_mm",)),
        returns=obj({"holding": {"type": "boolean"}, "width_mm": {"type": "number"}}, ("holding",)),
        preconditions=["gripper open", "hand at the grasp standoff"],
        postconditions=["part held"],
    ),
    SkillSpec(
        name="release", category="manipulation", docstring="Open the gripper.",
        params=obj({"open_mm": {"type": "number", "minimum": 0, "maximum": 80, "default": 80}}),
        returns=obj({"released": {"type": "boolean"}}, ("released",)),
    ),
    SkillSpec(name="done", category="terminal", terminal="done", docstring="End the program: the goal is met.",
              params=obj({"result": {"type": "object"}}), returns=obj({})),
    SkillSpec(name="abort", category="terminal", terminal="abort", docstring="End the program: give up.",
              params=obj({"reason": {"type": "string", "minLength": 1}}, ("reason",)), returns=obj({})),
    # In the catalog but not allowed for T2: the panel reader calls a model.
    SkillSpec(name="inspect_panel", category="perception", big_model=True,
              docstring="Read every port's LED and occupancy from the front camera.",
              params=obj({}), returns=obj({"ports": {"type": "object"}})),
]
SPECS = {spec.name: spec for spec in CATALOG}

TASK = TaskSpec(
    id="T2", title="rack_slot", goal_text="Put the 1U box in slot n of the rack.",
    end_condition="box in slot n: position error at most 3 mm, yaw error at most 2 deg, resting on both rails, "
                  "gripper released",
    checker_id="rack_slot_v1", scene_id="rack_cell",
    allowed_skills=["locate", "approach", "grasp", "release", "done", "abort"],
    nominal_geometry={"rack_origin": [0.62, 0.0, 0.0], "rack_yaw_deg": 0.0, "slot_heights_m": [0.1, 0.14445],
                      "tray_origin": [0.45, -0.3, 0.02]},
    budgets=Budgets(max_big_model_calls=0, max_usd=0.0, max_seconds=60.0),
    safety=Safety(workspace_bounds=AABB(min=(0.1, -0.6, 0.0), max=(0.95, 0.6, 0.9)), max_speed=0.25, max_force=70.0),
    params_schema={"slot": {"type": "integer", "minimum": 0, "maximum": 1},
                   "grip_force": {"type": "integer", "minimum": 0, "maximum": 70}},
    params={"slot": 1, "grip_force": 30},
    seed=7,
)


def make_task(**updates: Any) -> TaskSpec:
    return TaskSpec.model_validate({**TASK.model_dump(), **updates})


def edge(src: str, dst: str, on: str) -> dict[str, str]:
    return {"src": src, "dst": dst, "on": on}


def good_graph() -> dict[str, Any]:
    """T2: find the box, pick it, find slot n, place it, let go."""
    return {
        "nodes": [
            {"id": "locate_box", "skill": "locate", "params": {"target": "box", "camera": "overhead"}},
            {"id": "approach_box", "skill": "approach",
             "params": {"position": {"ref": "locate_box.position"}, "yaw": {"ref": "locate_box.yaw"}, "standoff": 0.05}},
            {"id": "grasp_box", "skill": "grasp",
             "params": {"width_mm": {"ref": "locate_box.width_mm"}, "force": {"ref": "task.grip_force"}}},
            {"id": "locate_slot", "skill": "locate", "params": {"target": "slot", "index": {"ref": "task.slot"}}},
            {"id": "approach_slot", "skill": "approach",
             "params": {"position": {"ref": "locate_slot.position"}, "yaw": {"ref": "locate_slot.yaw"},
                        "standoff": 0.0, "speed": 0.05}},
            {"id": "release_box", "skill": "release", "params": {"open_mm": 60}},
            {"id": "done", "skill": "done", "params": {}},
            {"id": "abort", "skill": "abort", "params": {"reason": "could not seat the box"}},
        ],
        "edges": [
            edge("locate_box", "approach_box", "success"), edge("locate_box", "abort", "failure"),
            edge("approach_box", "grasp_box", "success"), edge("approach_box", "abort", "failure"),
            edge("grasp_box", "locate_slot", "success"), edge("grasp_box", "abort", "failure"),
            edge("locate_slot", "approach_slot", "success"), edge("locate_slot", "abort", "failure"),
            edge("approach_slot", "release_box", "success"), edge("approach_slot", "abort", "failure"),
            edge("release_box", "done", "success"), edge("release_box", "abort", "failure"),
        ],
        "entry": "locate_box",
        "max_retries": 1,
        "terminal_nodes": ["done", "abort"],
    }


def node(g: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(n for n in g["nodes"] if n["id"] == node_id)


def set_edge(g: dict[str, Any], src: str, on: str, dst: str) -> None:
    for e in g["edges"]:
        if e["src"] == src and e["on"] == on:
            e["dst"] = dst
            return
    g["edges"].append(edge(src, dst, on))


def drop_edge(g: dict[str, Any], src: str, on: str) -> None:
    g["edges"] = [e for e in g["edges"] if not (e["src"] == src and e["on"] == on)]


def _rename_refs(value: Any, old: str, new: str) -> Any:
    if is_ref(value) and value["ref"].split(".", 1)[0] == old:
        return {"ref": new + "." + value["ref"].split(".", 1)[1]}
    if isinstance(value, dict):
        return {k: _rename_refs(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename_refs(v, old, new) for v in value]
    return value


def rename(g: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    """Rename a node everywhere: its id, edges, entry, terminal_nodes and refs to it."""
    for n in g["nodes"]:
        if n["id"] == old:
            n["id"] = new
        n["params"] = _rename_refs(n.get("params", {}), old, new)
    for e in g["edges"]:
        e["src"] = new if e["src"] == old else e["src"]
        e["dst"] = new if e["dst"] == old else e["dst"]
    g["terminal_nodes"] = [new if t == old else t for t in g["terminal_nodes"]]
    g["entry"] = new if g["entry"] == old else g["entry"]
    return g


def errors_for(data: dict[str, Any], task: TaskSpec | None = TASK, catalog: Any = CATALOG) -> list[GraphError]:
    """ProgramGraph.validate(catalog, task). The JSON-text path a writer's reply takes must agree with it."""
    errors = ProgramGraph.model_validate(data).validate(catalog, task)
    assert all(isinstance(e, GraphError) for e in errors)
    from_text = validate_graph(json.dumps(data), catalog, task)
    assert [str(e) for e in from_text] == [str(e) for e in errors]
    return errors


def expect_error(errors: list[GraphError], code: str, node_id: str | None, *mentions: str) -> GraphError:
    """The error with ``code`` for ``node_id`` (None: a graph-level error), checked as the writer reads it.

    The writer receives ``str(error)`` through format_errors, so the node name and
    every ``mentions`` string are looked for in that text. The fix must not be empty.
    """
    hits = [e for e in errors if e.code == code and e.node == node_id]
    assert hits, f"expected [{code}] on {node_id!r}, got:\n{format_errors(errors) or '(no errors)'}"
    err = hits[0]
    text = str(err)
    if node_id is None:
        assert text.startswith(f"[{code}] graph: "), text
    else:
        assert text.startswith(f"[{code}] node '{node_id}': "), text
    assert err.message.strip()
    assert err.fix.strip(), f"[{code}] has an empty fix"
    for word in mentions:
        assert word in text, f"{word!r} not in {text!r}"
    return err


def expect_error_on_one_of(errors: list[GraphError], code: str, node_ids: set[str]) -> GraphError:
    hits = [e for e in errors if e.code == code and e.node in node_ids]
    assert hits, f"expected [{code}] on one of {sorted(node_ids)}, got:\n{format_errors(errors) or '(no errors)'}"
    return expect_error(errors, code, hits[0].node)


# =========================================================== spec conformance
def _literal_values(annotation: Any) -> set[Any]:
    out: set[Any] = set()
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        out |= _literal_values(arg) if get_args(arg) else {arg}
    return out


def test_schema_fields_follow_the_architecture_spec():
    assert {"id", "title", "goal_text", "end_condition", "checker_id", "scene_id", "allowed_skills",
            "nominal_geometry", "budgets", "safety"} <= set(TaskSpec.model_fields)
    assert set(Budgets.model_fields) == {"max_big_model_calls", "max_usd", "max_seconds"}
    assert set(Safety.model_fields) == {"workspace_bounds", "max_speed", "max_force"}
    assert {"name", "docstring", "params", "returns", "preconditions", "postconditions"} <= set(SkillSpec.model_fields)
    assert set(ProgramGraph.model_fields) == {"nodes", "edges", "entry", "max_retries", "terminal_nodes"}
    assert ProgramGraph.model_fields["max_retries"].default == 1
    assert {"id", "skill", "params"} <= set(Node.model_fields)
    assert set(Edge.model_fields) == {"src", "dst", "on"}
    assert _literal_values(Edge.model_fields["on"].annotation) == {"success", "failure"}
    assert {"passed", "score", "evidence", "failure_class", "failing_node"} <= set(Verdict.model_fields)
    assert set(Evidence.model_fields) == {"metrics", "image_paths"}
    assert FAILURE_CLASSES == ("perception", "grasp", "alignment", "stopping", "timeout", "safety", "other")
    assert _literal_values(Verdict.model_fields["failure_class"].annotation) == set(FAILURE_CLASSES)
    assert set(TraceEvent.model_fields) == {"run_id", "ts", "seq", "kind", "payload"}
    assert EVENT_KINDS == ("model_call", "node_start", "node_end", "verdict", "patch", "intervention", "budget_stop")
    assert _literal_values(TraceEvent.model_fields["kind"].annotation) == set(EVENT_KINDS) == set(PAYLOAD_MODELS)
    assert set(ModelCallPayload.model_fields) == {
        "provider", "model", "role", "input_tokens", "output_tokens", "reasoning_tokens", "usd", "usd_source",
        "latency_ms", "cache_hit", "request_hash"}
    assert _literal_values(ModelCallPayload.model_fields["role"].annotation) == {"writer", "reader", "system1"}
    assert _literal_values(ModelCallPayload.model_fields["usd_source"].annotation) == {"provider", "computed"}
    assert set(NodeEndPayload.model_fields) == {"node_id", "skill", "ok", "outputs", "error", "sim_seconds"}
    assert set(PatchPayload.model_fields) == {"round", "graph_before_hash", "graph_after_hash", "failure_summary"}


# ================================================================ round trips
def test_taskspec_json_round_trip():
    text = TASK.model_dump_json()
    back = TaskSpec.model_validate_json(text)
    assert back == TASK
    assert back.model_dump_json() == text
    assert back.safety.workspace_bounds.min == (0.1, -0.6, 0.0)  # tuples come back as tuples


@pytest.mark.parametrize("name", sorted(SPECS))
def test_skillspec_json_round_trip(name):
    spec = SPECS[name]
    back = SkillSpec.model_validate_json(spec.model_dump_json())
    assert back == spec
    assert back.catalog_entry() == spec.catalog_entry()


def test_program_graph_json_round_trip():
    graph = ProgramGraph.model_validate(good_graph())
    text = graph.model_dump_json()
    back = ProgramGraph.model_validate_json(text)
    assert back == graph
    assert back.model_dump_json() == text
    assert back.graph_hash() == graph.graph_hash() and len(graph.graph_hash()) == 16
    assert back.validate(CATALOG, TASK) == []
    for source in (text, text.encode("utf-8"), json.loads(text), graph):
        parsed, errs = parse_graph(source)
        assert errs == [] and parsed == graph


def test_graph_hash_ignores_key_order_but_not_content():
    base = ProgramGraph.model_validate(good_graph()).graph_hash()
    g = good_graph()
    node(g, "approach_box")["params"] = dict(reversed(list(node(g, "approach_box")["params"].items())))
    assert ProgramGraph.model_validate(g).graph_hash() == base
    node(g, "approach_box")["params"]["standoff"] = 0.06
    assert ProgramGraph.model_validate(g).graph_hash() != base


def test_program_graph_navigation():
    g = good_graph()
    node(g, "grasp_box")["max_retries"] = 3
    graph = ProgramGraph.model_validate(g)
    assert graph.next_node("grasp_box", "success") == "locate_slot"
    assert graph.next_node("grasp_box", "failure") == "abort"
    assert graph.next_node("done", "success") is None
    assert graph.retries_for("grasp_box") == 3 and graph.retries_for("locate_box") == 1
    with pytest.raises(KeyError):
        graph.node("ghost")


def make_observation(with_pixels: bool = True) -> Observation:
    intr = CameraIntrinsics(width=64, height=48, fx=55.4, fy=55.4, cx=32.0, cy=24.0, fovy_deg=47.0)
    down = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    cameras = {}
    for i, (name, pos) in enumerate((("overhead", (0.5, 0.0, 1.2)), ("front", (1.1, 0.0, 0.5)),
                                     ("wrist", (0.31, 0.0, 0.55)))):
        rgb = np.full((48, 64, 3), 10 * (i + 1), dtype=np.uint8) if with_pixels else None
        cameras[name] = CameraImage(name=name, path=f"{name}_000012.png", intrinsics=intr,
                                    extrinsics=CameraExtrinsics(position=pos, rotation=down), rgb=rgb)
    return Observation(cameras=cameras, joint_positions=[0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8],
                       joint_velocities=[0.0] * 7, gripper_width=0.08, gripper_force=0.0,
                       tcp_position=(0.31, 0.0, 0.49), tcp_rotation=down, wrench_estimate=[0.0] * 6, sim_time=1.5)


def test_observation_json_round_trip_drops_pixels():
    obs = make_observation()
    assert obs.image("front").shape == (48, 64, 3)
    text = obs.model_dump_json()
    assert '"rgb"' not in text
    data = json.loads(text)
    assert set(data["cameras"]) == {"overhead", "front", "wrist"}
    assert all("rgb" not in cam for cam in data["cameras"].values())
    back = Observation.model_validate_json(text)
    assert all(cam.rgb is None for cam in back.cameras.values())
    assert back.model_dump() == obs.model_dump()  # every field but the pixels survives
    assert back.model_dump_json() == text
    with pytest.raises(ValueError):
        back.image("front")
    assert json.loads(json.dumps(obs.summary())) == data  # summary() is JSON-safe and has no pixels
    assert "rgb" not in json.dumps(Observation.model_json_schema())


def test_camera_geometry_helpers():
    cam = make_observation(with_pixels=False).cameras["front"]
    assert np.allclose(cam.extrinsics.base_to_camera() @ cam.extrinsics.camera_to_base(), np.eye(4))
    assert np.allclose(cam.intrinsics.matrix(), [[55.4, 0, 32.0], [0, 55.4, 24.0], [0, 0, 1]])


def test_verdict_json_round_trip():
    passed = Verdict(passed=True, score=1.0, task_id="T2", seed=500, checker_id="rack_slot_v1",
                     evidence=Evidence(metrics={"pos_err_mm": 1.2, "yaw_err_deg": 0.4, "on_rails": True},
                                       image_paths=["front_final.png"]))
    failed = Verdict(passed=False, score=0.35, failure_class="alignment", failing_node="approach_slot",
                     evidence=Evidence(metrics={"pos_err_mm": 6.8}), summary="box hit the left rail")
    for verdict in (passed, failed):
        back = Verdict.model_validate_json(verdict.model_dump_json())
        assert back == verdict


GRAPH_HASH = ProgramGraph.model_validate(good_graph()).graph_hash()
PAYLOADS: dict[str, dict[str, Any]] = {
    "model_call": {"provider": "fake", "model": "fake-writer", "role": "writer", "input_tokens": 1830,
                   "output_tokens": 412, "reasoning_tokens": None, "usd": 0.0, "usd_source": "computed",
                   "latency_ms": 12.5, "cache_hit": False,
                   "request_hash": hashlib.sha256(b"canonical request").hexdigest()},
    "node_start": {"node_id": "grasp_box", "skill": "grasp", "params": {"width_mm": 44.0, "force": 30}, "attempt": 1},
    "node_end": {"node_id": "grasp_box", "skill": "grasp", "ok": False, "outputs": {"holding": False},
                 "error": "fingers closed on nothing", "sim_seconds": 1.25},
    "verdict": {"passed": False, "score": 0.4, "failure_class": "alignment", "failing_node": "approach_slot",
                "metrics": {"pos_err_mm": 4.2}, "image_paths": ["front_final.png"]},
    "patch": {"round": 2, "graph_before_hash": GRAPH_HASH, "graph_after_hash": "fedcba9876543210",
              "failure_summary": "rack yaw +2.6 deg in all 4 failures"},
    "intervention": {"reason": "operator paused the run", "actor": "human"},
    "budget_stop": {"scope": "session", "cap_usd": 1.0, "spent_usd": 1.02, "reason": "projected cost over the cap"},
}


def event(kind: str, payload: dict[str, Any] | None = None, **fields: Any) -> TraceEvent:
    base = {"run_id": "run-0001", "ts": 1760745600.25, "seq": 3, "kind": kind,
            "payload": PAYLOADS[kind] if payload is None else payload}
    return TraceEvent(**{**base, **fields})


def test_payload_fixtures_cover_every_kind():
    assert set(PAYLOADS) == set(EVENT_KINDS) == set(PAYLOAD_MODELS)


@pytest.mark.parametrize("kind", EVENT_KINDS)
def test_trace_event_json_round_trip(kind):
    ev = event(kind)
    line = ev.model_dump_json()
    back = TraceEvent.model_validate_json(line)
    assert back == ev
    assert back.model_dump_json() == line
    assert {k: back.payload[k] for k in PAYLOADS[kind]} == PAYLOADS[kind]
    assert TraceEvent.model_validate(json.loads(line)) == ev  # a parsed events.jsonl line


# ========================================================= JSON Schema export
def _local_refs(schema: Any):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "$ref":
                yield value
            else:
                yield from _local_refs(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _local_refs(value)


def test_export_writes_one_valid_json_file_per_schema(tmp_path):
    written = export_json_schemas(tmp_path)
    expected = {f"{name}.json" for name in MODELS} | {f"TraceEvent.{kind}.json" for kind in EVENT_KINDS}
    assert {"TaskSpec", "SkillSpec", "ProgramGraph", "Observation", "Verdict", "TraceEvent"} == set(MODELS)
    assert sorted(p.name for p in written) == sorted(expected)
    assert {p.name for p in tmp_path.iterdir()} == expected
    for path in written:
        raw = path.read_bytes()
        assert not raw.startswith(codecs.BOM_UTF8) and b"\r\n" not in raw and raw.endswith(b"\n")
        schema = json.loads(raw.decode("utf-8"))
        assert schema["type"] == "object" and schema["properties"], path.name
        for ref in _local_refs(schema):
            assert ref.startswith("#/$defs/") and ref.split("/")[-1] in schema.get("$defs", {}), (path.name, ref)
    for name in MODELS:
        assert json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))["title"] == name


def test_committed_json_schemas_match_the_models(tmp_path):
    written = {p.name: p for p in export_json_schemas(tmp_path)}
    committed = {p.name: p for p in JSON_DIR.glob("*.json")}
    assert set(committed) == set(written)
    stale = sorted(n for n, p in written.items() if committed[n].read_text(encoding="utf-8") != p.read_text(encoding="utf-8"))
    assert not stale, f"rerun `uv run python -m loopkit.schemas`: {stale}"


def test_exported_schemas_accept_real_instances(tmp_path):
    export_json_schemas(tmp_path)
    load = lambda name: json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))  # noqa: E731
    instances = {
        "TaskSpec": TASK.model_dump(mode="json"),
        "SkillSpec": SPECS["locate"].model_dump(mode="json"),
        "ProgramGraph": ProgramGraph.model_validate(good_graph()).model_dump(mode="json"),
        "Observation": make_observation().model_dump(mode="json"),
        "Verdict": Verdict(passed=False, score=0.2, failure_class="grasp").model_dump(mode="json"),
        "TraceEvent": event("node_end").model_dump(mode="json"),
    }
    for name, data in instances.items():
        assert check_schema(data, load(name)) == [], name
    for kind in EVENT_KINDS:
        assert check_schema(event(kind).payload, load(f"TraceEvent.{kind}")) == [], kind
    assert check_schema({**instances["TaskSpec"], "seed_hint": 3}, load("TaskSpec"))  # extra fields are refused


# ==================================================== TaskSpec and SkillSpec
def test_taskspec_rules():
    with pytest.raises(ValidationError, match="not declared"):
        make_task(params={"slot": 1, "port": 3})
    with pytest.raises(ValidationError, match="below max"):
        AABB(min=(0.0, 0.0, 0.5), max=(1.0, 1.0, 0.5))
    with pytest.raises(ValidationError):
        Budgets(max_big_model_calls=0, max_usd=0.0, max_seconds=0.0)
    with pytest.raises(ValidationError):
        Budgets(max_big_model_calls=-1, max_usd=0.0, max_seconds=10.0)
    with pytest.raises(ValidationError):
        Safety(workspace_bounds=TASK.safety.workspace_bounds, max_speed=0.0, max_force=70.0)
    with pytest.raises(ValidationError):
        make_task(object_poses={"box": [0.5, 0.0, 0.1]})  # extra fields are refused
    box = TASK.safety.workspace_bounds
    assert box.contains((0.5, 0.0, 0.5)) and not box.contains((1.0, 0.0, 0.5))
    assert box.contains((0.96, 0.0, 0.5), margin=0.02)


def test_task_for_writer_hides_per_seed_values():
    view = TASK.for_writer()
    assert "seed" not in view and "params" not in view
    assert view["params_schema"] == TASK.params_schema
    assert view["allowed_skills"] == TASK.allowed_skills
    json.dumps(view)


def test_skillspec_rules_and_catalog_text():
    with pytest.raises(ValidationError):
        SkillSpec(name="Locate", docstring="d", params=obj({}), returns=obj({}))
    with pytest.raises(ValidationError):
        SkillSpec(name="grasp-box", docstring="d", params=obj({}), returns=obj({}))
    with pytest.raises(ValidationError):
        SkillSpec(name="halt", docstring="d", params=obj({}), returns=obj({}), terminal="stop")
    assert as_catalog(CATALOG) == as_catalog(SPECS) == SPECS
    text = catalog_text(CATALOG, allowed=TASK.allowed_skills)
    assert "### locate  (perception)" in text and "### abort" in text
    assert "inspect_panel" not in text
    entry = SPECS["locate"].catalog_entry()
    assert "- target (required): string" in entry and "- camera (optional)" in entry
    assert "returns:" in entry and "- found: boolean" in entry


# ===================================================================== Verdict
@pytest.mark.parametrize("kwargs", [
    dict(passed=True, score=1.0),
    dict(passed=True, score=1.0, failing_node=None, summary="seated"),
    dict(passed=False, score=0.0, failure_class="timeout"),
    dict(passed=False, score=0.6, failure_class="alignment", failing_node="approach_slot"),
])
def test_verdict_consistent_cases(kwargs):
    Verdict(**kwargs)


@pytest.mark.parametrize("kwargs,words", [
    (dict(passed=True, score=0.99), "score 1"),
    (dict(passed=True, score=1.0, failure_class="grasp"), "no failure_class"),
    (dict(passed=False, score=0.3), "needs a failure_class"),
    (dict(passed=False, score=0.3, failure_class="bad_luck"), "failure_class"),
    (dict(passed=False, score=1.2, failure_class="other"), "less than or equal to 1"),
    (dict(passed=False, score=-0.1, failure_class="other"), "greater than or equal to 0"),
    (dict(passed=True, score=float("nan")), "score"),
    (dict(passed=False, score=0.2, failure_class="grasp", confidence=0.9), "Extra inputs"),
])
def test_verdict_rejects_inconsistent_cases(kwargs, words):
    with pytest.raises(ValidationError, match=words):
        Verdict(**kwargs)


# ================================================================== TraceEvent
# A payload model keeps extra fields, so a payload that carries every field of
# another kind is accepted under it. These two (kind, payload-of) pairs are the
# only such supersets among the seven kinds.
SUPERSET_PAYLOADS = {("node_start", "node_end"), ("intervention", "budget_stop")}


@pytest.mark.parametrize("kind", EVENT_KINDS)
def test_trace_event_rejects_payload_of_another_kind(kind):
    for other in EVENT_KINDS:
        if other == kind or (kind, other) in SUPERSET_PAYLOADS:
            continue
        with pytest.raises(ValidationError):
            event(kind, PAYLOADS[other])
    with pytest.raises(ValidationError):
        event(kind, {})


@pytest.mark.parametrize("kind,changes", [
    ("model_call", {"usd": -0.01}),
    ("model_call", {"role": "critic"}),
    ("model_call", {"usd_source": "guess"}),
    ("model_call", {"latency_ms": -1.0}),
    ("node_end", {"ok": "maybe"}),
    ("node_end", {"sim_seconds": -0.5}),
    ("verdict", {"score": 1.5}),
    ("verdict", {"failure_class": "unlucky"}),
    ("patch", {"round": -1}),
    ("budget_stop", {"cap_usd": "a lot"}),
])
def test_trace_event_rejects_bad_payload_fields(kind, changes):
    with pytest.raises(ValidationError):
        event(kind, {**PAYLOADS[kind], **changes})


def test_trace_event_envelope_rules_and_extra_payload_fields():
    with pytest.raises(ValidationError):
        TraceEvent(run_id="run-0001", ts=0.0, seq=0, kind="note", payload=PAYLOADS["intervention"])
    with pytest.raises(ValidationError):
        event("intervention", seq=-1)
    with pytest.raises(ValidationError):
        event("intervention", host="lab-pc")
    ev = event("node_end", {**PAYLOADS["node_end"], "joint_error_max": 0.002})
    back = TraceEvent.model_validate_json(ev.model_dump_json())
    assert back.payload["joint_error_max"] == 0.002  # extra payload fields are kept
    start = event("node_start", {"node_id": "locate_box", "skill": "locate"})
    assert start.payload == {"node_id": "locate_box", "skill": "locate", "params": {}, "attempt": 0}


# ================================================================= Observation
OBSERVATION_FIELDS = {
    "cameras",  # images as paths plus pixels, with intrinsics and extrinsics
    "joint_positions", "joint_velocities", "gripper_width", "gripper_force",  # joint and gripper sensing
    "tcp_position", "tcp_rotation",  # forward kinematics of the arm's own joints
    "wrench_estimate",  # external wrench from joint torques
    "sim_time",
}


def test_observation_has_only_camera_and_proprioception_fields():
    assert set(Observation.model_fields) == OBSERVATION_FIELDS
    assert set(CameraImage.model_fields) == {"name", "path", "intrinsics", "extrinsics", "rgb"}
    assert set(CameraIntrinsics.model_fields) == {"width", "height", "fx", "fy", "cx", "cy", "fovy_deg"}
    assert set(CameraExtrinsics.model_fields) == {"position", "rotation"}
    for model in (Observation, CameraImage, CameraIntrinsics, CameraExtrinsics):
        assert model.model_config.get("extra") == "forbid", model.__name__
    # the only nested types are the camera models: nothing else can hide in the observation
    assert set(Observation.model_json_schema()["$defs"]) == {"CameraImage", "CameraIntrinsics", "CameraExtrinsics"}
    words = ("pose", "body", "object", "rack", "box", "plug", "port", "slot", "tray", "qpos", "xpos", "quat",
             "contact", "target", "world")
    names = set(Observation.model_fields) | set(CameraImage.model_fields)
    assert not [n for n in names for w in words if w in n.lower()]


def test_observation_refuses_privileged_extras():
    data = make_observation(with_pixels=False).model_dump()
    with pytest.raises(ValidationError):
        Observation(**data, box_pose=[0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])
    cam = {**data["cameras"]["front"], "object_pose": [0.5, 0.0, 0.1]}
    with pytest.raises(ValidationError):
        Observation(**{**data, "cameras": {"front": cam}})


# ======================================================= validate: bad graphs
# One test per bad-graph kind in docs/01_architecture.md and the task brief.
def test_good_graph_is_valid():
    assert errors_for(good_graph()) == []


# Code fences and prose around the object are accepted (decision 25, see
# test_fenced_or_prose_wrapped_reply_is_accepted); text with no object in it stays bad_json.
@pytest.mark.parametrize("make", [
    pytest.param(lambda t: t[:-12], id="truncated"),
    pytest.param(lambda t: "```json\n" + t[:-12] + "\n```", id="code_fences_around_broken_json"),
    pytest.param(lambda t: "Here is the program graph: none, sorry.", id="prose_only"),
    pytest.param(lambda t: t[:-1] + ",}", id="trailing_comma"),
    pytest.param(lambda t: str(json.loads(t)), id="python_repr"),
    pytest.param(lambda t: "", id="empty"),
    pytest.param(lambda t: "[" + t + "]", id="top_level_list"),
    pytest.param(lambda t: t[:-12].encode("utf-8"), id="truncated_bytes"),
])
def test_bad_json(make):
    source = make(json.dumps(good_graph()))
    errs = validate_graph(source, CATALOG, TASK)
    assert [e.code for e in errs] == ["bad_json"]
    expect_error(errs, "bad_json", None)
    graph, parse_errs = parse_graph(source)
    assert graph is None and [str(e) for e in parse_errs] == [str(e) for e in errs]


def test_unknown_skill():
    g = good_graph()
    node(g, "grasp_box")["skill"] = "grab"
    errs = errors_for(g)
    err = expect_error(errs, "unknown_skill", "grasp_box", "'grab'")
    assert [e.code for e in errs] == ["unknown_skill"]
    assert "Did you mean 'grasp'?" in err.fix


def _with_panel_check() -> dict[str, Any]:
    g = good_graph()
    g["nodes"].insert(3, {"id": "check_panel", "skill": "inspect_panel", "params": {}})
    set_edge(g, "grasp_box", "success", "check_panel")
    g["edges"] += [edge("check_panel", "locate_slot", "success"), edge("check_panel", "abort", "failure")]
    return g


def test_skill_not_allowed_by_task():
    errs = errors_for(_with_panel_check())
    err = expect_error(errs, "skill_not_allowed", "check_panel", "'inspect_panel'", "T2")
    assert [e.code for e in errs] == ["skill_not_allowed"]
    assert "inspect_panel" not in err.fix and "locate" in err.fix
    # the same graph is fine for a task that allows the skill, or with no task
    assert errors_for(_with_panel_check(), task=make_task(allowed_skills=TASK.allowed_skills + ["inspect_panel"])) == []
    assert errors_for(_with_panel_check(), task=None) == []


@pytest.mark.parametrize("node_id,param,value", [
    ("approach_box", "standoff", "close"),
    ("locate_box", "camera", 3),
    ("locate_slot", "index", True),  # a boolean is not an integer in JSON Schema
    ("locate_slot", "index", 1.5),
    ("grasp_box", "width_mm", None),
    ("abort", "reason", ["no", "box"]),
])
def test_param_failing_schema_type(node_id, param, value):
    g = good_graph()
    node(g, node_id)["params"][param] = value
    errs = errors_for(g)
    err = expect_error(errs, "bad_param", node_id, f"{param} is", "expected")
    assert [e.code for e in errs] == ["bad_param"]
    assert err.fix.startswith(f"Give '{param}' ")


@pytest.mark.parametrize("node_id,param,value,words", [
    ("approach_slot", "speed", 0.5, "above the maximum 0.25"),
    ("approach_slot", "speed", 0.0, "must be greater than 0"),
    ("approach_box", "standoff", -0.01, "below the minimum 0"),
    ("locate_slot", "index", 12, "above the maximum 11"),
    ("locate_box", "camera", "side", "expected one of"),
    ("approach_box", "position", [0.5, 0.0], "has 2 items, needs at least 3"),
    ("approach_box", "position", [0.5, 0.0, 0.1, 0.0], "has 4 items, allows at most 3"),
    ("abort", "reason", "", "shorter than 1"),
])
def test_param_out_of_range(node_id, param, value, words):
    g = good_graph()
    node(g, node_id)["params"][param] = value
    errs = errors_for(g)
    expect_error(errs, "bad_param", node_id, words)
    assert [e.code for e in errs] == ["bad_param"]


def test_missing_required_param():
    g = good_graph()
    del node(g, "locate_slot")["params"]["target"]
    errs = errors_for(g)
    err = expect_error(errs, "missing_param", "locate_slot", "'target'")
    assert [e.code for e in errs] == ["missing_param"]
    assert err.fix.startswith('Add "target": string') and "enum" in err.fix


def test_unknown_param():
    g = good_graph()
    node(g, "grasp_box")["params"]["speed"] = 0.1
    errs = errors_for(g)
    err = expect_error(errs, "unknown_param", "grasp_box", "'speed'")
    assert [e.code for e in errs] == ["unknown_param"]
    assert "width_mm" in err.fix and "force" in err.fix


def test_ref_to_unknown_node():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_bx.position"}
    errs = errors_for(g)
    err = expect_error(errs, "ref_unknown_node", "approach_box", "'locate_bx'")
    assert [e.code for e in errs] == ["ref_unknown_node"]
    assert "Did you mean 'locate_box'?" in err.fix


def test_ref_to_unknown_field():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_box.pose"}
    errs = errors_for(g)
    err = expect_error(errs, "ref_unknown_field", "approach_box", "'pose'", "locate_box")
    assert [e.code for e in errs] == ["ref_unknown_field"]
    assert "position" in err.fix and "yaw" in err.fix


def test_ref_to_node_that_is_not_upstream():
    g = good_graph()  # a downstream node
    node(g, "approach_box")["params"]["position"] = {"ref": "approach_slot.tcp_position"}
    errs = errors_for(g)
    expect_error(errs, "ref_not_upstream", "approach_box", "'approach_slot'")
    assert [e.code for e in errs] == ["ref_not_upstream"]

    g = good_graph()  # a node on a side branch that never leads here
    g["nodes"].insert(3, {"id": "regrip", "skill": "grasp", "params": {"width_mm": 44}})
    set_edge(g, "grasp_box", "failure", "regrip")
    g["edges"] += [edge("regrip", "abort", "success"), edge("regrip", "abort", "failure")]
    node(g, "release_box")["params"]["open_mm"] = {"ref": "regrip.width_mm"}
    expect_error(errors_for(g), "ref_not_upstream", "release_box", "'regrip'")

    g = good_graph()  # a terminal node never runs before anything
    node(g, "release_box")["params"]["open_mm"] = {"ref": "done.result"}
    expect_error(errors_for(g), "ref_not_upstream", "release_box", "'done'")


def test_ref_to_unknown_task_param():
    g = good_graph()
    node(g, "locate_slot")["params"]["index"] = {"ref": "task.port"}
    errs = errors_for(g)
    err = expect_error(errs, "ref_unknown_field", "locate_slot", "'port'")
    assert [e.code for e in errs] == ["ref_unknown_field"]
    assert "slot" in err.fix and "grip_force" in err.fix
    # with no task there is nothing to check task refs against
    assert errors_for(g, task=None) == []


def test_unreachable_node():
    g = good_graph()
    g["nodes"].append({"id": "locate_port", "skill": "locate", "params": {"target": "port", "index": 3}})
    g["edges"] += [edge("locate_port", "done", "success"), edge("locate_port", "abort", "failure")]
    errs = errors_for(g)
    expect_error(errs, "unreachable", "locate_port")
    assert [e.code for e in errs] == ["unreachable"]


def test_non_terminal_node_without_failure_edge():
    g = good_graph()
    drop_edge(g, "grasp_box", "failure")
    errs = errors_for(g)
    err = expect_error(errs, "no_failure_edge", "grasp_box")
    assert [e.code for e in errs] == ["no_failure_edge"]
    assert '"src": "grasp_box"' in err.fix and '"on": "failure"' in err.fix


def test_node_without_success_edge():
    g = good_graph()
    drop_edge(g, "release_box", "success")
    errs = errors_for(g)
    err = expect_error(errs, "no_success_edge", "release_box")
    assert '"src": "release_box"' in err.fix and '"on": "success"' in err.fix
    expect_error(errs, "unreachable", "done")  # the cut also strands done


def _grasp_retry_loop(bound_on: str | None = None) -> dict[str, Any]:
    """A failed grasp goes back to perception: locate_box -> approach_box -> grasp_box -(failure)-> locate_box."""
    g = good_graph()
    set_edge(g, "grasp_box", "failure", "locate_box")
    if bound_on:
        node(g, bound_on)["max_visits"] = 3
    return g


def test_cycle_without_max_visits():
    errs = errors_for(_grasp_retry_loop())
    err = expect_error(errs, "unbounded_cycle", "locate_box", "locate_box", "approach_box", "grasp_box")
    assert [e.code for e in errs] == ["unbounded_cycle"]
    assert "max_visits" in err.fix


def test_same_cycle_passes_once_max_visits_is_set():
    assert errors_for(_grasp_retry_loop("locate_box")) == []


def test_duplicate_node_ids():
    g = good_graph()
    g["nodes"].append({"id": "grasp_box", "skill": "grasp", "params": {"width_mm": 40}})
    errs = errors_for(g)
    expect_error(errs, "duplicate_node", "grasp_box")
    assert [e.code for e in errs] == ["duplicate_node"]


def test_unknown_entry():
    g = good_graph()
    g["entry"] = "start"
    errs = errors_for(g)
    err = expect_error(errs, "unknown_entry", None, "'start'")
    assert [e.code for e in errs] == ["unknown_entry"]
    assert "locate_box" in err.fix


def test_terminal_node_with_outgoing_edge():
    g = good_graph()
    g["edges"].append(edge("abort", "done", "success"))
    errs = errors_for(g)
    expect_error(errs, "terminal_has_edges", "abort", "success")
    assert [e.code for e in errs] == ["terminal_has_edges"]


def test_terminal_skill_not_listed_in_terminal_nodes():
    g = good_graph()
    g["terminal_nodes"] = ["done"]
    errs = errors_for(g)
    err = expect_error(errs, "terminal_not_listed", "abort", "'abort'")
    assert [e.code for e in errs] == ["terminal_not_listed"]
    assert "terminal_nodes" in err.fix


def test_no_reachable_done():
    g = good_graph()  # no done node at all
    g["nodes"] = [n for n in g["nodes"] if n["id"] != "done"]
    set_edge(g, "release_box", "success", "abort")
    g["terminal_nodes"] = ["abort"]
    errs = errors_for(g)
    expect_error(errs, "no_done", None, "done")
    assert [e.code for e in errs] == ["no_done"]

    g = good_graph()  # a done node that nothing leads to
    set_edge(g, "release_box", "success", "abort")
    errs = errors_for(g)
    expect_error(errs, "no_done", None)
    expect_error(errs, "unreachable", "done")


def test_failure_self_loop():
    g = good_graph()
    set_edge(g, "locate_box", "failure", "locate_box")  # "try perception again" as an edge
    errs = errors_for(g)
    err = expect_error(errs, "unbounded_cycle", "locate_box", "locate_box -> locate_box")
    assert [e.code for e in errs] == ["unbounded_cycle"]
    assert "max_retries" in err.fix


def test_success_self_loop():
    g = good_graph()
    set_edge(g, "release_box", "success", "release_box")
    expect_error(errors_for(g), "unbounded_cycle", "release_box", "release_box -> release_box")


def test_duplicate_edge():
    g = good_graph()
    g["edges"].append(edge("approach_box", "release_box", "success"))
    errs = errors_for(g)
    expect_error(errs, "duplicate_edge", "approach_box", "grasp_box", "release_box")
    assert [e.code for e in errs] == ["duplicate_edge"]


# ===================================================== validate: other codes
@pytest.mark.parametrize("mutate,node_id,words", [
    pytest.param(lambda g: g["edges"][0].update(on="always"), None, "edges.0.on", id="edge_on"),
    pytest.param(lambda g: node(g, "done").update(params=None), "done", "params", id="params_null"),
    pytest.param(lambda g: node(g, "grasp_box").update(max_visits=0), "grasp_box", "max_visits", id="max_visits_0"),
    pytest.param(lambda g: node(g, "grasp_box").update(max_retries=11), "grasp_box", "max_retries", id="retries_11"),
    pytest.param(lambda g: node(g, "grasp_box").update(retry=True), "grasp_box", "retry", id="extra_node_field"),
    pytest.param(lambda g: g.pop("terminal_nodes"), None, "terminal_nodes", id="missing_field"),
    pytest.param(lambda g: g.update(version=2), None, "version", id="extra_field"),
    pytest.param(lambda g: g.update(nodes={"locate_box": {}}), None, "nodes", id="nodes_not_a_list"),
])
def test_bad_shape(mutate, node_id, words):
    g = good_graph()
    mutate(g)
    errs = validate_graph(json.dumps(g), CATALOG, TASK)
    assert {e.code for e in errs} == {"bad_shape"}
    expect_error(errs, "bad_shape", node_id, words)


def test_empty_graph():
    errs = validate_graph({"nodes": [], "edges": [], "entry": "locate_box", "terminal_nodes": []}, CATALOG, TASK)
    expect_error(errs, "empty_graph", None)


@pytest.mark.parametrize("bad", ["release box", "2nd_release", "release.box", "r" * 65, "r\u00e9lease"])
def test_bad_node_id(bad):
    errs = errors_for(rename(good_graph(), "release_box", bad))
    expect_error(errs, "bad_node_id", bad)
    assert [e.code for e in errs] == ["bad_node_id"]


def test_unknown_terminal():
    g = good_graph()
    g["terminal_nodes"].append("finish")
    expect_error(errors_for(g), "unknown_terminal", "finish")


def test_non_terminal_skill_listed_as_terminal():
    g = good_graph()
    g["terminal_nodes"].append("release_box")
    expect_error(errors_for(g), "bad_terminal", "release_box", "'release'")


def test_edge_to_or_from_unknown_node():
    g = good_graph()
    set_edge(g, "release_box", "success", "finish")
    expect_error(errors_for(g), "unknown_edge_node", "release_box", "'finish'")
    g = good_graph()
    g["edges"].append(edge("start", "locate_box", "success"))
    errs = errors_for(g)
    expect_error(errs, "unknown_edge_node", None, "'start'")
    assert [e.code for e in errs] == ["unknown_edge_node"]


def test_ref_to_own_output():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": "approach_box.tcp_position"}
    expect_error(errors_for(g), "ref_self", "approach_box")


@pytest.mark.parametrize("ref", ["locate_box", "locate_box.", ".position", "locate box.position",
                                 "locate_box..position", "locate_box.position[0]", "$locate_box.position"])
def test_malformed_ref(ref):
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": ref}
    expect_error(errors_for(g), "bad_ref", "approach_box", f"'{ref}'")


def test_ref_type_mismatch():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_box.found"}
    expect_error(errors_for(g), "ref_type_mismatch", "approach_box", "boolean", "array")
    task = make_task(params_schema={**TASK.params_schema, "label": {"type": "string"}},
                     params={**TASK.params, "label": "U2"})
    g = good_graph()
    node(g, "grasp_box")["params"]["force"] = {"ref": "task.label"}
    expect_error(errors_for(g, task=task), "ref_type_mismatch", "grasp_box", "task.label", "string")


def test_ref_with_extra_keys_is_a_literal():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_box.position", "default": [0, 0, 0]}
    expect_error(errors_for(g), "bad_param", "approach_box", "expected array")


def test_every_error_is_numbered_for_the_writer():
    g = good_graph()
    drop_edge(g, "grasp_box", "failure")
    node(g, "grasp_box")["params"]["speed"] = 0.1
    errs = errors_for(g)
    lines = format_errors(errs).splitlines()
    assert len(lines) == len(errs) == 2
    assert lines[0].startswith("1. [") and lines[1].startswith("2. [")
    assert set(errs[0].to_dict()) == {"code", "node", "message", "fix"}


def test_is_ref_and_iter_refs():
    assert is_ref({"ref": "a.b"})
    assert not is_ref({"ref": "a.b", "x": 1}) and not is_ref({"ref": 3}) and not is_ref("a.b")
    assert list(iter_refs([{"ref": "a.b"}, {"k": ({"ref": "c.d"},)}, "e.f", {"ref": 1}])) == ["a.b", "c.d"]


# =================================================== validate: edge cases
def test_refs_nested_in_lists_and_objects():
    g = good_graph()
    node(g, "approach_slot")["params"]["position"] = [
        {"ref": "locate_slot.position.0"}, {"ref": "locate_slot.position.1"}, 0.12]
    node(g, "done")["params"]["result"] = {"released": {"ref": "release_box.released"}, "slot": {"ref": "task.slot"}}
    assert errors_for(g) == []
    for bad, code, words in (
        ([{"ref": "locate_slt.position.0"}, 0.0, 0.12], "ref_unknown_node", "'locate_slt'"),
        ([{"ref": "locate_slot.pos.0"}, 0.0, 0.12], "ref_unknown_field", "'pos'"),
        ([{"ref": "release_box.released"}, 0.0, 0.12], "ref_not_upstream", "'release_box'"),
        ([{"ref": "task.port"}, 0.0, 0.12], "ref_unknown_field", "'port'"),
        ([0.0, 0.0, {"ref": "locate_slot"}], "bad_ref", "'locate_slot'"),
        ([[{"ref": "ghost.x"}]], "ref_unknown_node", "'ghost'"),
    ):
        node(g, "approach_slot")["params"]["position"] = bad
        expect_error(errors_for(g), code, "approach_slot", words)


def test_integer_refs_into_number_params():
    g = good_graph()
    # grasp_box.width_mm (number) <- locate_box.width_mm (integer); grasp_box.force (number) <- task.grip_force (integer)
    assert node(g, "grasp_box")["params"] == {"width_mm": {"ref": "locate_box.width_mm"},
                                             "force": {"ref": "task.grip_force"}}
    assert SPECS["locate"].returns["properties"]["width_mm"]["type"] == "integer"
    assert TASK.params_schema["grip_force"]["type"] == "integer"
    assert errors_for(g) == []
    # the other way round is a real mismatch: a number need not be integral
    node(g, "locate_slot")["params"]["index"] = {"ref": "grasp_box.width_mm"}
    expect_error(errors_for(g), "ref_type_mismatch", "locate_slot", "number", "integer")


def _v_bounded_recovery_loop():
    """A failed slot approach backs off and tries again; the back-off node carries the bound."""
    g = good_graph()
    g["nodes"].insert(5, {"id": "back_off", "skill": "approach", "max_visits": 2,
                          "params": {"position": {"ref": "locate_slot.position"}, "standoff": 0.15}})
    set_edge(g, "approach_slot", "failure", "back_off")
    g["edges"] += [edge("back_off", "approach_slot", "success"), edge("back_off", "abort", "failure")]
    return g


def _v_diamond():
    """A slower second approach rejoins at grasp_box, which reads the shared upstream node."""
    g = good_graph()
    g["nodes"].insert(2, {"id": "approach_box_slow", "skill": "approach",
                          "params": {"position": {"ref": "locate_box.position"}, "speed": 0.02}})
    set_edge(g, "approach_box", "failure", "approach_box_slow")
    g["edges"] += [edge("approach_box_slow", "grasp_box", "success"), edge("approach_box_slow", "abort", "failure")]
    return g


def _v_two_done_nodes():
    g = good_graph()
    g["nodes"].append({"id": "done_unverified", "skill": "done", "params": {"result": {"note": "open not confirmed"}}})
    g["terminal_nodes"].append("done_unverified")
    set_edge(g, "release_box", "failure", "done_unverified")
    return g


def _v_minimal_params():
    g = good_graph()
    node(g, "locate_box")["params"] = {"target": "box"}
    node(g, "approach_box")["params"] = {"position": {"ref": "locate_box.position"}}
    node(g, "release_box")["params"] = {}
    del node(g, "done")["params"]
    return g


def _v_with(fn):
    def build():
        g = good_graph()
        fn(g)
        return g
    return build


VALID_VARIANTS = {
    "bounded_retry_loop": lambda: _grasp_retry_loop("locate_box"),
    "bounded_recovery_loop": _v_bounded_recovery_loop,
    "diamond_merge": _v_diamond,
    "two_done_nodes": _v_two_done_nodes,
    "minimal_params": _v_minimal_params,
    "entry_not_first": _v_with(lambda g: g["nodes"].reverse()),
    "hyphenated_ids": lambda: rename(good_graph(), "locate_box", "locate-box"),
    "integral_float_for_integer": _v_with(lambda g: node(g, "locate_slot")["params"].update(index=1.0)),
    "unicode_reason": _v_with(lambda g: node(g, "abort")["params"].update(reason="bo\u00eete introuvable \u7bb1")),
    "retry_overrides": _v_with(lambda g: (g.update(max_retries=0), node(g, "grasp_box").update(max_retries=3,
                                                                                               timeout_s=5.0))),
    "do_nothing": lambda: {"nodes": [{"id": "done", "skill": "done"}], "edges": [], "entry": "done",
                           "terminal_nodes": ["done"]},
}


@pytest.mark.parametrize("name", sorted(VALID_VARIANTS))
def test_valid_graphs_a_strict_validator_might_reject(name):
    assert errors_for(VALID_VARIANTS[name]()) == []


def test_valid_graph_without_task_and_from_text():
    assert ProgramGraph.model_validate(good_graph()).validate(CATALOG) == []
    assert validate_graph(json.dumps(good_graph()).encode("utf-8"), CATALOG, TASK) == []
    assert validate_graph(json.dumps(good_graph(), indent=2), SPECS, TASK) == []


class _Pose(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: tuple[float, float, float]
    yaw: float = 0.0


class _ApproachParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: _Pose
    standoff: float = Field(0.05, ge=0.0, le=0.2)
    speed: float = Field(0.1, gt=0.0, le=0.25)
    camera: Literal["overhead", "front", "wrist"] = "front"
    index: int | None = None


class _LocateReturns(BaseModel):
    pose: _Pose
    found: bool
    count: int


GENERATED = [
    SkillSpec(name="locate", docstring="Find a part.", params=obj({"target": {"type": "string"}}, ("target",)),
              returns=_LocateReturns.model_json_schema()),
    SkillSpec(name="approach", docstring="Go to a pose.", params=_ApproachParams.model_json_schema(),
              returns=obj({"reached": {"type": "boolean"}})),
    SPECS["done"], SPECS["abort"],
]


def _generated_graph(approach_params: dict[str, Any]) -> dict[str, Any]:
    return {"nodes": [{"id": "find", "skill": "locate", "params": {"target": "box"}},
                      {"id": "go", "skill": "approach", "params": approach_params},
                      {"id": "done", "skill": "done"}, {"id": "abort", "skill": "abort", "params": {"reason": "lost"}}],
            "edges": [edge("find", "go", "success"), edge("find", "abort", "failure"),
                      edge("go", "done", "success"), edge("go", "abort", "failure")],
            "entry": "find", "terminal_nodes": ["done", "abort"]}


@pytest.mark.parametrize("params", [
    {"target": {"ref": "find.pose"}},
    {"target": {"ref": "find.pose"}, "index": {"ref": "find.count"}, "camera": "wrist"},
    {"target": {"position": [0.5, 0.0, 0.1], "yaw": 0.3}, "index": None, "speed": 0.25},
    {"target": {"position": {"ref": "find.pose.position"}}},
])
def test_pydantic_generated_schemas_accept_valid_params(params):
    assert errors_for(_generated_graph(params), task=None, catalog=GENERATED) == []


def test_pydantic_generated_schemas_reject_bad_params():
    errs = errors_for(_generated_graph({"target": {"position": [0.5, 0.0], "yaw": "x", "roll": 1}}), None, GENERATED)
    assert [e.code for e in errs] == ["bad_param"] * 3
    # values are printed as JSON since decision 14 ("x", not Python's 'x')
    for words in ("target.position has 2 items", 'target.yaw is "x"', "unknown field 'roll'"):
        expect_error([e for e in errs if words in e.message], "bad_param", "go", words)
    errs = errors_for(_generated_graph({"target": {"ref": "find.pose"}, "speed": 0}), None, GENERATED)
    expect_error(errs, "bad_param", "go", "must be greater than 0")
    errs = errors_for(_generated_graph({"target": {"ref": "find.pose"}, "index": {"ref": "find.found"}}),
                      None, GENERATED)
    expect_error(errs, "ref_type_mismatch", "go", "boolean", "integer or null")


def _chain(n: int) -> dict[str, Any]:
    """n nodes in a line, each reading the first node's output."""
    nodes = [{"id": "n0", "skill": "locate", "params": {"target": "box"}}]
    nodes += [{"id": f"n{i}", "skill": "grasp", "params": {"width_mm": {"ref": "n0.width_mm"}}} for i in range(1, n)]
    nodes += [{"id": "done", "skill": "done"}, {"id": "abort", "skill": "abort", "params": {"reason": "chain broke"}}]
    edges = []
    for i in range(n):
        edges += [edge(f"n{i}", f"n{i + 1}" if i + 1 < n else "done", "success"), edge(f"n{i}", "abort", "failure")]
    return {"nodes": nodes, "edges": edges, "entry": "n0", "terminal_nodes": ["done", "abort"]}


def test_large_graphs():
    n = 1000
    start = time.perf_counter()
    g = _chain(n)
    assert errors_for(g) == []
    set_edge(g, f"n{n - 1}", "failure", "n0")  # one loop through all 1000 nodes
    errs = errors_for(g)
    expect_error(errs, "unbounded_cycle", "n0", f"n{n - 1} -> n0")
    assert [e.code for e in errs] == ["unbounded_cycle"]
    node(g, "n0")["max_visits"] = 2
    assert errors_for(g) == []
    g["nodes"].append({"id": "stray", "skill": "release"})
    g["edges"] += [edge("stray", "done", "success"), edge("stray", "abort", "failure")]
    errs = errors_for(g)
    expect_error(errs, "unreachable", "stray")
    assert [e.code for e in errs] == ["unreachable"]
    assert time.perf_counter() - start < 60


def _regrasp_loop(bound_on: str | None = None) -> dict[str, Any]:
    """A failed grasp opens and grasps again. The loop is only entered through grasp_box's failure edge."""
    g = good_graph()
    g["nodes"].insert(3, {"id": "regrasp", "skill": "release", "params": {"open_mm": 80}})
    set_edge(g, "grasp_box", "failure", "regrasp")
    g["edges"] += [edge("regrasp", "grasp_box", "success"), edge("regrasp", "abort", "failure")]
    if bound_on:
        node(g, bound_on)["max_visits"] = 2
    return g


def _wiggle_loop(bound_on: tuple[str, ...] = ()) -> dict[str, Any]:
    """A failed slot approach tries two recovery moves that hand over to each other on failure."""
    g = good_graph()
    g["nodes"][5:5] = [
        {"id": "wiggle_a", "skill": "approach", "params": {"position": {"ref": "locate_slot.position"}, "standoff": 0.01}},
        {"id": "wiggle_b", "skill": "approach", "params": {"position": {"ref": "locate_slot.position"}, "speed": 0.02}},
    ]
    set_edge(g, "approach_slot", "failure", "wiggle_a")
    g["edges"] += [edge("wiggle_a", "release_box", "success"), edge("wiggle_a", "wiggle_b", "failure"),
                   edge("wiggle_b", "release_box", "success"), edge("wiggle_b", "wiggle_a", "failure")]
    for node_id in bound_on:
        node(g, node_id)["max_visits"] = 2
    return g


def test_cycles_reachable_only_through_failure_edges():
    errs = errors_for(_regrasp_loop())
    expect_error(errs, "unbounded_cycle", "grasp_box", "grasp_box -> regrasp")
    assert [e.code for e in errs] == ["unbounded_cycle"]
    assert errors_for(_regrasp_loop("regrasp")) == []  # regrasp's success edge closes the loop, so its bound ends it
    errs = errors_for(_wiggle_loop())  # a loop made of failure edges only
    expect_error(errs, "unbounded_cycle", "wiggle_a", "wiggle_a -> wiggle_b")
    assert [e.code for e in errs] == ["unbounded_cycle"]


def test_node_id_task_is_reserved():
    g = rename(good_graph(), "release_box", "task")
    errs = errors_for(g)
    err = expect_error(errs, "reserved_node_id", "task", "reserved")
    assert [e.code for e in errs] == ["reserved_node_id"]
    assert "Rename" in err.fix
    # a ref meant for that node resolves against the task instead
    node(g, "done")["params"]["result"] = {"released": {"ref": "task.released"}}
    expect_error(errors_for(g), "ref_unknown_field", "done", "'released'")
    expect_error(errors_for(g, task=None), "reserved_node_id", "task")


# ============================================== brute-force cross-checks, fuzz
def _random_graph(rng: random.Random, n: int, refs: bool = False) -> dict[str, Any]:
    ids = [f"g{i}" for i in range(n)]
    nodes = []
    for i in ids:
        if refs:
            src = rng.choice([j for j in ids if j != i])
            nodes.append({"id": i, "skill": "grasp", "params": {"width_mm": {"ref": f"{src}.width_mm"}}})
        else:
            nodes.append({"id": i, "skill": "release", "params": {}})
    nodes += [{"id": "done", "skill": "done"}, {"id": "abort", "skill": "abort", "params": {"reason": "random"}}]
    rng.shuffle(nodes)
    edges = [edge(i, rng.choice(ids + ["done", "abort"]), on) for i in ids for on in ("success", "failure")]
    return {"nodes": nodes, "edges": edges, "entry": rng.choice(ids), "terminal_nodes": ["done", "abort"]}


def _adjacency(g: dict[str, Any]) -> dict[str, set[str]]:
    adj = {n["id"]: set() for n in g["nodes"]}
    for e in g["edges"]:
        adj[e["src"]].add(e["dst"])
    return adj


def _reach(adj: dict[str, set[str]], start: str, avoid: str | None = None) -> set[str]:
    """Nodes reachable from ``start`` in one or more steps without stepping on ``avoid``."""
    seen: set[str] = set()
    todo = [start]
    while todo:
        for w in adj[todo.pop()]:
            if w != avoid and w not in seen:
                seen.add(w)
                todo.append(w)
    return seen


def test_cycles_and_reachability_match_brute_force():
    rng = random.Random(1017)
    for _ in range(300):
        g = _random_graph(rng, rng.randint(1, 8))
        errs = validate_graph(g, CATALOG, TASK)
        adj = _adjacency(g)
        reach = {v: _reach(adj, v) for v in adj}
        loops: list[set[str]] = []
        grouped: set[str] = set()
        for v in adj:
            if v not in grouped:
                comp = {v} | {w for w in reach[v] if v in reach[w]}
                grouped |= comp
                if len(comp) > 1 or v in adj[v]:
                    loops.append(comp)
        flagged = [e.node for e in errs if e.code == "unbounded_cycle"]
        assert len(flagged) == len(loops), (g, format_errors(errs))
        assert all(sum(f in comp for f in flagged) == 1 for comp in loops), (g, format_errors(errs))
        live = {g["entry"]} | reach[g["entry"]]
        assert {e.node for e in errs if e.code == "unreachable"} == set(adj) - live, (g, format_errors(errs))
        assert any(e.code == "no_done" for e in errs) == ("done" not in live)


def _can_run_before_success(g: dict[str, Any], reader: str, src: str) -> bool:
    """Brute force over (node, src-has-succeeded) states: can ``reader`` run while ``src`` has not succeeded yet?"""
    nxt = {(e["src"], e["on"]): e["dst"] for e in g["edges"]}
    start = (g["entry"], False)
    seen, todo = {start}, [start]
    while todo:
        v, ok = todo.pop()
        for on in ("success", "failure"):
            w = nxt.get((v, on))
            state = (w, ok or (v == src and on == "success"))
            if w is not None and state not in seen:
                seen.add(state)
                todo.append(state)
    return (reader, False) in seen


def test_ref_upstream_rule_matches_brute_force():
    """A ref is flagged exactly when the reader can run before its source has succeeded (decision 2).

    Changed by decision 2: a node's outputs exist only after it succeeds, so a ref is
    fine only when every path from the entry to the reader leaves the source through a
    success edge. The earlier version also passed refs to the entry node or to a
    dominator reached through its failure edge. For a reader that never runs, a source
    that cannot reach it is still flagged.
    """
    rng = random.Random(2017)
    checked = 0
    for _ in range(300):
        g = _random_graph(rng, rng.randint(2, 8), refs=True)
        errs = validate_graph(g, CATALOG, TASK)
        flagged = {e.node for e in errs if e.code == "ref_not_upstream"}
        adj = _adjacency(g)
        entry = g["entry"]
        live = {entry} | _reach(adj, entry)
        for n in g["nodes"]:
            if n["skill"] != "grasp":
                continue
            me, src = n["id"], n["params"]["width_mm"]["ref"].split(".")[0]
            if me in live:
                assert (me in flagged) == _can_run_before_success(g, me, src), (me, src, g)
                checked += 1
            elif me not in _reach(adj, src):
                assert me in flagged, (me, src, g)
    assert checked > 500


KNOWN_CODES = {
    "bad_json", "bad_shape", "empty_graph", "bad_node_id", "reserved_node_id", "duplicate_node", "unknown_entry",
    "unknown_skill", "skill_not_allowed", "unknown_terminal", "bad_terminal", "terminal_not_listed",
    "unknown_edge_node", "terminal_has_edges", "duplicate_edge", "no_success_edge", "no_failure_edge",
    "unreachable", "no_done", "unbounded_cycle", "missing_param", "unknown_param", "bad_param", "bad_ref",
    "ref_unknown_field", "ref_unknown_node", "ref_self", "ref_not_upstream", "ref_type_mismatch",
    "success_path_no_done", "config_missing_skill",
}
WEIRD = [None, True, 0, -1, 1.5, 1e308, float("nan"), "", "x", "\n", [], {}, [1, 2], {"ref": "x"},
         {"ref": "locate_box.position"}, {"ref": "task.slot"}, {"ref": 5}, {"a": {"ref": "ghost.y"}},
         "done", "abort", "locate", "grasp", "ghost", "success", "failure", "locate_box", "release_box"]


def _mutate(g: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    def records(key: str) -> list[dict[str, Any]]:
        value = g.get(key)
        return [r for r in value if isinstance(r, dict)] if isinstance(value, list) else []

    for _ in range(rng.randint(1, 4)):
        where = rng.choice(["node", "edge", "top", "param"])
        if where == "node" and records("nodes"):
            n = rng.choice(records("nodes"))
            key = rng.choice(["id", "skill", "params", "max_visits", "max_retries", "timeout_s", "extra"])
            n.pop(key, None) if rng.random() < 0.2 else n.__setitem__(key, rng.choice(WEIRD))
        elif where == "edge" and records("edges"):
            rng.choice(records("edges"))[rng.choice(["src", "dst", "on"])] = rng.choice(WEIRD)
        elif where == "top":
            key = rng.choice(["nodes", "edges", "entry", "terminal_nodes", "max_retries"])
            g.pop(key, None) if rng.random() < 0.3 else g.__setitem__(key, rng.choice(WEIRD))
        elif where == "param" and records("nodes"):
            n = rng.choice(records("nodes"))
            if isinstance(n.get("params"), dict):
                n["params"][rng.choice(["position", "yaw", "target", "width_mm", "reason", "zzz"])] = rng.choice(WEIRD)
    return g


def test_fuzzed_graphs_always_get_readable_errors():
    rng = random.Random(4242)
    for _ in range(600):
        g = _mutate(good_graph(), rng)
        for source in (g, json.dumps(g)):
            errs = validate_graph(source, CATALOG, TASK)
            assert isinstance(errs, list)
            for e in errs:
                assert isinstance(e, GraphError) and e.code in KNOWN_CODES, e
                assert e.node is None or isinstance(e.node, str)
                assert e.message.strip() and e.fix.strip(), e
            assert len({str(e) for e in errs}) == len(errs)  # no repeated lines in the repair prompt


# ======================================================= fixed defects
# These were strict xfails until the S1 schema fixes. Each docstring names the defect.
def _bounded_failure_self_loop():
    g = good_graph()
    set_edge(g, "locate_box", "failure", "locate_box")
    node(g, "locate_box")["max_visits"] = 3
    return g


def _bounded_loop_beside_unbounded_loop():
    """Loop A (approach_box -failure-> locate_box) has no bound. Loop B (grasp_box -failure-> regrasp
    -success-> locate_box) is bounded by regrasp, whose failure edge leaves for abort. Both run through
    locate_box, so they share one strongly connected component."""
    g = good_graph()
    set_edge(g, "approach_box", "failure", "locate_box")
    g["nodes"].insert(3, {"id": "regrasp", "skill": "release", "params": {"open_mm": 80}, "max_visits": 2})
    set_edge(g, "grasp_box", "failure", "regrasp")
    g["edges"] += [edge("regrasp", "locate_box", "success"), edge("regrasp", "abort", "failure")]
    return g


@pytest.mark.parametrize("build,loop", [
    pytest.param(_bounded_failure_self_loop, {"locate_box"}, id="failure_self_loop"),
    pytest.param(lambda: _grasp_retry_loop("grasp_box"), {"locate_box", "approach_box", "grasp_box"},
                 id="bound_on_node_whose_failure_edge_closes_loop"),
    pytest.param(lambda: _regrasp_loop("grasp_box"), {"grasp_box", "regrasp"}, id="regrasp_bound_on_grasp"),
    pytest.param(lambda: _wiggle_loop(("wiggle_a", "wiggle_b")), {"wiggle_a", "wiggle_b"},
                 id="loop_of_failure_edges_only"),
    pytest.param(_bounded_loop_beside_unbounded_loop, {"locate_box", "approach_box"},
                 id="bounded_loop_hides_unbounded_loop"),
])
def test_max_visits_must_be_able_to_end_the_loop(build, loop):
    """Was: validate_graph only checked that some node of the strongly connected component set max_visits.
    Past its bound a node fails without running and takes its failure edge, so a bound ends a loop only if
    that failure edge leaves it, and one bounded loop must not hide another loop in the same component."""
    expect_error_on_one_of(errors_for(build()), "unbounded_cycle", loop)


@pytest.mark.parametrize("value", [
    pytest.param([{"ref": "locate_box.position.0"}, "high", 0.2], id="bad_item_type"),
    pytest.param([{"ref": "locate_box.position.0"}, 0.1], id="too_few_items"),
])
def test_literal_next_to_a_nested_ref_is_still_checked(value):
    """Was: a param value that contained a nested ref skipped the JSON Schema check entirely."""
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = value
    expect_error(errors_for(g), "bad_param", "approach_box", "position")


def _number_param_text(literal: str) -> str:
    return json.dumps(good_graph()).replace('"open_mm": 60', '"open_mm": ' + literal)


@pytest.mark.parametrize("text", [
    pytest.param(_number_param_text("1" + "0" * 309), id="int_beyond_float_range"),
    pytest.param(_number_param_text("1" * 4301), id="int_beyond_str_digits_limit"),
    pytest.param(_number_param_text("[" * 1500 + "]" * 1500), id="nesting_beyond_python_stack"),
    pytest.param(_number_param_text("[" * 20000 + "]" * 20000), id="nesting_beyond_json_parser"),
])
def test_parseable_but_hostile_output_gets_errors_not_an_exception(text):
    """Was: float() overflow on a 310-digit int, ValueError on a 4301-digit int, RecursionError from
    deep nesting in iter_refs/check and in json.loads."""
    crash, errs = None, []
    try:
        errs = validate_graph(text, CATALOG, TASK)
    except Exception as exc:
        crash = f"{type(exc).__name__}: {str(exc)[:160]}"
    if crash:  # fail outside the except block so pytest does not render a thousand-frame chained traceback
        pytest.fail(f"validate_graph raised {crash}", pytrace=False)
    assert errs and all(isinstance(e, GraphError) and e.fix for e in errs)


@pytest.mark.parametrize("case", ["node_id", "ref"])
def test_trailing_newline_is_not_a_plain_identifier(case):
    """Was: NODE_ID_RE and REF_RE ended in `$`, which also matches before a trailing newline."""
    if case == "node_id":
        expect_error(errors_for(rename(good_graph(), "release_box", "release_box\n")), "bad_node_id", "release_box\n")
    else:
        g = good_graph()
        node(g, "approach_box")["params"]["position"] = {"ref": "locate_box.position\n"}
        expect_error(errors_for(g), "bad_ref", "approach_box")


def test_bad_terminal_gets_one_consistent_fix():
    """Was: a non-terminal node listed in terminal_nodes also got terminal_has_edges, an opposite fix."""
    g = good_graph()
    g["terminal_nodes"].append("release_box")
    errs = errors_for(g)
    expect_error(errs, "bad_terminal", "release_box")
    assert not [e for e in errs if e.code == "terminal_has_edges" and e.node == "release_box"], format_errors(errs)


def test_cycle_message_follows_real_edges():
    """Was: unbounded_cycle printed the loop's nodes in declaration order, not edge order."""
    g = _grasp_retry_loop()
    g["nodes"][1], g["nodes"][2] = g["nodes"][2], g["nodes"][1]  # declare grasp_box before approach_box
    err = expect_error(errors_for(g), "unbounded_cycle", "locate_box")
    path = re.search(r"nodes (.+) form a loop", err.message).group(1).split(" -> ")
    edges = {(e["src"], e["dst"]) for e in g["edges"]}
    assert all((a, b) in edges for a, b in zip(path, path[1:])), path


def test_utf8_bom_bytes_are_accepted():
    """Was: bytes were decoded as utf-8, not utf-8-sig (PowerShell 5.1 writes a BOM)."""
    data = codecs.BOM_UTF8 + json.dumps(good_graph()).encode("utf-8")
    assert json.loads(data) == good_graph()
    assert validate_graph(data, CATALOG, TASK) == []


def test_refs_in_undeclared_params_are_checked():
    """Was: refs inside a param the skill does not declare were never checked."""
    wait = SkillSpec(name="wait", docstring="Hold still.", returns={"type": "object", "properties": {}},
                     params={"type": "object", "properties": {"seconds": {"type": "number", "minimum": 0}}})
    g = good_graph()
    g["nodes"].insert(1, {"id": "settle", "skill": "wait", "params": {"seconds": 0.5, "until": {"ref": "ghost.done"}}})
    set_edge(g, "locate_box", "success", "settle")
    g["edges"] += [edge("settle", "approach_box", "success"), edge("settle", "abort", "failure")]
    task = make_task(allowed_skills=TASK.allowed_skills + ["wait"])
    expect_error(errors_for(g, task=task, catalog=CATALOG + [wait]), "ref_unknown_node", "settle", "'ghost'")


def test_verdict_trace_payload_follows_verdict_rules():
    """Was: the verdict TraceEvent payload did not apply Verdict's rules."""
    with pytest.raises(ValidationError):
        Verdict(passed=True, score=0.0, failure_class="grasp")
    with pytest.raises(ValidationError):
        event("verdict", {"passed": True, "score": 0.0, "failure_class": "grasp"})


@pytest.mark.parametrize("build", [
    pytest.param(lambda: make_task(budgets={"max_big_model_calls": 0, "max_usd": float("inf"), "max_seconds": 60.0}),
                 id="task_max_usd_inf"),
    pytest.param(lambda: event("budget_stop", {**PAYLOADS["budget_stop"], "cap_usd": float("inf")}),
                 id="budget_stop_cap_inf"),
    pytest.param(lambda: event("intervention", ts=float("nan")), id="trace_ts_nan"),
])
def test_accepted_floats_survive_a_json_round_trip(build):
    """Was: inf and nan were accepted, then written to JSON as null, which the same model refused."""
    try:
        model = build()
    except ValidationError:
        return  # refusing inf and nan up front is fine too
    type(model).model_validate_json(model.model_dump_json())


def test_observations_with_pixels_compare_equal():
    """Was: the default __eq__ compared the rgb ndarrays with == and raised ValueError."""
    obs = make_observation()
    assert obs == obs.model_copy(deep=True)


def test_enum_and_const_keep_booleans_and_numbers_apart():
    """Was: enum and const used Python ==, where True == 1."""
    assert check_schema(1, {"enum": [0, 1]}) == [] and check_schema(True, {"enum": [True]}) == []
    assert check_schema(True, {"enum": [0, 1]})
    assert check_schema(True, {"const": 1})
    assert check_schema(1, {"enum": [True, False]})


def test_catalog_text_states_exclusive_bounds():
    """Was: the catalog text dropped exclusiveMinimum and exclusiveMaximum."""
    line = next(x for x in SPECS["approach"].catalog_entry().splitlines() if x.strip().startswith("- speed"))
    assert "exclusiveMinimum" in line or "greater than" in line or ">" in line, line


def test_failed_verdict_has_a_partial_score():
    """Was: a failed verdict could carry score 1.0 (docs/02_task_suite.md: '1 if passed, else partial')."""
    with pytest.raises(ValidationError):
        Verdict(passed=False, score=1.0, failure_class="stopping")


def test_task_params_satisfy_params_schema():
    """Was: TaskSpec checked that params are declared but not that their values satisfy params_schema."""
    with pytest.raises(ValidationError):
        make_task(params={"slot": 5, "grip_force": 30})


# ================================================== S1 schema decisions
# One block per numbered decision of the S1 schema fix. They are the contract the
# S3 interpreter implements.

# ---- 1. a loop is bounded only where max_visits can end it
_CYCLE_FIX = ("Set max_visits on a node of this loop whose failure edge leaves the loop, or remove the back edge. "
              "Retrying one node needs no loop: use max_retries")


def test_bounded_loop_does_not_hide_an_unbounded_one():
    errs = errors_for(_bounded_loop_beside_unbounded_loop())
    assert [e.code for e in errs] == ["unbounded_cycle"]
    err = expect_error(errs, "unbounded_cycle", "locate_box")
    assert "nodes locate_box -> approach_box -> locate_box form a loop with no visit bound" in err.message
    assert err.fix == _CYCLE_FIX


def test_each_loop_is_reported_once_in_declaration_order():
    g = _grasp_retry_loop()
    set_edge(g, "approach_slot", "failure", "locate_slot")  # a second, separate loop
    errs = errors_for(g)
    assert [(e.code, e.node) for e in errs] == [("unbounded_cycle", "locate_box"), ("unbounded_cycle", "locate_slot")]
    assert "locate_slot -> approach_slot -> locate_slot" in errs[1].message


def test_loop_message_names_the_bound_that_cannot_end_it():
    err = expect_error(errors_for(_bounded_failure_self_loop()), "unbounded_cycle", "locate_box",
                       "locate_box -> locate_box", "max_visits cannot end", "fails without running", "'locate_box'")
    assert err.fix == _CYCLE_FIX


def _runs_forever(g: dict[str, Any]) -> bool:
    """Brute force over run states (node, visit counts of bounded nodes): can some run go on forever?

    Each arrival at a node counts a visit. A bounded node visited more than max_visits
    times fails without running; any other node may succeed or fail.
    """
    nxt = {(e["src"], e["on"]): e["dst"] for e in g["edges"]}
    nodes = {n["id"]: n for n in g["nodes"]}
    terminal = set(g["terminal_nodes"])
    bounded = sorted(i for i, n in nodes.items() if n.get("max_visits"))

    def arrive(v: str, counts: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        if v in bounded:
            k = bounded.index(v)
            counts = counts[:k] + (min(counts[k] + 1, nodes[v]["max_visits"] + 1),) + counts[k + 1:]
        return v, counts

    def successors(state: tuple[str, tuple[int, ...]]) -> list[tuple[str, tuple[int, ...]]]:
        v, counts = state
        if v in terminal:
            return []
        exhausted = v in bounded and counts[bounded.index(v)] > nodes[v]["max_visits"]
        outcomes = ("failure",) if exhausted else ("success", "failure")
        return [arrive(nxt[(v, on)], counts) for on in outcomes if (v, on) in nxt]

    start = arrive(g["entry"], (0,) * len(bounded))
    color = {start: 1}
    stack = [(start, iter(successors(start)))]
    while stack:
        state, it = stack[-1]
        for child in it:
            if color.get(child) == 1:
                return True
            if child not in color:
                color[child] = 1
                stack.append((child, iter(successors(child))))
                break
        else:
            color[state] = 2
            stack.pop()
    return False


def test_loop_bound_rule_matches_brute_force():
    """A reachable unbounded_cycle is reported exactly when some run can go on forever, and every
    reported cycle is made of edges a run can take forever (no success edge out of a bounded node)."""
    rng = random.Random(3031)
    forever = 0
    for _ in range(400):
        g = _random_graph(rng, rng.randint(1, 7))
        for n in g["nodes"]:
            if n["skill"] == "release" and rng.random() < 0.4:
                n["max_visits"] = rng.randint(1, 3)
        errs = validate_graph(g, CATALOG, TASK)
        nxt = {(e["src"], e["on"]): e["dst"] for e in g["edges"]}
        bounded = {n["id"] for n in g["nodes"] if n.get("max_visits")}
        cycles = [re.search(r"nodes (.+?) form a loop", e.message).group(1).split(" -> ")
                  for e in errs if e.code == "unbounded_cycle"]
        for cycle in cycles:
            assert cycle[0] == cycle[-1], cycle
            for a, b in zip(cycle, cycle[1:]):
                assert any(nxt.get((a, on)) == b and not (on == "success" and a in bounded)
                           for on in ("success", "failure")), (a, b, cycle, g)
        live = {g["entry"]} | _reach(_adjacency(g), g["entry"])
        runs_forever = _runs_forever(g)
        forever += runs_forever
        assert runs_forever == any(set(c) & live for c in cycles), (g, format_errors(errs))
    assert forever > 50


# ---- 2. a node's outputs exist only after it succeeds
def _fallback_locate() -> dict[str, Any]:
    """A fallback locate runs only when locate_box fails, and approach_box reads the fallback."""
    g = good_graph()
    g["nodes"].insert(1, {"id": "locate_box_front", "skill": "locate", "params": {"target": "box", "camera": "front"}})
    set_edge(g, "locate_box", "failure", "locate_box_front")
    g["edges"] += [edge("locate_box_front", "approach_box", "success"), edge("locate_box_front", "abort", "failure")]
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_box_front.position"}
    return g


def test_ref_to_a_node_on_one_branch_is_not_upstream():
    errs = errors_for(_fallback_locate())
    err = expect_error(errs, "ref_not_upstream", "approach_box", "'locate_box_front'",
                       "'locate_box_front' does not always succeed before 'approach_box' runs")
    assert err.fix == "Refer only to a node that succeeds on every path to this node"
    # the fallback path also skips locate_box's success, so its readers are flagged too
    reads = {(e.node, re.search(r"reads node '([^']+)'", e.message).group(1)) for e in errs}
    assert reads == {("approach_box", "locate_box_front"), ("approach_box", "locate_box"), ("grasp_box", "locate_box")}


def test_ref_to_a_node_later_in_a_loop_is_not_upstream():
    g = _grasp_retry_loop("locate_box")
    node(g, "approach_box")["params"]["yaw"] = {"ref": "grasp_box.width_mm"}
    errs = errors_for(g)
    expect_error(errs, "ref_not_upstream", "approach_box", "'grasp_box'")
    assert [e.code for e in errs] == ["ref_not_upstream"]


def test_outputs_of_a_failed_node_cannot_be_read():
    """approach_guess runs only after locate_box failed. locate_box is the entry, so it always ran first,
    but its outputs never exist on that path."""
    g = good_graph()
    g["nodes"].insert(1, {"id": "approach_guess", "skill": "approach",
                          "params": {"position": {"ref": "locate_box.position"}}})
    set_edge(g, "locate_box", "failure", "approach_guess")
    g["edges"] += [edge("approach_guess", "grasp_box", "success"), edge("approach_guess", "abort", "failure")]
    errs = errors_for(g)
    expect_error(errs, "ref_not_upstream", "approach_guess", "'locate_box'")
    expect_error(errs, "ref_not_upstream", "grasp_box", "'locate_box'")  # reachable through the guess


# ---- 3. refs anywhere in a param value
def test_refs_nested_in_literals_are_type_checked_at_their_position():
    g = good_graph()
    node(g, "approach_slot")["params"]["position"] = [{"ref": "locate_slot.found"}, 0.0, 0.1]
    errs = errors_for(g)
    expect_error(errs, "ref_type_mismatch", "approach_slot", "param 'position[0]'", "boolean", "needs number")
    assert [e.code for e in errs] == ["ref_type_mismatch"]
    errs = errors_for(_generated_graph({"target": {"position": {"ref": "find.pose.yaw"}}}), None, GENERATED)
    expect_error(errs, "ref_type_mismatch", "go", "param 'target.position'", "find.pose.yaw is number", "array")
    node(g, "approach_slot")["params"]["position"] = [{"ref": "locate_slot.width_mm"}, 0.0, 0.1]  # integer -> number
    assert errors_for(g) == []


def test_a_ref_counts_as_one_item_of_a_literal_list():
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = [{"ref": "locate_box.position.0"}]
    expect_error(errors_for(g), "bad_param", "approach_box", "position has 1 items, needs at least 3")
    node(g, "approach_box")["params"]["position"] = [{"ref": "locate_box.position.0"},
                                                     {"ref": "locate_box.position.1"}, 0.1, 0.2]
    expect_error(errors_for(g), "bad_param", "approach_box", "position has 4 items, allows at most 3")


def test_check_wildcard_skips_only_the_marked_values():
    schema = {"type": "array", "items": {"type": "number"}, "minItems": 3}
    assert check_schema([{"ref": "a.b"}, 1, 2], schema, wildcard=is_ref) == []
    assert check_schema([{"ref": "a.b"}, "x", 2], schema, wildcard=is_ref) == ['value[1] is "x", expected number']
    assert check_schema([{"ref": "a.b"}], schema, wildcard=is_ref) == ["value has 1 items, needs at least 3"]
    assert check_schema([{"ref": "a.b"}, 1, 2], schema)  # without the wildcard a ref is just a dict


# ---- 4. dotted ref paths are resolved fully
@pytest.mark.parametrize("ref,words", [
    ("locate_box.position.7", ("locate_box.position has at most 3 items", "no item 7")),
    ("locate_box.position.x", ("locate_box.position is an array", "'x'")),
    ("locate_box.yaw.deg", ("locate_box.yaw is number", "no field 'deg'")),
    ("locate_box.found.value", ("locate_box.found is boolean", "no field 'value'")),
])
def test_ref_paths_are_resolved_against_the_returns_schema(ref, words):
    g = good_graph()
    node(g, "approach_box")["params"]["yaw"] = {"ref": ref}
    errs = errors_for(g)
    assert [e.code for e in errs] == ["ref_unknown_field"]
    expect_error(errs, "ref_unknown_field", "approach_box", *words)


def test_ref_paths_follow_refs_and_tuples_in_generated_schemas():
    ok = _generated_graph({"target": {"position": [{"ref": "find.pose.position.0"}, {"ref": "find.pose.position.2"},
                                                   0.1]}})
    assert errors_for(ok, None, GENERATED) == []
    errs = errors_for(_generated_graph({"target": {"position": {"ref": "find.pose.postion"}}}), None, GENERATED)
    err = expect_error(errs, "ref_unknown_field", "go", "find.pose has no field 'postion'")
    assert err.fix == "Did you mean 'position'? find.pose has: position, yaw"
    errs = errors_for(_generated_graph({"target": {"position": [{"ref": "find.pose.position.3"}, 0.0, 0.1]}}),
                      None, GENERATED)
    expect_error(errs, "ref_unknown_field", "go", "no item 3")


def test_task_ref_paths_are_resolved_too():
    task = make_task(params_schema={**TASK.params_schema,
                                    "offset": {"type": "array", "items": {"type": "number"}, "maxItems": 3}},
                     params={**TASK.params, "offset": [0.0, 0.01, 0.0]})
    g = good_graph()
    node(g, "approach_slot")["params"]["position"] = [{"ref": "task.offset.0"}, 0.0, 0.1]
    assert errors_for(g, task=task) == []
    node(g, "approach_slot")["params"]["position"] = [{"ref": "task.offset.3"}, 0.0, 0.1]
    expect_error(errors_for(g, task=task), "ref_unknown_field", "approach_slot", "task.offset has at most 3 items")
    node(g, "approach_slot")["params"]["position"] = [{"ref": "task.slot.x"}, 0.0, 0.1]
    expect_error(errors_for(g, task=task), "ref_unknown_field", "approach_slot", "task.slot is integer", "'x'")


# ---- 5. params are closed unless the schema opens them
class _DefaultExtraApproach(BaseModel):  # pydantic's default extra: the schema has no additionalProperties key
    position: tuple[float, float, float]
    standoff: float = Field(0.05, ge=0.0, le=0.2)


def test_params_are_closed_unless_the_schema_opens_them():
    params = _DefaultExtraApproach.model_json_schema()
    assert "additionalProperties" not in params
    approach = SkillSpec(name="approach", docstring="Move.", params=params, returns=SPECS["approach"].returns)
    others = [s for s in CATALOG if s.name != "approach"]
    g = good_graph()
    for n in g["nodes"]:
        if n["skill"] == "approach":
            n["params"] = {"position": n["params"]["position"]}
    node(g, "approach_box")["params"]["standof"] = 0.12
    errs = errors_for(g, catalog=[approach] + others)
    err = expect_error(errs, "unknown_param", "approach_box", "'standof'")
    assert [e.code for e in errs] == ["unknown_param"]
    assert err.fix == "Did you mean 'standoff'? approach takes: position, standoff"
    for extra, value, codes in ((True, 0.12, []), ({"type": "number", "maximum": 0.1}, 0.12, ["bad_param"]),
                                ({"type": "number"}, 0.05, [])):
        opened = approach.model_copy(update={"params": {**params, "additionalProperties": extra}})
        node(g, "approach_box")["params"]["standof"] = value
        assert [e.code for e in errors_for(g, catalog=[opened] + others)] == codes, extra


# ---- 6. map outputs
def test_map_returns_accept_any_field_name_typed_by_additional_properties():
    from pydantic import TypeAdapter

    dance = SkillSpec(name="dance", docstring="Fit an image-to-pose correction.", category="calibration",
                      params=obj({"n": {"type": "integer", "minimum": 1}}),
                      returns=TypeAdapter(dict[str, float]).json_schema())
    assert "returns: a map of number" in dance.catalog_entry()
    task = make_task(allowed_skills=TASK.allowed_skills + ["dance"])
    g = good_graph()
    g["nodes"].insert(3, {"id": "calib", "skill": "dance", "params": {"n": 6}})
    set_edge(g, "grasp_box", "success", "calib")
    g["edges"] += [edge("calib", "locate_slot", "success"), edge("calib", "abort", "failure")]
    node(g, "release_box")["params"]["open_mm"] = {"ref": "calib.dx_mm"}
    assert errors_for(g, task=task, catalog=CATALOG + [dance]) == []
    node(g, "locate_slot")["params"]["target"] = {"ref": "calib.anything"}
    expect_error(errors_for(g, task=task, catalog=CATALOG + [dance]), "ref_type_mismatch", "locate_slot",
                 "calib.anything is number", "needs string")


# ---- 7. union errors come from the branch of the value's type
class _OptionalLocateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    index: int | None = Field(default=None, ge=0, le=11)
    camera: Literal["overhead", "front", "wrist"] | None = None
    pose_hint: _Pose | None = None


@pytest.mark.parametrize("param,value,message,fix_words", [
    ("index", 12, "index is 12, above the maximum 11", ("integer or null", "maximum 11")),
    ("index", -1, "index is -1, below the minimum 0", ("minimum 0",)),
    ("index", "two", 'index is "two", expected integer or null', ("integer or null",)),
    ("camera", "side", 'camera is "side", expected one of ["overhead", "front", "wrist"]',
     ('enum ["overhead", "front", "wrist"]',)),
    ("pose_hint", {"position": [0.5, 0.0]}, "pose_hint.position has 2 items, needs at least 3",
     ("position (required)", "yaw")),
])
def test_optional_params_report_the_branch_of_the_values_type(param, value, message, fix_words):
    locate = SkillSpec(name="locate", docstring="Find a part.", params=_OptionalLocateParams.model_json_schema(),
                       returns=SPECS["locate"].returns)
    g = good_graph()
    node(g, "locate_box")["params"][param] = value
    errs = errors_for(g, catalog=[locate] + [s for s in CATALOG if s.name != "locate"])
    err = expect_error(errs, "bad_param", "locate_box", message)
    assert [e.code for e in errs] == ["bad_param"]
    for word in fix_words:
        assert word in err.fix, err.fix


def test_union_errors_come_from_the_branch_of_the_values_type():
    schema = {"anyOf": [{"type": "string", "maxLength": 2}, {"type": "integer", "maximum": 5}]}
    assert check_schema(9, schema) == ["value is 9, above the maximum 5"]
    assert check_schema("abc", schema) == ["value is longer than 2 characters"]
    assert check_schema(None, schema) == ["value is null, expected integer or string"]
    tagged = {"oneOf": [{"properties": {"kind": {"const": "a"}, "x": {"type": "integer"}}, "required": ["kind", "x"]},
                        {"properties": {"kind": {"const": "b"}, "y": {"type": "string"}}, "required": ["kind", "y"]}]}
    assert check_schema({"kind": "b", "x": 1}, tagged) == ["value is missing required field 'y'"]


# ---- 8. one describe helper for the catalog and the fix texts
@pytest.mark.parametrize("schema,text", [
    ({"type": "integer", "minimum": 0, "maximum": 11}, "integer, minimum 0, maximum 11"),
    ({"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}], "default": None},
     "integer or null, minimum 0, default null"),
    ({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}, "number, exclusiveMinimum 0, exclusiveMaximum 1"),
    ({"type": "boolean", "default": True}, "boolean, default true"),
    ({"type": "string", "enum": ["a", "b"]}, 'string, enum ["a", "b"]'),
    ({"enum": ["green", "amber"]}, 'string, enum ["green", "amber"]'),
    (VEC3, "array of number, 3 items"),
    ({"type": "array", "items": {"type": "string"}, "minItems": 1}, "array of string, at least 1 item"),
    ({"type": "array", "items": {"type": "number", "minimum": 0}, "maxItems": 4},
     "array of number (minimum 0), at most 4 items"),
    ({"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "minItems": 2, "maxItems": 2},
     "array [integer, string], 2 items"),
    ({"type": "object", "additionalProperties": {"type": "number"}}, "a map of number"),
    ({}, "any value"),
])
def test_describe(schema, text):
    assert describe(schema) == text


def test_describe_lists_object_fields_one_level_deep():
    params = _ApproachParams.model_json_schema()
    assert describe(params["properties"]["target"], params) == (
        "object {position (required): array of number, 3 items; yaw: number, default 0.0}")
    assert describe(_LocateReturns.model_json_schema()) == (
        "object {pose (required): object {position, yaw}; found (required): boolean; count (required): integer}")


def test_catalog_and_fix_texts_use_the_same_description():
    approach = GENERATED[1]
    target = describe(approach.params["properties"]["target"], approach.params)
    assert f"- target (required): {target}" in approach.catalog_entry()
    assert "- index (optional): integer or null, default null" in approach.catalog_entry()
    assert "- index (optional): integer, minimum 0, maximum 11, slot or port index" in SPECS["locate"].catalog_entry()
    err = expect_error(errors_for(_generated_graph({}), None, GENERATED), "missing_param", "go", "'target'")
    assert err.fix == f'Add "target": {target}'


# ---- 9. shape errors do not hide semantic errors
def test_extra_keys_and_null_params_do_not_hide_semantic_errors():
    g = good_graph()
    drop_edge(g, "grasp_box", "failure")
    node(g, "approach_box")["params"]["position"] = {"ref": "locate_bx.position"}
    for n in g["nodes"]:
        n["description"] = f"step {n['id']}"
    errs = validate_graph(json.dumps(g), CATALOG, TASK)
    assert [e.code for e in errs] == ["bad_shape"] * 8 + ["no_failure_edge", "ref_unknown_node"]
    assert parse_graph(json.dumps(g))[0] is None  # the graph as written still does not parse
    g = good_graph()
    drop_edge(g, "grasp_box", "failure")
    node(g, "done")["params"] = None
    g["comment"] = "v1"
    assert sorted(e.code for e in validate_graph(g, CATALOG, TASK)) == ["bad_shape", "bad_shape", "no_failure_edge"]


def test_other_shape_errors_still_stop_the_semantic_checks():
    g = good_graph()
    drop_edge(g, "grasp_box", "failure")
    g["edges"][0]["on"] = "always"
    assert {e.code for e in validate_graph(g, CATALOG, TASK)} == {"bad_shape"}


# ---- 10. a skill the task allows but the catalog lacks is a setup error
def test_allowed_skill_missing_from_the_catalog_is_a_config_error():
    task = make_task(allowed_skills=TASK.allowed_skills + ["slide_into", "slide_into"])
    g = good_graph()
    g["nodes"].insert(5, {"id": "slide", "skill": "slide_into", "params": {}})
    set_edge(g, "approach_slot", "success", "slide")
    g["edges"] += [edge("slide", "release_box", "success"), edge("slide", "abort", "failure")]
    errs = errors_for(g, task=task)
    assert [e.code for e in errs] == ["config_missing_skill", "unknown_skill"]
    expect_error(errs, "config_missing_skill", None, "'slide_into'", "task T2", "not in the skill catalog")
    err = expect_error(errs, "unknown_skill", "slide", "'slide_into'")
    assert "slide_into" not in err.fix and "release" in err.fix
    assert [e.code for e in errors_for(good_graph(), task=task)] == ["config_missing_skill"]
    err = expect_error(errors_for(_with_panel_check(), task=task), "skill_not_allowed", "check_panel")
    assert "slide_into" not in err.fix
    with pytest.raises(ValueError, match="slide_into"):
        catalog_text(CATALOG, allowed=task.allowed_skills)


# ---- 11. following success edges from the entry reaches done
def test_success_path_must_reach_done():
    g = good_graph()
    for e in g["edges"]:
        e["on"] = "failure" if e["on"] == "success" else "success"
    expect_error(errors_for(g), "success_path_no_done", None,
                 "following success edges from the entry never reaches done", "locate_box -> abort")
    g = good_graph()
    set_edge(g, "release_box", "success", "abort")
    set_edge(g, "release_box", "failure", "done")
    assert [e.code for e in errors_for(g)] == ["success_path_no_done"]


def test_a_loop_of_success_edges_never_reaches_done():
    g = good_graph()
    set_edge(g, "approach_slot", "success", "locate_slot")
    set_edge(g, "locate_slot", "failure", "release_box")
    node(g, "locate_slot")["max_visits"] = 2  # bounded, so not an unbounded_cycle
    errs = errors_for(g)
    assert [e.code for e in errs] == ["success_path_no_done"]
    expect_error(errs, "success_path_no_done", None, "loop locate_slot -> approach_slot -> locate_slot")


# ---- 12. strict graph numbers
@pytest.mark.parametrize("mutate,words", [
    pytest.param(lambda g: node(g, "grasp_box").update(max_visits=True), "max_visits", id="max_visits_bool"),
    pytest.param(lambda g: node(g, "grasp_box").update(max_visits=2.5), "max_visits", id="max_visits_fraction"),
    pytest.param(lambda g: node(g, "grasp_box").update(timeout_s=True), "timeout_s", id="timeout_bool"),
    pytest.param(lambda g: node(g, "grasp_box").update(timeout_s="inf"), "timeout_s", id="timeout_string"),
    pytest.param(lambda g: node(g, "grasp_box").update(timeout_s=float("inf")), "timeout_s", id="timeout_inf"),
    pytest.param(lambda g: node(g, "grasp_box").update(timeout_s=0), "timeout_s", id="timeout_zero"),
    pytest.param(lambda g: node(g, "grasp_box").update(max_retries="2"), "max_retries", id="retries_string"),
    pytest.param(lambda g: g.update(max_retries=False), "max_retries", id="top_retries_bool"),
])
def test_graph_numbers_are_strict(mutate, words):
    g = good_graph()
    mutate(g)
    errs = validate_graph(g, CATALOG, TASK)
    assert {e.code for e in errs} == {"bad_shape"} and any(words in str(e) for e in errs), format_errors(errs)


def test_integral_floats_are_counts_and_null_top_level_retries_mean_one():
    g = good_graph()
    node(g, "grasp_box").update(max_visits=2.0, max_retries=3.0, timeout_s=5)
    g["max_retries"] = None
    graph, errs = parse_graph(json.dumps(g))
    assert errs == [] and validate_graph(g, CATALOG, TASK) == []
    n = graph.node("grasp_box")
    assert (n.max_visits, n.max_retries, n.timeout_s) == (2, 3, 5.0)
    assert type(n.max_visits) is int and type(n.max_retries) is int and type(n.timeout_s) is float
    assert graph.max_retries == 1 and graph.retries_for("locate_box") == 1


def test_exported_graph_schema_agrees_on_strict_numbers():
    schema = ProgramGraph.model_json_schema()
    for bad in ({"max_visits": True}, {"max_visits": 0}, {"max_retries": "2"}, {"timeout_s": 0}):
        g = good_graph()
        node(g, "grasp_box").update(bad)
        assert check_schema(g, schema), bad
    g = good_graph()
    g["max_retries"] = None
    assert check_schema(g, schema) == []


# ---- 13. JSON Schema export
def test_trace_event_schema_ties_the_payload_to_its_kind(tmp_path):
    export_json_schemas(tmp_path)
    schema = json.loads((tmp_path / "TraceEvent.json").read_text(encoding="utf-8"))
    assert [b["properties"]["kind"]["const"] for b in schema["oneOf"]] == list(EVENT_KINDS)
    for kind in EVENT_KINDS:
        assert check_schema(event(kind).model_dump(mode="json"), schema) == [], kind
    bad = {"run_id": "r", "ts": 1.0, "seq": 0, "kind": "model_call", "payload": {}}
    errs = check_schema(bad, schema)
    assert errs and all(e.startswith("payload is missing required field") for e in errs), errs
    assert check_schema({**bad, "payload": PAYLOADS["intervention"]}, schema)


def test_exported_schemas_document_the_rules():
    graph = ProgramGraph.model_json_schema()
    node_props, top = graph["$defs"]["Node"]["properties"], graph["properties"]
    for props, name in ((top, "entry"), (top, "terminal_nodes"), (top, "max_retries"), (node_props, "max_retries"),
                        (node_props, "max_visits"), (node_props, "timeout_s"),
                        (graph["$defs"]["Edge"]["properties"], "on")):
        assert props[name].get("description"), name
    assert "fails without running" in node_props["max_visits"]["description"]
    for word in ("max_retries", "max_visits", "timeout_s", "fails without running"):
        assert word in GRAPH_FORMAT_HINT, word
    verdict = " ".join(Verdict.model_json_schema()["description"].split())
    assert "score 1" in verdict and "no failing_node" in verdict and "below 1" in verdict
    task = " ".join(TaskSpec.model_json_schema()["description"].split())
    assert "declared in" in task and "satisfies its schema" in task and "min is below its max" in task


def test_exported_schemas_use_only_json_schema_keywords(tmp_path):
    for path in export_json_schemas(tmp_path):
        text = path.read_text(encoding="utf-8")
        for key in ('"ge"', '"le"', '"gt"', '"lt"', '"strict"', '"allow_inf_nan"'):
            assert key not in text, (path.name, key)


# ---- 14. error text the writer can act on
def test_edge_shape_errors_name_the_edge():
    g = good_graph()
    g["edges"][5]["on"] = "fail"  # grasp_box -> abort
    errs = validate_graph(json.dumps(g), CATALOG, TASK)
    expect_error(errs, "bad_shape", None, "edges.5.on (edge grasp_box -> abort)", '"success" or "failure"',
                 'got "fail"')


def test_values_in_messages_are_json():
    g = good_graph()
    node(g, "locate_box")["params"]["camera"] = None
    expect_error(errors_for(g), "bad_param", "locate_box", "camera is null, expected string")
    g = good_graph()
    node(g, "approach_box")["params"]["standoff"] = True
    expect_error(errors_for(g), "bad_param", "approach_box", "standoff is true, expected number")
    assert check_schema("side", {"enum": ["front", None, True]}) == ['value is "side", expected one of ["front", null, true]']


def test_bad_json_reads_well_and_flags_a_cut_off_reply():
    text = json.dumps(good_graph(), indent=2)
    err = validate_graph(text[: int(len(text) * 0.7)], CATALOG, TASK)[0]
    assert err.code == "bad_json" and "at at" not in str(err) and "at line" in err.message, err
    assert err.fix.endswith("If your reply was cut off, return a shorter graph")
    for cut in ('{"nodes": [', '{"entry": tr', '{"entry": "loc', '```json\n{"nodes": [{"id": "a"'):
        assert validate_graph(cut, CATALOG, TASK)[0].fix.endswith("return a shorter graph"), cut
    for broken in (text[:-1] + ",}", "I cannot do that", "Here is the graph: {nodes: []}"):
        err = validate_graph(broken, CATALOG, TASK)[0]
        assert err.code == "bad_json" and "cut off" not in err.fix, broken


@pytest.mark.parametrize("value", [
    "locate_box.position", {"$ref": "locate_box.position"}, {"ref": "locate_box.position", "default": [0, 0, 0]},
])
def test_near_ref_literal_gets_the_ref_form(value):
    g = good_graph()
    node(g, "approach_box")["params"]["position"] = value
    err = expect_error(errors_for(g), "bad_param", "approach_box", "expected array")
    assert '{"ref": "<node_id>.<field>"}' in err.fix
    node(g, "approach_box")["params"]["position"] = "close"
    assert "ref" not in expect_error(errors_for(g), "bad_param", "approach_box").fix


def test_format_hint_is_printed_once_at_the_end():
    g = good_graph()
    g["edges"] = [{"from": e["src"], "to": e["dst"], "on": e["on"]} for e in g["edges"]]
    errs = validate_graph(g, CATALOG, TASK)
    text = format_errors(errs)
    assert len(errs) == 48 and text.count(GRAPH_FORMAT_HINT) == 1 and text.endswith(GRAPH_FORMAT_HINT)
    assert not any(GRAPH_FORMAT_HINT in e.fix for e in errs)
    assert 'Rename it to "src"' in text and 'Rename it to "dst"' in text


def test_node_fields_at_the_top_level_are_moved_onto_a_node():
    g = _grasp_retry_loop()
    g["max_visits"] = 3
    errs = validate_graph(g, CATALOG, TASK)
    err = expect_error(errs, "bad_shape", None, "max_visits")
    assert err.fix == '"max_visits" is a node field: move it onto a node'
    expect_error(errs, "unbounded_cycle", "locate_box")  # reported in the same round (decision 9)


# ---- 15. the verdict trace payload: flattened and consistent
def test_verdict_shaped_payload_is_flattened():
    v = Verdict(passed=False, score=0.3, failure_class="alignment", failing_node="approach_slot",
                evidence=Evidence(metrics={"pos_err_mm": 6.1}, image_paths=["closeup.png"]))
    back = TraceEvent.model_validate_json(event("verdict", v.model_dump(mode="json")).model_dump_json())
    assert back.payload["metrics"] == {"pos_err_mm": 6.1} and back.payload["image_paths"] == ["closeup.png"]
    assert "evidence" not in back.payload and back.payload["failing_node"] == "approach_slot"
    p = VerdictPayload.model_validate({"passed": True, "score": 1.0, "metrics": {"a": 1},
                                       "evidence": {"metrics": {"a": 2}, "note": "x"}})
    assert p.metrics == {"a": 1} and p.model_dump()["evidence"] == {"note": "x"}  # top-level keys win


@pytest.mark.parametrize("payload,words", [
    ({"passed": True, "score": 0.5}, "score 1"),
    ({"passed": True, "score": 1.0, "failure_class": "grasp"}, "no failure_class"),
    ({"passed": True, "score": 1.0, "failing_node": "grasp_box"}, "no failing_node"),
    ({"passed": False, "score": 0.5}, "needs a failure_class"),
    ({"passed": False, "score": 1.0, "failure_class": "other"}, "below 1"),
])
def test_verdict_payload_rules(payload, words):
    with pytest.raises(ValidationError, match=words):
        event("verdict", payload)


# ---- 16. Verdict: failed means a partial score, passed names no failing node
@pytest.mark.parametrize("kwargs,words", [
    (dict(passed=False, score=1.0, failure_class="stopping"), "below 1"),
    (dict(passed=True, score=1.0, failing_node="grasp_box"), "no failing_node"),
])
def test_verdict_rules_from_s1(kwargs, words):
    with pytest.raises(ValidationError, match=words):
        Verdict(**kwargs)


# ---- 17. token counts and camera names
def test_token_counts_are_not_negative():
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        with pytest.raises(ValidationError, match=field):
            event("model_call", {**PAYLOADS["model_call"], field: -1})
    event("model_call", {**PAYLOADS["model_call"], "input_tokens": None, "output_tokens": 0})


def test_camera_keys_are_camera_names():
    data = make_observation(with_pixels=False).model_dump()
    data["cameras"] = {"front": data["cameras"]["wrist"]}
    with pytest.raises(ValidationError, match="camera"):
        Observation.model_validate(data)


# ---- 18. integral floats
def test_graph_hash_treats_integral_floats_as_ints():
    g = good_graph()
    node(g, "locate_slot")["params"]["index"] = 1
    as_int = ProgramGraph.model_validate(g).graph_hash()
    node(g, "locate_slot")["params"]["index"] = 1.0
    assert ProgramGraph.model_validate(g).graph_hash() == as_int
    g = good_graph()
    node(g, "release_box")["params"]["open_mm"] = 60.0
    assert ProgramGraph.model_validate(g).graph_hash() == GRAPH_HASH


def test_normalize_params_gives_ints_where_the_schema_says_integer():
    g = good_graph()
    node(g, "locate_slot")["params"]["index"] = 1.0
    node(g, "release_box")["params"]["open_mm"] = 60.0
    graph = ProgramGraph.model_validate(g)
    fixed = normalize_params(graph, CATALOG)
    assert isinstance(fixed, ProgramGraph) and fixed is not graph
    assert fixed.node("locate_slot").params["index"] == 1 and type(fixed.node("locate_slot").params["index"]) is int
    assert type(fixed.node("release_box").params["open_mm"]) is float  # a number param keeps its float
    assert fixed.node("approach_box").params["position"] == {"ref": "locate_box.position"}
    assert type(graph.node("locate_slot").params["index"]) is float  # the input graph is not changed
    assert fixed.validate(CATALOG, TASK) == [] and fixed.graph_hash() == graph.graph_hash()
    pick = SkillSpec(name="pick", docstring="Pick by slot.", returns=obj({}), params=obj({
        "slots": {"type": "array", "items": {"type": "integer"}},
        "pair": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "number"}]},
        "opt": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "cfg": {"type": "object", "properties": {"n": {"type": "integer"}}},
    }))
    pg = ProgramGraph.model_validate({
        "nodes": [{"id": "p", "skill": "pick", "params": {"slots": [1.0, 2.0], "pair": [3.0, 4.0], "opt": 5.0,
                                                          "cfg": {"n": 6.0, "x": 7.0}}},
                  {"id": "done", "skill": "done"}],
        "edges": [edge("p", "done", "success"), edge("p", "done", "failure")], "entry": "p",
        "terminal_nodes": ["done"]})
    params = normalize_params(pg, [pick, SPECS["done"]]).node("p").params
    kinds = {k: [type(x).__name__ for x in v] if isinstance(v, list) else type(v).__name__ for k, v in params.items()}
    assert kinds == {"slots": ["int", "int"], "pair": ["int", "float"], "opt": "int", "cfg": "dict"}
    assert params["cfg"] == {"n": 6, "x": 7.0} and type(params["cfg"]["n"]) is int and type(params["cfg"]["x"]) is float


# ---- 19. declared floats are finite
@pytest.mark.parametrize("build", [
    pytest.param(lambda: AABB(min=(0.0, 0.0, float("-inf")), max=(1.0, 1.0, 1.0)), id="aabb"),
    pytest.param(lambda: Safety(workspace_bounds=TASK.safety.workspace_bounds, max_speed=float("inf"), max_force=70.0),
                 id="safety"),
    pytest.param(lambda: Verdict(passed=False, score=float("nan"), failure_class="other"), id="verdict"),
    pytest.param(lambda: CameraIntrinsics(width=64, height=48, fx=float("nan"), fy=55.4, cx=32.0, cy=24.0,
                                          fovy_deg=47.0), id="intrinsics"),
    pytest.param(lambda: CameraExtrinsics(position=(0.0, float("inf"), 0.0),
                                          rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
                 id="extrinsics"),
    pytest.param(lambda: Observation(**{**make_observation(with_pixels=False).model_dump(),
                                        "gripper_width": float("nan")}), id="observation"),
    pytest.param(lambda: event("model_call", {**PAYLOADS["model_call"], "latency_ms": float("inf")}), id="model_call"),
    pytest.param(lambda: event("node_end", {**PAYLOADS["node_end"], "sim_seconds": float("inf")}), id="node_end"),
    pytest.param(lambda: Node(id="n", skill="grasp", timeout_s=float("inf")), id="node_timeout"),
])
def test_declared_floats_are_finite(build):
    with pytest.raises(ValidationError):
        build()


def test_free_dicts_may_hold_nan():
    v = Verdict(passed=False, score=0.2, failure_class="other", evidence=Evidence(metrics={"err": float("nan")}))
    assert math.isnan(v.evidence.metrics["err"])
    event("node_end", {**PAYLOADS["node_end"], "outputs": {"x": float("nan")}})  # accepted (written as null)


# ---- 20. observations with pixels compare
def test_camera_image_equality_compares_pixels():
    obs = make_observation()
    other = obs.model_copy(deep=True)
    assert obs == other
    other.cameras["front"].rgb[0, 0, 0] += 1
    assert obs != other
    assert make_observation(with_pixels=False) == make_observation(with_pixels=False)
    assert make_observation(with_pixels=False) != obs


# ---- 21. hostile input never raises
def test_hostile_values_in_a_dict_source_get_errors():
    g = good_graph()
    node(g, "release_box")["params"]["open_mm"] = 10 ** 5000  # beyond float() and str() limits
    expect_error(validate_graph(g, CATALOG, TASK), "bad_param", "release_box", "above the maximum 80")
    deep: Any = 0.5
    for _ in range(32):
        deep = [deep]
    node(g, "release_box")["params"]["open_mm"] = deep  # 32 levels: an ordinary type error
    err = expect_error(validate_graph(g, CATALOG, TASK), "bad_param", "release_box", "expected number")
    node(g, "release_box")["params"]["open_mm"] = [deep]  # 33 levels
    expect_error(validate_graph(g, CATALOG, TASK), "bad_param", "release_box", "nested too deeply")
    loop: dict[str, Any] = {}
    loop["self"] = loop
    node(g, "release_box")["params"]["open_mm"] = loop
    expect_error(validate_graph(g, CATALOG, TASK), "bad_param", "release_box", "nested too deeply")
    assert "nested too deeply" not in err.message


def test_iter_refs_handles_deep_shared_and_self_referencing_values():
    deep: Any = {"ref": "a.b"}
    for _ in range(5000):
        deep = [deep]
    assert list(iter_refs(deep)) == ["a.b"]
    shared = [{"ref": "c.d"}]
    assert list(iter_refs([shared, {"k": shared}])) == ["c.d", "c.d"]  # every occurrence counts
    loop: dict[str, Any] = {"r": {"ref": "e.f"}}
    loop["self"] = loop
    assert list(iter_refs(loop)) == ["e.f"]


def test_parse_graph_never_raises_on_hostile_text():
    for text in (_number_param_text("1" * 4301), "[" * 20000, '{"nodes": ' + "[" * 3000):
        graph, errs = parse_graph(text)
        assert graph is None and [e.code for e in errs] == ["bad_json"], errs
        assert errs[0].fix


# ---- 22. a BOM in text input
def test_bom_in_text_is_accepted():
    assert validate_graph("\ufeff" + json.dumps(good_graph()), CATALOG, TASK) == []
    assert parse_graph(codecs.BOM_UTF8 + b"  " + json.dumps(good_graph()).encode("utf-8"))[1] == []


# ---- 23. ids and refs match to the very end
def test_identifier_patterns_anchor_at_the_very_end():
    from loopkit.schemas.graph import NODE_ID_RE, REF_RE

    assert NODE_ID_RE.match("release_box\n") is None and REF_RE.match("locate_box.position\n") is None
    assert NODE_ID_RE.fullmatch("release_box") and REF_RE.fullmatch("locate_box.position.0")


# ---- 24. enum and const compare as JSON
def test_enum_and_const_compare_nested_values_as_json():
    assert check_schema([1, True], {"enum": [[1, 1]]})
    assert check_schema({"a": 1.0}, {"const": {"a": 1}}) == []
    assert check_schema(0, {"const": False}) and check_schema(False, {"const": False}) == []


# ---- 25. fenced or prose-wrapped replies
@pytest.mark.parametrize("wrap", [
    pytest.param(lambda t: "```json\n" + t + "\n```", id="json_fence"),
    pytest.param(lambda t: "```\n" + t + "\n```", id="bare_fence"),
    pytest.param(lambda t: "Here is the program graph:\n" + t, id="prose_before"),
    pytest.param(lambda t: t + "\nThis graph locates the box first.", id="prose_after"),
    pytest.param(lambda t: "Sure.\n```json\n" + t + "\n```\nLet me know.", id="fence_and_prose"),
])
def test_fenced_or_prose_wrapped_reply_is_accepted(wrap):
    """Changed by decision 25 (these were bad_json, test_bad_json pinned two of them): the outermost
    JSON object is used, so a fenced reply does not cost a repair round."""
    text = wrap(json.dumps(good_graph(), indent=2))
    assert validate_graph(text, CATALOG, TASK) == []
    graph, errs = parse_graph(text)
    assert errs == [] and graph == ProgramGraph.model_validate(good_graph())


def test_a_fenced_array_is_still_not_a_graph():
    errs = validate_graph("```json\n[" + json.dumps(good_graph()) + "]\n```", CATALOG, TASK)
    assert [e.code for e in errs] == ["bad_json"]
    expect_error(errs, "bad_json", None, "the top level is an array")


# ---- 26. task param values satisfy their schema
def test_task_param_values_are_checked_against_their_schema():
    with pytest.raises(ValidationError, match="slot is 5, above the maximum 1"):
        make_task(params={"slot": 5, "grip_force": 30})
    with pytest.raises(ValidationError, match="grip_force"):
        make_task(params={"slot": 1, "grip_force": "strong"})
    assert make_task(params={"slot": 0}).params == {"slot": 0}
