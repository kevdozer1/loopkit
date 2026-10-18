"""The program writer's prompt.

Parts, in order: the role and the graph rules; the TaskSpec as plain text (the nominal
geometry as a JSON block, never JSON nested inside a string); the skill catalog built from
SkillSpecs only; the graph format with one tiny worked example on a different task; and,
in patch rounds, the current graph and a text summary of how it failed. The three camera
images go with the message.
"""

from __future__ import annotations

import json

from ..schemas import GRAPH_FORMAT_HINT, ProgramGraph, SkillSpec, TaskSpec, catalog_text

SYSTEM = """You write robot programs. A program is a graph of skill calls, given as one JSON object.

Rules:
- Use only the skills listed in the catalog for this task, with their params exactly as documented.
- Every node that is not done or abort has exactly one success edge and one failure edge.
- A node's outputs exist only after it succeeds. Read them with {"ref": "<node_id>.<field>"}, for example
  {"ref": "find_box.pose"}. Read a task parameter with {"ref": "task.<name>"}.
- A failed node is retried in place up to max_retries times (graph default, or set on the node), then its
  failure edge is taken. A loop through edges needs max_visits on a node of the loop whose failure edge
  leaves the loop; a node past its max_visits fails without running.
- The program runs with no model in the loop: it must finish on its own. End with a done node when the task
  is complete and an abort node when it cannot be.
- Answer with the JSON object only: no prose, no code fences."""

EXAMPLE_TASK = "Example task (not yours): look at the box, then hold the hand 10 cm in front of it."
EXAMPLE_GRAPH = {
    "nodes": [
        {"id": "find", "skill": "locate", "params": {"name": "box"}, "max_retries": 1},
        {"id": "hover", "skill": "approach", "params": {"target": {"ref": "find.pose"}, "offset_mm": 100}},
        {"id": "finish", "skill": "done", "params": {}},
        {"id": "give_up", "skill": "abort", "params": {"reason": "box not found or not reachable"}},
    ],
    "edges": [
        {"src": "find", "dst": "hover", "on": "success"},
        {"src": "find", "dst": "give_up", "on": "failure"},
        {"src": "hover", "dst": "finish", "on": "success"},
        {"src": "hover", "dst": "give_up", "on": "failure"},
    ],
    "entry": "find",
    "max_retries": 0,
    "terminal_nodes": ["finish", "give_up"],
}


def task_text(task: TaskSpec) -> str:
    w = task.for_writer()
    lines = [f"TASK {w['id']}: {w['title']}", f"Goal: {w['goal_text']}", f"End condition (what the checker scores): "
             f"{w['end_condition']}", "Allowed skills: " + ", ".join(w["allowed_skills"])]
    if w.get("params_schema"):
        lines.append("Task parameters (the value changes from run to run; read it with {\"ref\": \"task.<name>\"}):")
        for name, schema in w["params_schema"].items():
            desc = schema.get("description", "")
            rng = f" from {schema.get('minimum')} to {schema.get('maximum')}" if "minimum" in schema else ""
            lines.append(f"  - {name}: {schema.get('type', 'value')}{rng}. {desc}".rstrip())
    b = w["budgets"]
    lines.append(f"Budgets per run: {b['max_seconds']:.0f} s of robot time, {b['max_big_model_calls']} model calls "
                 f"while running, ${b['max_usd']:.2f} for those calls.")
    s = w["safety"]
    lines.append(f"Safety: tool speed at most {s['max_speed']} m/s, contact force at most {s['max_force']} N, the tool "
                 f"centre point stays in the box {s['workspace_bounds']['min']} to {s['workspace_bounds']['max']} m.")
    if w.get("output_schema"):
        lines.append("The program's result is the output of its read_panel node (the panel report).")
    lines.append("Nominal fixture geometry (what the robot may know; the real poses differ a little each run, so "
                 "locate things before using them):")
    lines.append(json.dumps(w["nominal_geometry"], indent=1, sort_keys=True))
    return "\n".join(lines)


def example_text() -> str:
    return (f"GRAPH FORMAT\n{GRAPH_FORMAT_HINT}\n\n{EXAMPLE_TASK}\nIts graph:\n"
            + json.dumps(EXAMPLE_GRAPH, indent=1))


def build_messages(task: TaskSpec, catalog: dict[str, SkillSpec], *, previous: ProgramGraph | None = None,
                   failure_summary: str | None = None, image_names: tuple[str, ...] = ()) -> list[dict]:
    parts = [task_text(task), "SKILL CATALOG\n" + catalog_text(catalog, task.allowed_skills), example_text()]
    if image_names:
        parts.append("The attached images are the " + ", ".join(image_names) + " camera views at the start of a "
                     "training run, in that order.")
    if previous is not None:
        parts.append("YOUR CURRENT PROGRAM\n" + json.dumps(previous.model_dump(mode="json", exclude_none=True), indent=1))
    if failure_summary:
        parts.append("HOW IT DID IN REHEARSAL\n" + failure_summary)
        parts.append("Write an improved program graph. Change what the rehearsal shows is wrong; keep what works.")
    else:
        parts.append("Write the program graph for this task.")
    parts.append("Return the JSON object only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def repair_message(errors_text: str) -> dict:
    return {"role": "user", "content": "The graph does not validate:\n" + errors_text
            + "\nReturn the corrected graph as one JSON object, nothing else."}
