"""Hand-written reference graphs for T1 to T3.

They are not model output. Tests and FakeModel dry runs use them, and they measure what
the skill library can do with a sensible program, so a writer's result can be read
against them.
"""

from __future__ import annotations

import copy

_T1 = {
    "nodes": [
        {"id": "read", "skill": "read_panel", "params": {}, "max_retries": 1},
        {"id": "finish", "skill": "done", "params": {}},
        {"id": "give_up", "skill": "abort", "params": {"reason": "could not read the panel"}},
    ],
    "edges": [
        {"src": "read", "dst": "finish", "on": "success"},
        {"src": "read", "dst": "give_up", "on": "failure"},
    ],
    "entry": "read", "max_retries": 0, "terminal_nodes": ["finish", "give_up"],
}

_T2 = {
    "nodes": [
        {"id": "find_box", "skill": "locate", "params": {"name": "box"}, "max_retries": 1},
        {"id": "pick_box", "skill": "grasp", "params": {"part": "box", "pose": {"ref": "find_box.pose"}}, "max_retries": 1},
        {"id": "find_slot", "skill": "locate", "params": {"name": "slot", "index": {"ref": "task.target_slot"}},
         "max_retries": 1},
        {"id": "slide", "skill": "slide_into", "params": {"target": {"ref": "find_slot.pose"}}},
        {"id": "let_go", "skill": "release", "params": {}},
        {"id": "back_off", "skill": "retreat", "params": {"distance_mm": 80}},
        {"id": "finish", "skill": "done", "params": {}},
        {"id": "give_up", "skill": "abort", "params": {"reason": "a step failed"}},
    ],
    "edges": [
        {"src": "find_box", "dst": "pick_box", "on": "success"},
        {"src": "find_box", "dst": "give_up", "on": "failure"},
        {"src": "pick_box", "dst": "find_slot", "on": "success"},
        {"src": "pick_box", "dst": "give_up", "on": "failure"},
        {"src": "find_slot", "dst": "slide", "on": "success"},
        {"src": "find_slot", "dst": "give_up", "on": "failure"},
        {"src": "slide", "dst": "let_go", "on": "success"},
        {"src": "slide", "dst": "let_go", "on": "failure"},
        {"src": "let_go", "dst": "back_off", "on": "success"},
        {"src": "let_go", "dst": "give_up", "on": "failure"},
        {"src": "back_off", "dst": "finish", "on": "success"},
        {"src": "back_off", "dst": "finish", "on": "failure"},
    ],
    "entry": "find_box", "max_retries": 0, "terminal_nodes": ["finish", "give_up"],
}

_T3 = {
    "nodes": [
        {"id": "find_plug", "skill": "locate", "params": {"name": "plug"}, "max_retries": 1},
        {"id": "pick_plug", "skill": "grasp", "params": {"part": "plug", "pose": {"ref": "find_plug.pose"}},
         "max_retries": 1},
        {"id": "find_port", "skill": "locate_port", "params": {"n": {"ref": "task.target_port"}}, "max_retries": 1},
        {"id": "push", "skill": "insert", "params": {"target": {"ref": "find_port.pose"}}},
        {"id": "let_go", "skill": "release", "params": {}},
        {"id": "back_off", "skill": "retreat", "params": {"distance_mm": 60}},
        {"id": "finish", "skill": "done", "params": {}},
        {"id": "give_up", "skill": "abort", "params": {"reason": "a step failed"}},
    ],
    "edges": [
        {"src": "find_plug", "dst": "pick_plug", "on": "success"},
        {"src": "find_plug", "dst": "give_up", "on": "failure"},
        {"src": "pick_plug", "dst": "find_port", "on": "success"},
        {"src": "pick_plug", "dst": "give_up", "on": "failure"},
        {"src": "find_port", "dst": "push", "on": "success"},
        {"src": "find_port", "dst": "give_up", "on": "failure"},
        {"src": "push", "dst": "let_go", "on": "success"},
        {"src": "push", "dst": "give_up", "on": "failure"},
        {"src": "let_go", "dst": "back_off", "on": "success"},
        {"src": "let_go", "dst": "give_up", "on": "failure"},
        {"src": "back_off", "dst": "finish", "on": "success"},
        {"src": "back_off", "dst": "finish", "on": "failure"},
    ],
    "entry": "find_plug", "max_retries": 0, "terminal_nodes": ["finish", "give_up"],
}

REFERENCE = {"T1": _T1, "T2": _T2, "T3": _T3}


def reference_graph(task_id: str) -> dict:
    return copy.deepcopy(REFERENCE[task_id])


def t3_with_dance() -> dict:
    """T3 with dance before the insert (aligned_insert uses its correction)."""
    g = reference_graph("T3")
    for n in g["nodes"]:
        if n["id"] == "push":
            n["skill"] = "aligned_insert"
    g["nodes"].insert(3, {"id": "calibrate", "skill": "dance", "params": {"target": {"ref": "find_port.pose"}}})
    g["edges"] = [e for e in g["edges"] if not (e["src"] == "find_port" and e["on"] == "success")]
    g["edges"] += [{"src": "find_port", "dst": "calibrate", "on": "success"},
                   {"src": "calibrate", "dst": "push", "on": "success"},
                   {"src": "calibrate", "dst": "give_up", "on": "failure"}]
    return g
