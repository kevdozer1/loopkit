"""TaskSpecs T1 (inspect), T2 (rack_slot) and T3 (plug_insert) on rack_cell.

``task_spec(task_id, seed)`` fills the per-seed params (target slot or port) from the
scene randomization. The writer sees ``TaskSpec.for_writer()``: the same spec without
the seed and the per-seed values, which a graph reads as ``{"ref": "task.<param>"}``.
"""

from __future__ import annotations

from ..schemas import AABB, Budgets, Safety, TaskSpec
from . import geometry as G
from .rack_cell import SCENE_ID, sample_config

TASK_IDS = ("T1", "T2", "T3")
SAFETY = Safety(
    workspace_bounds=AABB(min=G.WORKSPACE_MIN, max=G.WORKSPACE_MAX),
    max_speed=G.MAX_SPEED,
    max_force=G.MAX_FORCE,
)
LED_ENUM = list(G.LED_STATES)
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        str(n): {
            "type": "object",
            "properties": {"occupied": {"type": "boolean"}, "led": {"type": "string", "enum": LED_ENUM}},
            "required": ["occupied", "led"],
            "additionalProperties": False,
        }
        for n in range(1, G.N_PORTS + 1)
    },
    "required": [str(n) for n in range(1, G.N_PORTS + 1)],
    "additionalProperties": False,
}

# Skill names are the S3 skill library. done and abort are always allowed.
SKILLS_T1 = ["read_panel", "done", "abort"]
SKILLS_MOTION = ["move_to", "approach", "retreat", "grasp", "release"]
SKILLS_T2 = ["locate", *SKILLS_MOTION, "slide_into", "done", "abort"]
SKILLS_T3 = ["locate", "locate_port", *SKILLS_MOTION, "insert", "dance", "aligned_insert", "done", "abort"]


def _base(task_id: str) -> dict:
    nominal = G.nominal_geometry()
    if task_id == "T1":
        return dict(
            id="T1", title="inspect: read the patch panel",
            goal_text="Report, for each of the 12 patch-panel ports, whether a plug is in it and the colour of the "
                      "LED above it. Ports are numbered 1 to 12 on the label strip under the ports.",
            end_condition="A JSON report {\"1\": {\"occupied\": bool, \"led\": \"green|amber|red|off\"}, ..., \"12\": {...}} "
                          "for all 12 ports. Pass only when all 24 fields are right; the score is the fraction right.",
            checker_id="T1_inspect", allowed_skills=SKILLS_T1,
            budgets=Budgets(max_big_model_calls=1, max_usd=0.25, max_seconds=20.0),
            params_schema={}, output_schema=REPORT_SCHEMA,
            nominal_geometry={k: nominal[k] for k in ("frames", "rack", "ports", "appearance")},
        )
    if task_id == "T2":
        return dict(
            id="T2", title="rack_slot: slide the 1U box into slot n",
            goal_text="Pick the 1U box from its stand on the parts tray and slide it into shelf slot n of the rack "
                      "until it rests on both rails against the back stop, then let go and move the hand away.",
            end_condition="Box resting on both rails of slot n: position error at most 3 mm and yaw error at most "
                          "2 deg from the seated pose (against the back stop), gripper released.",
            checker_id="T2_rack_slot", allowed_skills=SKILLS_T2,
            budgets=Budgets(max_big_model_calls=0, max_usd=0.0, max_seconds=90.0),
            params_schema={"target_slot": {"type": "integer", "minimum": 1, "maximum": G.N_SLOTS,
                                           "description": "shelf slot to fill, 1 (lower) or 2 (upper)"}},
            nominal_geometry={k: nominal[k] for k in ("frames", "rack", "ports", "slots", "box", "tray", "robot", "appearance")},
        )
    if task_id == "T3":
        return dict(
            id="T3", title="plug_insert: insert the keyed plug into port n",
            goal_text="Pick the keyed plug from its holder on the parts tray and insert it into patch-panel port n "
                      "(key up), then let go and move the hand away.",
            end_condition="Plug in port n: tip at least 8 mm past the faceplate, lateral error at most 0.5 mm, "
                          "plug released and still seated after 1 s.",
            checker_id="T3_plug_insert", allowed_skills=SKILLS_T3,
            budgets=Budgets(max_big_model_calls=0, max_usd=0.0, max_seconds=90.0),
            params_schema={"target_port": {"type": "integer", "minimum": 1, "maximum": G.N_PORTS,
                                           "description": "port to fill, 1 to 12 (always free)"}},
            nominal_geometry={k: nominal[k] for k in ("frames", "rack", "ports", "plug", "tray", "robot", "appearance")},
        )
    raise ValueError(f"unknown task {task_id!r}; known: {', '.join(TASK_IDS)}")


def task_spec(task_id: str, seed: int | None = None) -> TaskSpec:
    """The TaskSpec for ``task_id``; with a seed, ``params`` holds that seed's target."""
    data = _base(task_id)
    params = {}
    if seed is not None:
        cfg = sample_config(seed)
        if task_id == "T2":
            params = {"target_slot": cfg.target_slot}
        elif task_id == "T3":
            params = {"target_port": cfg.target_port}
    return TaskSpec(scene_id=SCENE_ID, safety=SAFETY, params=params, seed=seed, **data)


def all_task_specs(seed: int | None = None) -> dict[str, TaskSpec]:
    return {t: task_spec(t, seed) for t in TASK_IDS}
