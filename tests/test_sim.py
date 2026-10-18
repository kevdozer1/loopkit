"""S2: rack_cell scene, SimBackend, TaskSpecs, checkers, oracles, and the privileged-state guard."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from loopkit import paths
from loopkit.runtime.backend import Grip, MoveTo, Pose
from loopkit.schemas import Observation, TaskSpec
from loopkit.sim import geometry as G
from loopkit.sim import rack_cell
from loopkit.sim.backend import SimBackend
from loopkit.sim.checkers import RunInfo, check
from loopkit.sim.oracles import do_nothing, oracle_t2, oracle_t3, run_program
from loopkit.sim.tasks import TASK_IDS, task_spec

pytestmark = pytest.mark.skipif(not (paths.panda_dir() / "panda.xml").is_file(), reason="Menagerie Panda not cloned")


# ------------------------------------------------------------------ scene and randomization
def test_same_seed_same_scene_bit_for_bit():
    a, b = rack_cell.make_scene(42), rack_cell.make_scene(42)
    assert a.xml == b.xml and a.cfg == b.cfg
    for field in ("body_pos", "body_quat", "geom_pos", "geom_size", "geom_rgba", "light_pos", "light_dir", "light_diffuse"):
        assert np.array_equal(getattr(a.model, field), getattr(b.model, field)), field
    assert rack_cell.make_scene(43).xml != a.xml


def test_same_seed_same_observation_bit_for_bit():
    hashes = []
    for _ in range(2):
        backend = SimBackend()
        obs = backend.reset(7)
        hashes.append((tuple(obs.joint_positions), b"".join(obs.image(c).tobytes() for c in sorted(obs.cameras))))
        backend.close()
    assert hashes[0] == hashes[1]


def test_randomization_ranges():
    rx, ry, ryaw = G.RACK_NOMINAL
    for seed in range(100):
        cfg = rack_cell.sample_config(seed)
        x, y, yaw = cfg.rack_pose
        assert abs(x - rx) <= G.RACK_JITTER_XY + 1e-9 and abs(y - ry) <= G.RACK_JITTER_XY + 1e-9
        assert abs(yaw - ryaw) <= G.RACK_JITTER_YAW + 1e-9
        assert 2 <= len(cfg.lights) <= 4
        assert 0 <= len(cfg.occupied) <= 6 and len(set(cfg.occupied)) == len(cfg.occupied)
        assert cfg.target_port not in cfg.occupied and 1 <= cfg.target_port <= G.N_PORTS
        assert cfg.target_slot in (1, 2)
        assert set(cfg.led_states) <= set(G.LED_STATES) and len(cfg.led_states) == G.N_PORTS
        for (px, py, pyaw), (nx, ny, nyaw) in ((cfg.box_pose, G.BOX_NOMINAL), (cfg.plug_pose, G.PLUG_NOMINAL)):
            assert abs(px - nx) <= G.PART_JITTER_XY + 1e-9 and abs(py - ny) <= G.PART_JITTER_XY + 1e-9
            assert abs(pyaw - nyaw) <= G.PART_JITTER_YAW + 1e-9
    occupancy = [len(rack_cell.sample_config(s).occupied) for s in range(200)]
    assert min(occupancy) == 0 and max(occupancy) == 6


# ------------------------------------------------------------------ observation and privileged-state guard
ALLOWED_OBSERVATION_FIELDS = {
    "cameras", "joint_positions", "joint_velocities", "gripper_width", "gripper_force",
    "tcp_position", "tcp_rotation", "wrench_estimate", "sim_time",
}


def test_observation_carries_no_object_pose():
    assert set(Observation.model_fields) == ALLOWED_OBSERVATION_FIELDS
    backend = SimBackend()
    obs = backend.reset(3)
    assert set(obs.cameras) == {"overhead", "front", "wrist"}
    dumped = obs.model_dump(mode="json")
    text = str(dumped)
    for forbidden in ("box", "plug", "rack", "port", "slot", "qpos"):
        assert forbidden not in text, forbidden
    for cam in obs.cameras.values():
        assert cam.rgb.shape == (480, 640, 3) and cam.rgb.dtype == np.uint8
    backend.close()


def test_backend_hides_mujoco_state():
    backend = SimBackend()
    backend.reset(1)
    public = {n for n in dir(backend) if not n.startswith("_")}
    assert "model" not in public and "data" not in public and "scene" not in public
    # the only door to simulator truth is the documented privileged view (checkers and oracles)
    assert hasattr(backend, "privileged")
    backend.close()


# What each guarded package may import from the simulator side. Skills and the interpreter get
# everything through SkillContext (Observations and act) and the TaskSpec. The program package may
# build TaskSpecs and hand a graph to the harness's episode grader (loopkit.sim.episode), whose
# plain-data record is all it sees: checker verdicts and the scene facts that docs/01_architecture.md
# asks failure summaries to report. Nothing in these packages may touch the backend's privileged view.
SIM_IMPORTS_ALLOWED = {
    "skills": (),
    "runtime": (),
    "program": ("loopkit.sim.tasks", "loopkit.sim.episode"),
}
FORBIDDEN_NAMES = ("privileged", "PrivilegedView", "_scene", "_arm")


def _imports(tree: ast.AST, module: str) -> set[str]:
    """Fully qualified names a module imports (modules, or module.symbol for from-imports)."""
    out = set()
    pkg = module.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
            out |= {base if a.name == "*" else f"{base}.{a.name}" for a in node.names}
    return out


def _is_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_skills_cannot_reach_privileged_state():
    """CLAUDE.md rule 10: skills see Observations only. Checkers and oracles may read privileged state."""
    offenders = []
    root = paths.REPO_ROOT / "loopkit"
    for pkg, allowed in SIM_IMPORTS_ALLOWED.items():
        for py in (root / pkg).rglob("*.py"):
            module = "loopkit." + ".".join(py.relative_to(root).with_suffix("").parts)
            rel = py.relative_to(paths.REPO_ROOT)
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for name in _imports(tree, module):
                if _is_under(name, "mujoco") or (_is_under(name, "loopkit.sim")
                                                 and not any(_is_under(name, a) for a in allowed)):
                    offenders.append(f"{rel} imports {name}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                    offenders.append(f"{rel}:{node.lineno} uses .{node.attr}")
                elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                    offenders.append(f"{rel}:{node.lineno} uses {node.id}")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_NAMES:
                    offenders.append(f"{rel}:{node.lineno} names {node.value!r}")
    assert not offenders, offenders


def test_guard_catches_a_skill_that_peeks():
    """The guard itself works: a skill that reads the privileged view is reported."""
    src = ("from ..sim.checkers import check\n"
           "from loopkit.sim import rack_cell\n"
           "def s(ctx):\n"
           "    return ctx.backend.privileged\n")
    tree = ast.parse(src)
    names = _imports(tree, "loopkit.skills.peek")
    assert "loopkit.sim.checkers.check" in names and "loopkit.sim.rack_cell" in names
    assert any(isinstance(n, ast.Attribute) and n.attr == "privileged" for n in ast.walk(tree))


def test_camera_model_projects_leds_onto_their_pixels():
    """Intrinsics and extrinsics in the Observation are right: LED centres land on LED-coloured pixels."""
    want = {"green": (0, 255, 0), "red": (255, 0, 0)}
    for seed in (2, 9):
        backend = SimBackend(cameras=("front",))
        obs = backend.reset(seed)
        cam = obs.cameras["front"]
        K = cam.intrinsics.matrix()
        Tcb = cam.extrinsics.base_to_camera()
        img = obs.image("front").astype(int)
        pv = backend.privileged
        checked = 0
        for n in range(1, G.N_PORTS + 1):
            state = pv.cfg.led_states[n - 1]
            if state not in want:
                continue
            p = pv.rack_to_world((-G.PANEL_THICK - 0.0008, G.port_y(n), G.PORT_Z + G.LED_DZ))
            pc = Tcb[:3, :3] @ p + Tcb[:3, 3]
            u, v, w = K @ pc
            px = img[int(round(v / w)), int(round(u / w))]
            assert np.abs(px - np.array(want[state])).max() <= 40, (seed, n, state, px)
            checked += 1
        assert checked > 0
        backend.close()


# ------------------------------------------------------------------ tasks
def test_task_specs_round_trip_and_params():
    for seed in (0, 5):
        cfg = rack_cell.sample_config(seed)
        for tid in TASK_IDS:
            spec = task_spec(tid, seed)
            assert TaskSpec.model_validate_json(spec.model_dump_json()) == spec
            assert "seed" not in spec.for_writer() and "params" not in spec.for_writer()
            assert {"done", "abort"} <= set(spec.allowed_skills)
        assert task_spec("T2", seed).params == {"target_slot": cfg.target_slot}
        assert task_spec("T3", seed).params == {"target_port": cfg.target_port}
    assert task_spec("T3").nominal_geometry["ports"]["count"] == 12


# ------------------------------------------------------------------ backend motion and safety
def test_move_outside_workspace_is_a_safety_violation():
    backend = SimBackend()
    backend.reset(0)
    r = backend.act(MoveTo(pose=Pose(position=(0.5, 0.0, 1.5), yaw_deg=0.0)))
    assert not r.ok and r.safety_violation
    backend.close()


def test_grip_reports_width_and_force_on_the_box():
    backend = SimBackend()
    backend.reset(3)
    pv = backend.privileged
    T = pv.body_pose("box")
    yaw = math.atan2(T[1, 0], T[0, 0])
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    grasp = T[:3, 3] + fwd * (-G.BOX_SIZE[0] / 2 + 0.012)
    backend.act(Grip(width=0.075))
    assert backend.act(MoveTo(pose=Pose(position=tuple(grasp - 0.07 * fwd), yaw_deg=math.degrees(yaw)), speed=0.15)).ok
    assert backend.act(MoveTo(pose=Pose(position=tuple(grasp), yaw_deg=math.degrees(yaw)), speed=0.05)).ok
    r = backend.act(Grip(force=70.0))
    assert abs(r.gripper_width - G.BOX_SIZE[2]) < 0.002
    assert abs(r.gripper_force - 70.0) < 1.0
    backend.close()


# ------------------------------------------------------------------ checkers, oracles, do-nothing
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_do_nothing_scores_zero(task_id):
    for seed in (1, 2):
        rec = run_program(task_id, seed, do_nothing)
        assert rec["score"] == 0.0 and not rec["passed"]


@pytest.mark.parametrize("task_id,program", [("T2", oracle_t2), ("T3", oracle_t3)])
def test_oracle_passes(task_id, program):
    rec = run_program(task_id, 4, program)
    assert rec["passed"], rec["summary"]
    assert rec["score"] == 1.0 and rec["error"] is None


def test_wrong_port_is_a_perception_failure():
    seed = 4
    spec = task_spec("T3", seed)
    wrong = next(p for p in range(1, 13) if p != spec.params["target_port"] and p not in rack_cell.sample_config(seed).occupied)
    backend = SimBackend(safety=spec.safety, render=False)
    backend.reset(seed)
    oracle_t3(backend, spec.model_copy(update={"params": {"target_port": wrong}}))
    verdict = check(spec, backend, {}, RunInfo())
    assert not verdict.passed and verdict.failure_class == "perception"
    assert verdict.evidence.metrics["in_port"] == wrong
    backend.close()


def test_t1_checker_scores_fields():
    backend = SimBackend(render=False)
    backend.reset(5)
    spec = task_spec("T1", 5)
    from loopkit.sim.checkers import truth_report

    truth = truth_report(backend)
    assert check(spec, backend, {"report": truth}).passed
    flipped = {k: dict(v) for k, v in truth.items()}
    flipped["3"]["occupied"] = not flipped["3"]["occupied"]
    v = check(spec, backend, {"report": flipped})
    assert not v.passed and v.failure_class == "perception" and abs(v.score - 23 / 24) < 1e-9
    backend.close()
