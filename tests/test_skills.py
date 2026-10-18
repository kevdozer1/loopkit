"""S3: perception accuracy against simulator truth (tests may read privileged state; skills never do)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from loopkit import paths
from loopkit.sim.backend import SimBackend
from loopkit.sim.tasks import task_spec
from loopkit.skills import perception as P

pytestmark = pytest.mark.skipif(not (paths.panda_dir() / "panda.xml").is_file(), reason="Menagerie Panda not cloned")

# dev seeds, including the ones whose box estimate was worst before the face refinement (103, 146)
SEEDS = (3, 103, 146, 152)


def _yaw(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


@pytest.fixture(scope="module")
def scenes():
    out = {}
    geom = {**task_spec("T3").nominal_geometry, **task_spec("T2").nominal_geometry}
    for seed in SEEDS:
        b = SimBackend(cameras=("overhead", "front"))
        obs = b.reset(seed)
        pv = b.privileged
        out[seed] = (obs, geom, pv.body_pose("box"), pv.body_pose("plug"), pv.rack_T())
        b.close()
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_box_estimate_within_rail_clearance(scenes, seed):
    """The rails leave 1 mm per side, so the box's width position must be well inside that."""
    obs, geom, T, _, _ = scenes[seed]
    e = P.locate_box(obs, geom)
    c, s = math.cos(_yaw(T)), math.sin(_yaw(T))
    dx, dy = e.x - T[0, 3], e.y - T[1, 3]
    along_depth, along_width = c * dx + s * dy, -s * dx + c * dy
    assert abs(along_width) < 0.0005, along_width
    assert abs(along_depth) < 0.001, along_depth
    assert abs(math.degrees(P._wrap(e.yaw - _yaw(T)))) < 0.4
    assert e.extra["face_refined"]


@pytest.mark.parametrize("seed", SEEDS)
def test_plug_estimate(scenes, seed):
    obs, geom, _, T, _ = scenes[seed]
    e = P.locate_plug(obs, geom)
    assert np.hypot(e.x - T[0, 3], e.y - T[1, 3]) < 0.0015
    assert abs(math.degrees(P._wrap(e.yaw - _yaw(T)))) < 1.5


@pytest.mark.parametrize("seed", SEEDS)
def test_rack_pose_estimate(scenes, seed):
    obs, geom, _, _, RT = scenes[seed]
    est = P.estimate_rack_pose(obs, geom)
    assert np.hypot(est.x - RT[0, 3], est.y - RT[1, 3]) < 0.001
    assert abs(math.degrees(P._wrap(est.yaw - _yaw(RT)))) < 0.2
