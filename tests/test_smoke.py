from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from loopkit import paths
from loopkit.sim import smoke

pytestmark = pytest.mark.skipif(
    not (paths.panda_dir() / "scene.xml").is_file(), reason="Menagerie Panda not cloned into third_party/"
)


def test_panda_loads_steps_and_renders(tmp_path):
    out = tmp_path / "smoke.png"
    result = smoke.run_smoke(out=out, steps=100)
    assert result["finite_qpos"]
    assert result["shape"] == [480, 640, 3]
    assert result["nu"] >= 8  # 7 arm actuators plus the gripper
    image = np.asarray(Image.open(out))
    assert image.shape == (480, 640, 3)
    assert image.std() > 5, "frame is blank"
    # the arm itself is in view: hiding its visual geoms changes a real share of pixels
    assert result["robot_pixel_share"] > 0.03, result
