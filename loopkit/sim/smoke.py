"""S0 smoke: load the Menagerie Panda, step 100 times, render one 640 x 480 frame.

``uv run python -m loopkit.sim.smoke`` writes ``LOOPKIT_DATA/runs/smoke.png``.
MUJOCO_GL is left unset: the default WGL/GLFW path works on Windows, egl does not.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ..paths import panda_dir, runs_dir

WIDTH, HEIGHT = 640, 480
PANDA_VISUAL_GROUP = 2  # Menagerie Panda: class "visual" geoms are group 2, collision group 3


def load_panda_scene() -> mujoco.MjModel:
    scene = panda_dir() / "scene.xml"
    if not scene.is_file():
        raise FileNotFoundError(
            f"{scene} is missing. Clone it with: git clone --depth 1 "
            "https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie"
        )
    return mujoco.MjModel.from_xml_path(str(scene))


def fixed_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.25, 0.0, 0.45]
    cam.distance = 1.9
    cam.azimuth = 140.0
    cam.elevation = -22.0
    return cam


def render(model: mujoco.MjModel, data: mujoco.MjData, show_robot: bool = True) -> np.ndarray:
    option = mujoco.MjvOption()
    option.geomgroup[PANDA_VISUAL_GROUP] = 1 if show_robot else 0
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)  # height first
    try:
        renderer.update_scene(data, camera=fixed_camera(), scene_option=option)
        return renderer.render().copy()
    finally:
        renderer.close()


def run_smoke(out: Path | None = None, steps: int = 100) -> dict:
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ.pop("MUJOCO_GL")
    model = load_panda_scene()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    rgb = render(model, data, show_robot=True)
    empty = render(model, data, show_robot=False)
    diff = np.abs(rgb.astype(np.int16) - empty.astype(np.int16)).max(axis=2)
    robot_pixel_share = float((diff > 25).mean())
    out = Path(out) if out is not None else runs_dir() / "smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    return {
        "png": str(out),
        "shape": list(rgb.shape),
        "steps": steps,
        "sim_time_s": round(float(data.time), 4),
        "timestep_s": float(model.opt.timestep),
        "nq": int(model.nq),
        "nu": int(model.nu),
        "robot_pixel_share": round(robot_pixel_share, 4),
        "finite_qpos": bool(np.isfinite(data.qpos).all()),
        "mujoco": mujoco.__version__,
    }


if __name__ == "__main__":
    print(run_smoke(), flush=True)
