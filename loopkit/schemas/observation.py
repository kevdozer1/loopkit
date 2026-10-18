"""Observation: everything a skill may see. Cameras and proprioception only.

There are no body poses here. ``tcp_position`` and ``tcp_rotation`` come from the
robot's own joint encoders through its kinematic model, which a real arm reports
too; ``wrench_estimate`` is the external wrench estimated from joint torques, as
the Panda reports it. Nothing about the rack, the box or the plug is in this
type except what the camera pixels show.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics in pixels (OpenCV convention)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fovy_deg: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


class CameraExtrinsics(BaseModel):
    """Camera pose in the robot base frame. ``rotation`` columns are the camera's
    x (right), y (down) and z (forward, optical axis) axes in base coordinates."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    position: Vec3
    rotation: Mat3

    def camera_to_base(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = np.asarray(self.rotation, dtype=float)
        T[:3, 3] = np.asarray(self.position, dtype=float)
        return T

    def base_to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_base())


class CameraImage(BaseModel):
    """One camera's frame. Equal when every field matches and the pixels are equal (np.array_equal)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, allow_inf_nan=False)

    name: str
    path: str | None = None
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    rgb: SkipJsonSchema[np.ndarray | None] = Field(default=None, exclude=True, repr=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CameraImage):
            return NotImplemented
        if (self.name, self.path, self.intrinsics, self.extrinsics) != (
                other.name, other.path, other.intrinsics, other.extrinsics):
            return False
        if self.rgb is None or other.rgb is None:
            return self.rgb is None and other.rgb is None
        return bool(np.array_equal(self.rgb, other.rgb))


class Observation(BaseModel):
    """What a skill sees. Each key of ``cameras`` is that camera's name."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, allow_inf_nan=False)

    cameras: dict[str, CameraImage]
    joint_positions: list[float]
    joint_velocities: list[float]
    gripper_width: float = Field(description="Distance between the finger pads, m")
    gripper_force: float = Field(description="Grip force estimate from the gripper actuator, N")
    tcp_position: Vec3 = Field(description="Tool centre point from forward kinematics, base frame, m")
    tcp_rotation: Mat3 = Field(description="Tool frame axes (columns) from forward kinematics, base frame")
    wrench_estimate: list[float] | None = Field(
        default=None, description="External force (N) and torque (Nm) at the hand estimated from joint torques"
    )
    sim_time: float

    @model_validator(mode="after")
    def _camera_keys_are_names(self) -> "Observation":
        wrong = [f"cameras[{key!r}] holds the camera named {cam.name!r}"
                 for key, cam in self.cameras.items() if key != cam.name]
        if wrong:
            raise ValueError("each cameras key must be its camera's name: " + "; ".join(wrong))
        return self

    def image(self, name: str) -> np.ndarray:
        cam = self.cameras[name]
        if cam.rgb is None:
            raise ValueError(f"camera {name!r} has no pixels in this observation")
        return cam.rgb

    def summary(self) -> dict[str, Any]:
        """JSON-safe description without pixels."""
        return self.model_dump(mode="json")
