"""rack_cell geometry: the one source of truth for the scene, checkers, oracles and TaskSpecs.

Frames
------
* world: the robot base frame. Panda link0 at the origin, x forward, y left, z up.
* rack frame: origin on the floor below the centre of the rack front plane, x into
  the rack (away from the robot), y left, z up. Nominally at ``RACK_NOMINAL``; per
  seed it is shifted up to 2 cm in x and y and yawed up to 3 deg about z.
* part frames (box, plug): origin at the part centre, x along the insertion
  direction (away from the robot), y left, z up.

Real-world references (docs/02_task_suite.md): 19-inch panel 482.6 mm wide, 1U is
44.45 mm. The 1U box is scaled to 200 x 150 x 44.45 mm and 1.0 kg so the Panda can
carry it. Ports are 12 x 9 mm with 0.5 mm total clearance to the plug.
"""

from __future__ import annotations

import math

import numpy as np

U = 0.04445  # one rack unit, m

# --------------------------------------------------------------------------- rack
# The rack faces the robot obliquely: a horizontal Panda hand reaches poorly when it points
# straight away from the base (joint 6 cannot hyperextend), so every approach direction in the
# cell is 25 to 55 deg off the radial line from the base. This keeps the whole task on one arm branch.
RACK_NOMINAL = (0.60, 0.12, math.radians(50.0))   # x, y, yaw (rad) of the rack frame in world
RACK_JITTER_XY = 0.02                    # +- m
RACK_JITTER_YAW = math.radians(3.0)      # +- rad
RACK_Z0 = 0.19                           # bottom of slot 1 (rack frame z)
SLOT_PITCH = 1.5 * U                     # each shelf slot is 1.5U tall
N_SLOTS = 2
PANEL_BOTTOM = RACK_Z0 + 3 * U           # the patch panel row is the top 1U of the 4U section
PANEL_TOP = RACK_Z0 + 4 * U
PANEL_HALF_WIDTH = 0.4826 / 2
PANEL_THICK = 0.003                      # faceplate, x from -PANEL_THICK to 0
POST_Y = 0.2286                          # front post centre |y|
POST_SIZE = (0.03, 0.0127)               # x, y
RACK_DEPTH = 0.30

# shelf rails (L profile), rack frame
BOX_SIZE = (0.150, 0.200, U)             # depth (x), width (y), height (z)
BOX_MASS = 1.0
BOX_BEZEL_W = 0.012                       # magenta strip along the top front edge
RAIL_LATERAL_CLEARANCE = 0.001           # per side
RAIL_THICK = 0.003                       # horizontal flange thickness
RAIL_FLANGE_WIDTH = 0.015                # horizontal flange width under the box edge
RAIL_GUIDE_HEIGHT = 0.010                # vertical flange height above the rail top
RAIL_GUIDE_THICK = 0.002
RAIL_X0 = 0.0                            # rails start at the rack front plane
RAIL_FLARE_LEN = 0.015                   # entry flare length
RAIL_FLARE_OPEN = 0.006                  # extra half-width at the very front of the flare
BOX_PROTRUSION = 0.025                   # a seated box sticks out this far in front of the rack plane
STOP_X = BOX_SIZE[0] - BOX_PROTRUSION    # back stop face (rack frame x) = 0.125
RAIL_X1 = STOP_X + 0.003

# patch panel ports, rack frame
N_PORTS = 12
PORT_PITCH = 0.016                       # "about 15 mm": 16 mm leaves room for the mouth chamfer between ports
PORT_W, PORT_H = 0.012, 0.009            # opening width (y) and height (z)
PORT_DEPTH = 0.012                       # straight bore behind the mouth chamfer
PORT_CHAMFER = 0.0015                    # mouth chamfer, 45 deg, in the faceplate
PORT_WALL = 0.0015                       # bore wall thickness
PORT_Z = PANEL_BOTTOM + 0.0235           # port centre height
KEY_NOTCH_W, KEY_NOTCH_H = 0.004, 0.0013
LED_SIZE = (0.0015, 0.004, 0.003)        # x (protrusion), y, z full sizes
LED_DZ = 0.0115                          # LED centre above the port centre
LABEL_DZ = -0.0125                       # label strip centre below the port centre
LABEL_H = 0.0085

# plug (part frame: origin at the body centre, x towards the tip)
PLUG_BODY = (0.020, 0.0115, 0.0085)      # length (x), width (y), height (z)
PLUG_KEY = (0.016, 0.003, 0.001)         # ridge on top of the body
# The boot is a cable strain relief, narrower than the Panda fingertip V-groove (5 mm between the
# flank pads), so it nests in the groove: the centre pads squeeze its top and bottom and the flank
# pads centre it sideways to about +-0.3 mm.
PLUG_BOOT = (0.025, 0.0030, 0.007)       # strain-relief boot behind the body (the grasp handle)
PLUG_MASS_BODY = 0.008
PLUG_MASS_BOOT = 0.003
PLUG_SEATED_DEPTH = 0.012                # tip depth past the faceplate front when fully seated in an occupied port

LED_STATES = ("green", "amber", "red", "off")
LED_RGBA = {
    "green": (0.0, 1.0, 0.0, 1.0),
    "amber": (1.0, 0.6, 0.0, 1.0),
    "red": (1.0, 0.0, 0.0, 1.0),
    "off": (0.13, 0.13, 0.13, 1.0),
}
PANEL_RGBA = (0.62, 0.64, 0.67, 1.0)     # light steel, clearly lighter than an off LED

# ------------------------------------------------------------------------- tray
TRAY_CENTER = (0.46, -0.37)              # world x, y of the tray plate centre
TRAY_SIZE = (0.32, 0.42)                 # x, y
TRAY_TOP_Z = 0.06                        # low, so the hand body below a slot-1 approach clears it
# Part stands are tall because the Franka hand body spans +-104 mm along the finger axis, 40 to 130 mm
# behind the fingertips: with the fingers vertical, the space under the grasp line must stay clear.
BOX_RISER_SIZE = (0.10, 0.16, 0.180)     # under the back of the box: the front edge overhangs for the bottom finger
BOX_RISER_DX = 0.005                     # riser centre ahead of the box centre (box frame x), so 30 mm of the front overhangs
BOX_NOMINAL = (0.47, -0.34, math.radians(-5.0))   # world x, y, yaw of the box centre (resting on its riser)
PLUG_HOLDER_SIZE = (0.022, 0.012, 0.195)  # under the plug body only; the boot overhangs towards the robot
PLUG_HOLDER_DX = 0.001                   # holder centre ahead of the plug body centre (plug frame x)
# The plug stand sits beyond the box, away from the rack's approach corridor (the held box and the arm's
# wrist sweep near (0.4, -0.1) on slot-1 approaches).
PLUG_NOMINAL = (0.42, -0.52, math.radians(-15.0))  # world x, y, yaw of the plug body centre
PART_JITTER_XY = 0.005
PART_JITTER_YAW = math.radians(10.0)

BOX_REST_Z = TRAY_TOP_Z + BOX_RISER_SIZE[2] + BOX_SIZE[2] / 2
PLUG_REST_Z = TRAY_TOP_Z + PLUG_HOLDER_SIZE[2] + PLUG_BODY[2] / 2

# ------------------------------------------------------------------------ robot
TCP_OFFSET = 0.1034                      # hand frame z of the fingertip pad centres
GRIPPER_MAX_WIDTH = 0.08
WORKSPACE_MIN = (0.15, -0.60, 0.05)
WORKSPACE_MAX = (0.80, 0.45, 0.70)
MAX_SPEED = 0.25                         # m/s
MAX_FORCE = 60.0                         # N, estimated contact force at the hand

# --------------------------------------------------------------------- cameras
CAMERA_W, CAMERA_H = 640, 480


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_matrix(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw)
    T[:3, 3] = (x, y, z)
    return T


def port_y(n: int) -> float:
    """Rack frame y of port ``n`` (1..12). Port 1 is on the left as seen from the robot."""
    if not 1 <= n <= N_PORTS:
        raise ValueError(f"port {n} is not in 1..{N_PORTS}")
    return (N_PORTS + 1) / 2 * PORT_PITCH - n * PORT_PITCH


def port_mouth_rack(n: int) -> np.ndarray:
    """Rack frame point at the centre of port ``n``'s opening on the faceplate front."""
    return np.array([-PANEL_THICK, port_y(n), PORT_Z])


def slot_rail_top(k: int) -> float:
    """Rack frame z of the rail top (the seated box bottom) in slot ``k`` (1 or 2)."""
    if not 1 <= k <= N_SLOTS:
        raise ValueError(f"slot {k} is not in 1..{N_SLOTS}")
    return RACK_Z0 + (k - 1) * SLOT_PITCH + RAIL_THICK


def box_seated_rack(k: int) -> np.ndarray:
    """Rack frame centre of a box fully seated against the back stop in slot ``k``."""
    return np.array([STOP_X - BOX_SIZE[0] / 2, 0.0, slot_rail_top(k) + BOX_SIZE[2] / 2])


def plug_seated_rack(n: int, depth: float = PLUG_SEATED_DEPTH) -> np.ndarray:
    """Rack frame centre of a plug body whose tip is ``depth`` past the faceplate front in port ``n``."""
    tip_x = -PANEL_THICK + depth
    return np.array([tip_x - PLUG_BODY[0] / 2, port_y(n), PORT_Z])


def nominal_geometry() -> dict:
    """The fixture geometry a robot may know (the TaskSpec "BIM"), all nominal."""
    rx, ry, ryaw = RACK_NOMINAL
    return {
        "frames": "world = robot base (x forward, y left, z up). rack frame: origin on the floor below the centre "
                  "of the rack front plane, x into the rack, y left, z up. Distances in metres, angles in degrees.",
        "rack": {
            "nominal_pose": {"x": rx, "y": ry, "yaw_deg": math.degrees(ryaw)},
            "pose_tolerance": {"xy": RACK_JITTER_XY, "yaw_deg": math.degrees(RACK_JITTER_YAW)},
            "front_plane_x": 0.0,
            "faceplate_front_x": -PANEL_THICK,
            "panel": {"bottom_z": PANEL_BOTTOM, "top_z": PANEL_TOP, "half_width": PANEL_HALF_WIDTH},
        },
        "ports": {
            "count": N_PORTS,
            "pitch": PORT_PITCH,
            "opening_w": PORT_W,
            "opening_h": PORT_H,
            "bore_depth": PORT_DEPTH,
            "mouth_chamfer": PORT_CHAMFER,
            "center_z": PORT_Z,
            "center_y": {str(n): round(port_y(n), 6) for n in range(1, N_PORTS + 1)},
            "numbering": "port 1 is leftmost as seen from the robot (largest rack y)",
            "led_above_port_dz": LED_DZ,
            "label_below_port_dz": LABEL_DZ,
            "key": "the plug key is on top; the port's notch is in the top wall",
        },
        "slots": {
            "count": N_SLOTS,
            "rail_top_z": {str(k): round(slot_rail_top(k), 6) for k in range(1, N_SLOTS + 1)},
            "rail_inner_half_width": BOX_SIZE[1] / 2 + RAIL_LATERAL_CLEARANCE,
            "rail_start_x": RAIL_X0,
            "entry_flare": {"length": RAIL_FLARE_LEN, "extra_half_width": RAIL_FLARE_OPEN},
            "back_stop_x": STOP_X,
            "seated_box_center": {str(k): [round(v, 6) for v in box_seated_rack(k)] for k in range(1, N_SLOTS + 1)},
        },
        "box": {
            "size": {"depth_x": BOX_SIZE[0], "width_y": BOX_SIZE[1], "height_z": BOX_SIZE[2]},
            "mass_kg": BOX_MASS,
            "grasp": "pinch the top and bottom faces near the front edge; the seated box protrudes "
                     f"{BOX_PROTRUSION} m in front of the rack plane",
        },
        "plug": {
            "body": {"length_x": PLUG_BODY[0], "width_y": PLUG_BODY[1], "height_z": PLUG_BODY[2]},
            "boot": {"length_x": PLUG_BOOT[0], "width_y": PLUG_BOOT[1], "height_z": PLUG_BOOT[2]},
            "key": {"length_x": PLUG_KEY[0], "width_y": PLUG_KEY[1], "height_z": PLUG_KEY[2]},
            "grasp": "pinch the boot's top and bottom faces; the body points into the port",
        },
        "tray": {
            "center": list(TRAY_CENTER),
            "top_z": TRAY_TOP_Z,
            "box_nominal": {"x": BOX_NOMINAL[0], "y": BOX_NOMINAL[1], "yaw_deg": math.degrees(BOX_NOMINAL[2]),
                            "center_z": BOX_REST_Z},
            "plug_nominal": {"x": PLUG_NOMINAL[0], "y": PLUG_NOMINAL[1], "yaw_deg": math.degrees(PLUG_NOMINAL[2]),
                             "center_z": PLUG_REST_Z},
            "part_pose_tolerance": {"xy": PART_JITTER_XY, "yaw_deg": math.degrees(PART_JITTER_YAW)},
        },
        "robot": {
            "tcp": "between the fingertip pads; tool z is the approach axis, tool y the finger closing axis",
            "gripper_max_width": GRIPPER_MAX_WIDTH,
        },
        "appearance": {
            "box": {"top": "blue", "bezel": "magenta strip along the top of the front edge (the edge facing "
                    "the robot) and on the front face", "bezel_width": BOX_BEZEL_W,
                    "top_z": BOX_REST_Z + BOX_SIZE[2] / 2},
            "plug": {"body": "yellow, key on top", "boot": "dark green", "top_z": PLUG_REST_Z + PLUG_BODY[2] / 2},
            "ports": "a free port is a dark opening; an occupied port shows a yellow plug with a dark green boot",
            "leds": {"green": [0, 1, 0], "amber": [1, 0.6, 0], "red": [1, 0, 0], "off": "dark grey"},
            "panel": "light grey faceplate, white label strip with the port numbers under the ports",
        },
    }
