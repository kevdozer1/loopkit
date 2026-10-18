"""Importing this module registers every skill of the rack_cell library."""

from . import calibration, manipulation, motion, perceive, terminal  # noqa: F401

SKILL_NAMES = ("locate", "locate_port", "read_panel", "move_to", "approach", "retreat", "grasp", "release",
               "slide_into", "insert", "dance", "aligned_insert", "done", "abort")
