"""loopkit: a small model-agnostic loop for robot task programs.

A frontier model writes a program graph from a typed skill library, the graph is
rehearsed in MuJoCo, a privileged checker scores it, failures go back to the
writer as text, and every event is traced.
"""

__version__ = "0.1.0"
