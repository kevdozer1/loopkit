"""The rack_cell skill library.

Every skill is a typed function whose signature and docstring generate its SkillSpec.
Skills see camera images, proprioception and the TaskSpec's nominal geometry only.
"""

from .base import REGISTRY, Pose, SkillContext, SkillFailure, call_skill, catalog
from .library import SKILL_NAMES

__all__ = ["REGISTRY", "Pose", "SKILL_NAMES", "SkillContext", "SkillFailure", "call_skill", "catalog"]
