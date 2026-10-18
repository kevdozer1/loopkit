"""The five typed interfaces between loopkit's parts (pydantic v2).

``uv run python -m loopkit.schemas`` writes their JSON Schema to ``loopkit/schemas/json/``.
"""

from .describe import describe
from .graph import (
    GRAPH_FORMAT_HINT,
    Edge,
    GraphError,
    Node,
    ProgramGraph,
    format_errors,
    is_ref,
    iter_refs,
    looks_like_ref,
    normalize_params,
    parse_graph,
    validate_graph,
)
from .observation import CameraExtrinsics, CameraImage, CameraIntrinsics, Observation
from .skill import SkillSpec, as_catalog, catalog_text
from .task import AABB, Budgets, Safety, TaskSpec
from .trace import (
    EVENT_KINDS,
    PAYLOAD_MODELS,
    BudgetStopPayload,
    InterventionPayload,
    ModelCallPayload,
    NodeEndPayload,
    NodeStartPayload,
    PatchPayload,
    TraceEvent,
    VerdictPayload,
)
from .verdict import FAILURE_CLASSES, Evidence, Verdict

__all__ = [
    "AABB", "Budgets", "BudgetStopPayload", "CameraExtrinsics", "CameraImage", "CameraIntrinsics", "EVENT_KINDS",
    "Edge", "Evidence", "FAILURE_CLASSES", "GRAPH_FORMAT_HINT", "GraphError", "InterventionPayload",
    "ModelCallPayload", "Node", "NodeEndPayload", "NodeStartPayload", "Observation", "PAYLOAD_MODELS",
    "PatchPayload", "ProgramGraph", "Safety", "SkillSpec", "TaskSpec", "TraceEvent", "Verdict", "VerdictPayload",
    "as_catalog", "catalog_text", "describe", "format_errors", "is_ref", "iter_refs", "looks_like_ref",
    "normalize_params", "parse_graph", "validate_graph",
]
