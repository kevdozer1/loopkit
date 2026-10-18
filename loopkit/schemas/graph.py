"""ProgramGraph: the program a writer produces, and its validator.

A graph is nodes (one skill call each), edges taken on ``success`` or
``failure``, an ``entry`` node and the ``terminal_nodes`` (the ``done`` and
``abort`` nodes). A param is a literal or a reference ``{"ref": "<node_id>.<field>"}``
to an earlier node's output, or ``{"ref": "task.<param>"}`` to a task parameter.
A ref path may go deeper into an output: ``locate.pose.position.0``.

Run semantics, which the interpreter follows:

* ``max_retries`` (graph default, node override): a failed node is re-run in
  place up to this many times before its failure edge is taken. In-place
  retries are always bounded.
* ``max_visits`` (node): a node visited more than ``max_visits`` times fails
  without running and takes its failure edge. That is how a loop through edges
  ends, so the validator removes the success edges of every node that sets
  ``max_visits`` and reports each cycle that is left (``unbounded_cycle``).
* A node's outputs exist only after it succeeds. A ref from node R to node S is
  valid only if every path from the entry to R leaves S through a success edge.
* Following success edges from the entry must reach a done node.
* Params are handed on as validated JSON. ``normalize_params(graph, catalog)``
  turns integral floats into ints wherever the skill's schema says integer; the
  interpreter calls it before running.

``validate(catalog)`` returns readable errors. Each one names the node and says
how to fix it, because the text goes back to the writer. Codes that start with
``config_`` are setup errors (the task and the catalog disagree), not the writer's.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, Strict, ValidationError, field_validator

from .describe import describe, unwrap_optional
from .jsonschema_lite import check as check_schema
from .jsonschema_lite import is_number, resolve, short, type_names
from .skill import SkillSpec, as_catalog
from .task import TaskSpec

NODE_ID_RE = re.compile(r"\A[A-Za-z][A-Za-z0-9_\-]{0,63}\Z")
REF_RE = re.compile(r"\A(?P<node>[A-Za-z][A-Za-z0-9_\-]*)\.(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\Z")
TASK_REF = "task"
MAX_RETRIES_LIMIT = 10
MAX_VISITS_LIMIT = 50
MAX_PARAM_DEPTH = 32
REF_FORM = '{"ref": "<node_id>.<field>"}'

GRAPH_FORMAT_HINT = (
    'The top level is {"nodes": [...], "edges": [...], "entry": "<node id>", "max_retries": 1, '
    '"terminal_nodes": [...]}. A node is {"id": "...", "skill": "...", "params": {...}} and may also set '
    '"max_retries" (in-place retries after a failure, 0 to 10), "max_visits" (1 to 50: a node visited more '
    'than max_visits times fails without running and takes its failure edge, which is how a loop ends) and '
    '"timeout_s" (seconds, greater than 0). An edge is {"src": "<node id>", "dst": "<node id>", "on": '
    '"success" or "failure"}. A param is a literal or {"ref": "<node_id>.<field>"} for an output of a node '
    'that has succeeded earlier, or {"ref": "task.<param>"}.'
)

_FORMAT_CODES = frozenset({"bad_json", "bad_shape", "empty_graph"})
_CYCLE_FIX = ("Set max_visits on a node of this loop whose failure edge leaves the loop, or remove the back edge. "
              "Retrying one node needs no loop: use max_retries")
_SUCCESS_PATH_FIX = ("Point each node's success edge toward the goal, so that when every node succeeds the program "
                     "ends at a done node. Failure edges go to a recovery node or an abort node")
_JSON_FIX = "Return one JSON object with double-quoted keys and strings: no comments, no trailing commas"
_CUT_OFF_FIX = "If your reply was cut off, return a shorter graph"


# ------------------------------------------------------------------------ refs
def is_ref(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"ref"} and isinstance(value["ref"], str)


def _children(value: Any) -> Iterable[Any]:
    return value.values() if isinstance(value, dict) else value


def iter_refs(value: Any) -> Iterator[str]:
    """Every ``{"ref": ...}`` string inside a param value, in order.

    Iterative, so deep nesting cannot overflow the stack, and a container that holds
    itself is not entered again.
    """
    if is_ref(value):
        yield value["ref"]
        return
    if not isinstance(value, (dict, list, tuple)):
        return
    stack = [iter(_children(value))]
    path = [id(value)]  # the containers being walked, outermost first
    on_path = {id(value)}
    while stack:
        for item in stack[-1]:
            if is_ref(item):
                yield item["ref"]
            elif isinstance(item, (dict, list, tuple)) and id(item) not in on_path:
                stack.append(iter(_children(item)))
                path.append(id(item))
                on_path.add(id(item))
                break
        else:
            stack.pop()
            on_path.discard(path.pop())


def looks_like_ref(value: Any) -> bool:
    """A literal a writer probably meant as a ref: a "node.field" string, ``{"$ref": ...}``,
    or a dict with "ref" and other keys."""
    if isinstance(value, str):
        return REF_RE.fullmatch(value) is not None
    if isinstance(value, dict):
        return "$ref" in value or ("ref" in value and not is_ref(value))
    return False


def _contains_near_ref(value: Any) -> bool:
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if looks_like_ref(item):
            return True
        if depth < MAX_PARAM_DEPTH:
            if isinstance(item, dict):
                stack.extend((v, depth + 1) for v in item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend((v, depth + 1) for v in item)
    return False


def _too_deep(value: Any, limit: int = MAX_PARAM_DEPTH) -> bool:
    """True when ``value`` nests containers more than ``limit`` levels deep (iterative)."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children: Iterable[Any] = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        else:
            continue
        if depth > limit:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False


# ---------------------------------------------------------------------- models
def _whole_number(value: Any) -> Any:
    """1.0 -> 1, so an integral float counts as a whole number. Booleans and strings fail the strict int."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _count(lo: int, hi: int) -> Any:
    """A strict whole number from ``lo`` to ``hi``. The bounds sit inside the type, so the JSON Schema says
    minimum/maximum (pydantic cannot move Field bounds through the before-validator of an Optional)."""
    return Annotated[int, Strict(), Field(ge=lo, le=hi), BeforeValidator(_whole_number)]


Retries = _count(0, MAX_RETRIES_LIMIT)
Visits = _count(1, MAX_VISITS_LIMIT)
Seconds = Annotated[float, Strict(), Field(gt=0, allow_inf_nan=False)]


class Node(BaseModel):
    """One skill call."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(description="Unique node id: a letter, then letters, digits, '_' or '-', at most 64 characters")
    skill: str = Field(description="The name of a skill in the catalog")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="The skill's params. Each value is a literal or {\"ref\": \"<node_id>.<field>\"}, an output of "
                    "a node that succeeds on every path to this one, or {\"ref\": \"task.<param>\"}",
    )
    max_retries: Retries | None = Field(
        default=None,
        description="In-place retries: a failed node runs again up to this many times before its failure edge is "
                    "taken. Overrides the graph's max_retries",
    )
    max_visits: Visits | None = Field(
        default=None,
        description="Loop bound: a node visited more than max_visits times fails without running and takes its "
                    "failure edge. A loop through edges ends only at a node with max_visits whose failure edge "
                    "leaves the loop",
    )
    timeout_s: Seconds | None = Field(default=None, description="Seconds the node may run before it fails")


class Edge(BaseModel):
    """After ``src`` finishes with outcome ``on``, ``dst`` runs next."""

    model_config = ConfigDict(extra="forbid")

    src: str = Field(description="The id of the node this edge leaves")
    dst: str = Field(description="The id of the node that runs next")
    on: Literal["success", "failure"] = Field(
        description="The outcome of src that takes this edge. Every non-terminal node has one success edge and "
                    "one failure edge",
    )


@dataclass(frozen=True)
class GraphError:
    code: str
    node: str | None
    message: str
    fix: str

    def __str__(self) -> str:
        where = f"node '{self.node}'" if self.node else "graph"
        return f"[{self.code}] {where}: {self.message}. Fix: {self.fix}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramGraph(BaseModel):
    """A robot task program: skill calls joined by success and failure edges, run from ``entry`` until a
    done or abort node. A node visited more than its max_visits fails without running and takes its failure
    edge; a node's outputs can be read only by nodes that run after it has succeeded."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[Node] = Field(description="The skill calls")
    edges: list[Edge] = Field(description="One success edge and one failure edge per non-terminal node")
    entry: str = Field(description="The id of the first node to run")
    max_retries: Retries | None = Field(
        default=1,
        description="Default in-place retries for every node (a node's own max_retries overrides it). null means 1",
    )
    terminal_nodes: list[str] = Field(
        description="The ids of the done and abort nodes. A terminal node ends the program and has no edges",
    )

    @field_validator("max_retries", mode="before")
    @classmethod
    def _null_means_default(cls, value: Any) -> Any:
        return 1 if value is None else value

    # ----------------------------------------------------------- navigation
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"no node {node_id!r}")

    def next_node(self, node_id: str, on: str) -> str | None:
        for e in self.edges:
            if e.src == node_id and e.on == on:
                return e.dst
        return None

    def retries_for(self, node_id: str) -> int:
        n = self.node(node_id)
        if n.max_retries is not None:
            return n.max_retries
        return 1 if self.max_retries is None else self.max_retries

    # ------------------------------------------------------------ identity
    def canonical_json(self) -> str:
        """Compact sorted JSON. Integral floats in params are written as ints, so 1 and 1.0 hash the same."""
        data = self.model_dump(mode="json", exclude_none=True)
        for node in data.get("nodes", []):
            if isinstance(node.get("params"), dict):
                node["params"] = _integral_floats_to_int(node["params"])
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def graph_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    # ---------------------------------------------------------- validation
    def validate(self, catalog: Mapping[str, SkillSpec] | Iterable[SkillSpec],  # type: ignore[override]
                 task: TaskSpec | None = None) -> list[GraphError]:
        """Readable errors against ``catalog`` (and the task's allowed skills and params)."""
        return validate_graph(self, catalog, task)


def _integral_floats_to_int(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _integral_floats_to_int(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_integral_floats_to_int(v) for v in value]
    return value


_NODE_FIELDS = frozenset(Node.model_fields) - frozenset(ProgramGraph.model_fields)
_EDGE_FIELDS = frozenset(Edge.model_fields)
_EDGE_ALIASES = {"from": "src", "source": "src", "start": "src", "to": "dst", "target": "dst", "dest": "dst",
                 "destination": "dst", "end": "dst", "when": "on", "condition": "on", "outcome": "on",
                 "type": "on", "label": "on"}
_NODE_ALIASES = {"name": "id", "node_id": "id", "skill_name": "skill", "action": "skill", "args": "params",
                 "arguments": "params", "parameters": "params", "inputs": "params", "retries": "max_retries",
                 "retry": "max_retries", "timeout": "timeout_s", "max_visit": "max_visits"}
_KNOWN_FIELDS = frozenset(ProgramGraph.model_fields) | frozenset(Node.model_fields) | _EDGE_FIELDS


# --------------------------------------------------------------------- parsing
_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _ends_early(exc: json.JSONDecodeError) -> bool:
    """The parse failed at the end of the text (JSON had started and only a token fragment is left):
    the reply was probably cut off."""
    if exc.msg.startswith("Unterminated string"):
        return True
    head, tail = exc.doc[:exc.pos], exc.doc[exc.pos:].strip()
    if not any(c in head for c in "{["):
        return False
    return len(tail) <= 20 and not any(c in tail for c in '{}[],:"')


def _json_error(exc: json.JSONDecodeError) -> GraphError:
    at = f"line {exc.lineno}, column {exc.colno}"
    detail = f"{exc.msg} {at}" if exc.msg.endswith(" at") else f"{exc.msg} at {at}"
    fix = _JSON_FIX + (". " + _CUT_OFF_FIX if _ends_early(exc) else "")
    return GraphError("bad_json", None, f"the output is not valid JSON ({detail})", fix)


def _unreadable(exc: BaseException) -> GraphError:
    if isinstance(exc, RecursionError):
        why = "it is nested too deeply"
    elif "digits" in str(exc):
        why = "an integer literal has too many digits"
    else:
        why = f"{type(exc).__name__}: {short(str(exc), 120)}"
    return GraphError("bad_json", None, f"the output could not be read as JSON ({why})",
                      f"Return one JSON object of ordinary numbers and strings, nested at most {MAX_PARAM_DEPTH} "
                      "levels")


def _load_json(text: str) -> tuple[Any, GraphError | None]:
    """The JSON value in a writer's reply: the whole text, else a fenced block, else the outermost object."""
    if text.startswith("\ufeff"):
        text = text[1:]
    body = text.strip()
    if not body:
        return None, GraphError("bad_json", None, "the output is empty", "Return one JSON object")
    try:
        return json.loads(body), None
    except json.JSONDecodeError as exc:
        error = exc
    except (ValueError, RecursionError) as exc:
        return None, _unreadable(exc)
    for match in _FENCE_RE.finditer(body):
        try:
            return json.loads(match.group(1)), None
        except json.JSONDecodeError:
            continue
        except (ValueError, RecursionError) as exc:
            return None, _unreadable(exc)
    start = body.find("{")
    if start >= 0:
        try:
            value, _end = json.JSONDecoder().raw_decode(body, start)
        except json.JSONDecodeError as exc:
            error = exc
        except (ValueError, RecursionError) as exc:
            return None, _unreadable(exc)
        else:
            if isinstance(value, dict):
                return value, None
    return None, _json_error(error)


def _kind_phrase(value: Any) -> str:
    if isinstance(value, bool):
        return "a boolean"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "an array"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, (int, float)):
        return "a number"
    return f"a {type(value).__name__}"


def _parse(source: Any) -> tuple[ProgramGraph | None, list[GraphError], dict[str, Any] | None]:
    """(graph, errors, lenient copy). The lenient copy is the data without unknown keys and with
    null params as {}, given only when those are the only shape errors."""
    if isinstance(source, ProgramGraph):
        return source, [], None
    data: Any = source
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = bytes(source).decode("utf-8-sig", errors="replace")
    if isinstance(source, str):
        data, error = _load_json(source)
        if error is not None:
            return None, [error], None
    if not isinstance(data, dict):
        return None, [GraphError("bad_json", None, f"the top level is {_kind_phrase(data)}, not a JSON object",
                                 "Return one JSON object with nodes, edges, entry and terminal_nodes")], None
    try:
        return ProgramGraph.model_validate(data), [], None
    except ValidationError as exc:
        return None, _shape_errors(exc, data), _lenient_copy(exc, data)
    except (ValueError, RecursionError, OverflowError, TypeError) as exc:
        return None, [_unreadable(exc)], None


def parse_graph(source: ProgramGraph | str | bytes | dict[str, Any]) -> tuple[ProgramGraph | None, list[GraphError]]:
    """A graph from a model, JSON text or a dict, or the errors that stop it parsing.

    Text may carry a UTF-8 BOM, code fences or prose around the JSON; the outermost
    JSON object is used. Hostile input (huge numbers, deep nesting) gives bad_json,
    never an exception.
    """
    try:
        graph, errors, _lenient = _parse(source)
    except (ValueError, RecursionError, OverflowError, TypeError) as exc:
        return None, [_unreadable(exc)]
    return graph, errors


def _record(data: dict[str, Any], key: str, index: int) -> dict[str, Any] | None:
    records = data.get(key)
    if isinstance(records, list) and 0 <= index < len(records) and isinstance(records[index], dict):
        return records[index]
    return None


def _node_name(data: dict[str, Any], index: int) -> str:
    rec = _record(data, "nodes", index)
    node_id = rec.get("id") if rec is not None else None
    return node_id if isinstance(node_id, str) and node_id else f"nodes[{index}]"


def _edge_name(data: dict[str, Any], index: int) -> str:
    rec = _record(data, "edges", index)
    if rec is None:
        return f"#{index}"

    def end(*keys: str) -> str:
        for k in keys:
            if k in rec:
                v = rec[k]
                return v if isinstance(v, str) else short(v, 30)
        return "?"

    return f"{end('src', 'from', 'source')} -> {end('dst', 'to', 'target')}"


def _extra_fix(loc: tuple[Any, ...]) -> str:
    key = loc[-1] if loc else None
    if len(loc) == 1:
        if key in _NODE_FIELDS:
            return f'"{key}" is a node field: move it onto a node'
        if key in _EDGE_FIELDS:
            return f'"{key}" is an edge field: move it onto an edge'
    elif loc[0] == "edges" and key in _EDGE_ALIASES:
        return f'Rename it to "{_EDGE_ALIASES[key]}"'
    elif loc[0] == "nodes" and key in _NODE_ALIASES:
        return f'Rename it to "{_NODE_ALIASES[key]}"'
    elif loc[0] == "nodes" and key in _EDGE_FIELDS:
        return f'"{key}" is an edge field: write the transition as an edge in "edges"'
    return "Remove that field"


def _shape_fix(loc: tuple[Any, ...], kind: str) -> str:
    if kind == "extra_forbidden":
        return _extra_fix(loc)
    field = next((p for p in reversed(loc) if isinstance(p, str) and p in _KNOWN_FIELDS), None)
    if kind == "missing":
        return f'Add "{field}"' if field else "Add that field"
    if field == "on":
        return 'An edge\'s "on" is "success" or "failure"'
    if field == "params":
        return 'Give "params" a JSON object: {} when the skill takes no params'
    if field == "max_retries":
        return f"Use a whole number from 0 to {MAX_RETRIES_LIMIT}"
    if field == "max_visits":
        return f"Use a whole number from 1 to {MAX_VISITS_LIMIT}"
    if field == "timeout_s":
        return "Use a number of seconds greater than 0"
    if field in ("nodes", "edges", "terminal_nodes"):
        return f"Give {field} a JSON array, as in the graph format"
    return "Follow the graph format"


def _shape_errors(exc: ValidationError, data: dict[str, Any]) -> list[GraphError]:
    errors: list[GraphError] = []
    for e in exc.errors(include_url=False):
        loc = tuple(e.get("loc", ()))
        kind = e.get("type", "")
        node = _node_name(data, loc[1]) if len(loc) >= 2 and loc[0] == "nodes" and isinstance(loc[1], int) else None
        where = ".".join(str(p) for p in loc) or "graph"
        if len(loc) >= 2 and loc[0] == "edges" and isinstance(loc[1], int):
            where += f" (edge {_edge_name(data, loc[1])})"
        msg = str(e.get("msg", "invalid"))
        if kind == "literal_error":
            msg = msg.replace("'", '"')
        if kind not in ("missing", "extra_forbidden") and "input" in e:
            msg += f", got {short(e['input'])}"
        errors.append(GraphError("bad_shape", node, f"{where}: {msg}", _shape_fix(loc, kind)))
    return errors


def _lenient_copy(exc: ValidationError, data: dict[str, Any]) -> dict[str, Any] | None:
    """``data`` without unknown keys and with null params as {}, when those are its only shape errors."""
    for e in exc.errors(include_url=False):
        loc = tuple(e.get("loc", ()))
        if e.get("type") == "extra_forbidden":
            continue
        if len(loc) == 3 and loc[0] == "nodes" and loc[2] == "params" and e.get("input", 0) is None:
            continue
        return None

    def strip(record: Any, fields: Iterable[str], is_node: bool) -> Any:
        if not isinstance(record, dict):
            return record
        out = {k: v for k, v in record.items() if k in fields}
        if is_node and "params" in out and out["params"] is None:
            out["params"] = {}
        return out

    lenient = {k: v for k, v in data.items() if k in ProgramGraph.model_fields}
    if isinstance(lenient.get("nodes"), list):
        lenient["nodes"] = [strip(n, Node.model_fields, True) for n in lenient["nodes"]]
    if isinstance(lenient.get("edges"), list):
        lenient["edges"] = [strip(e, Edge.model_fields, False) for e in lenient["edges"]]
    return lenient


# ------------------------------------------------------------------ validation
def _suggest(name: str, options: Iterable[str]) -> str:
    options = sorted(set(options))
    if not options:
        return "There is nothing to choose from: check the skill catalog and the task's allowed_skills"
    close = difflib.get_close_matches(name, options, n=1)
    head = f"Did you mean '{close[0]}'? " if close else ""
    return f"{head}Use one of: {', '.join(options)}"


def _types(names: set[str]) -> str:
    return " or ".join(sorted(names)) or "any value"


def _types_compatible(have: set[str], want: set[str]) -> bool:
    have = {t for t in have if t != "null"}
    if not have or not want:
        return True
    for t in have:
        if t in want or (t == "integer" and "number" in want):
            return True
    return False


def _sccs(nodes: list[str], adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components, iterative."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adjacency.get(root, [])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(adjacency.get(w, []))))
                    advanced = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def _cycle_from(start: str, members: set[str], adjacency: dict[str, list[str]]) -> list[str]:
    """A concrete cycle start -> ... -> (back to start) inside one strongly connected component, by DFS."""
    stack = [(start, iter(adjacency.get(start, [])))]
    seen = {start}
    while stack:
        v, it = stack[-1]
        for w in it:
            if w not in members:
                continue
            if w == start:
                return [x for x, _ in stack]
            if w not in seen:
                seen.add(w)
                stack.append((w, iter(adjacency.get(w, []))))
                break
        else:
            stack.pop()
    return [start]


def _container(schema: Any, root: dict[str, Any], value: Any) -> dict[str, Any]:
    """The schema that describes the dict or list ``value`` at its position (through $ref, unions and allOf)."""
    s = resolve(schema, root) if isinstance(schema, dict) else {}
    kind = "object" if isinstance(value, dict) else "array"
    for _ in range(8):
        branches = s.get("anyOf") or s.get("oneOf")
        if isinstance(branches, list) and branches:
            fits = [resolve(b, root) for b in branches
                    if isinstance(b, dict) and (not type_names(b, root) or kind in type_names(b, root))]
            if not fits:
                return {}
            passing = [b for b in fits if not check_schema(value, b, root=root, wildcard=is_ref)]
            s = (passing or fits)[0]
            continue
        if isinstance(s.get("allOf"), list):
            merged = {k: v for k, v in s.items() if k != "allOf"}
            for b in s["allOf"]:
                for k, v in (resolve(b, root) if isinstance(b, dict) else {}).items():
                    if k == "properties" and isinstance(v, dict):
                        merged["properties"] = {**(merged.get("properties") or {}), **v}
                    else:
                        merged.setdefault(k, v)
            s = merged
            continue
        break
    return s


def _item_schemas(s: dict[str, Any]) -> tuple[list[Any], Any]:
    prefix, items = s.get("prefixItems"), s.get("items")
    if isinstance(items, list):  # the older tuple form
        prefix, items = items, s.get("additionalItems")
    return (prefix if isinstance(prefix, list) else []), items


def _ref_slots(value: Any, schema: Any, root: dict[str, Any], path: str) -> Iterator[tuple[str, Any, str]]:
    """(ref, the schema at the ref's position, its param path) for every ref in a param value."""
    if is_ref(value):
        yield value["ref"], schema, path
    elif isinstance(value, dict):
        s = _container(schema, root, value)
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        extra = s.get("additionalProperties")
        for key, item in value.items():
            sub = props[key] if key in props else (extra if isinstance(extra, dict) else {})
            yield from _ref_slots(item, sub, root, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        s = _container(schema, root, value)
        prefix, items = _item_schemas(s)
        for i, item in enumerate(value):
            sub = prefix[i] if i < len(prefix) else (items if isinstance(items, dict) else {})
            yield from _ref_slots(item, sub, root, f"{path}[{i}]")


def _resolve_path(schema: Any, root: dict[str, Any], segments: list[str]) -> tuple[Any, tuple[int, str, Any] | None]:
    """(the schema at ``segments``, None) or (None, (index of the first missing segment, why, detail)).

    Follows properties (closed unless additionalProperties is true or a schema; an
    object with neither properties nor additionalProperties is of unknown shape),
    numeric indexes into arrays (items, prefixItems, maxItems) and $ref.
    """
    cur = schema
    for i, seg in enumerate(segments):
        s, _nullable = unwrap_optional(cur if isinstance(cur, dict) else {}, root)
        branches = s.get("anyOf") or s.get("oneOf")
        if isinstance(branches, list) and branches and "type" not in s:
            first: tuple[int, str, Any] | None = None
            for branch in branches:
                if not isinstance(branch, dict) or type_names(branch, root) == {"null"}:
                    continue
                found, miss = _resolve_path(branch, root, segments[i:])
                if miss is None:
                    return found, None
                first = first or (miss[0] + i, miss[1], miss[2])
            return None, first or (i, "field", [])
        names = type_names(s, root) - {"null"}
        props = s.get("properties") if isinstance(s.get("properties"), dict) else None
        extra = s.get("additionalProperties")
        is_object = "object" in names or props is not None or "additionalProperties" in s
        is_array = "array" in names or "items" in s or "prefixItems" in s
        if is_object and not (is_array and props is None and extra is None):
            if props is not None and seg in props:
                cur = props[seg]
            elif isinstance(extra, dict):
                cur = extra
            elif extra is True or (props is None and extra is None):
                cur = {}
            else:
                return None, (i, "field", list(props or {}))
            continue
        if is_array:
            if not seg.isdigit():
                return None, (i, "index", None)
            idx = int(seg)
            hi = s.get("maxItems")
            prefix, items = _item_schemas(s)
            if is_number(hi) and idx >= hi:
                return None, (i, "range", int(hi))
            if idx < len(prefix):
                cur = prefix[idx]
            elif prefix and items is False:
                return None, (i, "range", len(prefix))
            else:
                cur = items if isinstance(items, dict) else {}
            continue
        if not names:
            cur = {}
            continue
        return None, (i, "scalar", names)
    return cur, None


def _missing_text(where: str, prefix: str, seg: str, kind: str, detail: Any) -> tuple[str, str]:
    """(message, fix) for the first segment of a ref path that does not exist."""
    if kind == "field":
        names = [str(n) for n in detail or []]
        close = difflib.get_close_matches(seg, names, n=1)
        head = f"Did you mean '{close[0]}'? " if close else ""
        return f"{where}: {prefix} has no field '{seg}'", f"{head}{prefix} has: {', '.join(names) or 'no fields'}"
    if kind == "index":
        return (f"{where}: {prefix} is an array, so '{seg}' is not one of its items",
                f"Use an item number, for example {prefix}.0")
    if kind == "range":
        n = int(detail)
        fix = f"Use an item number from 0 to {n - 1}" if n > 0 else f"{prefix} has no items to refer to"
        return f"{where}: {prefix} has at most {n} items, so it has no item {seg}", fix
    return f"{where}: {prefix} is {_types(set(detail))}, so it has no field '{seg}'", f"Refer to {prefix} itself"


class _Validator:
    """One validate_graph run over a parsed graph."""

    def __init__(self, graph: ProgramGraph, specs: dict[str, SkillSpec], task: TaskSpec | None):
        self.graph = graph
        self.specs = specs
        self.task = task
        self.allowed = set(task.allowed_skills) if task is not None else None
        self.errors: list[GraphError] = []
        self.listed = set(graph.terminal_nodes)
        self.by_id: dict[str, Node] = {}
        self.out: dict[tuple[str, str], str] = {}
        self.live: set[str] = set()
        self._without_success: dict[str, set[str]] = {}
        self._after_success: dict[str, set[str]] = {}

    def add(self, code: str, node: str | None, message: str, fix: str) -> None:
        self.errors.append(GraphError(code, node, message, fix))

    def run(self) -> list[GraphError]:
        if not self.graph.nodes:
            return [GraphError("empty_graph", None, "the graph has no nodes",
                               "Add nodes: the skill calls and a done node, as in the graph format")]
        self.check_ids()
        self.check_skills()
        self.check_terminals()
        self.check_edges()
        self.check_reachability()
        self.check_cycles()
        self.check_params()
        return self.errors

    # ------------------------------------------------------------- structure
    def check_ids(self) -> None:
        for n in self.graph.nodes:
            if not NODE_ID_RE.fullmatch(n.id):
                self.add("bad_node_id", n.id, f"node id {short(n.id, 80)} is not a plain identifier",
                         "Use letters, digits, '_' or '-', starting with a letter, at most 64 characters")
            if n.id == TASK_REF:
                self.add("reserved_node_id", n.id, "'task' is reserved for task parameter references", "Rename the node")
            if n.id in self.by_id:
                self.add("duplicate_node", n.id, f"two nodes share the id '{n.id}'", "Give every node a unique id")
            else:
                self.by_id[n.id] = n
        if self.graph.entry not in self.by_id:
            self.add("unknown_entry", None, f"entry '{self.graph.entry}' is not a node id",
                     f"Set entry to the first node to run, one of: {', '.join(self.by_id)}")

    def check_skills(self) -> None:
        pool = self.allowed if self.allowed is not None else self.specs
        usable = [name for name in pool if name in self.specs]
        for n in self.graph.nodes:
            if n.skill not in self.specs:
                self.add("unknown_skill", n.id, f"skill '{n.skill}' is not in the skill catalog", _suggest(n.skill, usable))
            elif self.allowed is not None and n.skill not in self.allowed:
                self.add("skill_not_allowed", n.id, f"skill '{n.skill}' is not allowed for task {self.task.id}",
                         _suggest(n.skill, usable))

    def terminal_kind(self, node: Node) -> str | None:
        spec = self.specs.get(node.skill)
        return spec.terminal if spec is not None else None

    def is_terminal(self, node: Node) -> bool:
        """A node whose skill is done or abort. A node with an unknown skill counts as terminal when listed."""
        spec = self.specs.get(node.skill)
        if spec is not None:
            return spec.terminal is not None
        return node.id in self.listed

    def check_terminals(self) -> None:
        listed = self.listed
        for t in self.graph.terminal_nodes:
            if t not in self.by_id:
                self.add("unknown_terminal", t, f"terminal_nodes lists '{t}', which is not a node",
                         "List only the ids of nodes whose skill is done or abort")
                continue
            spec = self.specs.get(self.by_id[t].skill)
            if spec is not None and spec.terminal is None:
                self.add("bad_terminal", t,
                         f"node '{t}' is listed as terminal but its skill '{self.by_id[t].skill}' is not done or abort",
                         "Only done and abort nodes are terminal: remove it from terminal_nodes")
        for n in self.graph.nodes:
            spec = self.specs.get(n.skill)
            if spec is not None and spec.terminal is not None and n.id not in listed:
                self.add("terminal_not_listed", n.id,
                         f"node '{n.id}' uses the terminal skill '{n.skill}' but is not in terminal_nodes",
                         f"Add '{n.id}' to terminal_nodes")

    def check_edges(self) -> None:
        ids = ", ".join(self.by_id)
        for e in self.graph.edges:
            label = f"edge {e.src} -> {e.dst} ({e.on})"
            if e.src not in self.by_id:
                self.add("unknown_edge_node", None, f"{label} starts at '{e.src}', which is not a node",
                         f"Edges connect existing node ids: {ids}")
            if e.dst not in self.by_id:
                self.add("unknown_edge_node", e.src if e.src in self.by_id else None,
                         f"{label} points to '{e.dst}', which is not a node", f"Edges connect existing node ids: {ids}")
            if e.src in self.by_id and self.is_terminal(self.by_id[e.src]):
                self.add("terminal_has_edges", e.src, f"terminal node '{e.src}' has an outgoing {e.on} edge",
                         "A done or abort node ends the program. Remove edges that start at it")
            key = (e.src, e.on)
            if key in self.out and self.out[key] != e.dst:
                self.add("duplicate_edge", e.src,
                         f"node '{e.src}' has two {e.on} edges ({self.out[key]} and {e.dst})",
                         f"Keep exactly one {e.on} edge per node")
            self.out.setdefault(key, e.dst)
        for n in self.graph.nodes:
            if self.by_id.get(n.id) is not n or self.is_terminal(n):
                continue
            if (n.id, "success") not in self.out:
                self.add("no_success_edge", n.id, f"node '{n.id}' ({n.skill}) has no success edge",
                         f'Add {{"src": "{n.id}", "dst": "<next node>", "on": "success"}}')
            if (n.id, "failure") not in self.out:
                self.add("no_failure_edge", n.id, f"node '{n.id}' ({n.skill}) has no failure edge",
                         f'Add {{"src": "{n.id}", "dst": "<a recovery node or an abort node>", "on": "failure"}}')

    def reach(self, starts: Iterable[str | None], skip: tuple[str, str] | None = None) -> set[str]:
        """Nodes reachable from ``starts`` (included) over the graph's edges, never taking ``skip``."""
        seen = {s for s in starts if s in self.by_id}
        queue = deque(seen)
        while queue:
            v = queue.popleft()
            for on in ("success", "failure"):
                if (v, on) == skip:
                    continue
                w = self.out.get((v, on))
                if w is not None and w in self.by_id and w not in seen:
                    seen.add(w)
                    queue.append(w)
        return seen

    def check_reachability(self) -> None:
        entry = self.graph.entry
        if entry not in self.by_id:
            return
        self.live = self.reach([entry])
        for n in self.graph.nodes:
            if self.by_id.get(n.id) is n and n.id not in self.live:
                self.add("unreachable", n.id, f"node '{n.id}' can never run: no path leads to it from the entry",
                         "Connect it with an edge from a reachable node, or delete it")
        if not any(self.terminal_kind(self.by_id[v]) == "done" for v in self.live):
            self.add("no_done", None, "no done node is reachable from the entry",
                     "Add a node with skill done, list it in terminal_nodes, and route the success path to it")
            return  # the success path cannot reach done either; no_done says so
        self.check_success_path()

    def check_success_path(self) -> None:
        cur = self.graph.entry
        path = [cur]
        position = {cur: 0}
        while True:
            node = self.by_id[cur]
            if self.is_terminal(node):
                kind = self.terminal_kind(node)
                if kind is not None and kind != "done":
                    self.add("success_path_no_done", None,
                             f"following success edges from the entry never reaches done: {' -> '.join(path)} "
                             f"ends at the {kind} node '{cur}'", _SUCCESS_PATH_FIX)
                return
            nxt = self.out.get((cur, "success"))
            if nxt is None or nxt not in self.by_id:
                return  # no_success_edge or unknown_edge_node already says so
            if nxt in position:
                loop = path[position[nxt]:] + [nxt]
                self.add("success_path_no_done", None,
                         f"following success edges from the entry never reaches done: the success edges loop "
                         f"{' -> '.join(loop)}", _SUCCESS_PATH_FIX)
                return
            position[nxt] = len(path)
            path.append(nxt)
            cur = nxt

    def check_cycles(self) -> None:
        """Remove the success edges of nodes with max_visits; every cycle left can run forever."""
        ids = list(self.by_id)
        order = {node_id: k for k, node_id in enumerate(ids)}
        bounded = {node_id for node_id, n in self.by_id.items() if n.max_visits is not None}
        reduced: dict[str, list[str]] = {node_id: [] for node_id in ids}
        for (src, on), dst in self.out.items():
            if src in self.by_id and dst in self.by_id and not (on == "success" and src in bounded):
                if dst not in reduced[src]:
                    reduced[src].append(dst)
        loops = []
        for comp in _sccs(ids, reduced):
            if len(comp) > 1 or comp[0] in reduced[comp[0]]:
                start = min(comp, key=order.__getitem__)
                loops.append((order[start], _cycle_from(start, set(comp), reduced)))
        for _, cycle in sorted(loops, key=lambda item: item[0]):
            text = " -> ".join(cycle + [cycle[0]])
            held = [c for c in cycle if c in bounded]
            if held:
                names = ", ".join(f"'{c}'" for c in held)
                why = (f"form a loop that max_visits cannot end: past its bound a node fails without running and "
                       f"takes its failure edge, and the failure edge of {names} stays in this loop")
            else:
                why = "form a loop with no visit bound"
            self.add("unbounded_cycle", cycle[0], f"nodes {text} {why}", _CYCLE_FIX)

    # ---------------------------------------------------------------- params
    def check_params(self) -> None:
        for n in self.graph.nodes:
            spec = self.specs.get(n.skill)
            if spec is None or self.by_id.get(n.id) is not n:
                continue
            schema = spec.params if isinstance(spec.params, dict) else {}
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            required = schema.get("required") if isinstance(schema.get("required"), list) else []
            for name in required:
                if name not in n.params:
                    self.add("missing_param", n.id,
                             f"node '{n.id}' ({n.skill}) is missing the required param '{name}'",
                             f"Add \"{name}\": {describe(props.get(name, {}), schema)}")
            for name, value in n.params.items():
                try:
                    self.check_param(n, name, value, schema, props)
                except (ValueError, RecursionError, OverflowError, TypeError) as exc:
                    self.add("bad_param", n.id,
                             f"node '{n.id}' ({n.skill}): param '{name}' could not be checked ({type(exc).__name__})",
                             f"Give '{name}' a plain JSON value")

    def check_param(self, n: Node, name: str, value: Any, schema: dict[str, Any], props: dict[str, Any]) -> None:
        if _too_deep(value):
            self.add("bad_param", n.id,
                     f"node '{n.id}' ({n.skill}): {name} is nested too deeply (more than {MAX_PARAM_DEPTH} levels)",
                     f"Give '{name}' a value as the skill's params describe it")
            return
        extra = schema.get("additionalProperties")
        sub: Any
        if name in props:
            sub = props[name]
        elif isinstance(extra, dict):
            sub = extra
        elif extra is True:
            sub = {}
        else:  # params are closed unless the schema opens them
            sub = None
            close = difflib.get_close_matches(name, list(props), n=1)
            head = f"Did you mean '{close[0]}'? " if close else ""
            self.add("unknown_param", n.id, f"node '{n.id}' ({n.skill}) has an unknown param '{name}'",
                     f"{head}{n.skill} takes: {', '.join(props) or 'no params'}")
        for ref, slot, path in _ref_slots(value, sub if sub is not None else {}, schema, name):
            err = self.check_ref(ref, n, path, slot, schema)
            if err is not None:
                self.errors.append(err)
        if sub is None:
            return
        problems = check_schema(value, sub, path=name, root=schema, wildcard=is_ref)
        if problems:
            fix = f"Give '{name}' {describe(sub, schema)}"
            if _contains_near_ref(value):
                fix += f". To use another node's output, write {REF_FORM} as the whole value, with no other keys"
            for problem in problems:
                self.add("bad_param", n.id, f"node '{n.id}' ({n.skill}): {problem}", fix)

    def succeeds_before(self, src: str, reader: str) -> bool:
        """Every path from the entry to ``reader`` leaves ``src`` through its success edge."""
        if self.graph.entry in self.by_id and reader in self.live:
            if src not in self._without_success:
                self._without_success[src] = self.reach([self.graph.entry], skip=(src, "success"))
            return reader not in self._without_success[src]
        # the reader never runs from the entry: judge by the structure after src's success
        if src not in self._after_success:
            self._after_success[src] = self.reach([self.out.get((src, "success"))])
        return reader in self._after_success[src]

    def check_ref(self, ref: str, node: Node, param: str, slot: Any, slot_root: dict[str, Any]) -> GraphError | None:
        where = f"node '{node.id}' ({node.skill}) param '{param}'"
        m = REF_RE.fullmatch(ref)
        if not m:
            return GraphError("bad_ref", node.id, f"{where}: '{ref}' is not a reference",
                              'Write {"ref": "<node_id>.<field>"} for an earlier node\'s output or '
                              '{"ref": "task.<param>"} for a task parameter')
        src, segments = m["node"], m["path"].split(".")
        want = type_names(slot, slot_root) if isinstance(slot, dict) and slot else set()
        if src == TASK_REF:
            return self.check_task_ref(ref, node, param, where, segments, want)
        if src not in self.by_id:
            return GraphError("ref_unknown_node", node.id,
                              f"{where}: '{ref}' refers to node '{src}', which does not exist", _suggest(src, self.by_id))
        if src == node.id:
            return GraphError("ref_self", node.id, f"{where}: a node cannot read its own outputs", "Refer to an earlier node")
        if not self.succeeds_before(src, node.id):
            return GraphError("ref_not_upstream", node.id,
                              f"{where}: '{ref}' reads node '{src}', but '{src}' does not always succeed before "
                              f"'{node.id}' runs", "Refer only to a node that succeeds on every path to this node")
        spec = self.specs.get(self.by_id[src].skill)
        if spec is None:
            return None
        returns = spec.returns if isinstance(spec.returns, dict) else {}
        found, miss = _resolve_path(returns, returns, segments)
        if miss is not None:
            i, kind, detail = miss
            if i == 0 and kind == "field":
                names = [str(x) for x in detail or []]
                close = difflib.get_close_matches(segments[0], names, n=1)
                head = f"Did you mean '{close[0]}'? " if close else ""
                return GraphError("ref_unknown_field", node.id,
                                  f"{where}: skill {spec.name} (node '{src}') returns no field '{segments[0]}'",
                                  f"{head}{spec.name} returns: {', '.join(names) or 'nothing'}")
            message, fix = _missing_text(where, ".".join([src] + segments[:i]), segments[i], kind, detail)
            return GraphError("ref_unknown_field", node.id, message, fix)
        have = type_names(found, returns) if isinstance(found, dict) else set()
        if not _types_compatible(have, want):
            return GraphError("ref_type_mismatch", node.id,
                              f"{where}: {ref} is {_types(have)} but '{param}' needs {_types(want)}",
                              "Refer to an output of the right type")
        return None

    def check_task_ref(self, ref: str, node: Node, param: str, where: str, segments: list[str],
                       want: set[str]) -> GraphError | None:
        if self.task is None:
            return None
        declared = self.task.params_schema
        field = segments[0]
        if field not in declared:
            return GraphError("ref_unknown_field", node.id, f"{where}: the task has no param '{field}'",
                              f"Task params: {', '.join(declared) or 'none'}")
        root = declared[field] if isinstance(declared[field], dict) else {}
        found, miss = _resolve_path(root, root, segments[1:])
        if miss is not None:
            i, kind, detail = miss
            message, fix = _missing_text(where, ".".join([TASK_REF, field] + segments[1:1 + i]), segments[1 + i],
                                         kind, detail)
            return GraphError("ref_unknown_field", node.id, message, fix)
        have = type_names(found, root) if isinstance(found, dict) else set()
        if not _types_compatible(have, want):
            return GraphError("ref_type_mismatch", node.id,
                              f"{where}: {ref} is {_types(have)} but '{param}' needs {_types(want)}",
                              "Refer to a value of the right type")
        return None


def _config_errors(specs: dict[str, SkillSpec], task: TaskSpec | None) -> list[GraphError]:
    if task is None:
        return []
    return [GraphError("config_missing_skill", None,
                       f"task {task.id} allows the skill '{name}', which is not in the skill catalog",
                       "This is a setup error, not the program's: add the skill to the catalog or remove it "
                       "from the task's allowed_skills")
            for name in dict.fromkeys(task.allowed_skills) if name not in specs]


def _unique(errors: list[GraphError]) -> list[GraphError]:
    """One error per distinct text, in discovery order."""
    out: list[GraphError] = []
    seen: set[str] = set()
    for e in errors:
        text = str(e)
        if text not in seen:
            seen.add(text)
            out.append(e)
    return out


def validate_graph(source: ProgramGraph | str | bytes | dict[str, Any],
                   catalog: Mapping[str, SkillSpec] | Iterable[SkillSpec],
                   task: TaskSpec | None = None) -> list[GraphError]:
    """All errors for ``source``. An empty list means the graph can run.

    Shape errors that are only unknown keys or a null params are reported together
    with the semantic errors of the graph without them. Hostile input never raises:
    it gives bad_json or bad_param.
    """
    specs = as_catalog(catalog)
    errors = _config_errors(specs, task)
    try:
        graph, shape, lenient = _parse(source)
    except (ValueError, RecursionError, OverflowError, TypeError) as exc:
        return _unique(errors + [_unreadable(exc)])
    errors += shape
    if graph is None and lenient is not None:
        try:
            graph = ProgramGraph.model_validate(lenient)
        except (ValueError, RecursionError, OverflowError, TypeError):
            graph = None
    if graph is not None:
        try:
            errors += _Validator(graph, specs, task).run()
        except (ValueError, RecursionError, OverflowError, TypeError) as exc:
            errors.append(GraphError("bad_param", None,
                                     f"the graph could not be checked ({type(exc).__name__}: {short(str(exc), 120)})",
                                     f"Use plain JSON values: ordinary numbers and strings, nested at most "
                                     f"{MAX_PARAM_DEPTH} levels"))
    return _unique(errors)


# ------------------------------------------------------------------- params
def _to_ints(value: Any, schema: Any, root: dict[str, Any], depth: int = 0) -> Any:
    if depth > MAX_PARAM_DEPTH or is_ref(value) or not isinstance(schema, dict):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            names = type_names(schema, root)
            if "integer" in names and "number" not in names:
                return int(value)
        return value
    if isinstance(value, dict):
        s = _container(schema, root, value)
        props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
        extra = s.get("additionalProperties")
        return {k: _to_ints(v, props[k] if k in props else (extra if isinstance(extra, dict) else {}), root, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        s = _container(schema, root, value)
        prefix, items = _item_schemas(s)
        out = [_to_ints(v, prefix[i] if i < len(prefix) else (items if isinstance(items, dict) else {}), root, depth + 1)
               for i, v in enumerate(value)]
        return out if isinstance(value, list) else tuple(out)
    return value


def normalize_params(graph: ProgramGraph, catalog: Mapping[str, SkillSpec] | Iterable[SkillSpec]) -> ProgramGraph:
    """A copy of ``graph`` whose params hold ints wherever the skill's schema says integer (1.0 -> 1).

    The interpreter calls this before running, so a skill typed ``index: int`` gets an
    int. Refs, nodes with unknown skills and values the schema does not type stay as they are.
    """
    specs = as_catalog(catalog)
    nodes = []
    for n in graph.nodes:
        spec = specs.get(n.skill)
        if spec is None or not n.params:
            nodes.append(n)
            continue
        schema = spec.params if isinstance(spec.params, dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        extra = schema.get("additionalProperties")
        params = {}
        for name, value in n.params.items():
            sub = props[name] if name in props else (extra if isinstance(extra, dict) else None)
            params[name] = _to_ints(value, sub, schema) if sub is not None else value
        nodes.append(n.model_copy(update={"params": params}))
    return graph.model_copy(update={"nodes": nodes})


def format_errors(errors: list[GraphError]) -> str:
    """Numbered error list for the writer's repair prompt. When an error is about the graph's
    format, the format follows once at the end."""
    lines = [f"{i}. {e}" for i, e in enumerate(errors, 1)]
    if any(e.code in _FORMAT_CODES for e in errors):
        lines.append(f"Graph format: {GRAPH_FORMAT_HINT}")
    return "\n".join(lines)
