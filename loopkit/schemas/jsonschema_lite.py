"""A small JSON Schema checker for the subset loopkit uses.

Skill params and returns are JSON Schema generated from typed functions. Graph
validation checks literal params against them and the messages go back to the
program writer, so they are short and say what was expected. Supported: type,
enum, const, minimum/maximum (and exclusive), minLength/maxLength, pattern,
items/prefixItems, minItems/maxItems, properties, required, additionalProperties,
anyOf, oneOf (checked like anyOf), allOf and local ``$ref`` into ``$defs``.

Values in messages are JSON (``true``, ``null``, ``"text"``), because the reader
wrote JSON. ``enum`` and ``const`` compare the way JSON Schema does, so a boolean
is never equal to a number. When no branch of an anyOf/oneOf passes, the errors
are those of the branch whose type matches the value (a null branch is ignored),
so ``Optional[int]`` with a range reports the range, not a type error.

``check(..., wildcard=pred)`` treats every value for which ``pred`` is true (a
graph's ``{"ref": ...}``) as valid at its position; everything around it, such as
the other items of a list and the list's length, is still checked.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

Wildcard = Callable[[Any], bool]

_TYPES = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": (list, tuple),
    "null": type(None),
}


# ------------------------------------------------------------------ JSON values
def is_number(value: Any) -> bool:
    """An int or a float, but not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_type(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return name == "boolean"
    if name == "integer":
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if name == "number":
        # ints are always finite; never call float() on one (a 310-digit int overflows)
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    py = _TYPES.get(name)
    return py is not None and isinstance(value, py)


def json_type(value: Any) -> str | None:
    """The JSON type name of a Python value ("integer" for ints), or None when it is not JSON."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def json_equal(a: Any, b: Any) -> bool:
    """Equality as JSON Schema's enum and const use it: true is not 1, 1 is 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if is_number(a) and is_number(b):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:  # an ndarray compares element-wise
        return False


def _fallback_text(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"<an integer of about {int(abs(value).bit_length() * 0.30103) + 1} digits>"
    try:
        return repr(value)
    except Exception:  # deep nesting, an int too long for str()
        return f"<{type(value).__name__}>"


def short(value: Any, limit: int = 60) -> str:
    """``value`` as compact JSON for a message (repr when it is not JSON), cut to ``limit`` characters."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        text = _fallback_text(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_short = short  # older name


# ---------------------------------------------------------------------- schemas
def _pointer_part(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` (``#/$defs/Name``) against ``root``. Keys next to the ref win."""
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema and seen < 16:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return {}
        node: Any = root
        for part in ref[2:].split("/"):
            node = node.get(_pointer_part(part), {}) if isinstance(node, dict) else {}
        if not isinstance(node, dict):
            return {}
        schema = {**node, **{k: v for k, v in schema.items() if k != "$ref"}}
        seen += 1
    return schema if isinstance(schema, dict) else {}


def type_names(schema: dict[str, Any], root: dict[str, Any] | None = None) -> set[str]:
    """The JSON types a schema admits (empty set means any)."""
    root = root if root is not None else schema
    schema = resolve(schema, root)
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    branches = [b for key in ("anyOf", "oneOf") for b in (schema.get(key) or []) if isinstance(b, dict)]
    if branches:
        out: set[str] = set()
        for branch in branches:
            names = type_names(branch, root)
            if not names:
                return set()
            out |= names
        return out
    for branch in schema.get("allOf") or []:
        if isinstance(branch, dict):
            names = type_names(branch, root)
            if names:
                return names
    if "const" in schema:
        kind = json_type(schema["const"])
        return {kind} if kind else set()
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        kinds = {json_type(v) for v in schema["enum"]}
        return set() if None in kinds else kinds  # type: ignore[return-value]
    return set()


def _admits(value: Any, names: set[str]) -> bool:
    return not names or any(_is_type(value, n) for n in names)


def _discriminators_match(value: Any, branch: dict[str, Any], root: dict[str, Any]) -> bool:
    """False when ``value`` disagrees with a const property of ``branch`` (a tagged union's tag)."""
    if not isinstance(value, dict):
        return True
    props = resolve(branch, root).get("properties")
    if not isinstance(props, dict):
        return True
    for name, sub in props.items():
        sub = resolve(sub, root) if isinstance(sub, dict) else {}
        if "const" in sub and name in value and not json_equal(value[name], sub["const"]):
            return False
    return True


def _union_errors(value: Any, failures: list[tuple[dict[str, Any], list[str]]], root: dict[str, Any],
                  where: str) -> list[str]:
    candidates = []
    for branch, errs in failures:
        names = type_names(branch, root)
        if names == {"null"}:
            continue
        if _admits(value, names):
            candidates.append((branch, errs))
    if not candidates:
        kinds: set[str] = set()
        for branch, _ in failures:
            kinds |= type_names(branch, root)
        expected = " or ".join(sorted(kinds)) or "one of the allowed forms"
        return [f"{where} is {short(value)}, expected {expected}"]
    if len(candidates) > 1:
        tagged = [c for c in candidates if _discriminators_match(value, c[0], root)]
        candidates = tagged or candidates
    return min(candidates, key=lambda c: len(c[1]))[1]


def check(value: Any, schema: dict[str, Any], path: str = "", root: dict[str, Any] | None = None, *,
          wildcard: Wildcard | None = None) -> list[str]:
    """Error strings for ``value`` against ``schema``. Empty list means valid."""
    root = root if root is not None else schema
    if wildcard is not None and wildcard(value):
        return []
    schema = resolve(schema, root)
    if not schema:
        return []
    where = path or "value"
    errors: list[str] = []

    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            failures = []
            for branch in branches:
                errs = check(value, branch, path, root, wildcard=wildcard)
                if not errs:
                    break
                failures.append((branch, errs))
            else:
                return errors + _union_errors(value, failures, root, where)
    for branch in schema.get("allOf") or []:
        if isinstance(branch, dict):
            errors += check(value, branch, path, root, wildcard=wildcard)

    t = schema.get("type")
    if t is not None:
        names = [t] if isinstance(t, str) else [n for n in t if isinstance(n, str)]
        if names and not any(_is_type(value, n) for n in names):
            return errors + [f"{where} is {short(value)}, expected {' or '.join(names)}"]

    if "const" in schema and not json_equal(value, schema["const"]):
        errors.append(f"{where} must be {short(schema['const'], 200)}, got {short(value)}")
    enum = schema.get("enum")
    if isinstance(enum, list) and not any(json_equal(value, option) for option in enum):
        errors.append(f"{where} is {short(value)}, expected one of {short(enum, 400)}")

    if is_number(value):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if is_number(lo) and value < lo:
            errors.append(f"{where} is {short(value)}, below the minimum {short(lo)}")
        if is_number(hi) and value > hi:
            errors.append(f"{where} is {short(value)}, above the maximum {short(hi)}")
        xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if is_number(xlo) and value <= xlo:
            errors.append(f"{where} is {short(value)}, must be greater than {short(xlo)}")
        if is_number(xhi) and value >= xhi:
            errors.append(f"{where} is {short(value)}, must be less than {short(xhi)}")

    if isinstance(value, str):
        lo, hi = schema.get("minLength"), schema.get("maxLength")
        if is_number(lo) and len(value) < lo:
            errors.append(f"{where} is shorter than {lo} characters")
        if is_number(hi) and len(value) > hi:
            errors.append(f"{where} is longer than {hi} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                matched = re.search(pattern, value) is not None
            except re.error:
                matched = True
            if not matched:
                errors.append(f"{where} {short(value)} does not match {pattern}")

    if isinstance(value, (list, tuple)):
        n = len(value)
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        if is_number(lo) and n < lo:
            errors.append(f"{where} has {n} items, needs at least {lo}")
        if is_number(hi) and n > hi:
            errors.append(f"{where} has {n} items, allows at most {hi}")
        prefix, items = schema.get("prefixItems"), schema.get("items")
        if isinstance(items, list):  # the older tuple form
            prefix, items = items, schema.get("additionalItems")
        start = 0
        if isinstance(prefix, list):
            for i, (item, sub) in enumerate(zip(value, prefix)):
                if isinstance(sub, dict):
                    errors += check(item, sub, f"{where}[{i}]", root, wildcard=wildcard)
            start = len(prefix)
        if isinstance(items, dict):
            for i in range(start, n):
                errors += check(value[i], items, f"{where}[{i}]", root, wildcard=wildcard)
        elif items is False and n > start and not (is_number(hi) and n > hi):
            errors.append(f"{where} has {n} items, allows at most {start}")

    if isinstance(value, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        required = schema.get("required")
        for name in required if isinstance(required, list) else []:
            if name not in value:
                errors.append(f"{where} is missing required field '{name}'")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            sub_path = f"{where}.{name}" if path else str(name)
            if name in props:
                errors += check(item, props[name], sub_path, root, wildcard=wildcard)
            elif extra is False:
                errors.append(f"{where} has unknown field '{name}'; allowed: {short(sorted(props), 400)}")
            elif isinstance(extra, dict):
                errors += check(item, extra, sub_path, root, wildcard=wildcard)
    return errors
