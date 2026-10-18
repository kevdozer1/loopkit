"""Plain-text descriptions of JSON Schemas for the program writer.

One helper serves the skill catalog text (``SkillSpec.catalog_entry``) and the
validator's fix texts (``graph.py``), so the writer reads the same words in both:

>>> describe({"anyOf": [{"type": "integer", "minimum": 0, "maximum": 11}, {"type": "null"}], "default": None})
'integer or null, minimum 0, maximum 11, default null'
>>> describe({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3})
'array of number, 3 items'

An Optional (anyOf with a null branch) is described by its non-null branch. Objects
list their fields one level deep (``$ref`` resolved); an object whose only schema is
``additionalProperties`` is "a map of <type>". Values are JSON (true, null).
"""

from __future__ import annotations

from typing import Any

from .jsonschema_lite import is_number, resolve, short, type_names

_BOUNDS = ("minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum", "minLength", "maxLength")


def unwrap_optional(schema: dict[str, Any], root: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    """``(non-null branch, True)`` for anyOf/oneOf ``[X, null]`` (or type ``[X, "null"]``), else ``(schema, False)``.

    Keys next to the anyOf (default, description) are kept on the returned schema.
    """
    root = root if root is not None else schema
    s = resolve(schema, root) if isinstance(schema, dict) else {}
    for key in ("anyOf", "oneOf"):
        branches = s.get(key)
        if isinstance(branches, list) and branches:
            rest = [b for b in branches if isinstance(b, dict) and type_names(b, root) != {"null"}]
            if len(rest) == 1 and len(rest) < len(branches):
                inner = resolve(rest[0], root)
                outer = {k: v for k, v in s.items() if k not in (key, "title")}
                return {**inner, **outer}, True
    t = s.get("type")
    if isinstance(t, list) and "null" in t:
        rest = [x for x in t if x != "null"]
        if len(rest) == 1:
            return {**s, "type": rest[0]}, True
    return s, False


def _items(n: Any) -> str:
    return f"{n} item" if n == 1 else f"{n} items"


def _size(lo: Any, hi: Any) -> str:
    lo = lo if is_number(lo) else None
    hi = hi if is_number(hi) else None
    if lo is not None and lo == hi:
        return _items(lo)
    if lo and hi is not None:
        return f"{lo} to {_items(hi)}"
    if lo:
        return f"at least {_items(lo)}"
    if hi is not None:
        return f"at most {_items(hi)}"
    return ""


def _item_kind(schema: Any, root: dict[str, Any]) -> str:
    """A compact description for an array item or a map value: 'number (minimum 0)'."""
    if not isinstance(schema, dict):
        return "any value"
    text = describe(schema, root, fields=False, defaults=False)
    head, _, rest = text.partition(", ")
    return f"{head} ({rest})" if rest else head


def _array_head(s: dict[str, Any], root: dict[str, Any]) -> str:
    prefix, items = s.get("prefixItems"), s.get("items")
    if isinstance(items, list):  # the older tuple form
        prefix, items = items, s.get("additionalItems")
    if isinstance(prefix, list) and prefix:
        kinds = [_item_kind(p, root) for p in prefix]
        if len(set(kinds)) == 1 and not isinstance(items, dict):
            head = f"array of {kinds[0]}"
        else:
            head = "array [" + ", ".join(kinds) + "]"
            if isinstance(items, dict):
                head += f", then {_item_kind(items, root)}"
    elif isinstance(items, dict):
        head = f"array of {_item_kind(items, root)}"
    else:
        head = "array"
    size = _size(s.get("minItems"), s.get("maxItems"))
    return f"{head}, {size}" if size else head


def _object_head(s: dict[str, Any], root: dict[str, Any], fields: bool) -> str:
    props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
    extra = s.get("additionalProperties")
    if not props:
        return f"a map of {_item_kind(extra, root)}" if isinstance(extra, dict) else "object"
    if fields:
        required = s.get("required") if isinstance(s.get("required"), list) else []
        parts = []
        for name, sub in props.items():
            flag = " (required)" if name in required else ""
            parts.append(f"{name}{flag}: {describe(sub, root, fields=False)}")
        head = "object {" + "; ".join(parts) + "}"
    else:
        head = "object {" + ", ".join(str(n) for n in props) + "}"
    if isinstance(extra, dict):
        head += f", other fields: {_item_kind(extra, root)}"
    return head


def _head(s: dict[str, Any], root: dict[str, Any], fields: bool) -> str:
    if "type" not in s:
        for key in ("anyOf", "oneOf"):
            branches = s.get(key)
            if isinstance(branches, list) and branches:
                parts = []
                for branch in branches:
                    text = "null" if type_names(branch, root) == {"null"} else _item_kind(branch, root)
                    if text not in parts:
                        parts.append(text)
                return " or ".join(parts)
    names = type_names(s, root)
    if names == {"array"} or (not names and ("items" in s or "prefixItems" in s)):
        return _array_head(s, root)
    if names == {"object"} or (not names and ("properties" in s or "additionalProperties" in s)):
        return _object_head(s, root, fields)
    if not names:
        return "any value"
    return " or ".join(sorted(names))


def describe(schema: dict[str, Any], root: dict[str, Any] | None = None, *, fields: bool = True,
             defaults: bool = True, descriptions: bool = False) -> str:
    """One line for ``schema``: types, then enum, bounds, default and (optionally) its description.

    ``fields`` lists an object's fields with their own descriptions (one level
    deep); nested objects show only their field names.
    """
    root = root if root is not None else schema
    s, nullable = unwrap_optional(schema, root)
    head = _head(s, root, fields)
    bits = [head + " or null" if nullable and head != "any value" else head]
    if "const" in s:
        bits.append(f"always {short(s['const'], 200)}")
    if isinstance(s.get("enum"), list):
        bits.append(f"enum {short(s['enum'], 400)}")
    for key in _BOUNDS:
        if is_number(s.get(key)):
            bits.append(f"{key} {short(s[key])}")
    if isinstance(s.get("pattern"), str):
        bits.append(f"pattern {s['pattern']}")
    if defaults and "default" in s:
        bits.append(f"default {short(s['default'], 200)}")
    if descriptions and isinstance(s.get("description"), str) and s["description"].strip():
        bits.append(s["description"].strip())
    return ", ".join(bits)
