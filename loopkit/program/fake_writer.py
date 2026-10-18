"""Scripted FakeModel behaviour for dry runs and tests.

``reference_writer`` answers a writer prompt with the hand-written reference graph for the task
named in the prompt, and any other role with the smallest answer that fits its JSON schema. A
result produced this way measures the skill library and the loop, never a model.
"""

from __future__ import annotations

import re

from ..router.adapters.fake import from_schema
from .reference import reference_graph

TASK_RE = re.compile(r"TASK (T\d+):")


def task_of(messages: list[dict]) -> str | None:
    for msg in messages:
        m = TASK_RE.search(str(msg.get("content", "")))
        if m:
            return m.group(1)
    return None


def reference_writer(req):
    if req.role == "writer":
        task = task_of(req.messages)
        if task is not None:
            return reference_graph(task)
    return from_schema(req.json_schema) if req.json_schema else "fake response"
