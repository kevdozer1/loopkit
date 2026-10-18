"""Write each schema's JSON Schema to ``loopkit/schemas/json/``."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import ProgramGraph
from .observation import Observation
from .skill import SkillSpec
from .task import TaskSpec
from .trace import PAYLOAD_MODELS, TraceEvent
from .verdict import Verdict

JSON_DIR = Path(__file__).resolve().parent / "json"
MODELS = {
    "TaskSpec": TaskSpec,
    "SkillSpec": SkillSpec,
    "ProgramGraph": ProgramGraph,
    "Observation": Observation,
    "Verdict": Verdict,
    "TraceEvent": TraceEvent,
}


def schemas() -> dict[str, dict]:
    out = {name: model.model_json_schema() for name, model in MODELS.items()}
    for kind, model in PAYLOAD_MODELS.items():
        out[f"TraceEvent.{kind}"] = model.model_json_schema()
    return out


def export_json_schemas(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or JSON_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in schemas().items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in export_json_schemas():
        print(p, flush=True)
