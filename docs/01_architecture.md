# 01. Architecture and interface spec

Level 1: five typed interfaces sit between the parts, so swapping a model is one line of config. Only the program writer is rented. Everything that makes it reliable is loopkit's.

```
TaskSpec ──► program writer (any model, via router) ──► ProgramGraph JSON
                 ▲                                            │ validate(catalog), repair up to 3x
                 │ failure summaries as text                  ▼
            rehearse-and-patch  ◄── Verdicts ◄── checker ◄── interpreter on Backend (Sim now, SO-101 Sunday)
                                                              │
                                                    TraceEvents ──► reports, exporters (SFT, prefs, RL env, decisions)
```

## Package layout

```
loopkit/
  paths.py        LOOPKIT_DATA, run dirs, disk-floor check
  config.py       .env loader (CRLF safe), model and bakeoff yaml loaders
  schemas/        TaskSpec, SkillSpec, ProgramGraph, Verdict, TraceEvent, Observation
  router/         chat(), adapters, disk cache, spend ledger
  traces/         TraceWriter, load_runs()
  sim/            rack_cell scene builder, randomization, cameras, SimBackend, checkers, oracles
  skills/         perception, motion, manipulation, calibration, catalog generation
  runtime/        Backend protocol, interpreter
  program/        writer prompt, parse, validate-and-repair
  bakeoff/        run.py (resumable), report.py, figure
  export/         sft, prefs, rl_env, decisions
tests/
config/           models.yaml, bakeoff.yaml
sessions/         S0 to S7 prompts
reports/          one md per session plus PROGRESS.md
```

## Schemas (pydantic v2, each exports JSON Schema to `schemas/json/`)

**TaskSpec**
- `id`, `title`, `goal_text`
- `end_condition`: plain text a checker can score, for example "plug seated in port 7, depth at least 8 mm, lateral error at most 0.5 mm"
- `checker_id`, `scene_id`
- `allowed_skills`: list of skill names
- `nominal_geometry`: the "BIM" the robot is allowed to know. Rack nominal pose, port pitch, slot heights, tray location. The real pose differs by the per-seed randomization, so skills still have to localize.
- `budgets`: `max_big_model_calls`, `max_usd`, `max_seconds`
- `safety`: `workspace_bounds` (AABB in base frame), `max_speed` (m/s), `max_force` (N)

**SkillSpec**: `name`, `docstring`, `params` (JSON Schema), `returns` (JSON Schema), `preconditions`, `postconditions`. Generated from the typed function and its docstring. The catalog text sent to the writer is built from these only.

**ProgramGraph**
- `nodes`: `[{id, skill, params}]`
- `edges`: `[{src, dst, on: "success" | "failure"}]`
- `entry`, `max_retries` (per node, default 1), `terminal_nodes` (ids of `done` and `abort` nodes)
- Data flow: a param may be a literal or a reference `{"ref": "<node_id>.<field>"}` to an earlier node's return value. The interpreter keeps a blackboard of node outputs.
- `validate(catalog)` returns readable errors for: bad JSON, unknown skills, params failing the skill's JSON Schema or range, refs to unknown nodes or fields, unreachable nodes, a non-terminal node with no failure edge, and cycles with no retry bound. Each error names the node and says how to fix it, because the text goes back to the writer.

**Observation**: camera images (overhead, front, wrist) as paths plus arrays, camera intrinsics and extrinsics, joint state, gripper width, gripper force estimate, sim time. No body poses.

**Verdict**: `passed`, `score` 0 to 1, `evidence {metrics: dict, image_paths: list}`, `failure_class` in `perception, grasp, alignment, stopping, timeout, safety, other`, `failing_node` (optional).

**TraceEvent**: `run_id`, `ts`, `seq`, `kind` in `model_call, node_start, node_end, verdict, patch, intervention, budget_stop`, `payload`.
- `model_call` payload: `provider, model, role (writer|reader|system1), input_tokens, output_tokens, reasoning_tokens, usd, usd_source (provider|computed), latency_ms, cache_hit, request_hash`
- `node_end` payload: `node_id, skill, ok, outputs, error, sim_seconds`
- `patch` payload: `round, graph_before_hash, graph_after_hash, failure_summary`

## Router

`chat(model_key, messages, images=None, json_schema=None, role="writer", max_tokens=..., reasoning="high") -> Response`

- `model_key` is a key in `config/models.yaml`. The adapter and real model id come from there.
- Adapters: `openrouter` (default), `elorian`, `jev`, `fake`. Optional direct `anthropic`, `openai`, `gemini`.
- OpenRouter: send `"usage": {"include": true}` and record `usage.cost` as `usd` with `usd_source: provider`. Otherwise compute from yaml prices.
- Reasoning: the program writer uses each model's highest reasoning setting and records it. Small JSON calls (the T1 reader, system1 probes) disable reasoning, because statebench found reasoning truncated JSON answers on 19 of 24 calls.
- Disk cache under `LOOPKIT_DATA/cache/` keyed by SHA-256 of the canonical request (model id, messages, image hashes, schema, params). A hit costs $0 and is traced with `cache_hit: true`.
- Retries with exponential backoff on 429 and 5xx, a per-call timeout, and a clear error naming the missing env var when a key is absent.
- Spend ledger at `LOOPKIT_DATA/spend_ledger.jsonl`. Before every call the router projects worst-case cost (input tokens plus `max_tokens` at yaml prices, or Astra prices if the price is unknown) and refuses the call if it would cross the active cap. Caps can be set per session and per model.

## Interpreter

Runs a ProgramGraph on a `Backend` protocol with `observe() -> Observation`, `act(command) -> ActResult`, `stop()`, `reset(seed)`. `SimBackend` wraps the rack_cell scene. `SO101Backend` is a stub until S5. The interpreter enforces per-node timeouts, total `max_seconds`, workspace bounds and speed limits, and emits `node_start` and `node_end` per node. A safety violation ends the run with `failure_class: safety`.

## Rehearse-and-patch

1. Run the graph on N training seeds.
2. Summarize failures as text: failing node, its error and outputs, checker metrics, and what differed in the scene from passing seeds (for example "rack yaw +2.6 deg in all 4 failures"). No raw images in the patch prompt by default. CaP-X found raw images hurt and text about what changed helped every model.
3. Send the summary with the current graph to the writer. Validate the new graph.
4. Stop after 5 rounds or when the pass rate fails to improve for 2 rounds. Keep the best graph.

## Design choices and the evidence behind them

- Graphs over free Python. In GaP's ablation one LLM writing plain Python scored 0. A graph can be validated before it moves a motor.
- Skills see cameras only. That keeps sim pass rates honest and transferable.
- Failures go back as text. CaP-X result above.
- Every call goes through the router with a cache. Replays cost nothing and a rerun on a new model release is one command.
- Prices live in config and reports build from traces alone. Every number in a report traces back to a logged event.
