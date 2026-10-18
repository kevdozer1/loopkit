# 06. Reuse map: Kevin's own repos

Both repos below are Kevin's prior work. Read them, copy the pattern, rewrite it inside loopkit. Do not import across repos and do not edit them.

## statebench (`C:\Users\Kevin\projects\statebench`)

| Need in loopkit | Look at | What to take |
|---|---|---|
| `.env` loading and data root | `src/statebench/config.py`, `env.ps1` | CRLF-safe dotenv parse, `data_root()` from an env var, `free_space_gb()` for the disk floor |
| Backend registry and credential errors | `src/statebench/backends/base.py` | Name-to-class registry, response record with measured latency and tokens, `load_secret()` that names the missing variable |
| Jev calls | `src/statebench/backends/jev.py` | Endpoint `https://api.typesafe.ai/v1/systemone`, model `jev-latest`, env var `TYPESAFE_JEV_API_KEY`, no cost field in the API (compute from tokens) |
| OpenRouter calls | `src/statebench/backends/llm.py` | Request shape, `usage.include`, reasoning toggle |
| Spend cap across invocations | `src/statebench/spend_guard.py` | Ledger file plus projected-cost check before each call. loopkit seeds from its own ledger, not OpenRouter `/credits` |
| Rendering and mp4 | `src/statebench/render.py`, `video19.py` | `mujoco.Renderer` on Windows, frame writer, mp4 via imageio |
| MuJoCo scene with a fixed executor and physical verifier | `src/statebench/env.py`, `verify.py`, `executor.py` | The three-way split (decision maker, fixed executor, fixed verifier) that loopkit's writer, interpreter and checker mirror |
| Seed discipline and stats | `notes/HANDOFF.md` section on seeds, `stats.py` | Paired seeds across conditions, Wilson intervals |

## robolabel (`C:\Users\Kevin\projects\robolabel`)

| Need in loopkit | Look at | What to take |
|---|---|---|
| Price table | `src/robolabel/prices.yaml`, `src/robolabel/eval/prices.py` | Field layout (`price_in`, `price_out`, `cached_in`, source and access date). Prices for Opus 5.5, GPT-6 Sol, GPT-6 Astra, GPT-6 Luna, DeepSeek V4.1 Flash come from its 2026-09-27 OpenRouter read |
| Provider layer | `src/robolabel/providers/` (`base.py`, `openrouter.py`, `gemini.py`, `openai.py`, `mock.py`) | Adapter shape and a mock provider for offline tests |

## SO-101 (for S5 only)

- Bring-up script: `C:\Users\Kevin\projects\kevin_bringup.ps1`. Robot id `p4_follower`, data home `D:\so101_data`, LeRobot runs from a conda env named `lerobot`.
- Port check: `C:\Users\Kevin\projects\so101_check.py`.
- LeRobot source: `C:\Users\Kevin\projects\lerobot`.
- `SO101Backend` should call LeRobot from that conda env, either as a subprocess bridge or by running loopkit's backend module inside the env. Decide in S5, not before.
