# loopkit: rules for every agent session

loopkit is a small model-agnostic loop for robot task programs. A frontier model writes a program graph from a skill library, the graph is rehearsed in MuJoCo, a privileged checker scores it, failures go back to the writer as text, and every event is traced. A bake-off runs the same loop across models and reports pass rate, cost per success and big-model calls.

## Hard rules

1. Personal project and prior IP of Kevin Hopkins. No company code, data or confidential information ever goes in this repo. Nothing here names any employer.
2. `context/` is private reference material and is gitignored. Read it for intent. Never copy text from it into committed files, commit messages or reports.
3. Commit after each working step with a clear message. Never push. Never add a remote.
4. Large outputs (images, videos, traces, caches, bake-off results) go under `LOOPKIT_DATA` (default `D:\loopkit_data`). Never into the repo. No `D:` path appears literally in committed code. Read it through `loopkit/paths.py`.
5. API keys come from `.env` through `env.ps1` or `loopkit/config.py`. Never print, log or commit a key.
6. Respect the spend cap in each session file. Every paid call goes through the router, which records cost and refuses a call that would cross the cap.
7. Videos are mp4. No gifs.
8. Every session ends by writing `reports/<session>.md`: what ran, results with numbers, open issues, and Kevin's exact next action. Also update `reports/PROGRESS.md`.
9. Report honestly. Record measured latency, tokens and cost per call, never a vendor claim. If something did not run, say so. Never write "first", "novel" or "nobody has".
10. Skills see camera images, proprioception and the nominal fixture geometry in the TaskSpec only. They never read privileged sim state. Checkers may read privileged state. A test enforces this.
11. Seeds: 0 to 999 for development. 500 to 519 are the S3 held-out set. 1000 to 1009 are bake-off training seeds. 2000 to 2019 are bake-off held-out seeds and are never used before the bake-off.
12. Disk floor: if free space on C: or D: drops below 10 GB, stop and report.

## Stack

- Python 3.12 with uv (same as statebench on this machine). `UV_LINK_MODE=copy`. The venv lives in the repo on C:.
- MuJoCo (statebench runs 3.13.0 here), Menagerie Franka Panda from `third_party/mujoco_menagerie`.
- pydantic v2 for schemas, pytest for tests, httpx for HTTP, pandas plus pyarrow for trace tables.
- Model calls: OpenRouter is the default route for frontier models (one key, `usage.cost` returned). Direct adapters exist for Elorian (Glimpse) and Typesafe (Jev). Direct Anthropic, OpenAI and Gemini adapters are optional.

## Read before coding

1. `KICKOFF.md` if you were started from it
2. `docs/01_architecture.md`, the interface spec
3. `docs/02_task_suite.md`, the scene and tasks
4. `docs/03_bakeoff_protocol.md`
5. `docs/05_machine_gotchas.md`, Windows traps that cost real time on this machine
6. `docs/06_reuse_map.md`, patterns to copy from Kevin's own statebench and robolabel repos
7. `docs/04_prior_art.md`, when designing skills, the writer prompt or the patch loop
8. The session file you were asked to run, under `sessions/`

## Running things

```powershell
. C:\Users\Kevin\projects\loopkit\env.ps1
Set-Location C:\Users\Kevin\projects\loopkit
uv run pytest -q
```

Background commands buffer stdout. Use `print(..., flush=True)` for progress lines. Use the Write tool for any substantial file. Large shell heredocs fail to parse here.

## Other repos on this machine

You may read `C:\Users\Kevin\projects\statebench` and `C:\Users\Kevin\projects\robolabel` (Kevin's own) and copy patterns from them, rewritten inside loopkit. Do not import from them and do not modify them. Do not read or borrow from any other project folder on this machine.
