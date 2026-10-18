# 05. Machine gotchas (Kevin's Windows box)

Learned the hard way in statebench. Each one cost real time.

## Disk and filesystem

- `D:` is exFAT. No symlinks, no hardlinks, 512 KB clusters (small files waste space, so batch images into fewer files or keep frame counts modest). uv cannot install interpreters there. git needs `safe.directory` for repos on D:. Keep the repo and venv on C:, data and caches on D:, and set `UV_LINK_MODE=copy`.
- Disk floor: stop and report if C: or D: has less than 10 GB free.
- The harness blocks `Remove-Item` on top-level `D:\` paths. Use `[IO.File]::Delete` or `[IO.Directory]::Delete` inside your own subfolders.

## PowerShell

- Always load env with `. .\env.ps1`. Never a bash loader. `.env` files here have CRLF line endings, and a bash loader produced paths ending in `\r` that failed with a confusing `WinError 123`.
- PowerShell 5.1 `-Encoding utf8` writes a BOM, which breaks JSON parsers. Use `[IO.File]::WriteAllText(path, text, [Text.UTF8Encoding]::new($false))`, or write files from Python.
- Piping a long command into `Select-Object -First N` can kill it and report exit 255 on a successful run. Capture to a variable instead.
- Background stdout is block-buffered. Print progress with `flush=True`.
- Large heredocs with tables or backticks fail to parse. Use the Write tool for substantial files.

## MuJoCo rendering

- Do not set `MUJOCO_GL=egl` on Windows. The default WGL/GLFW path with `mujoco.Renderer` works. Unset it if a tool sets it.
- `mujoco.Renderer(model, 480, 640)` takes height first.

## Model APIs

- OpenRouter: disable reasoning for small JSON answers (`"reasoning": {"enabled": false}`). Gemini Flash spent 327 of 535 completion tokens reasoning and truncated the JSON on 19 of 24 calls. For the program writer, use high reasoning and a generous `max_tokens`.
- Do not nest text inside another JSON string. Escaping inflated prompt tokens. Send state as a plain text block.
- Request `"usage": {"include": true}` to get `usage.cost` back.
- The OpenRouter key in statebench is shared with other projects, so OpenRouter `/credits` usage is not loopkit's spend. loopkit keeps its own ledger. Best: a dedicated OpenRouter key for loopkit with a credit limit set on the key as a hard backstop.

## Git

- Pin LF with `.gitattributes`. statebench lost file hashes to CRLF working copies.
