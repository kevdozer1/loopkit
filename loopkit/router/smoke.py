"""Live smoke: one small call to every models.yaml model that has an id and a key.

    uv run python -m loopkit.router.smoke              # every entry except fake
    uv run python -m loopkit.router.smoke --only opus,luna
    uv run python -m loopkit.router.smoke --fake       # also the offline FakeModel ($0), to check the path

Vision models get one test image (a dark grey panel with 12 LEDs in a row, 5 of them
green) and "How many ports have a green LED? Answer with JSON {"green": <integer>}."
with reasoning disabled, max_tokens 300 and role reader. Text-only models get the
same panel as a short text state: Jev as one typed ``choice`` question with options
"0" to "12", other text models as the same question in text.

Prints a table (model key, real id, latency, tokens, usd, usd_source, answer and the
expected answer) and writes it to ``LOOPKIT_DATA/runs/s1_smoke/live_table.json``.
Models with no id or no key are listed as skipped with the reason. Before the first
paid call it sets the S1 spend session ($1 cap, $21 total) unless a session is
already active. Calls bypass the cache so every number is measured live. Any error
for one model (a router error or anything else) becomes that model's row, so one
model never stops the table. Nothing a key could hide in is printed: router errors
are scrubbed, and the output and the returned rows are checked against the loaded
keys once more before they are printed, written or returned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from .. import config, paths
from ..config import ModelEntry, ModelsConfig
from .adapters import available_adapters, get_adapter
from .core import chat, resolve_route
from .errors import ModelCallError
from .http import HttpClient
from .ledger import Ledger
from .redact import redact, strip_secret

SESSION = "S1"
SESSION_CAP_USD = 1.0
TOTAL_CAP_USD = 21.0
MAX_TOKENS = 300
QUESTION = 'How many ports have a green LED? Answer with JSON {"green": <integer>}.'
# left to right, ports 1 to 12: five green, the rest amber, red or off
LED_STATES = ("green", "amber", "green", "off", "red", "green", "off", "amber", "green", "red", "off", "green")
EXPECTED_GREEN = sum(1 for s in LED_STATES if s == "green")
LED_RGB = {"green": (0, 255, 0), "amber": (255, 153, 0), "red": (255, 0, 0), "off": (96, 96, 96)}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"green": {"type": "integer", "minimum": 0, "maximum": len(LED_STATES)}},
    "required": ["green"],
    "additionalProperties": False,
}
JEV_QUESTION_NAME = "green_led_ports"
IMAGE_STATE = (f"Front panel of a network switch with {len(LED_STATES)} ports in a row, shown in the image, "
               "with one LED above each port.")


def smoke_dir() -> Path:
    path = paths.runs_dir() / "s1_smoke"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_led_image(path: Path) -> Path:
    """A dark grey panel with 12 small LEDs above 12 ports, ``EXPECTED_GREEN`` of them green."""
    from PIL import Image, ImageDraw

    pitch, width, height = 72, 72 * len(LED_STATES) + 48, 220
    img = Image.new("RGB", (width, height), (34, 34, 36))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 20, width - 12, height - 20], fill=(52, 52, 56), outline=(80, 80, 84), width=2)
    for i, state in enumerate(LED_STATES):
        cx = 24 + pitch // 2 + i * pitch
        draw.ellipse([cx - 10, 52, cx + 10, 72], fill=LED_RGB[state], outline=(20, 20, 20))
        draw.rectangle([cx - 22, 96, cx + 22, 140], fill=(14, 14, 16), outline=(90, 90, 94), width=2)
        draw.text((cx - 6, 158), str(i + 1), fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def text_state() -> str:
    leds = ", ".join(f"port {i + 1} {state}" for i, state in enumerate(LED_STATES))
    return (f"Front panel of a network switch with {len(LED_STATES)} ports in a row. "
            f"LED above each port, left to right: {leds}.")


def jev_questions() -> dict[str, Any]:
    n = len(LED_STATES)
    criteria = {str(k): f"exactly {k} of the {n} ports have a green LED" for k in range(n + 1)}
    return {JEV_QUESTION_NAME: {"type": "choice", "instructions": "How many ports have a green LED?",
                                "criteria": criteria}}


def skip_reason(entry: ModelEntry) -> str | None:
    routed, _note = resolve_route(entry)
    if routed.adapter not in available_adapters():
        return f"adapter {routed.adapter} is not implemented (route it through openrouter)"
    if routed.adapter != "fake" and not routed.id:
        return "no id in config/models.yaml (verified: false)"
    if routed.adapter == "elorian" and not routed.base_url:
        return "no base_url in config/models.yaml"
    names = get_adapter(routed.adapter).env_vars(routed)
    if names and not config.has_secret(names):
        others = f" (or {', '.join(names[1:])})" if len(names) > 1 else ""
        alt = getattr(entry, "alt_route", None)
        alt_hint = ""
        if alt is not None and getattr(alt, "id", None) and alt.adapter in available_adapters():
            alt_vars = get_adapter(alt.adapter).default_env_vars
            if alt_vars:
                alt_hint = f", or {alt_vars[0]} for its alt_route {alt.id}"
        return f"no key: set {names[0]}{others}{alt_hint} in .env"
    return None


def _answer(adapter: str, response: Any) -> Any:
    value = response.json
    if adapter == "jev":
        block = value.get(JEV_QUESTION_NAME) if isinstance(value, dict) else None
        choice = block.get("choice") if isinstance(block, dict) else None
        try:
            return int(choice)
        except (TypeError, ValueError):
            return choice
    if isinstance(value, dict) and "green" in value:
        return value["green"]
    return None


def call_one(key: str, entry: ModelEntry, image_path: Path, *, models: ModelsConfig, ledger: Ledger,
             http: HttpClient | None = None, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """One model's row. Never raises for that model: any error (router or not) becomes the row's ``error``."""
    try:
        return _call_one(key, entry, image_path, models=models, ledger=ledger, http=http, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - one model must never stop the table
        return {"model_key": key, "model_id": getattr(entry, "id", None), "real_id": getattr(entry, "id", None),
                "adapter": getattr(entry, "adapter", None), "expected": EXPECTED_GREEN, "ok": False,
                "error": redact(f"{type(exc).__name__}: {exc}")}


def _call_one(key: str, entry: ModelEntry, image_path: Path, *, models: ModelsConfig, ledger: Ledger,
              http: HttpClient | None = None, max_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    routed, route_note = resolve_route(entry)
    row: dict[str, Any] = {"model_key": key, "model_id": routed.id, "real_id": routed.id,
                           "adapter": routed.adapter, "route": route_note, "expected": EXPECTED_GREEN, "ok": False}
    kwargs: dict[str, Any] = dict(max_tokens=max_tokens, reasoning="disabled", models=models, ledger=ledger,
                                  use_cache=False, phase="smoke", http=http)
    if routed.adapter == "jev":  # the typed-question protocol (Jev, and d1 through its own endpoint)
        kwargs.update(role="system1", extra={"questions": jev_questions()})
        if entry.vision:
            kwargs.update(messages=IMAGE_STATE, images=[image_path])
            row["input"] = "image + typed choice question"
        else:
            kwargs.update(messages=text_state())
            row["input"] = "text state + typed choice question"
    elif entry.vision:
        kwargs.update(messages=QUESTION, images=[image_path], json_schema=ANSWER_SCHEMA, role="reader")
        row["input"] = "image + question"
    else:
        role = "system1" if "system1" in entry.roles else "reader"
        kwargs.update(messages=f"{text_state()}\n\n{QUESTION}", json_schema=ANSWER_SCHEMA, role=role)
        row["input"] = "text state + question"
    notes: list[str] = []
    try:
        try:
            response = chat(key, **kwargs)
        except ModelCallError as exc:
            # some reasoning models refuse reasoning: disabled; that refusal costs nothing, so retry once with low
            if exc.status == 400 and "reason" in str(exc).lower():
                notes.append("the model refused reasoning disabled; retried with low")
                kwargs["reasoning"] = "low"
                response = chat(key, **kwargs)
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - any error is this model's row, never the end of the table
        row.update(error=redact(f"{type(exc).__name__}: {exc}"), note="; ".join(notes) or None)
        return row
    answer = _answer(routed.adapter, response)
    row.update(
        ok=True, real_id=response.model, latency_ms=round(response.latency_ms, 1),
        input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens, usd=response.usd, usd_source=response.usd_source,
        answer=answer, correct=answer == EXPECTED_GREEN, json_mode=response.json_mode,
        json_error=response.json_error, reasoning_setting=response.reasoning_setting,
        finish_reason=response.finish_reason, text=response.text[:300], note="; ".join(notes) or None,
    )
    return row


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def format_table(rows: list[dict[str, Any]]) -> str:
    head = ("model key", "real id", "latency ms", "in tok", "out tok", "usd", "usd_source",
            f"answer (expected {EXPECTED_GREEN})")
    body = []
    for r in rows:
        if r.get("ok"):
            answer = f"{_fmt(r.get('answer'))} ({'right' if r.get('correct') else 'wrong'})"
            if r.get("json_error"):
                answer += f" json_error: {str(r['json_error'])[:60]}"
        else:
            answer = f"ERROR {str(r.get('error'))[:90]}"
        body.append((r["model_key"], _fmt(r.get("real_id")), _fmt(r.get("latency_ms"), ".0f"),
                     _fmt(r.get("input_tokens")), _fmt(r.get("output_tokens")), _fmt(r.get("usd"), ".6f"),
                     _fmt(r.get("usd_source")), answer))
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
    lines = [" | ".join(h.ljust(w) for h, w in zip(head, widths)),
             "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(lines)


def _free(entry: ModelEntry) -> bool:
    """A call that projects $0 (the FakeModel at its zero prices) needs no spend session."""
    return entry.adapter == "fake" and not (entry.price_in or entry.price_out)


def _loaded_keys(models: ModelsConfig) -> list[str]:
    names: set[str] = set()
    for entry in models.models.values():
        if entry.adapter in available_adapters():
            names.update(get_adapter(entry.adapter).env_vars(entry))
    return [v for v in (config.env(n) for n in sorted(names)) if v and len(v) >= 6]


def _guard(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def run_smoke(*, include_fake: bool = False, only: list[str] | None = None, models: ModelsConfig | None = None,
              ledger: Ledger | None = None, http: HttpClient | None = None, out_dir: Path | None = None,
              max_tokens: int = MAX_TOKENS, echo: bool = True) -> dict[str, Any]:
    """Run the smoke and return ``{"rows", "skipped", ...}`` (also written to live_table.json)."""
    cfg = models if models is not None else config.load_models()
    led = ledger if ledger is not None else Ledger()
    out = Path(out_dir) if out_dir is not None else smoke_dir()
    out.mkdir(parents=True, exist_ok=True)
    image_path = make_led_image(out / "leds.png")
    secrets = _loaded_keys(cfg)

    def say(text: str = "") -> None:
        if echo:
            print(_guard(text, secrets), flush=True)

    plan: list[tuple[str, ModelEntry]] = []
    skipped: list[dict[str, str]] = []
    for key, entry in cfg.models.items():
        if entry.adapter == "fake" and not include_fake:
            continue
        if only and key not in only:
            continue
        try:
            reason = skip_reason(entry)
        except Exception as exc:  # noqa: BLE001 - one model must never stop the table
            reason = redact(f"could not check this entry: {type(exc).__name__}: {exc}")
        if reason:
            skipped.append({"model_key": key, "reason": reason})
        else:
            plan.append((key, entry))

    session_note = None
    if any(not _free(entry) for _key, entry in plan):
        active = led.caps().get("active_session")
        if active:
            session_note = f"using the active spend session {active}"
        else:
            led.set_active_session(SESSION, SESSION_CAP_USD, total_cap_usd=TOTAL_CAP_USD)
            session_note = f"set spend session {SESSION}: cap ${SESSION_CAP_USD:.2f}, total cap ${TOTAL_CAP_USD:.2f}"
        say(session_note)
    spent_before = led.spent()

    rows: list[dict[str, Any]] = []
    for key, entry in plan:
        say(f"calling {key} ({entry.id}) ...")
        row = call_one(key, entry, image_path, models=cfg, ledger=led, http=http, max_tokens=max_tokens)
        for secret in secrets:  # no loaded key in a returned row either, wherever a server echoed it
            row = strip_secret(row, secret)
        rows.append(row)

    spent = led.spent() - spent_before
    result = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": QUESTION, "expected_green": EXPECTED_GREEN, "led_states": list(LED_STATES),
        "image": str(image_path), "max_tokens": max_tokens, "reasoning": "disabled",
        "rows": rows, "skipped": skipped, "spend_usd_this_run": round(spent, 8), "session": session_note,
    }
    table_path = out / "live_table.json"
    table_path.write_text(_guard(json.dumps(result, indent=2, default=str), secrets) + "\n", encoding="utf-8")

    say()
    say(format_table(rows) if rows else "No model was called (every entry was skipped).")
    if skipped:
        say()
        say("Skipped:")
        for s in skipped:
            say(f"  {s['model_key']}: {s['reason']}")
    say()
    say(f"Spend this run: ${spent:.6f}. Table: {table_path}")
    result["table_path"] = str(table_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m loopkit.router.smoke", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated model keys to call (default: all)")
    parser.add_argument("--fake", action="store_true", help="also call the offline FakeModel ($0)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args(argv)
    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    result = run_smoke(include_fake=args.fake, only=only, max_tokens=args.max_tokens)
    failed = [r for r in result["rows"] if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
