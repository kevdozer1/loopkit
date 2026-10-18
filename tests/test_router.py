"""Router tests. All offline: HTTP goes through httpx.MockTransport, and the default
client is replaced by one that fails the test if anything tries the network."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from loopkit import config
from loopkit.config import ModelEntry, ModelsConfig
from loopkit.router import (
    DiskCache,
    FileLock,
    HttpClient,
    Ledger,
    MissingCredentialError,
    ModelCallError,
    ModelNotConfigured,
    RouterError,
    SpendCapExceeded,
    chat,
    clear_fake_script,
    fake_calls,
    fake_script,
    from_schema,
    normalize_image,
    parse_json_text,
    set_active_session,
    set_fake_script,
)
from loopkit.router import core as router_core
from loopkit.router import http as router_http
from loopkit.router.adapters import get_adapter
from loopkit.router.adapters.base import AdapterRequest
from loopkit.router.adapters.elorian import build_elorian_payload, normalize_elorian_reasoning
from loopkit.router.adapters.jev import DEFAULT_URL as JEV_URL
from loopkit.router.adapters.openrouter import API_URL as OPENROUTER_URL
from loopkit.router.adapters.openrouter import OpenRouterAdapter, build_payload, normalize_reasoning
from loopkit.router.core import normalize_messages
from loopkit.router.media import normalize_images
from loopkit.schemas.jsonschema_lite import check
from loopkit.schemas.trace import ModelCallPayload

# every key variable .env.example documents, plus the ones these tests use
KEY_VARS = tuple(sorted({k for k in config.dotenv_values(config.REPO_ROOT / ".env.example") if k.endswith("_KEY")}
                        | {"OPENROUTER_API_KEY", "ELORIAN_API_KEY", "TYPESAFE_JEV_API_KEY", "TYPESAFE_API_KEY",
                           "LK_TEST_SYS1_KEY"}))
# test keys, built at runtime so no literal looks like a credential; neither is a real key
PATTERN_KEY = "sk-or-" + "test-" + "k" * 40  # also matches the redaction patterns
PLAIN_KEY = "lk-test-" + "Q7" * 20  # matches no pattern: only exact-value redaction catches it
GREEN_SCHEMA = {"type": "object", "properties": {"green": {"type": "integer", "minimum": 0, "maximum": 12}},
                "required": ["green"], "additionalProperties": False}
HAS_ALT_ROUTE = "alt_route" in ModelEntry.model_fields


def _entry(**kw):
    base = dict(display="test entry", adapter="openrouter", id="vendor/test-model", price_in=1.0, price_out=2.0,
                vision=True, roles=["writer", "reader", "system1"], verified=True, source="test", accessed=None)
    base.update(kw)
    return base


def _models() -> ModelsConfig:
    models = {
        "fake": _entry(adapter="fake", id="fake", price_in=0.0, price_out=0.0),
        "paid_fake": _entry(adapter="fake", id="fake-paid", price_in=10.0, price_out=50.0),
        "orm": _entry(id="vendor/test-model", price_in=2.0, price_out=10.0),
        "orm_hi": _entry(id="vendor/test-model-hi", price_in=2.0, price_out=10.0, reasoning={"effort": "xhigh"}),
        "unset": _entry(id=None, price_in=None, price_out=None, verified=False),
        "unpriced": _entry(id="vendor/unpriced", price_in=None, price_out=None, verified=False),
        "astra": _entry(id="vendor/astra", price_in=10.0, price_out=50.0),
        "el": _entry(adapter="elorian", id="elorian-test", base_url="https://elorian.invalid/v1",
                     env_key="ELORIAN_API_KEY", price_in=0.6, price_out=3.0, reasoning={"effort": "xhigh"}),
        "el_nourl": _entry(adapter="elorian", id="elorian-test", base_url=None, price_in=0.6, price_out=3.0),
        "jev": _entry(adapter="jev", id="jev-latest", env_key="TYPESAFE_JEV_API_KEY", price_in=0.042,
                      price_out=0.0, vision=False, roles=["system1"]),
        "sys1_vision": _entry(adapter="jev", id="sys1-vision", endpoint="https://decisions.invalid/systemone",
                              env_key="LK_TEST_SYS1_KEY", price_in=0.04, price_out=0.0, vision=True,
                              roles=["system1"]),
        "free_or": _entry(id="vendor/free-tier", price_in=0.0, price_out=0.0, notes="free tier (rate limited)"),
        "orm_cache": _entry(id="vendor/cache-model", price_in=2.0, price_out=10.0, cached_in=0.2),
    }
    if HAS_ALT_ROUTE:
        models["el_alt"] = _entry(adapter="elorian", id="elorian-test", base_url="https://elorian.invalid/v1",
                                  env_key="ELORIAN_API_KEY", price_in=0.6, price_out=3.0,
                                  alt_route={"adapter": "openrouter", "id": "vendor/elorian-alt"})
    return ModelsConfig.model_validate({"version": "test", "currency": "USD", "unit": "usd_per_million_tokens",
                                        "models": models})


MODELS = _models()


def _refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a test tried the network: {request.method} {request.url}")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real data root, no keys from the environment or .env, no network, no leftover fake script."""
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.setattr(router_http, "_DEFAULT_CLIENT", HttpClient(transport=httpx.MockTransport(_refuse_network)))
    clear_fake_script()
    yield
    clear_fake_script()


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    led = Ledger(tmp_path / "ledger" / "spend_ledger.jsonl", tmp_path / "ledger" / "spend_caps.json")
    led.set_active_session("T", 1.0)
    return led


@pytest.fixture
def cache(tmp_path) -> DiskCache:
    return DiskCache(tmp_path / "cache")


@pytest.fixture
def ask(ledger, cache):
    def _ask(model_key, messages="hello world", **kw):
        kw.setdefault("models", MODELS)
        kw.setdefault("ledger", ledger)
        kw.setdefault("cache", cache)
        return chat(model_key, messages, **kw)

    return _ask


class Recorder:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload):
        self.events.append((kind, payload))


class Server:
    """A canned HTTP server: each item is (status, body[, headers]), "timeout", or a callable(request)."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if item == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if item == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if callable(item):
            return item(request)
        status, body, *rest = item
        headers = rest[0] if rest else {}
        if isinstance(body, (dict, list)):
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(status, content=str(body).encode("utf-8"), headers=headers)

    def client(self, sleeps: list | None = None) -> HttpClient:
        return HttpClient(transport=httpx.MockTransport(self),
                          sleep=sleeps.append if sleeps is not None else (lambda s: None))

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def or_response(content='{"green": 5}', *, cost=0.00123, prompt=600, completion=3, reasoning=0, cached=0,
                gen_id="gen-test-1", model="vendor/test-model", finish="stop", **extra):
    # the default tokens cost exactly the default $0.00123 at the orm prices (2.00 / 10.00), so the router's
    # check of a provider cost against its token counts keeps it
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
             "completion_tokens_details": {"reasoning_tokens": reasoning},
             "prompt_tokens_details": {"cached_tokens": cached}}
    if cost is not None:
        usage["cost"] = cost
    body = {"id": gen_id, "model": model, "provider": "TestUpstream", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": usage}
    body.update(extra)
    return body


def make_request(model_key, messages="Count the green LEDs.", *, images=None, json_schema=None, reasoning="high",
                 max_tokens=300, temperature=None, extra=None) -> AdapterRequest:
    entry = MODELS.get(model_key)
    adapter = get_adapter(entry.adapter)
    msgs = normalize_messages(messages)
    imgs = normalize_images(images)
    target = max(i for i, m in enumerate(msgs) if m["role"] == "user") if imgs else None
    return AdapterRequest(model_key=model_key, model_id=entry.id, entry=entry, role="reader", messages=msgs,
                          images=imgs, image_message_index=target, json_schema=json_schema, schema_name="answer",
                          max_tokens=max_tokens, reasoning=adapter.normalize_reasoning(reasoning, entry),
                          reasoning_requested=reasoning, temperature=temperature, extra=extra or {})


def led_array() -> np.ndarray:
    arr = np.full((24, 48, 3), 40, dtype=np.uint8)
    arr[8:16, 4:12] = (0, 255, 0)
    arr[8:16, 20:28] = (255, 153, 0)
    return arr


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- FakeModel and cache
def test_fake_round_trip_tokens_and_zero_cost(ask, ledger):
    r = ask("fake", "hello world", role="reader")
    assert r.text == "fake response"
    assert r.input_tokens == math.ceil(len("hello world") / 4)
    assert r.output_tokens == math.ceil(len("fake response") / 4)
    assert r.usd == 0.0 and r.usd_source == "computed" and r.projected_usd == 0.0
    assert r.provider == "fake" and r.model == "fake" and r.model_key == "fake" and r.role == "reader"
    assert r.cache_hit is False and len(r.request_hash) == 64
    assert r.reasoning_setting == {"effort": "high"}
    assert r.json is None and r.json_error is None and r.image_count == 0
    with_image = ask("fake", "hello world", images=[led_array()])
    assert with_image.input_tokens == math.ceil(len("hello world") / 4) + 765
    assert with_image.image_count == 1 and with_image.est_input_tokens == math.ceil(11 / 3.5) + 1600
    kinds = [rec["type"] for rec in ledger.records()]
    assert kinds == ["reserve", "settle", "reserve", "settle"]  # the fake still goes through the ledger
    assert ledger.spent() == 0.0


def test_repeat_is_a_cache_hit_and_salt_is_a_miss_all_traced(ask, ledger):
    rec = Recorder()
    r1 = ask("fake", "same request", trace=rec, phase="setup")
    r2 = ask("fake", "same request", trace=rec, phase="setup")
    assert r2.cache_hit and r2.usd == 0.0 and r2.usd_source == "computed"
    assert r2.text == r1.text and r2.request_hash == r1.request_hash
    assert r2.raw["original_usd"] == 0.0
    r3 = ask("fake", "same request", trace=rec, cache_salt="repeat-1")
    assert not r3.cache_hit and r3.request_hash != r1.request_hash
    assert [rec_["type"] for rec_ in ledger.records()].count("reserve") == 2  # the hit wrote nothing
    assert [k for k, _ in rec.events] == ["model_call"] * 3
    payloads = [p for _, p in rec.events]
    assert [p["cache_hit"] for p in payloads] == [False, True, False]
    for p in payloads:
        ModelCallPayload.model_validate(p)
        for field in ("model_key", "adapter", "ok", "error", "finish_reason", "reasoning_setting", "image_count",
                      "est_input_tokens", "projected_usd", "phase", "json_mode", "http_status", "attempts"):
            assert field in p, field
    assert payloads[0]["phase"] == "setup" and payloads[0]["ok"] is True and payloads[1]["usd"] == 0.0
    assert payloads[1]["attempts"] == 0


def test_use_cache_false_neither_reads_nor_writes(ask, cache):
    r1 = ask("fake", "no cache please", use_cache=False)
    assert cache.get(r1.request_hash) is None
    ask("fake", "no cache please")
    r3 = ask("fake", "no cache please", use_cache=False)
    assert not r3.cache_hit


def test_paid_cache_hit_keeps_original_cost_in_raw(ask, ledger):
    r1 = ask("paid_fake", "price me", max_tokens=100)
    assert r1.usd > 0 and r1.usd_source == "computed"
    assert r1.usd == pytest.approx((r1.input_tokens * 10.0 + r1.output_tokens * 50.0) / 1e6)
    assert r1.usd < r1.projected_usd
    r2 = ask("paid_fake", "price me", max_tokens=100)
    assert r2.cache_hit and r2.usd == 0.0 and r2.raw["original_usd"] == pytest.approx(r1.usd)
    assert ledger.spent() == pytest.approx(r1.usd)


# --------------------------------------------------------------------------- ledger and caps
def test_session_cap_refuses_before_any_reservation(ask, ledger):
    ledger.set_active_session("T", 0.001)
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "x" * 100, max_tokens=1000)
    e = err.value
    assert e.scope == "session" and e.cap_usd == 0.001 and e.spent_usd == 0.0
    assert e.projected_usd == pytest.approx(math.ceil(100 / 3.5) * 10.0 / 1e6 + 1000 * 50.0 / 1e6)
    assert ledger.records() == []


def test_model_cap_refuses(ask, ledger):
    ledger.set_active_session("T", 1.0, model_caps={"paid_fake": 0.01})
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "hello", max_tokens=1000)
    assert err.value.scope == "model" and err.value.cap_usd == 0.01
    ask("fake", "hello")  # another model in the same session is not affected
    assert ledger.caps()["model_caps"] == {"T": {"paid_fake": 0.01}}


def test_total_cap_counts_every_session(ask, ledger):
    ledger.set_active_session("A", 1.0, total_cap_usd=0.06)
    with fake_script(["x" * 4000]):  # 1000 output tokens at $50 per million: about $0.05 spent in session A
        first = ask("paid_fake", "first", max_tokens=1000)
    assert first.usd == pytest.approx((2 * 10.0 + 1000 * 50.0) / 1e6)
    caps = ledger.set_active_session("B", 1.0)  # total cap kept, session A's cap kept
    assert caps["total_cap_usd"] == 0.06 and caps["session_caps"]["A"] == 1.0
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "second", max_tokens=400)  # projects about $0.02: fine alone, over the total with A's spend
    assert err.value.scope == "total" and err.value.spent_usd == pytest.approx(first.usd)
    assert err.value.projected_usd < 0.06
    summary = ledger.summary()
    assert summary["by_session"] == {"A": pytest.approx(first.usd)}
    assert summary["by_model"]["paid_fake"] == pytest.approx(first.usd)


def test_no_active_session_refuses_paid_but_allows_fake(ask, tmp_path):
    bare = Ledger(tmp_path / "bare" / "spend_ledger.jsonl")
    assert bare.caps_path == tmp_path / "bare" / "spend_caps.json"
    with pytest.raises(SpendCapExceeded) as err:
        ask("paid_fake", "hello", ledger=bare)
    assert err.value.scope == "no_session" and "set_active_session" in str(err.value)
    r = ask("fake", "hello", ledger=bare)
    assert r.usd == 0.0
    assert [rec["session"] for rec in bare.records() if rec["type"] == "reserve"] == [None]


def test_unsettled_reservation_counts_as_spent(ask, ledger):
    rec = Recorder()
    with fake_script([RuntimeError("upstream vanished")]):
        with pytest.raises(RuntimeError, match="upstream vanished"):
            ask("paid_fake", "will fail", max_tokens=200, trace=rec)
    reserve = [r for r in ledger.records() if r["type"] == "reserve"]
    assert len(reserve) == 1 and not [r for r in ledger.records() if r["type"] == "settle"]
    assert ledger.spent() == pytest.approx(reserve[0]["usd"]) and reserve[0]["usd"] > 0
    assert ledger.summary()["unsettled"]["count"] == 1
    (kind, payload), = rec.events
    assert kind == "model_call" and payload["ok"] is False and "upstream vanished" in payload["error"]
    assert payload["usd"] == pytest.approx(reserve[0]["usd"])
    ModelCallPayload.model_validate(payload)


def test_rejected_before_generation_settles_at_zero(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((400, {"error": {"code": 400, "message": "vendor/test-model is not a valid model ID"}}))
    rec = Recorder()
    with pytest.raises(ModelCallError) as err:
        ask("orm", "hi", http=server.client(), trace=rec)
    assert err.value.status == 400 and err.value.rejected_before_generation and err.value.attempts == 1
    settles = [r for r in ledger.records() if r["type"] == "settle"]
    assert len(settles) == 1 and settles[0]["usd"] == 0.0
    assert ledger.spent() == 0.0
    assert rec.events[0][1]["usd"] == 0.0 and rec.events[0][1]["http_status"] == 400


def test_5xx_after_retries_stays_reserved(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    sleeps: list[float] = []
    server = Server((502, {"error": {"message": "bad gateway"}}))
    with pytest.raises(ModelCallError) as err:
        ask("orm", "hi", http=server.client(sleeps))
    assert err.value.status == 502 and err.value.attempts == 3 and not err.value.rejected_before_generation
    assert len(sleeps) == 2 and len(server.requests) == 3
    reserve = [r for r in ledger.records() if r["type"] == "reserve"]
    # every attempt was reserved before it was sent, and each lost one stays counted at the projection
    assert len(reserve) == 3 and reserve[0]["usd"] > 0 and not [r for r in ledger.records() if r["type"] == "settle"]
    assert ledger.spent() == pytest.approx(3 * reserve[0]["usd"])


def test_set_active_session_keeps_other_sessions(tmp_path):
    caps_path = tmp_path / "caps" / "spend_caps.json"
    set_active_session("S1", 1.0, total_cap_usd=21.0, path=caps_path)
    set_active_session("S4", 5.0, model_caps={"writer_a": 2.0}, path=caps_path)
    caps = json.loads(caps_path.read_text(encoding="utf-8"))
    assert caps["active_session"] == "S4"
    assert caps["session_caps"] == {"S1": 1.0, "S4": 5.0}
    assert caps["model_caps"] == {"S4": {"writer_a": 2.0}} and caps["total_cap_usd"] == 21.0
    with pytest.raises(ValueError):
        set_active_session("S5", -1.0, path=caps_path)


def test_ledger_tolerates_a_truncated_last_line(ledger):
    rid = ledger.reserve(0.25, "orm", "a" * 64)
    with open(ledger.path, "ab") as f:
        f.write(b'{"type": "reserve", "id": "dead", "usd": 9')  # a crash mid-write
    assert ledger.spent() == pytest.approx(0.25)
    ledger.settle(rid, 0.10, "provider")
    assert ledger.spent() == pytest.approx(0.10)
    assert [r["type"] for r in ledger.records()] == ["reserve", "settle"]
    summary = ledger.summary()
    assert summary["total_usd"] == pytest.approx(0.10) and summary["unsettled"]["count"] == 0
    assert summary["by_session_model"] == {"T": {"orm": pytest.approx(0.10)}}


def test_ledger_lock_is_an_os_lock_on_a_persistent_file(ledger):
    # R5: a lock file left on disk (by any earlier process, however old) blocks nobody; only a held OS lock does
    ledger.lock_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.lock_path.write_text("12345 0\n", encoding="ascii")
    old = time.time() - 3600
    os.utime(ledger.lock_path, (old, old))
    ledger.reserve(0.0, "fake", "b" * 64)
    assert ledger.lock_path.exists()  # never deleted, so no holder can lose its lock to someone's release()
    holder = FileLock(ledger.lock_path)
    holder.acquire()
    try:
        old = time.time() - 3600  # however old the holder's lock looks, a waiter never breaks it
        os.utime(ledger.lock_path, (old, old))
        impatient = Ledger(ledger.path, ledger.caps_path, lock_timeout_s=0.2)
        with pytest.raises(RouterError, match="lock"):
            impatient.reserve(0.0, "fake", "c" * 64)
        assert ledger.lock_path.exists() and holder.held
    finally:
        holder.release()
    impatient.reserve(0.0, "fake", "c" * 64)
    assert [r["request_hash"][0] for r in ledger.records()] == ["b", "c"]


def test_ledger_lock_is_released_when_its_holder_dies(ledger):
    code = ("import os, sys\nfrom loopkit.router.ledger import FileLock\nFileLock(sys.argv[1]).acquire()\n"
            "print('held', flush=True)\nos._exit(3)\n")  # dies holding the lock: no release, no cleanup
    proc = subprocess.run([sys.executable, "-c", code, str(ledger.lock_path)], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 3 and proc.stdout.strip() == "held", proc.stderr[-500:]
    t0 = time.monotonic()
    Ledger(ledger.path, ledger.caps_path, lock_timeout_s=5.0).reserve(0.0, "fake", "d" * 64)
    assert time.monotonic() - t0 < 5.0 and ledger.lock_path.exists()


def test_ledger_lock_excludes_threads_of_one_process(tmp_path):
    path = tmp_path / "locks" / "x.lock"
    state = {"inside": 0, "overlaps": 0, "entries": 0}

    def work():
        for _ in range(25):
            with FileLock(path, timeout_s=30):
                state["inside"] += 1
                state["entries"] += 1
                if state["inside"] > 1:
                    state["overlaps"] += 1
                time.sleep(0.001)
                state["inside"] -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert state == {"inside": 0, "overlaps": 0, "entries": 100}
    lock = FileLock(path)
    lock.acquire()
    with pytest.raises(RouterError, match="already held"):
        lock.acquire()
    lock.release()
    lock.release()  # a second release is harmless


# --------------------------------------------------------------------------- configuration errors
def test_missing_key_names_openrouter_variable_before_any_spend(ask, ledger):
    with pytest.raises(MissingCredentialError) as err:
        ask("orm", "hi")
    assert "OPENROUTER_API_KEY" in str(err.value)
    assert ledger.records() == []
    with pytest.raises(MissingCredentialError) as err:
        ask("jev", "state", extra={"questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert "TYPESAFE_JEV_API_KEY" in str(err.value)


def test_model_without_id_is_not_configured(ask, ledger):
    with pytest.raises(ModelNotConfigured) as err:
        ask("unset", "hi")
    assert "unset" in str(err.value) and "unverified" in str(err.value)
    with pytest.raises(ModelNotConfigured, match="base_url"):
        ask("el_nourl", "hi")
    assert ledger.records() == []


def test_bad_arguments_fail_before_spend(ask, ledger, monkeypatch):
    with pytest.raises(ValueError):
        ask("fake", "hi", role="critic")
    with pytest.raises(ValueError):
        ask("fake", [{"role": "robot", "content": "hi"}])
    with pytest.raises(ValueError):
        ask("fake", "hi", reasoning="maximum")
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    with pytest.raises(ValueError):  # Jev ignores reasoning, but a bad value still fails
        ask("jev", "state", reasoning="maximum", extra={"questions": {"q": {"type": "noul"}}})
    with pytest.raises(RouterError, match="text-only"):
        ask("jev", "state", images=[led_array()], extra={"questions": {"q": {"type": "noul"}}})
    bad_url = _models()
    bad_url.models["el"].base_url = "elorian.invalid/v1"
    with pytest.raises(ModelNotConfigured, match="http"):
        ask("el", "hi", models=bad_url)
    assert ledger.records() == []


def test_errors_keep_their_fields_through_pickling():
    import pickle

    cap = pickle.loads(pickle.dumps(SpendCapExceeded("over", scope="model", cap_usd=2.0, spent_usd=1.5,
                                                     projected_usd=0.7)))
    assert (cap.scope, cap.cap_usd, cap.spent_usd, cap.projected_usd, str(cap)) == ("model", 2.0, 1.5, 0.7, "over")
    call = pickle.loads(pickle.dumps(ModelCallError("failed", status=402, attempts=1,
                                                    rejected_before_generation=True)))
    assert (call.status, call.attempts, call.rejected_before_generation) == (402, 1, True)
    assert isinstance(call, RouterError) and isinstance(call, RuntimeError)


# --------------------------------------------------------------------------- OpenRouter adapter
def test_openrouter_reasoning_mapping():
    plain, hi = MODELS.get("orm"), MODELS.get("orm_hi")
    assert normalize_reasoning("disabled", plain) == {"enabled": False}
    assert normalize_reasoning(None, plain) == {"enabled": False}
    assert normalize_reasoning(False, plain) == {"enabled": False}
    assert normalize_reasoning("low", plain) == {"effort": "low"}
    assert normalize_reasoning("high", plain) == {"effort": "high"}
    assert normalize_reasoning("highest", plain) == {"effort": "high"}
    assert normalize_reasoning("highest", hi) == {"effort": "xhigh"}
    assert normalize_reasoning({"max_tokens": 2000}, plain) == {"max_tokens": 2000}
    with pytest.raises(ValueError):
        normalize_reasoning("extreme", plain)


def test_openrouter_payload_shape():
    arr = led_array()
    req = make_request("orm", [{"role": "system", "content": "You read panels."},
                               {"role": "user", "content": "Count the green LEDs."}],
                       images=[arr], json_schema=GREEN_SCHEMA, reasoning="disabled")
    body = build_payload(req, "json_schema")
    assert body["model"] == "vendor/test-model" and body["max_tokens"] == 300
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"enabled": False}
    assert "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": "You read panels."}
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Count the green LEDs."}
    assert parts[1]["type"] == "image_url"
    url = parts[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG") and np.array_equal(np.asarray(Image.open(io.BytesIO(decoded))), arr)
    assert body["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": False, "schema": GREEN_SCHEMA}}
    assert body["provider"] == {"require_parameters": True}

    obj = build_payload(req, "json_object_schema_in_prompt")
    assert obj["response_format"] == {"type": "json_object"}
    assert obj["messages"][1]["content"][-1]["text"].startswith("Return only JSON that matches this JSON Schema")
    plain = build_payload(req, "plain_schema_in_prompt")
    assert "response_format" not in plain and "provider" not in plain
    assert "JSON Schema" in plain["messages"][1]["content"][-1]["text"]
    no_schema = build_payload(make_request("orm", "hi", temperature=0.2, reasoning="highest"), None)
    assert no_schema["temperature"] == 0.2 and no_schema["reasoning"] == {"effort": "high"}
    assert "response_format" not in no_schema and no_schema["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert OpenRouterAdapter().headers() == {"X-Title": "loopkit"}


def test_openrouter_send_with_injected_transport_reports_provider_cost(ask, ledger, monkeypatch):
    server = Server((200, or_response(reasoning=3, cached=200)))
    req = make_request("orm", images=[led_array()], json_schema=GREEN_SCHEMA, reasoning="disabled")
    result = OpenRouterAdapter().send(req, PLAIN_KEY, server.client())
    assert result.usd == 0.00123 and result.text == '{"green": 5}' and result.json_mode == "json_schema"
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens, result.cached_tokens) == (600, 3, 3, 200)
    assert result.generation_id == "gen-test-1" and result.upstream_provider == "TestUpstream"
    sent = server.requests[0]
    assert str(sent.url) == OPENROUTER_URL
    assert sent.headers["authorization"] == "Bearer " + PLAIN_KEY and sent.headers["x-title"] == "loopkit"
    assert PLAIN_KEY.encode() not in sent.content

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server2 = Server((200, or_response()))
    r = ask("orm", "Count the green LEDs.", images=[led_array()], json_schema=GREEN_SCHEMA, reasoning="disabled",
            role="reader", http=server2.client())
    assert r.usd == 0.00123 and r.usd_source == "provider" and r.json == {"green": 5} and r.json_error is None
    assert r.reasoning_setting == {"enabled": False} and r.json_mode == "json_schema" and r.model == "vendor/test-model"
    settle = [x for x in ledger.records() if x["type"] == "settle"][0]
    assert settle["usd"] == 0.00123 and settle["usd_source"] == "provider"


def test_openrouter_json_mode_falls_back(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    refuse = (400, {"error": {"message": "This model does not support response_format json_schema"}})
    server = Server(refuse, (200, or_response(content='{"green": 4}')))
    r = ask("orm", "count", json_schema=GREEN_SCHEMA, http=server.client())
    assert r.json_mode == "json_object_schema_in_prompt" and r.json == {"green": 4}
    first, second = server.bodies()
    assert first["response_format"]["type"] == "json_schema"
    assert second["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in second["messages"][-1]["content"][-1]["text"]
    assert len(r.raw["modes_tried"]) == 1
    server3 = Server(refuse, (404, {"error": {"message": "No endpoints found that can handle the requested "
                                                         "parameters"}}), (200, or_response()))
    r3 = ask("orm", "count again", json_schema=GREEN_SCHEMA, http=server3.client())
    assert r3.json_mode == "plain_schema_in_prompt" and "response_format" not in server3.bodies()[2]
    # R1: every POST in every mode was reserved before it was sent; the refused modes settled at $0
    assert len([x for x in ledger.records() if x["type"] == "reserve"]) == 5
    assert [x["usd"] for x in ledger.records() if x["type"] == "settle"] == [0.0, 0.00123, 0.0, 0.0, 0.00123]
    assert ledger.spent() == pytest.approx(2 * 0.00123)


def test_unpriced_model_projects_at_astra_prices(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(cost=None, prompt=100, completion=10)))
    r = ask("unpriced", "x" * 35, max_tokens=1000, http=server.client())
    assert r.est_input_tokens == 10
    assert r.projected_usd == pytest.approx(10 * 10.0 / 1e6 + 1000 * 50.0 / 1e6)
    assert r.usd_source == "computed" and r.usd == pytest.approx((100 * 10.0 + 10 * 50.0) / 1e6)
    assert r.raw["price_fallback"] == "astra"


def test_200_without_usage_stays_reserved_at_projection(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = or_response()
    del body["usage"]
    r = ask("orm", "hi", http=Server((200, body)).client())
    assert r.usd == pytest.approx(r.projected_usd) and r.usd_source == "computed" and "usd_note" in r.raw
    assert ledger.spent() == pytest.approx(r.projected_usd)
    assert ledger.summary()["unsettled"] == {"count": 1, "usd": pytest.approx(r.projected_usd)}


# --------------------------------------------------------------------------- HTTP retries
def test_retry_on_429_then_success_without_sleeping():
    sleeps: list[float] = []
    server = Server((429, {"error": {"message": "rate limited"}}, {"Retry-After": "2"}), (200, {"ok": True}))
    res = server.client(sleeps).post_json("https://api.invalid/x", {"a": 1}, key=PLAIN_KEY, timeout_s=5)
    assert res.ok and res.status == 200 and res.data == {"ok": True} and res.attempts == 2
    assert sleeps == [2.0]
    assert [a["status"] for a in res.attempt_log] == [429, 200]


def test_retry_policy_backoff_caps_and_no_retry_on_400():
    sleeps: list[float] = []
    server = Server((503, "busy"), (520, "edge"), (200, {"ok": 1}))
    res = server.client(sleeps).post_json("https://api.invalid/x", {}, timeout_s=5)
    assert res.ok and res.attempts == 3 and len(sleeps) == 2
    assert 1.5 * 0.7 <= sleeps[0] <= 1.5 * 1.3 and 3.0 * 0.7 <= sleeps[1] <= 3.0 * 1.3
    sleeps.clear()
    res = Server((429, "", {"Retry-After": "120"}), (200, {})).client(sleeps).post_json("https://a.invalid", {})
    assert sleeps == [20.0]
    sleeps.clear()
    res = Server((400, {"error": {"message": "bad"}})).client(sleeps).post_json("https://a.invalid", {})
    assert res.status == 400 and res.attempts == 1 and sleeps == []
    res = Server("timeout", (200, {"ok": 2})).client(sleeps).post_json("https://a.invalid", {})
    assert res.ok and res.attempts == 2 and len(sleeps) == 1
    res = Server("connect").client().post_json("https://a.invalid", {})
    assert res.status is None and res.attempts == 3 and "connection" in res.error


def test_per_call_deadline_covers_a_trickling_body():
    def trickle(request):
        def chunks():
            for _ in range(5):
                time.sleep(0.03)
                yield b" "
            yield b"{}"
        return httpx.Response(200, content=chunks())

    res = Server(trickle).client().post_json("https://a.invalid", {}, timeout_s=0.05)
    assert res.status is None and "timeout" in res.error and res.attempts == 3


# --------------------------------------------------------------------------- secrets
def test_key_never_in_errors_traces_cache_or_ledger(ask, ledger, cache, monkeypatch):
    for key in (PLAIN_KEY, PATTERN_KEY):
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
        echo = {"error": {"message": f"key {key} refused, request was data:image/png;base64,{'A' * 300}"}}
        rec = Recorder()
        with pytest.raises(ModelCallError) as err:
            ask("orm", "hi", images=[led_array()], http=Server((401, echo)).client(), trace=rec)
        dumped = json.dumps(err.value.raw)
        for text in (str(err.value), repr(err.value), dumped, json.dumps(rec.events)):
            assert key not in text
            assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in text
        assert err.value.rejected_before_generation

        leaky = or_response(echo_header=f"Bearer {key}", note=f"served for {key}")
        r = ask("orm", f"hello {len(key)}", http=Server((200, leaky)).client(), trace=rec)
        record_text = cache.path_for(r.request_hash).read_text(encoding="utf-8")
        assert key not in record_text and key not in json.dumps(r.raw) and key not in json.dumps(rec.events)
        assert key not in ledger.path.read_text(encoding="utf-8")
        assert r.text == '{"green": 5}'


def test_cache_hit_needs_no_key(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    r1 = ask("orm", "cache me", http=Server((200, or_response())).client())
    monkeypatch.delenv("OPENROUTER_API_KEY")
    r2 = ask("orm", "cache me")
    assert r2.cache_hit and r2.text == r1.text and r2.usd == 0.0 and r2.raw["original_usd"] == 0.00123


# --------------------------------------------------------------------------- JSON answers
def test_json_parsing_strips_fences_and_prose():
    assert parse_json_text('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert parse_json_text("```\n[1, 2]\n```") == ([1, 2], None)
    assert parse_json_text('Here is the answer:\n{"a": {"b": [1]}}\nHope that helps.') == ({"a": {"b": [1]}}, None)
    assert parse_json_text('```json\n{"a": 1}') == ({"a": 1}, None)  # unclosed fence (truncated)
    assert parse_json_text("I see 5 [green] LEDs: {\"green\": 5}") == ({"green": 5}, None)
    value, err = parse_json_text("no json here")
    assert value is None and err.startswith("not JSON")
    assert parse_json_text("") == (None, "empty answer")


def test_json_schema_violation_reported_not_repaired(ask):
    with fake_script(['```json\n{"green": "five"}\n```', 'Sure! {"green": 5}', "nothing useful"]):
        bad = ask("fake", "one", json_schema=GREEN_SCHEMA)
        good = ask("fake", "two", json_schema=GREEN_SCHEMA)
        junk = ask("fake", "three", json_schema=GREEN_SCHEMA)
    assert bad.json == {"green": "five"} and "green" in bad.json_error and "integer" in bad.json_error
    assert good.json == {"green": 5} and good.json_error is None and good.json_ok
    assert junk.json is None and junk.json_error.startswith("not JSON")


# --------------------------------------------------------------------------- Jev, System One and Elorian
def test_jev_payload_and_cost_from_input_tokens(ask, ledger, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    questions = {"green": {"type": "choice", "instructions": "How many ports have a green LED?",
                           "criteria": {str(n): f"{n} green" for n in range(13)}}}
    answers = {"green": {"type": "choice", "choice": "5", "probabilities": {"5": 0.9, "4": 0.1}, "confidence": 0.8}}
    server = Server((200, {"id": "sys1-req-1", "model": "jev-latest", "answers": answers,
                           "usage": {"input_tokens": 1000, "output_tokens": 3}}))
    r = ask("jev", [{"role": "system", "content": "ignored"}, {"role": "user", "content": "panel: 12 ports"}],
            role="system1", extra={"questions": questions}, http=server.client())
    assert str(server.requests[0].url) == JEV_URL
    assert server.bodies()[0] == {"model": "jev-latest", "state": "panel: 12 ports", "questions": questions}
    assert server.requests[0].headers["authorization"] == "Bearer " + PLAIN_KEY
    assert r.usd == pytest.approx(1000 * 0.042 / 1e6) and r.usd_source == "computed"
    assert r.text == json.dumps(answers) and r.json == answers and r.json_error is None
    assert r.reasoning_setting is None and r.provider == "jev" and r.json_mode == "typed_questions"


def test_jev_needs_typed_questions_before_spend(ask, ledger, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    with pytest.raises(RouterError, match="typed questions"):
        ask("jev", "panel state")
    assert ledger.records() == []


def test_jev_fallback_variable_and_system_one_images(ask, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", PLAIN_KEY)  # the fallback name
    ok = {"answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 10, "output_tokens": 1}}
    r = ask("jev", "state", extra={"questions": {"q": {"type": "noul", "instructions": "x"}}},
            http=Server((200, ok)).client())
    assert r.json == ok["answers"]
    monkeypatch.setenv("LK_TEST_SYS1_KEY", PLAIN_KEY)
    server = Server((200, ok))
    ask("sys1_vision", "panel in the image", images=[led_array()],
        extra={"questions": {"q": {"type": "noul", "instructions": "x"}}}, http=server.client())
    body = server.bodies()[0]
    assert str(server.requests[0].url) == "https://decisions.invalid/systemone"
    assert body["model"] == "sys1-vision" and body["images"][0].startswith("data:image/png;base64,")


def test_elorian_payload_and_computed_cost(ask, ledger, monkeypatch):
    entry = MODELS.get("el")
    assert normalize_elorian_reasoning("disabled", entry) == "low"
    assert normalize_elorian_reasoning("medium", entry) == "medium"
    assert normalize_elorian_reasoning("high", entry) == "xhigh"
    assert normalize_elorian_reasoning("highest", entry) == "xhigh"
    assert normalize_elorian_reasoning({"reasoning_effort": "low"}, entry) == "low"
    body = build_elorian_payload(make_request("el", images=[led_array()], json_schema=GREEN_SCHEMA), "json_schema")
    assert "usage" not in body and "provider" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 300 and body["reasoning_effort"] == "xhigh"
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["response_format"]["json_schema"]["strict"] is False

    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(cost=None, prompt=2000, completion=100, model="elorian-test")))
    r = ask("el", "count", json_schema=GREEN_SCHEMA, reasoning="disabled", http=server.client())
    assert str(server.requests[0].url) == "https://elorian.invalid/v1/chat/completions"
    assert r.usd_source == "computed" and r.usd == pytest.approx((2000 * 0.6 + 100 * 3.0) / 1e6)
    assert r.reasoning_setting == "low" and server.bodies()[0]["reasoning_effort"] == "low"


@pytest.mark.skipif(not HAS_ALT_ROUTE, reason="config has no alt_route field")
def test_alt_route_used_only_when_the_primary_key_is_missing(ask, monkeypatch):
    rec = Recorder()
    with pytest.raises(MissingCredentialError) as err:
        ask("el_alt", "hi")
    assert "ELORIAN_API_KEY" in str(err.value) and "OPENROUTER_API_KEY" in str(err.value)
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(model="vendor/elorian-alt")))
    r = ask("el_alt", "hi", http=server.client(), trace=rec)
    assert str(server.requests[0].url) == OPENROUTER_URL and server.bodies()[0]["model"] == "vendor/elorian-alt"
    assert r.provider == "openrouter" and r.usd_source == "provider" and "route_note" in r.raw
    assert rec.events[0][1]["route"] == "alt_route" and rec.events[0][1]["model_key"] == "el_alt"
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    server2 = Server((200, or_response(cost=None, model="elorian-test")))
    r2 = ask("el_alt", "hi", http=server2.client())
    assert str(server2.requests[0].url).startswith("https://elorian.invalid/") and r2.provider == "elorian"


# --------------------------------------------------------------------------- images, cache files, schemas, scripts
def test_images_normalize_and_hash(tmp_path):
    arr = led_array()
    a = normalize_image(arr)
    assert a.mime == "image/png" and a.data.startswith(b"\x89PNG") and normalize_image(arr.copy()).sha256 == a.sha256
    assert normalize_image(Image.fromarray(arr)).sha256 == a.sha256
    raw_png = png_bytes(arr)
    assert normalize_image(raw_png).data == raw_png
    jpeg = io.BytesIO()
    Image.fromarray(arr).save(jpeg, format="JPEG")
    j = normalize_image(jpeg.getvalue())
    assert j.mime == "image/jpeg" and j.data == jpeg.getvalue()
    path = tmp_path / "panel.png"
    path.write_bytes(raw_png)
    assert normalize_image(path).sha256 == normalize_image(str(path)).sha256
    with pytest.raises(ValueError):
        normalize_image(arr.astype(np.float32))
    with pytest.raises(ValueError):
        normalize_image(tmp_path / "missing.png")
    assert "bytes=" in repr(a) and "base64" not in repr(a)


def test_image_bytes_change_the_request_hash(ask):
    other = led_array()
    other[0, 0] = (1, 2, 3)
    r1 = ask("fake", "look", images=[led_array()])
    r2 = ask("fake", "look", images=[other])
    r3 = ask("fake", "look", images=[led_array()])
    assert r1.request_hash != r2.request_hash and r3.cache_hit


def test_disk_cache_layout_corruption_and_atomic_write(tmp_path):
    cache = DiskCache(tmp_path / "c")
    h = "ab" + "0" * 62
    path = cache.put(h, {"text": "hello", "usd": 0.5})
    assert path == tmp_path / "c" / "ab" / f"{h}.json"
    assert cache.get(h)["text"] == "hello" and h in cache
    assert not list((tmp_path / "c" / "ab").glob("*.tmp"))
    path.write_text("{ not json", encoding="utf-8")
    assert cache.get(h) is None
    path.write_text('["a list"]', encoding="utf-8")
    assert cache.get(h) is None
    assert cache.get("cd" + "1" * 62) is None
    with pytest.raises(ValueError):
        cache.get("../../escape")
    with pytest.warns(RuntimeWarning):
        assert cache.put("ef" + "2" * 62, {"text": "x", "raw": {"score": float("nan")}}) is None


def test_default_cache_and_ledger_live_under_the_data_root(tmp_path):
    r = chat("fake", "where do I go", models=MODELS)
    data = tmp_path / "data"
    assert (data / "cache" / "router" / r.request_hash[:2] / f"{r.request_hash}.json").is_file()
    assert (data / "spend_ledger.jsonl").is_file()


def test_from_schema_gives_valid_instances():
    graph_schema = json.loads((Path(config.REPO_ROOT) / "loopkit" / "schemas" / "json" / "ProgramGraph.json")
                              .read_text(encoding="utf-8"))
    for schema in (
        graph_schema,
        GREEN_SCHEMA,
        {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 9},
                                          "x": {"type": "number", "exclusiveMinimum": 1.0},
                                          "s": {"type": "string", "minLength": 6},
                                          "led": {"enum": ["green", "amber"]},
                                          "opt": {"anyOf": [{"type": "null"}, {"type": "boolean"}]},
                                          "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2}}},
        {"$defs": {"P": {"type": "object", "properties": {"k": {"const": 7}}, "required": ["k"]}},
         "type": "array", "items": {"$ref": "#/$defs/P"}},
    ):
        value = from_schema(schema)
        assert check(value, schema) == [], (schema, value)
    assert from_schema(GREEN_SCHEMA) == {"green": 0}


def test_fake_script_list_callable_and_exhaustion(ask):
    seen = []

    def script(req):
        seen.append(req)
        return {"echo": req.text, "images": req.image_count}

    set_fake_script(script)
    r = ask("fake", "first", images=[led_array()], json_schema={"type": "object"}, extra={"k": 1}, max_tokens=99)
    assert json.loads(r.text) == {"echo": "first", "images": 1}
    req = seen[0]
    assert req.call_index == 0 and req.model_key == "fake" and req.max_tokens == 99 and req.extra == {"k": 1}
    assert len(req.images) == 1 and len(req.images[0]) == 64 and req.json_schema == {"type": "object"}
    set_fake_script(["one", {"two": 2}])
    assert ask("fake", "a").text == "one"
    assert ask("fake", "b").text == '{"two": 2}'
    assert fake_calls() == 2
    with pytest.raises(RouterError, match="exhausted"):
        ask("fake", "c")
    clear_fake_script()
    assert ask("fake", "d").text == "fake response"


def test_fake_script_scopes_the_cache(ask):
    with fake_script(["from script A"]):
        assert ask("fake", "identical").text == "from script A"
        assert ask("fake", "identical").cache_hit  # same script: a hit
    with fake_script(["from script B"]):
        r = ask("fake", "identical")
        assert r.text == "from script B" and not r.cache_hit  # another script never sees A's reply
    with fake_script(["outer"]):
        with fake_script(["inner"]):
            assert ask("fake", "nested").text == "inner"
        assert ask("fake", "nested 2").text == "outer"


# --------------------------------------------------------------------------- live smoke, offline
def test_smoke_runs_offline_and_sets_the_session(tmp_path, monkeypatch):
    from loopkit.router import smoke

    models = _models()
    del models.models["paid_fake"]  # keep the first run free, so it must not open a spend session
    bare = Ledger(tmp_path / "smoke" / "spend_ledger.jsonl")
    out = smoke.run_smoke(include_fake=True, models=models, ledger=bare, out_dir=tmp_path / "s1", echo=False)
    assert [r["model_key"] for r in out["rows"]] == ["fake"]
    assert out["rows"][0]["ok"] and bare.caps()["active_session"] is None
    reasons = {s["model_key"]: s["reason"] for s in out["skipped"]}
    assert "OPENROUTER_API_KEY" in reasons["orm"] and "verified: false" in reasons["unset"]
    assert "base_url" in reasons["el_nourl"] and "LK_TEST_SYS1_KEY" in reasons["sys1_vision"]
    assert (tmp_path / "s1" / "leds.png").is_file() and (tmp_path / "s1" / "live_table.json").is_file()

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response(content='{"green": 5}')))
    out = smoke.run_smoke(only=["orm"], models=models, ledger=bare, out_dir=tmp_path / "s1", http=server.client(),
                          echo=False)
    (row,) = out["rows"]
    assert row["answer"] == smoke.EXPECTED_GREEN == 5 and row["correct"] and row["usd_source"] == "provider"
    caps = bare.caps()
    assert caps["active_session"] == "S1" and caps["session_caps"]["S1"] == 1.0 and caps["total_cap_usd"] == 21.0
    body = server.bodies()[0]
    assert body["reasoning"] == {"enabled": False} and body["max_tokens"] == 300
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert PLAIN_KEY not in (tmp_path / "s1" / "live_table.json").read_text(encoding="utf-8")
    assert PLAIN_KEY not in smoke.format_table(out["rows"])


# =========================================================================== S1 fixes: one regression test group per decision
def _reserves(ledger):
    return [r for r in ledger.records() if r["type"] == "reserve"]


def _settles(ledger):
    return [r for r in ledger.records() if r["type"] == "settle"]


# --------------------------------------------------------------------------- R1: every HTTP attempt is reserved
def test_r1_lost_attempt_stays_reserved_when_a_retry_answers(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    for first in ("timeout", "connect", (524, "origin timed out")):
        before = ledger.spent()
        rec = Recorder()
        server = Server(first, (200, or_response()))
        r = ask("orm", f"write the program {first!r}", max_tokens=8000, http=server.client(), trace=rec)
        assert len(server.requests) == 2 and r.usd == 0.00123 and r.usd_source == "provider"
        reserves = _reserves(ledger)[-2:]
        assert [x["usd"] for x in reserves] == [pytest.approx(r.projected_usd)] * 2
        assert [x["tags"]["attempt"] for x in reserves] == [1, 2] and len({x["tags"]["call"] for x in reserves}) == 1
        assert _settles(ledger)[-1]["id"] == reserves[1]["id"] and _settles(ledger)[-1]["usd"] == 0.00123
        # the lost attempt may have been generated and billed: it stays counted at its projection
        assert ledger.spent() - before == pytest.approx(r.projected_usd + 0.00123)
        p = rec.events[0][1]
        assert p["usd"] == 0.00123 and p["ledger_usd"] == pytest.approx(r.projected_usd + 0.00123)
        assert (p["reserved_attempts"], p["unreconciled_attempts"]) == (2, 1)
        assert p["unreconciled_reserved_usd"] == pytest.approx(r.projected_usd) and r.raw["unreconciled_attempts"] == 1
    summary = ledger.summary()
    assert summary["calls"] == 3 and summary["reservations"] == 6 and summary["unsettled"]["count"] == 3
    hit = ask("orm", "write the program 'timeout'", max_tokens=8000)  # a hit is not charged for the lost attempt
    assert hit.cache_hit and hit.usd == 0.0 and "ledger_usd" not in hit.raw and "unreconciled_attempts" not in hit.raw
    assert hit.raw["original_ledger_usd"] == pytest.approx(hit.projected_usd + 0.00123)


def test_r1_each_retry_is_cap_checked_before_it_is_sent(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    ledger.set_active_session("T", 0.70)  # orm: 35 chars and 30000 max_tokens project $0.30002 per attempt
    rec = Recorder()
    server = Server("timeout")
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm", "x" * 35, max_tokens=30000, http=server.client(), trace=rec)
    assert len(server.requests) == 2 and err.value.scope == "session"  # the third POST was never sent
    assert err.value.spent_usd == pytest.approx(2 * 0.30002) and ledger.spent() == pytest.approx(2 * 0.30002)
    (kind, p), = rec.events
    assert kind == "model_call" and p["ok"] is False and "SpendCapExceeded" in p["error"] and p["attempts"] == 2
    assert p["usd"] == pytest.approx(2 * 0.30002) and p["unreconciled_attempts"] == 2
    ModelCallPayload.model_validate(p)


def test_r1_rate_limit_refusals_settle_at_zero_and_the_retry_is_reserved(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    sleeps: list[float] = []
    server = Server((429, {"error": {"message": "rate limited"}}, {"Retry-After": "1"}), (200, or_response()))
    r = ask("orm", "rate limited once", http=server.client(sleeps))
    assert len(server.requests) == 2 and sleeps == [1.0] and r.usd == 0.00123
    assert [x["usd"] for x in _settles(ledger)] == [0.0, 0.00123] and len(_reserves(ledger)) == 2
    assert ledger.spent() == pytest.approx(0.00123)


def test_r1_writer_calls_default_to_a_900_s_deadline(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    seen: list[float] = []

    def capture(request):
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json=or_response())

    rec = Recorder()
    for role, timeout_s in (("writer", None), ("reader", None), ("system1", None), ("writer", 42.0)):
        ask("orm", f"{role} {timeout_s}", role=role, timeout_s=timeout_s, http=Server(capture).client(), trace=rec)
    assert seen == [900.0, 300.0, 300.0, 42.0]
    assert [p["timeout_s"] for _, p in rec.events] == [900.0, 300.0, 300.0, 42.0]
    with pytest.raises(ValueError):
        ask("orm", "bad", timeout_s=float("inf"))


def test_r1_deadline_keeps_a_body_that_already_arrived(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = json.dumps(or_response()).encode()

    def late(request):  # the whole answer is in the last chunk, which lands just after the deadline
        def chunks():
            yield body[:40]
            time.sleep(0.6)
            yield body[40:]
        return httpx.Response(200, content=chunks())

    server = Server(late)
    r = ask("orm", "late but complete", timeout_s=0.4, http=server.client())
    assert len(server.requests) == 1 and r.text == '{"green": 5}'
    assert [a["status"] for a in r.raw["attempt_log"]] == [200] and ledger.spent() == pytest.approx(0.00123)

    def sized(complete):  # not JSON, so only its Content-Length can say it is whole
        def handler(request):
            def chunks():
                yield b"plain text "
                time.sleep(0.6)
                yield b"answer"
            n = len(b"plain text answer") + (0 if complete else 10)
            return httpx.Response(200, content=chunks(), headers={"Content-Length": str(n)})
        return handler

    def one_try(handler):
        return HttpClient(transport=httpx.MockTransport(handler), sleep=lambda s: None, max_attempts=1)

    res = one_try(sized(True)).post_json("https://a.invalid/x", {}, timeout_s=0.4)
    assert res.ok and res.text == "plain text answer" and res.attempts == 1
    res = one_try(sized(False)).post_json("https://a.invalid/x", {}, timeout_s=0.4)
    assert res.status is None and "timeout" in res.error and res.attempt_log[0]["outcome"] == "lost"


# --------------------------------------------------------------------------- R2: rejected only when nothing was lost
def test_r2_refusal_after_a_lost_attempt_is_not_rejected_before_generation(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    for first in ("timeout", "connect"):
        before = ledger.spent()
        rec = Recorder()
        server = Server(first, (402, {"error": {"code": 402, "message": "Insufficient credits"}}))
        with pytest.raises(ModelCallError) as err:
            ask("orm", f"hi {first}", http=server.client(), trace=rec)
        assert err.value.status == 402 and err.value.attempts == 2 and not err.value.rejected_before_generation
        lost = _reserves(ledger)[-2]
        assert _settles(ledger)[-1]["usd"] == 0.0 and _settles(ledger)[-1]["id"] == _reserves(ledger)[-1]["id"]
        assert ledger.spent() - before == pytest.approx(lost["usd"]) and rec.events[0][1]["usd"] == pytest.approx(
            lost["usd"])
    # across JSON modes: a lost 503 in the first mode, then refusals, is still not "rejected before generation"
    no_route = (503, {"error": {"message": "No available providers support the requested parameters"}})
    refuse_mode = (400, {"error": {"message": "response_format is not supported"}})
    refuse = (400, {"error": {"message": "invalid request"}})
    with pytest.raises(ModelCallError) as err:
        ask("orm", "modes", json_schema=GREEN_SCHEMA, http=Server(no_route, refuse_mode, refuse).client())
    assert err.value.attempts == 3 and not err.value.rejected_before_generation
    assert [entry["outcome"] for entry in err.value.raw["attempt_log"]] == ["lost", "rejected", "rejected"]
    # every attempt refused, in every mode: rejected, and nothing is counted
    before = ledger.spent()
    with pytest.raises(ModelCallError) as err:
        ask("orm", "modes refused", json_schema=GREEN_SCHEMA, http=Server(refuse_mode, refuse).client())
    assert err.value.rejected_before_generation and err.value.attempts == 2 and ledger.spent() == pytest.approx(before)


# --------------------------------------------------------------------------- R3: ledger I/O
def test_r3_ledger_records_are_ascii_and_split_on_newline_only(ledger):
    odd = "rehearse round 1\x85end é"
    rid = ledger.reserve(0.30, "orm", "d" * 64, tags={"note": odd})
    ledger.settle(rid, 0.25, "provider")
    raw = ledger.path.read_bytes()
    assert raw.isascii() and raw.count(b"\n") == 2
    assert [r["type"] for r in ledger.records()] == ["reserve", "settle"] and ledger.records()[0]["tags"]["note"] == odd
    assert ledger.spent() == pytest.approx(0.25)
    # a record an older writer left with raw separators inside a string still reads back whole
    old = {"type": "reserve", "id": "old1", "usd": 0.5, "session": "T", "model_key": "orm", "tags": {"phase": odd}}
    with open(ledger.path, "ab") as f:
        f.write(json.dumps(old, ensure_ascii=False).encode("utf-8") + b"\r\n")
    assert ledger.spent() == pytest.approx(0.75) and ledger.summary()["corrupt_lines"] == 0


def test_r3_session_names_model_keys_and_phases_must_be_single_line(ask, ledger):
    for bad in ("rehearse round 1", "a\nb", "x\x85y", "tab\there", "p q"):
        with pytest.raises(ValueError):
            ask("fake", "hi", phase=bad)
        with pytest.raises(ValueError):
            ledger.set_active_session(bad, 1.0)
        with pytest.raises(ValueError):
            ledger.reserve(0.1, bad, "e" * 64)
        with pytest.raises(ValueError):
            ledger.reserve(0.1, "orm", "e" * 64, tags={"phase": bad})
        with pytest.raises(ValueError):
            chat(bad, "hi", models=MODELS, ledger=ledger)
    assert ledger.records() == [] and ledger.caps()["active_session"] == "T"
    assert ask("fake", "hi", phase="rehearse round 1").text == "fake response"  # plain spaces are fine
    with pytest.raises(ValueError):  # a cache hit never reaches the ledger, and is refused all the same
        ask("fake", "hi", phase="rehearse round 1")
    ledger.caps_path.write_text(json.dumps({"active_session": "S3\u0085bakeoff", "session_caps": {}}), encoding="utf-8")
    with pytest.raises(RouterError, match="active_session"):
        ledger.reserve(0.1, "orm", "f" * 64)


def test_r3_orphan_settles_are_counted_and_warned(ledger):
    rid = ledger.reserve(0.2, "orm", "f" * 64)
    ledger.settle(rid, 0.1, "provider")
    ledger.settle("no-such-reserve", 0.4, "provider")
    with pytest.warns(RuntimeWarning, match="no matching reserve"):
        ledger.reserve(0.1, "orm", "g" * 64)  # the next reservation says so
    with pytest.warns(RuntimeWarning, match="no matching reserve"):
        summary = ledger.summary()  # and so does every summary
    assert summary["orphan_settles"] == {"count": 1, "usd": 0.4}
    assert summary["total_usd"] == pytest.approx(0.2) and summary["calls"] == 2


# --------------------------------------------------------------------------- R4: BYOK and cost sanity
def test_r4_byok_upstream_cost_is_added(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    body = or_response(cost=0.0005, prompt=5000, completion=15000)  # orm: 5000 * 2 + 15000 * 10 = $0.16
    body["usage"].update(is_byok=True, cost_details={"upstream_inference_cost": 0.16})
    r = ask("orm", "byok", max_tokens=16000, http=Server((200, body)).client())
    assert r.usd == pytest.approx(0.1605) and r.usd_source == "provider" and ledger.spent() == pytest.approx(0.1605)
    assert r.raw["cost_breakdown"] == {"usage_cost": 0.0005, "upstream_inference_cost": 0.16, "is_byok": True,
                                       "cost": pytest.approx(0.1605)}


def test_r4_provider_cost_far_below_token_cost_settles_at_computed(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    tokens = dict(prompt=5000, completion=15000)
    r = ask("orm", "zero", max_tokens=16000, http=Server((200, or_response(cost=0.0, **tokens))).client())
    assert r.usd == pytest.approx(0.16) and r.usd_source == "computed" and "below half" in r.raw["usd_note"]
    fee_only = or_response(cost=0.008, **tokens)
    fee_only["usage"]["is_byok"] = True  # BYOK without the upstream amount: the fee alone is not the cost
    r2 = ask("orm", "fee only", max_tokens=16000, http=Server((200, fee_only)).client())
    assert r2.usd == pytest.approx(0.16) and r2.usd_source == "computed" and "note" in r2.raw["cost_breakdown"]
    r3 = ask("orm", "just over half", max_tokens=16000, http=Server((200, or_response(cost=0.081, **tokens))).client())
    assert r3.usd == 0.081 and r3.usd_source == "provider"
    assert [s["usd"] for s in _settles(ledger)] == [pytest.approx(0.16), pytest.approx(0.16), 0.081]
    # cached tokens count at cached_in: 1000 * 2 + 9000 * 0.2 + 100 * 10 = $0.0048 is a plausible provider cost
    cached = or_response(cost=0.0048, prompt=10000, completion=100, cached=9000, model="vendor/cache-model")
    r4 = ask("orm_cache", "cached", http=Server((200, cached)).client())
    assert r4.usd == 0.0048 and r4.usd_source == "provider"


# --------------------------------------------------------------------------- R5: see test_ledger_lock_* above


# --------------------------------------------------------------------------- R6: reasoning tokens per adapter
def test_r6_reasoning_tokens_follow_the_adapter_property(ask, monkeypatch):
    assert get_adapter("openrouter").reasoning_in_output is True and get_adapter("elorian").reasoning_in_output is None
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    el = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=6000, model="elorian-test")))
    r = ask("el", "count", max_tokens=8000, http=el.client())
    assert r.usd == pytest.approx((2000 * 0.6 + (120 + 6000) * 3.0) / 1e6) and "separately" in r.raw["usd_note"]
    el2 = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=50, model="elorian-test")))
    r2 = ask("el", "count again", max_tokens=8000, http=el2.client())
    assert r2.usd == pytest.approx((2000 * 0.6 + 170 * 3.0) / 1e6) and "unknown" in r2.raw["usd_note"]
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)  # OpenRouter counts them inside completion_tokens
    orx = Server((200, or_response(cost=None, prompt=2000, completion=120, reasoning=100)))
    r3 = ask("orm", "count", http=orx.client())
    assert r3.usd == pytest.approx((2000 * 2.0 + 120 * 10.0) / 1e6)


# --------------------------------------------------------------------------- R7: nothing raises after a paid call
def test_r7_degenerate_json_is_a_json_error_never_an_exception(ask, ledger, monkeypatch):
    rec = Recorder()
    num_schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    replies = ['{"x": 1' + "0" * 400 + "}", '{"x": ' + "5" * 5000 + "}", "[" * 3000,
               '{"x": ' + "[" * 1500 + "]" * 1500 + "}"]
    with fake_script(replies):
        answers = [ask("paid_fake", f"q{i}", json_schema=num_schema, max_tokens=50, trace=rec) for i in range(4)]
    assert all(a.json_error for a in answers[1:]) and answers[1].json_error.startswith("not JSON")
    assert len(rec.events) == 4 and all(p["ok"] for _, p in rec.events)
    assert [x["type"] for x in ledger.records()] == ["reserve", "settle"] * 4
    # a validator that blows up (OverflowError for a huge number, RecursionError for deep nesting) is a json_error
    for boom in (OverflowError("int too large to convert to float"), RecursionError("maximum recursion depth")):
        def raising_check(value, schema, _boom=boom):
            raise _boom
        monkeypatch.setattr(router_core, "check", raising_check)
        with fake_script(['{"x": 1}']):
            r = ask("paid_fake", f"boom {type(boom).__name__}", json_schema=num_schema, max_tokens=50)
        assert r.json == {"x": 1} and r.json_error.startswith("schema check failed")
    for text in ('{"a": ' + "9" * 5000 + "}", "[" * 100000 + "]" * 100000, None, "```json\n" + "{" * 5000):
        value, err = parse_json_text(text)
        assert value is None and err.startswith(("not JSON", "empty answer"))
    # a degenerate first block does not stop the parser from reading a good one after it
    two_blocks = "```\n" + "[" * 3000 + "\n```\nThe corrected answer:\n```json\n{\"x\": 1}\n```"
    assert parse_json_text(two_blocks) == ({"x": 1}, None)


def test_r7_failure_after_the_paid_call_settles_and_traces_before_raising(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)

    def broken_put(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(cache, "put", broken_put)
    rec = Recorder()
    with pytest.raises(RuntimeError, match="disk on fire"):
        ask("orm", "paid then broken", http=Server((200, or_response())).client(), trace=rec)
    (kind, p), = rec.events
    assert p["ok"] is False and "disk on fire" in p["error"] and p["usd"] == pytest.approx(0.00123)
    assert [s["usd"] for s in _settles(ledger)] == [0.00123]
    ModelCallPayload.model_validate(p)

    def broken_cost(*a, **k):
        raise ArithmeticError("cost model broke")

    monkeypatch.setattr(router_core, "settled_cost", broken_cost)
    rec = Recorder()
    with pytest.raises(ArithmeticError):
        ask("orm", "cost broke", http=Server((200, or_response())).client(), trace=rec)
    (kind, p), = rec.events
    last = _reserves(ledger)[-1]
    assert p["ok"] is False and p["usd"] == pytest.approx(last["usd"]) and p["unreconciled_attempts"] == 1
    assert last["id"] not in {s["id"] for s in _settles(ledger)}  # unknown cost: stays counted at the projection


# --------------------------------------------------------------------------- R8: reasoning budget in the projection
def test_r8_a_reasoning_token_budget_is_projected(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    ledger.set_active_session("T", 0.05)
    server = Server((200, or_response()))
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm", "x" * 35, max_tokens=1000, reasoning={"max_tokens": 32000}, http=server.client())
    assert err.value.projected_usd == pytest.approx((10 * 2.0 + 33000 * 10.0) / 1e6) and not server.requests
    budget_models = _models()
    budget_models.models["orm_hi"].reasoning = {"max_tokens": 32000}
    with pytest.raises(SpendCapExceeded) as err:
        ask("orm_hi", "x" * 35, max_tokens=1000, reasoning="highest", models=budget_models, http=server.client())
    assert err.value.projected_usd == pytest.approx((10 * 2.0 + 33000 * 10.0) / 1e6) and not server.requests
    ledger.set_active_session("T", 1.0)
    rec = Recorder()
    r = ask("orm", "x" * 35, max_tokens=1000, reasoning={"type": "enabled", "budget_tokens": 5000},
            http=server.client(), trace=rec)
    assert r.projected_usd == pytest.approx((10 * 2.0 + 6000 * 10.0) / 1e6) and rec.events[0][1]["reasoning_budget"] == 5000
    assert server.bodies()[0]["reasoning"] == {"type": "enabled", "budget_tokens": 5000}


# --------------------------------------------------------------------------- R9: every non-fake adapter is paid
def test_r9_a_zero_priced_paid_model_still_needs_a_session_and_respects_caps(tmp_path, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    bare = Ledger(tmp_path / "bare" / "spend_ledger.jsonl")

    def call(text, cost=0.25):
        return chat("free_or", text, max_tokens=16000, models=MODELS, ledger=bare, cache=cache, use_cache=False,
                    http=Server((200, or_response(cost=cost, model="vendor/free-tier"))).client())

    with pytest.raises(SpendCapExceeded) as err:
        call("no session")
    assert err.value.scope == "no_session" and bare.records() == []
    bare.set_active_session("Z", 0.10, total_cap_usd=0.10)
    r = call("first")  # projects $0, admitted, and the provider's $0.25 is settled
    assert r.projected_usd == 0.0 and r.usd == 0.25 and bare.spent() == 0.25
    with pytest.raises(SpendCapExceeded) as err:  # the session is over its cap, so even a $0 projection is refused
        call("second")
    assert err.value.scope == "session" and len(_reserves(bare)) == 1
    at_cap = Ledger(tmp_path / "at_cap" / "spend_ledger.jsonl")
    at_cap.set_active_session("A", 0.25)  # no model or total cap: only "at the cap" can refuse the next call
    assert chat("free_or", "fills the cap", models=MODELS, ledger=at_cap, cache=cache, use_cache=False,
                http=Server((200, or_response(cost=0.25))).client()).usd == 0.25
    with pytest.raises(SpendCapExceeded) as err:  # exactly at the cap counts as reached
        chat("free_or", "one more", models=MODELS, ledger=at_cap, cache=cache, use_cache=False,
             http=Server((200, or_response(cost=0.0))).client())
    assert err.value.scope == "session" and err.value.spent_usd == pytest.approx(0.25)
    assert chat("fake", "the free fake needs nothing", models=MODELS, ledger=Ledger(tmp_path / "b2" / "l.jsonl"),
                cache=cache).usd == 0.0


def test_r9_model_entries_reject_non_finite_and_unmarked_zero_prices(tmp_path):
    good = _entry(id="vendor/x", price_in=1.0, price_out=2.0)
    for field, bad in (("price_in", float("nan")), ("price_out", float("inf")), ("cached_in", float("nan")),
                       ("price_in", -1.0)):
        with pytest.raises(ValidationError):
            ModelEntry.model_validate({**good, field: bad})
    with pytest.raises(ValidationError, match="free"):
        ModelEntry.model_validate({**good, "price_in": 0.0, "price_out": 0.0})
    for adapter in ("openrouter", "elorian", "jev"):
        with pytest.raises(ValidationError, match="free"):
            ModelEntry.model_validate({**good, "adapter": adapter, "price_in": 0, "price_out": 0})
    ModelEntry.model_validate({**good, "price_in": 0.0, "price_out": 0.0, "notes": "Free during the preview"})
    ModelEntry.model_validate({**good, "adapter": "fake", "price_in": 0.0, "price_out": 0.0})
    ModelEntry.model_validate({**good, "price_in": 0.04, "price_out": 0.0})  # input-only billing is not 0 / 0
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text("version: t\ncurrency: USD\nunit: usd_per_million_tokens\nmodels:\n  el:\n"
                         "    display: e\n    adapter: elorian\n    id: e-1\n    base_url: https://e.invalid/v1\n"
                         "    price_in: 0.6\n    price_out: 3.0\n    cached_in: .nan\n    vision: true\n"
                         "    roles: [reader]\n    verified: true\n    source: t\n    accessed: null\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        config.load_models(yaml_path)
    config.load_models()  # the committed models.yaml stays valid


# --------------------------------------------------------------------------- R10: NaN
def test_r10_caps_file_rejects_nan_infinity_and_bad_caps(ledger):
    for text in ('{"active_session": "T", "session_caps": {"T": NaN}}',
                 '{"active_session": "T", "session_caps": {"T": 1.0}, "total_cap_usd": Infinity}',
                 '{"active_session": "T", "session_caps": {"T": 1.0}, "total_cap_usd": -Infinity}',
                 '{"active_session": "T", "session_caps": {"T": -1}}',
                 '{"active_session": "T", "session_caps": {"T": 1e999}}',
                 '{"active_session": "T", "session_caps": {"T": true}}',
                 '{"active_session": "T", "session_caps": {"T": "1.0"}}',
                 '{"active_session": "T", "session_caps": {"T": 1.0}, "model_caps": {"T": {"orm": NaN}}}',
                 '{"active_session": "T", "session_caps": {"T": 1.0}, "note": NaN}'):  # NaN anywhere is refused
        ledger.caps_path.write_text(text, encoding="utf-8")
        with pytest.raises(RouterError):
            ledger.reserve(0.5, "orm", "i" * 64)
        with pytest.raises(RouterError):
            ledger.set_active_session("T", 1.0)
    assert ledger.records() == []


def test_r10_non_finite_computed_cost_settles_at_the_projection(ask, ledger, monkeypatch):
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    models = _models()
    models.models["el"].cached_in = float("nan")  # past validation, as a hand-built entry could be
    server = Server((200, or_response(cost=None, prompt=50000, completion=8000, model="elorian-test")))
    r = ask("el", "count", models=models, http=server.client())
    assert r.usd == pytest.approx(r.projected_usd) and r.usd > 0 and "not a finite" in r.raw["usd_note"]
    assert ledger.spent() == pytest.approx(r.projected_usd)


# --------------------------------------------------------------------------- R11: failed generations
def test_r11_a_failed_generation_raises_and_is_never_cached(ask, ledger, cache, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    partial = '{"nodes": [{"id": "locate", "skill": "find_ra'
    failed = or_response(content=partial, finish="error", cost=0.0004, prompt=100, completion=30)
    rec = Recorder()
    with pytest.raises(ModelCallError, match="failed upstream") as err:
        ask("orm", "write the graph", http=Server((200, failed)).client(), trace=rec)
    assert err.value.status == 200 and err.value.usd == 0.0004 and not err.value.rejected_before_generation
    assert not list(cache.root.rglob("*.json")) and ledger.spent() == pytest.approx(0.0004)
    assert rec.events[0][1]["ok"] is False and rec.events[0][1]["finish_reason"] == "error"
    server = Server((200, or_response(content='{"nodes": []}')))
    r = ask("orm", "write the graph", http=server.client())  # the identical request goes to the provider again
    assert not r.cache_hit and len(server.requests) == 1 and r.text == '{"nodes": []}'
    no_cost = or_response(content="partial")
    no_cost["choices"][0]["error"] = {"message": "upstream timed out", "code": 502}
    del no_cost["usage"]["cost"]
    before = ledger.spent()
    with pytest.raises(ModelCallError, match="upstream timed out"):
        ask("orm", "error object", http=Server((200, no_cost)).client())
    assert ledger.spent() - before == pytest.approx(_reserves(ledger)[-1]["usd"])  # unknown cost: the projection
    # a failed generation left in the cache by an older router is not served either
    hit_hash = r.request_hash
    record = json.loads(cache.path_for(hit_hash).read_text(encoding="utf-8"))
    record["finish_reason"] = "error"
    cache.path_for(hit_hash).write_text(json.dumps(record), encoding="utf-8")
    again = Server((200, or_response(content='{"nodes": [1]}')))
    assert not ask("orm", "write the graph", http=again.client()).cache_hit and len(again.requests) == 1


# --------------------------------------------------------------------------- R12: pre-send failures
def test_r12_numpy_values_in_schema_and_extra_are_sent_as_plain_json(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    schema = {"type": "object", "properties": {"green": {"type": "integer", "minimum": 0,
                                                         "maximum": np.int64(12)}}, "required": ["green"]}
    server = Server((200, or_response()))
    r = ask("orm", "count", json_schema=schema, http=server.client())
    assert r.json == {"green": 5} and r.json_error is None
    sent = server.bodies()[0]["response_format"]["json_schema"]["schema"]
    assert sent["properties"]["green"]["maximum"] == 12 and type(sent["properties"]["green"]["maximum"]) is int
    server2 = Server((200, or_response()))
    ask("orm", "warm", temperature=np.float32(0.5), reasoning={"max_tokens": np.int64(100)}, http=server2.client())
    assert server2.bodies()[0]["temperature"] == 0.5 and server2.bodies()[0]["reasoning"] == {"max_tokens": 100}
    with pytest.raises(ValueError):
        ask("orm", "nan", json_schema={"type": "number", "maximum": float("nan")})
    with pytest.raises(ValueError):
        ask("orm", "nan", temperature=float("nan"))


def test_r12_unparseable_url_is_not_configured_before_any_spend(ask, ledger, monkeypatch):
    monkeypatch.setenv("ELORIAN_API_KEY", PLAIN_KEY)
    for bad in ("https://api.elorian.invalid:443v1", "https://[::1/v1"):
        models = _models()
        models.models["el"].base_url = bad
        with pytest.raises(ModelNotConfigured, match="not a valid URL"):
            ask("el", "hi", models=models)
    assert ledger.records() == []


def test_r12_a_request_that_cannot_be_built_is_rejected_and_settled_at_zero(ask, ledger, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    server = Server((200, or_response()))
    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "headers", lambda self: {"X-Title": "loopkit “beta”"})
        with pytest.raises(ModelCallError) as err:
            ask("orm", "bad header", http=server.client())
    assert err.value.rejected_before_generation and err.value.status is None and "nothing was sent" in str(err.value)

    def bad_body(self, request, mode=None):
        raise TypeError("cannot build this")

    with monkeypatch.context() as m:
        m.setattr(OpenRouterAdapter, "build_payload", bad_body)
        with pytest.raises(ModelCallError, match="could not be built") as err:
            ask("orm", "bad body", http=server.client())
    assert err.value.rejected_before_generation and not server.requests
    assert len(_reserves(ledger)) == 2 and [s["usd"] for s in _settles(ledger)] == [0.0, 0.0]
    assert ledger.spent() == 0.0
    res = HttpClient(transport=httpx.MockTransport(_refuse_network)).post_json("https://a.invalid/x",
                                                                               {"x": float("nan")})
    assert not res.sent and res.attempts == 0 and "not plain JSON" in res.error
    res = HttpClient(transport=httpx.MockTransport(_refuse_network)).post_json("https://a.invalid:443v1/x", {})
    assert not res.sent and "URL is not valid" in res.error


def test_r12_smoke_survives_any_per_model_error(tmp_path, monkeypatch):
    from loopkit.router import smoke

    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    real_chat = smoke.chat

    def flaky(key, **kw):
        if key == "orm":
            raise KeyError("not a router error")
        return real_chat(key, **kw)

    monkeypatch.setattr(smoke, "chat", flaky)
    bare = Ledger(tmp_path / "smoke" / "spend_ledger.jsonl")
    out = smoke.run_smoke(only=["orm", "orm_hi"], models=MODELS, ledger=bare, out_dir=tmp_path / "s1",
                          http=Server((200, or_response(content='{"green": 5}'))).client(), echo=False)
    rows = {r["model_key"]: r for r in out["rows"]}
    assert not rows["orm"]["ok"] and "KeyError" in rows["orm"]["error"] and rows["orm_hi"]["ok"]
    assert rows["orm"]["input"] == "image + question"  # the failed row keeps what was asked
    assert (tmp_path / "s1" / "live_table.json").is_file()
    monkeypatch.setattr(smoke, "_call_one", lambda *a, **k: 1 / 0)  # even a failure outside chat() is one row
    out = smoke.run_smoke(only=["orm"], models=MODELS, ledger=bare, out_dir=tmp_path / "s1", echo=False)
    assert out["rows"][0]["ok"] is False and "ZeroDivisionError" in out["rows"][0]["error"]


# --------------------------------------------------------------------------- R13: the cache key
def test_r13_schema_property_order_and_destination_change_the_request_hash(ask, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", PLAIN_KEY)
    first = {"type": "object", "properties": {"green": {"type": "integer"}, "reasoning": {"type": "string"}},
             "required": ["green", "reasoning"]}
    second = {"type": "object", "properties": {"reasoning": {"type": "string"}, "green": {"type": "integer"}},
              "required": ["green", "reasoning"]}
    r1 = ask("orm", "count", json_schema=first, http=Server((200, or_response(content='{"green": 4, '
                                                                                   '"reasoning": "4"}'))).client())
    s2 = Server((200, or_response(content='{"reasoning": "five", "green": 5}')))
    r2 = ask("orm", "count", json_schema=second, http=s2.client())
    assert r1.request_hash != r2.request_hash and not r2.cache_hit and len(s2.requests) == 1
    assert ask("orm", "count", json_schema=json.loads(json.dumps(first))).cache_hit  # the same schema is a hit
    elsewhere = _models()
    elsewhere.models["orm"].base_url = "https://staging.router.invalid/api/v1"
    s3 = Server((200, or_response()))
    r3 = ask("orm", "count", json_schema=first, models=elsewhere, http=s3.client())
    assert not r3.cache_hit and r3.request_hash != r1.request_hash
    assert str(s3.requests[0].url) == "https://staging.router.invalid/api/v1/chat/completions"


# --------------------------------------------------------------------------- R14: keys that cannot go in a header
def test_r14_a_non_ascii_key_is_refused_by_name_before_any_reservation(ask, ledger, monkeypatch):
    core_secret = "sk-or-v1-" + "a1" * 32
    server = Server((200, or_response()))
    for bad in ("“" + core_secret + "”", core_secret[:9] + "​" + core_secret[9:],
                core_secret[:9] + " " + core_secret[9:]):
        monkeypatch.setenv("OPENROUTER_API_KEY", bad)
        with pytest.raises(MissingCredentialError) as err:
            ask("orm", "hi", http=server.client())
        for text in (str(err.value), repr(err.value), repr(err.value.args)):
            assert core_secret not in text and core_secret[9:30] not in text
        assert "OPENROUTER_API_KEY" in str(err.value) and err.value.__cause__ is None and err.value.__suppress_context__
    assert not server.requests and ledger.records() == []


def test_r14_header_build_errors_are_redacted_and_not_chained(ask, monkeypatch):
    secret = "lk-test-“" + "Q7" * 20
    res = Server((200, {})).client().post_json("https://a.invalid/x", {"a": 1}, key=secret)
    assert not res.sent and res.attempts == 0 and "ASCII" in res.error and "Q7Q7" not in res.error
    with pytest.raises(ModelCallError) as err:
        OpenRouterAdapter().send(make_request("orm"), secret, Server((200, or_response())).client())
    assert err.value.rejected_before_generation and err.value.__cause__ is None and err.value.__context__ is None
    for text in (str(err.value), repr(err.value), json.dumps(err.value.raw)):
        assert "Q7Q7" not in text


# --------------------------------------------------------------------------- R15: the Jev path scrubs the parsed answer
def test_r15_a_key_echoed_in_a_jev_answer_is_scrubbed_from_json_and_cache(ask, cache, monkeypatch):
    monkeypatch.setenv("TYPESAFE_JEV_API_KEY", PLAIN_KEY)
    questions = {"q": {"type": "choice", "instructions": "How many green LEDs?", "criteria": {"5": "five"}}}
    answers = {"q": {"type": "choice", "choice": "5", "debug": f"authorized as Bearer {PLAIN_KEY}",
                     PLAIN_KEY: ["as a key", {"deeper": PLAIN_KEY}]}}
    server = Server((200, {"id": "r1", "model": "jev-latest", "answers": answers,
                           "usage": {"input_tokens": 10, "output_tokens": 1}}))
    r = ask("jev", "panel", role="system1", extra={"questions": questions}, http=server.client())
    assert PLAIN_KEY not in json.dumps(r.json) and PLAIN_KEY not in r.text and PLAIN_KEY not in json.dumps(r.raw)
    assert r.json["q"]["choice"] == "5" and r.json["q"]["debug"] == "authorized as Bearer [redacted]"
    assert r.json["q"]["[redacted]"] == ["as a key", {"deeper": "[redacted]"}]
    assert PLAIN_KEY not in cache.path_for(r.request_hash).read_text(encoding="utf-8")
    hit = ask("jev", "panel", role="system1", extra={"questions": questions})
    assert hit.cache_hit and PLAIN_KEY not in json.dumps(hit.json) and hit.json == r.json
