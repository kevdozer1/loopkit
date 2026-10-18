"""loopkit's model router: ``chat()``, adapters, disk cache and spend ledger.

Every model call goes through :func:`chat`, which looks the model up in
``config/models.yaml``, serves a cached answer for free when it has one, projects
the worst-case cost and reserves it in the spend ledger (refusing a call that
would cross a cap), sends the call through the entry's adapter, settles the
measured cost, validates a JSON answer and emits one ``model_call`` trace event.

Adapters: ``openrouter`` (default), ``elorian``, ``jev`` and ``fake`` (offline, $0).
Start each session with :func:`set_active_session` so paid calls are allowed.
"""

from ..config import MissingCredentialError
from .adapters import available_adapters, get_adapter, register_adapter
from .adapters.fake import FakeRequest, clear_fake_script, fake_calls, fake_script, from_schema, set_fake_script
from .cache import DiskCache
from .core import Response, canonical_json, chat, parse_json_text, request_hash, resolve_route, validate_json
from .errors import ModelCallError, ModelNotConfigured, RouterError, SpendCapExceeded
from .http import HttpClient, HttpResult, default_client, set_default_client
from .ledger import FileLock, Ledger, read_caps, set_active_session
from .media import ImageData, normalize_image

__all__ = [
    "DiskCache",
    "FakeRequest",
    "FileLock",
    "HttpClient",
    "HttpResult",
    "ImageData",
    "Ledger",
    "MissingCredentialError",
    "ModelCallError",
    "ModelNotConfigured",
    "Response",
    "RouterError",
    "SpendCapExceeded",
    "available_adapters",
    "canonical_json",
    "chat",
    "clear_fake_script",
    "default_client",
    "fake_calls",
    "fake_script",
    "from_schema",
    "get_adapter",
    "normalize_image",
    "parse_json_text",
    "read_caps",
    "register_adapter",
    "request_hash",
    "resolve_route",
    "set_active_session",
    "set_default_client",
    "set_fake_script",
    "validate_json",
]
