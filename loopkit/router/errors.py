"""Router errors.

Every router error subclasses :class:`RouterError` (a ``RuntimeError``). Messages
say what to do next and never carry a key: HTTP error text is scrubbed before it
reaches an exception. :class:`MissingCredentialError` is loopkit.config's own
class, re-exported here so callers can catch everything from one module.
"""

from __future__ import annotations

from typing import Any

from ..config import MissingCredentialError

__all__ = [
    "MissingCredentialError",
    "ModelCallError",
    "ModelNotConfigured",
    "RouterError",
    "SpendCapExceeded",
]


class RouterError(RuntimeError):
    """Base class for the errors the router raises itself."""


class ModelNotConfigured(RouterError):
    """The models.yaml entry cannot be called yet: no id, no base_url, or an adapter loopkit lacks."""


class SpendCapExceeded(RouterError):
    """A call's projected cost would cross a spend cap, or no spend session is active.

    ``scope`` is ``session``, ``model``, ``total`` or ``no_session``.
    """

    def __init__(self, message: str, *, scope: str = "session", cap_usd: float = 0.0,
                 spent_usd: float = 0.0, projected_usd: float = 0.0):
        super().__init__(message)
        self.scope = scope
        self.cap_usd = float(cap_usd)
        self.spent_usd = float(spent_usd)
        self.projected_usd = float(projected_usd)

    def __reduce__(self):  # keep the attributes through pickling (multiprocessing)
        return (_rebuild, (type(self), str(self), {"scope": self.scope, "cap_usd": self.cap_usd,
                                                   "spent_usd": self.spent_usd,
                                                   "projected_usd": self.projected_usd}))


class ModelCallError(RouterError):
    """The provider call failed after the router's retries.

    ``status`` is the last HTTP status (None for a timeout, a connection error or a
    request that could not be built) and ``attempts`` the number of HTTP attempts
    made over every JSON mode. ``rejected_before_generation`` is True only when no
    attempt of the call was lost or answered: each one was refused before generating
    anything (HTTP 400/401/402/403/404/429 with no generation id) or never sent. The
    ledger settles each refused attempt at $0 and keeps every other attempt reserved
    at its projection. ``usd`` is a provider-reported cost when the failed response
    carried one (for example a 200 whose generation failed upstream).
    """

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 0,
                 rejected_before_generation: bool = False, generation_id: str | None = None,
                 usd: float | None = None, raw: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.attempts = int(attempts)
        self.rejected_before_generation = bool(rejected_before_generation)
        self.generation_id = generation_id
        self.usd = usd
        self.raw = raw or {}

    def __reduce__(self):
        return (_rebuild, (type(self), str(self), {
            "status": self.status, "attempts": self.attempts,
            "rejected_before_generation": self.rejected_before_generation,
            "generation_id": self.generation_id, "usd": self.usd, "raw": self.raw}))


def _rebuild(cls: type, message: str, kwargs: dict[str, Any]) -> Exception:
    return cls(message, **kwargs)
