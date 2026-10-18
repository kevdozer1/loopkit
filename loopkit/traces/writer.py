"""TraceWriter: one run's append-only event log, with its images and metadata.

A run directory is ``paths.run_dir(run_id)``, or ``root / run_id`` when a root is
given (tests, the bake-off's own traces folder)::

    <run_id>/
      events.jsonl   one compact JSON line per TraceEvent, seq 0, 1, 2, ...
      meta.json      run metadata (task_id, model_key, seed, repeat, ...), replaced atomically
      images/        PNGs from save_image, never overwritten
      .writer.lock   held locked by the run's open TraceWriter

One writer per run. A TraceWriter holds an OS lock on ``.writer.lock``
(``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere) from the moment it
opens until ``close()``, and the OS drops the lock if the process dies. A second
TraceWriter for the same run, in another process or in this one, raises
RunLockedError naming the run. Within a process, a writer is safe to share
between threads.

A Ctrl+C while a writer opens or closes never leaves the run locked with no
writer to close. An open that it interrupts gives back everything it took, and a
Ctrl+C that lands inside close() still releases the run. ``closed`` turns True
only once the run is released: if it is still False after a KeyboardInterrupt,
close() never got started, and calling it again finishes the job.

Opening a run that already has events resumes it. A partial last line left by a
crash stays where it is (readers skip it) and the next event starts on a fresh
line, so nothing written earlier is rewritten or lost. Each event's seq is
reserved just before its line is written: a write that fails or is interrupted
leaves a gap in seq, never a duplicate.

emit is not re-entrant. A signal handler that logs an event while the same
thread is inside emit gets RuntimeError instead of a deadlock. In a handler, set
a flag and log from the main loop.
"""

from __future__ import annotations

import errno
import io
import json
import math
import os
import re
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from .. import paths
from ..schemas.trace import EVENT_KINDS, PAYLOAD_MODELS, TraceEvent
from ..schemas.verdict import Verdict
from .jsonsafe import dumps, to_jsonable

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = [
    "EVENTS_FILE", "IMAGES_DIR", "LOCK_FILE", "MAX_SEQ", "META_FILE", "RunLockedError", "TraceWriter",
    "check_run_id", "iter_lines", "parse_event_line", "read_meta", "run_path_under", "safe_image_name",
]

EVENTS_FILE = "events.jsonl"
META_FILE = "meta.json"
IMAGES_DIR = "images"
LOCK_FILE = ".writer.lock"
MAX_SEQ = 2**63 - 1  # load_runs keeps seq in an int64 column; a line with a larger seq is malformed

# Verdict fields outside VerdictPayload that a verdict event always carries.
VERDICT_EXTRA_FIELDS = ("task_id", "seed", "checker_id", "summary")

_BAD_RUN_ID_CHARS = set('\\/:*?"<>|')
_UTF8_BOM = b"\xef\xbb\xbf"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MODES = {"1", "L", "LA", "I", "I;16", "P", "RGB", "RGBA"}
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
# Names Windows maps to a device in any folder, in any case, with or without an extension.
_SUPERSCRIPT_DIGITS = ("\N{SUPERSCRIPT ONE}", "\N{SUPERSCRIPT TWO}", "\N{SUPERSCRIPT THREE}")
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in (*"0123456789", *_SUPERSCRIPT_DIGITS)}
)
_RESERVED_NAMES = _DEVICE_NAMES  # the image-name guard uses the same list
_MAX_IMAGE_STEM = 100
# Windows byte-range locks are mandatory, so the lock sits past the holder note (the
# file's first line, which a refused writer reads to name the holder).
_LOCK_OFFSET = 1 << 20
_LOCK_HELD_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK}


class RunLockedError(RuntimeError):
    """Another TraceWriter has this run open. One writer per run."""


class _BadMeta(Exception):
    """meta.json exists but is not a UTF-8 JSON object. The message completes "<path> ..."."""


# ------------------------------------------------------------------------ run ids
def check_run_id(run_id: str) -> str:
    """``run_id`` when it is a plain directory name, else ValueError.

    Rejected: empty, ``.`` and ``..``, slashes and ``:*?"<>|``, control characters,
    lone surrogates, surrounding spaces, a trailing dot, a ``.jsonl`` suffix (the
    readers take such a string as the path of an events file) and Windows device
    names in any case, with or without an extension (CON, PRN, AUX, NUL, COM0 to
    COM9, LPT0 to LPT9, CONIN$, CONOUT$), which Windows maps to a device instead of
    a folder. ``paths.run_dir`` applies the same check.
    """
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id != run_id.strip()
        or run_id in (".", "..")
        or run_id.endswith(".")
        or any(c in _BAD_RUN_ID_CHARS or ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in run_id)
    ):
        raise ValueError(
            f"run_id {run_id!r} is not a plain directory name (no slashes, no :*?\"<>|, no control "
            "characters or lone surrogates, no surrounding spaces, no trailing dot)"
        )
    if run_id.lower().endswith(".jsonl"):
        raise ValueError(f"run_id {run_id!r} ends in .jsonl, which the trace readers take as an events file path")
    device = _device_name(run_id)
    if device is not None:
        raise ValueError(f"run_id {run_id!r} is the Windows device name {device}, which Windows will not use as a folder")
    return run_id


def _device_name(name: str) -> str | None:
    """The device ``name`` stands for on Windows (``nul.txt`` and ``Nul .log`` are NUL), else None."""
    stem = name.split(".", 1)[0].rstrip(" ").upper()
    return stem if stem in _DEVICE_NAMES else None


def run_path_under(root: str | Path, run_id: str) -> Path:
    """``root / run_id`` once ``run_id`` passes check_run_id and the path, made
    absolute, is still a direct child of ``root``.

    The second check is a backstop for names Windows maps elsewhere when a path is
    made absolute: ``<root>/nul`` becomes the NUL device, and a run there reports
    success while its events go nowhere. ValueError otherwise; nothing is created.
    """
    check_run_id(run_id)
    path = Path(root) / run_id
    resolved = os.path.abspath(path)
    expected = os.path.join(os.path.abspath(root), run_id)
    if os.path.normcase(resolved) != os.path.normcase(expected):
        raise ValueError(f"run_id {run_id!r} does not resolve to a folder inside {root}: the path becomes {resolved}")
    return path


# ------------------------------------------------------------------------ lines
def iter_lines(data: bytes) -> Iterator[tuple[int, bytes]]:
    """``(line_number, line)`` for each non-blank line, split on ``\\n`` only."""
    for number, raw in enumerate(data.split(b"\n"), 1):
        line = raw.strip()
        if number == 1 and line.startswith(_UTF8_BOM):
            line = line[len(_UTF8_BOM):].strip()
        if line:
            yield number, line


def parse_event_line(line: bytes | str) -> TraceEvent:
    """One events.jsonl line as a TraceEvent: the rule every trace reader applies.

    pydantic's JSON parser goes first. When it rejects the line, the line is read
    with json.loads and made JSON safe (lone surrogates become U+FFFD), which
    recovers lines that older writers wrote with a lone surrogate escape such as
    ``\\ud83d``. A seq beyond int64 is rejected. Raises ValueError.
    """
    try:
        event = TraceEvent.model_validate_json(line)
    except Exception as first:
        try:
            event = TraceEvent.model_validate(to_jsonable(json.loads(line)))
        except Exception:
            raise ValueError(f"not a valid trace event: {first}") from None
    if event.seq > MAX_SEQ:
        raise ValueError(f"seq {event.seq} does not fit in int64")
    return event


def _plain_seq(line: bytes) -> int | None:
    """The seq of ``line`` when it is written in the writer's own form (a plain int in
    range), whether or not a reader accepts the rest of the line; else None."""
    try:
        obj = json.loads(line)
    except Exception:  # any failure, RecursionError included
        return None
    seq = obj.get("seq") if isinstance(obj, dict) else None
    return seq if type(seq) is int and 0 <= seq <= MAX_SEQ else None


def _scan_events(path: Path) -> tuple[int, int, bool]:
    """(next seq, malformed lines, needs a newline before appending) for an events file.

    Each line goes through parse_event_line, the rule read_events applies, so the
    malformed count is the one read_events reports and resume agrees with the
    readers on every seq (1.0, "2", a surrogate escape). The next seq is one past
    the highest seq on any line: every seq a reader accepts, plus the plain int seq
    of a line it rejects, so a seq is never handed out twice.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return 0, 0, False
    top, bad = -1, 0
    for _, line in iter_lines(data):
        try:
            seq = parse_event_line(line).seq
        except Exception:  # any failure on one line, RecursionError included, costs that line only
            bad += 1
            seq = _plain_seq(line)
            if seq is None:
                continue
        top = max(top, seq)
    return top + 1, bad, bool(data) and not data.endswith(b"\n")


# ------------------------------------------------------------------------ meta
def _read_file(path: Path) -> bytes:
    return path.read_bytes()


def _retry_permission(fn: Callable[..., Any], *args: Any, attempts: int = 5, step: float = 0.02) -> Any:
    """``fn(*args)``, retried while Windows reports the file in use (PermissionError).

    Reading meta.json in the instant it is replaced, or replacing it while a reader
    has it open, fails on Windows for a moment. The last error is raised.
    """
    for attempt in range(attempts):
        try:
            return fn(*args)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(step * (attempt + 1))
    raise AssertionError("unreachable")


def _json_type(value: Any) -> str:
    names = {list: "array", str: "string", bool: "boolean", int: "number", float: "number", type(None): "null"}
    return names.get(type(value), type(value).__name__)


def _load_meta(path: Path) -> dict[str, Any]:
    """The JSON object in ``path``, made JSON safe; ``{}`` when there is no file.

    NaN and infinity become None and lone surrogates U+FFFD, so the result can
    always be written back. Raises OSError when the file cannot be read
    (PermissionError is retried first) and _BadMeta when it is not a UTF-8 JSON
    object.
    """
    try:
        raw = _retry_permission(_read_file, path)
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except UnicodeError as exc:
        raise _BadMeta(f"is not UTF-8 text ({exc})") from None
    except RecursionError:
        raise _BadMeta("is nested too deeply to parse") from None
    except ValueError as exc:
        raise _BadMeta(f"is not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise _BadMeta(f"holds a JSON {_json_type(data)}, not an object")
    try:
        return to_jsonable(data)
    except RecursionError:
        raise _BadMeta("is nested too deeply to parse") from None


def read_meta(run_dir: str | Path) -> dict[str, Any]:
    """A run's ``meta.json`` as a dict, made JSON safe (NaN and infinity become None).

    ``{}`` when there is no meta.json. When the file cannot be read, or is not a
    UTF-8 JSON object (PowerShell 5.1 saves ANSI or UTF-16, a hand edit leaves a
    trailing comma), this warns with the file's path and returns ``{}``: the run
    loses its meta columns and nothing else fails.
    """
    path = Path(run_dir) / META_FILE
    try:
        return _load_meta(path)
    except _BadMeta as exc:
        problem = str(exc)
    except OSError as exc:
        problem = f"could not be read ({type(exc).__name__}: {exc})"
    warnings.warn(f"{path} {problem}; treating the run's metadata as empty", stacklevel=2)
    return {}


def _move_aside(path: Path) -> Path | None:
    """Rename ``path`` to the first free ``<name>.corrupt-<n>``. None when it is already gone."""
    n = 1
    while True:
        target = path.with_name(f"{path.name}.corrupt-{n}")
        if not os.path.lexists(target):
            try:
                _retry_permission(os.rename, path, target)  # Windows refuses to replace an existing target
                return target
            except FileExistsError:
                pass
            except FileNotFoundError:
                return None
        n += 1


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a sibling temp file, fsync it, then replace ``path`` in one step."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _retry_permission(os.replace, tmp, path, attempts=10, step=0.05)  # a reader or a scanner has the target open
    finally:
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------------ the run lock
# Ctrl+C reaches Python code right after a C call returns or as a Python function
# starts, so it can land between any two steps of taking or giving back a run. Each
# step below leaves a state that _RunLock.release() undoes, and a release() that Ctrl+C
# cut short is finished by calling it again.
_OPEN_RUNS: dict[str, "_RunLock"] = {}
# Re-entrant: release() also runs from a finalizer or a signal handler, possibly on
# the thread that holds this lock.
_OPEN_RUNS_LOCK = threading.RLock()


class _RunLock:
    """This process's claim on one run: its registry entry plus the OS lock on ``.writer.lock``.

    Built holding nothing, so its owner has it in hand before acquire() takes anything;
    release() gives back whatever part of the claim is still held.
    """

    def __init__(self, run_dir: Path, run_id: str):
        self.key = os.path.normcase(str(run_dir))
        self.run_dir = run_dir
        self.path = run_dir / LOCK_FILE
        self.run_id = run_id
        self._fh: io.FileIO | None = None  # the open lock file: closing it drops the OS lock
        self._releasing = False  # a release() is under way in this thread (see release)

    @property
    def held(self) -> bool:
        """True while the lock file is open or the registry lists this claim."""
        fh = self._fh
        return (fh is not None and not fh.closed) or _OPEN_RUNS.get(self.key) is self

    def acquire(self) -> None:
        """Take the OS lock, then the registry entry. RunLockedError when another writer has the run.

        When this raises, Ctrl+C included, whatever it took is recorded here for the
        owner's release() to give back.
        """
        with _OPEN_RUNS_LOCK:
            if self.key in _OPEN_RUNS:
                raise RunLockedError(
                    f"run {self.run_id!r} already has an open TraceWriter in this process ({self.run_dir}). "
                    "One writer per run: share that writer, or close it before opening another."
                )
            self._fh = _lock_file(self.path, self.run_id)
            _OPEN_RUNS[self.key] = self

    def release(self) -> None:
        """Give back what this claim holds: close the lock file, then drop the registry entry.

        Idempotent, and safe to repeat after a call that Ctrl+C cut short: each step
        looks at what is still held, and neither can be left half done. FileIO.close
        marks the file closed in the same C call that closes the descriptor, so a
        repeat never closes a descriptor twice, and the registry check runs no Python
        code, so no signal handler can run between it and the pop. The two steps sit
        in separate finally blocks: an interrupt in one still runs the other.

        A call that re-enters a release already under way (a signal handler that closes
        the writer) returns at once and leaves the work to the outer call. Otherwise the
        handler could close the lock file, open another writer on the freed descriptor
        number, and the outer call would then unlock that new writer's lock.
        """
        if self._releasing:
            return
        try:
            self._releasing = True
            with _OPEN_RUNS_LOCK:
                try:
                    fh = self._fh
                    if fh is not None and not fh.closed:
                        try:
                            _unlock_file(fh)
                        finally:
                            fh.close()
                finally:
                    if self.key in _OPEN_RUNS and _OPEN_RUNS[self.key] is self:
                        _OPEN_RUNS.pop(self.key, None)
        finally:
            self._releasing = False


def _open_lock_file(path: Path) -> io.FileIO:
    """``path`` open for reading and writing, created when missing, never truncated.

    io.FileIO opens the file and takes the descriptor in one C call, so no Ctrl+C can
    fall between the two and leave a descriptor open that nothing will close.
    """
    try:
        return io.FileIO(path, "r+")
    except FileNotFoundError:
        pass
    try:
        return io.FileIO(path, "x+")
    except FileExistsError:  # another writer created it in between
        return io.FileIO(path, "r+")


def _lock_file(path: Path, run_id: str) -> io.FileIO:
    """``path`` open, holding its exclusive OS lock. RunLockedError when someone else holds it.

    Every way out but the return closes the file first, Ctrl+C included, so the lock is
    never left held by a descriptor that nothing owns.
    """
    fh = _open_lock_file(path)
    try:
        fd = fh.fileno()
        if os.name == "nt":
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        if exc.errno not in _LOCK_HELD_ERRNOS:
            raise
        raise RunLockedError(
            f"run {run_id!r} is already open in another TraceWriter{_holder(path)}: {path} is locked. "
            "One writer per run: wait for that writer to close, or stop it."
        ) from None
    except BaseException:
        fh.close()
        raise
    try:  # the holder note a refused writer reads
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid {os.getpid()}".ljust(31).encode("ascii") + b"\n")
    except OSError:
        pass
    except BaseException:  # Ctrl+C: give the lock back before passing the interrupt on
        fh.close()
        raise
    return fh


def _unlock_file(fh: io.FileIO) -> None:
    """Drop the OS lock ``fh`` holds. Best effort: closing ``fh`` drops it as well."""
    try:
        fd = fh.fileno()
        if os.name == "nt":
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except (OSError, ValueError):  # not locked, or fh closed meanwhile (ValueError)
        pass


def _holder(path: Path) -> str:
    """Who holds the lock, from the lock file's note: ' in this process', ' (process <pid>)' or ''."""
    try:
        words = path.read_bytes()[:64].decode("ascii", "replace").split()
    except OSError:
        return ""
    if len(words) >= 2 and words[0] == "pid" and words[1].isdigit():
        pid = int(words[1])
        return " in this process" if pid == os.getpid() else f" (process {pid})"
    return ""


# ------------------------------------------------------------------------ payload helpers
def safe_image_name(name: str) -> str:
    """``name`` reduced to ``[A-Za-z0-9_.-]`` with a ``.png`` suffix."""
    text = _UNSAFE_NAME_CHARS.sub("_", str(name))
    if text.lower().endswith(".png"):
        text = text[:-4]
    text = text.strip(".")  # no hidden files, no "..", no trailing dot
    if not text:
        text = "image"
    if text.split(".")[0].upper() in _RESERVED_NAMES:  # device names Windows will not create
        text = "_" + text
    return text[:_MAX_IMAGE_STEM] + ".png"


def _jsonable_dict(value: Any, what: str) -> dict[str, Any]:
    """``value`` (a dict, pydantic model or dataclass; None means empty) as a JSON-safe dict."""
    try:
        out = to_jsonable({} if value is None else value)
    except RecursionError:
        raise ValueError(f"{what} is nested too deeply or contains itself") from None
    except Exception as exc:  # a pydantic serializer, a failing __str__, ...
        raise ValueError(f"{what} is not JSON serializable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, dict):
        raise ValueError(f"{what} must be a dict, got {type(value).__name__}")
    return out


def _timestamp(ts: Any) -> float:
    if isinstance(ts, (bool, np.bool_)) or not isinstance(ts, (int, float, np.integer, np.floating)):
        raise ValueError(f"ts must be Unix time in seconds, got {ts!r}")
    value = float(ts)
    if not math.isfinite(value):
        raise ValueError(f"ts must be a finite number, got {ts!r}")
    return value


def _describe(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err.get("loc", ())) or "payload"
        parts.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return "; ".join(parts) or str(exc)


def _flatten_verdict(body: dict[str, Any]) -> dict[str, Any]:
    """Verdict-shaped dicts to VerdictPayload fields: evidence.metrics and
    evidence.image_paths move to the top level (top-level keys win), and the
    Verdict fields task_id, seed, checker_id and summary are always present."""
    out = dict(body)
    evidence = out.pop("evidence", None)
    if isinstance(evidence, dict):
        out.setdefault("metrics", evidence.get("metrics", {}))
        out.setdefault("image_paths", evidence.get("image_paths", []))
        rest = {k: v for k, v in evidence.items() if k not in ("metrics", "image_paths")}
        if rest:
            out["evidence"] = rest
    elif evidence is not None:
        out["evidence"] = evidence
    for name in VERDICT_EXTRA_FIELDS:
        out.setdefault(name, Verdict.model_fields[name].default)
    return out


def _png_bytes(image: Any) -> bytes:
    from PIL import Image

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if data.startswith(_PNG_SIGNATURE):
            return data
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                return _encode_png(img)
        except Exception as exc:  # Pillow raises several error types here
            raise ValueError("image bytes are neither a PNG nor an image Pillow can read") from exc
    if isinstance(image, Image.Image):
        return _encode_png(image)
    if isinstance(image, np.ndarray):
        return _encode_png(Image.fromarray(_image_array(image)))
    raise ValueError(f"save_image takes an HxWx3 uint8 array, a PIL image or PNG bytes, got {type(image).__name__}")


def _image_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    shape_ok = arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (3, 4))
    if arr.dtype != np.uint8 or not shape_ok or arr.size == 0:
        raise ValueError(
            f"image array must be HxWx3 uint8 (HxW and HxWx4 also work), got dtype {arr.dtype} shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _encode_png(img: Any) -> bytes:
    if img.mode not in _PNG_MODES:
        img = img.convert("RGBA" if img.mode.upper().endswith("A") else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_all(fh: io.FileIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = fh.write(view)
        if not written:
            raise OSError(f"no bytes written to {fh.name}")
        view = view[written:]


def _reentered(action: str, run_id: str, instead: str) -> RuntimeError:
    """The RuntimeError for a call a signal handler made while this thread was inside emit,
    marked so that emit passes it on unchanged wherever it surfaces."""
    error = RuntimeError(
        f"{action} called re-entrantly on run {run_id!r}: a signal handler ran while this thread was inside "
        f"emit; set a flag in the handler and {instead} from the main loop"
    )
    error._loopkit_reentrant = True
    return error


# ---------------------------------------------------------------------- the writer
class TraceWriter:
    """Append-only JSONL events for one run.

    ``TraceWriter(run_id, meta=None, root=None)`` writes to ``root / run_id`` when
    ``root`` is given, else to ``paths.run_dir(run_id)``. ``meta`` is merged into
    the run's ``meta.json`` (new keys win). A meta.json that is not a UTF-8 JSON
    object is first moved to ``meta.json.corrupt-<n>``, with a warning naming both
    files. Raises RunLockedError when another TraceWriter, in any process, has the
    run open.

    Attributes: ``run_id``, ``dir`` (the run directory), ``path`` (events.jsonl),
    ``meta_path``, ``images_dir``, ``lock_path``, ``next_seq`` (the seq the next
    event gets), ``resumed`` (the file already held events), ``closed`` and
    ``malformed_on_open`` (lines in the existing file that no reader accepts, such
    as a crash's partial line: the count ``read_events`` reports as ``malformed``).
    """

    def __init__(self, run_id: str, meta: Mapping[str, Any] | None = None, root: str | Path | None = None):
        self.run_id = check_run_id(run_id)
        # checked before anything is created
        new_meta = None if meta is None else _jsonable_dict(meta, f"meta for run {run_id!r}")
        if root is None:
            run_path = paths.run_dir(run_id)  # the same checks as run_path_under
        else:
            run_path = run_path_under(root, run_id)
            run_path.mkdir(parents=True, exist_ok=True)
        self.dir = Path(os.path.abspath(run_path))
        self.path = self.dir / EVENTS_FILE
        self.meta_path = self.dir / META_FILE
        self.images_dir = self.dir / IMAGES_DIR
        self.lock_path = self.dir / LOCK_FILE
        # An RLock only so that a signal handler's emit on the thread inside emit gets
        # past the lock and is refused by the _busy check instead of deadlocking.
        self._lock = threading.RLock()
        self._busy = False
        self._image_next: dict[str, int] = {}
        self._fh: io.FileIO | None = None
        self._meta: dict[str, Any] = {}
        self._next_seq, self.malformed_on_open, self._needs_newline = 0, 0, False
        self.resumed = False
        self._run_lock = _RunLock(self.dir, self.run_id)  # holds nothing yet: acquire() takes the run
        try:
            self._release_run = weakref.finalize(self, self._run_lock.release)  # frees a writer dropped unclosed
            self._run_lock.acquire()  # one writer per run, held until close()
            self._next_seq, self.malformed_on_open, self._needs_newline = _scan_events(self.path)
            self.resumed = self._next_seq > 0
            # Unbuffered: each event is one write straight to the OS, so a failed write
            # never leaves bytes in a Python buffer to be flushed later.
            self._fh = open(self.path, "ab", buffering=0)  # held open until close()
            if self._needs_newline:  # end a crash's partial last line so the next event starts on its own line
                _write_all(self._fh, b"\n")
                self._needs_newline = False
            if new_meta is None:
                self._meta = read_meta(self.dir)
            else:
                self._write_meta(new_meta)  # read, merge, write: all under the run lock
        except BaseException:
            try:
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
            finally:
                # Twice: a Ctrl+C that stops the first call, even at its first line, cannot stop both.
                try:
                    self._run_lock.release()  # gives back whatever acquire() took, however far it got
                finally:
                    self._run_lock.release()
            raise

    # ------------------------------------------------------------------ state
    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def closed(self) -> bool:
        """True once close() has finished: the events file is closed and the run released."""
        return self._fh is None and not self._run_lock.held

    @property
    def meta(self) -> dict[str, Any]:
        """A copy of the run's merged metadata."""
        return json.loads(json.dumps(self._meta))

    def __repr__(self) -> str:
        return f"TraceWriter(run_id={self.run_id!r}, next_seq={self._next_seq}, dir={str(self.dir)!r})"

    # ------------------------------------------------------------------ meta
    def update_meta(self, meta: Mapping[str, Any]) -> dict[str, Any]:
        """Merge ``meta`` into ``meta.json`` (new keys win) and return the merged dict.

        A malformed meta.json is moved aside first, as on open. A closed writer
        raises ValueError: it no longer holds the run lock.
        """
        new = _jsonable_dict(meta, f"meta for run {self.run_id!r}")
        with self._lock:
            if self._busy:
                raise _reentered("update_meta", self.run_id, "update the metadata")
            try:
                self._busy = True
                if self._fh is None:  # the run lock is gone with the writer
                    raise ValueError(f"TraceWriter for run {self.run_id!r} is closed")
                self._write_meta(new)
            finally:
                self._busy = False
        return self.meta

    def _write_meta(self, new: dict[str, Any]) -> None:
        """Merge ``new`` into meta.json. Only called while this writer holds the run lock."""
        try:
            old = _load_meta(self.meta_path)
        except _BadMeta as exc:  # never merge into {} silently: keep the old file beside the new one
            moved = _move_aside(self.meta_path)
            if moved is not None:
                warnings.warn(
                    f"{self.meta_path} {exc}; moved it to {moved} and wrote the new metadata in its place",
                    stacklevel=3,
                )
            old = {}
        except OSError as exc:  # could not read it: never write over what was not read
            exc.add_note(f"could not read {self.meta_path} to merge new metadata into it; nothing was written")
            raise
        merged = {**old, **new}
        _atomic_write_text(self.meta_path, json.dumps(merged, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        self._meta = merged

    # ------------------------------------------------------------------ events
    def emit(self, kind: str, payload: Mapping[str, Any] | None, ts: float | None = None) -> TraceEvent:
        """Validate one event, append it as a JSON line, and return it.

        Raises ValueError for an unknown ``kind``, a payload that fails the kind's
        model or cannot be made JSON safe, or a closed writer. Such an event writes
        nothing and uses no seq. The seq is reserved just before the line is
        written, so a write that fails or is interrupted (a full disk, Ctrl+C)
        leaves a gap in seq, never a duplicate.

        Not re-entrant: from a signal handler that interrupted emit on the same
        thread, emit raises RuntimeError instead of deadlocking. Set a flag in the
        handler and log from the main loop.
        """
        if not isinstance(kind, str) or kind not in PAYLOAD_MODELS:
            raise ValueError(f"unknown trace event kind {kind!r}; expected one of {', '.join(EVENT_KINDS)}")
        when = None if ts is None else _timestamp(ts)
        what = f"{kind} payload for run {self.run_id!r}"
        body = _jsonable_dict(payload, what)
        if kind == "verdict":
            body = _flatten_verdict(body)
        with self._lock:
            if self._busy:  # only the emit (or close, update_meta) this thread is already inside sets it
                raise _reentered("emit", self.run_id, "log")
            try:
                self._busy = True
                if self._fh is None:
                    raise ValueError(f"TraceWriter for run {self.run_id!r} is closed")
                if when is None:  # stamped under the lock, so default timestamps follow seq order across threads
                    when = time.time()
                seq = self._next_seq
                if seq > MAX_SEQ:  # readers drop such a line, so never write one
                    raise ValueError(f"run {self.run_id!r} has no seq left: the next would be {seq}, beyond int64")
                try:
                    event = TraceEvent(run_id=self.run_id, ts=when, seq=seq, kind=kind, payload=body)
                except ValidationError as exc:
                    raise ValueError(f"invalid {what}: {_describe(exc)}") from exc
                try:
                    line = dumps(event.model_dump(mode="json")).encode("utf-8") + b"\n"
                except Exception as exc:
                    if getattr(exc, "_loopkit_reentrant", False):  # a signal handler's refused call, not the payload
                        raise
                    raise ValueError(f"{what} is not JSON serializable: {type(exc).__name__}: {exc}") from exc
                if self._needs_newline:  # an earlier write failed partway: end that partial line first
                    line = b"\n" + line
                self._next_seq = seq + 1  # reserved: from here on a failure leaves a gap, never a duplicate
                self._needs_newline = True  # until the whole line is down
                _write_all(self._fh, line)
                self._fh.flush()
                self._needs_newline = False
            finally:
                self._busy = False
        return event

    def model_call(self, **payload: Any) -> TraceEvent:
        return self.emit("model_call", payload)

    def node_start(self, node_id: str, skill: str, params: Mapping[str, Any] | None = None, attempt: int = 0,
                   **extra: Any) -> TraceEvent:
        body = {"node_id": node_id, "skill": skill, "params": {} if params is None else params, "attempt": attempt}
        return self.emit("node_start", {**body, **extra})

    def node_end(self, node_id: str, skill: str, ok: bool, outputs: Mapping[str, Any] | None = None,
                 error: str | BaseException | None = None, sim_seconds: float = 0.0, **extra: Any) -> TraceEvent:
        body = {"node_id": node_id, "skill": skill, "ok": ok, "outputs": {} if outputs is None else outputs,
                "error": error, "sim_seconds": sim_seconds}
        return self.emit("node_end", {**body, **extra})

    def verdict(self, verdict: Verdict | Mapping[str, Any], **extra: Any) -> TraceEvent:
        """A Verdict (or a dict) flattened to passed, score, failure_class, failing_node,
        metrics, image_paths, task_id, seed, checker_id and summary. ``extra`` wins."""
        if isinstance(verdict, BaseModel):
            try:  # Python mode: numpy values in metrics are made JSON safe by emit
                data = verdict.model_dump(mode="python")
            except Exception as exc:
                raise ValueError(
                    f"verdict payload for run {self.run_id!r} is not JSON serializable: {type(exc).__name__}: {exc}"
                ) from exc
        elif isinstance(verdict, Mapping):
            data = dict(verdict)
        else:
            raise ValueError(f"verdict must be a loopkit.schemas.Verdict or a dict, got {type(verdict).__name__}")
        return self.emit("verdict", {**data, **extra})

    def patch(self, round: int, graph_before_hash: str | None, graph_after_hash: str | None,
              failure_summary: str, **extra: Any) -> TraceEvent:
        body = {"round": round, "graph_before_hash": graph_before_hash, "graph_after_hash": graph_after_hash,
                "failure_summary": failure_summary}
        return self.emit("patch", {**body, **extra})

    def intervention(self, reason: str, actor: str = "human", **extra: Any) -> TraceEvent:
        return self.emit("intervention", {"reason": reason, "actor": actor, **extra})

    def budget_stop(self, scope: str, cap_usd: float, spent_usd: float, reason: str, **extra: Any) -> TraceEvent:
        body = {"scope": scope, "cap_usd": cap_usd, "spent_usd": spent_usd, "reason": reason}
        return self.emit("budget_stop", {**body, **extra})

    # ------------------------------------------------------------------ images
    def save_image(self, name: str, image: Any) -> str:
        """Write ``images/<safe name>.png`` and return its absolute path.

        ``image`` is an HxWx3 uint8 array (HxW and HxWx4 also work), a PIL image
        or PNG bytes. An existing file is never overwritten: the name gets a
        numeric suffix (``_1``, ``_2``, ...) instead.
        """
        data = _png_bytes(image)
        filename = safe_image_name(name)
        stem = filename[: -len(".png")]
        self.images_dir.mkdir(parents=True, exist_ok=True)
        n = self._image_next.get(stem, 0)
        while True:
            target = self.images_dir / (filename if n == 0 else f"{stem}_{n}.png")
            try:
                fh = open(target, "xb")  # exclusive create: the no-overwrite guarantee
            except FileExistsError:
                n += 1
                continue
            try:
                with fh:
                    fh.write(data)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            self._image_next[stem] = n + 1
            return str(target)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Flush and close the events file and release the run lock. Safe to call twice.

        A Ctrl+C that lands inside close() still releases the run. One that stops it
        before it starts leaves the writer open (``closed`` stays False), and calling
        close() again finishes the job.
        """
        with self._lock:
            if self._busy:
                raise _reentered("close", self.run_id, "close the writer")
            try:
                self._busy = True
                fh, self._fh = self._fh, None
                if fh is not None:
                    try:
                        fh.flush()
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                    finally:
                        fh.close()
            finally:
                self._busy = False
                if self._fh is None:  # the events file is let go: let go of the run as well
                    # Directly, not through the finalizer, which is spent once called, done or not. Twice:
                    # a Ctrl+C that stops the first call, even at its first line, cannot stop both.
                    try:
                        self._run_lock.release()
                    finally:
                        self._run_lock.release()
                    self._release_run.detach()  # only now: until the run is released, dropping the writer frees it

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
