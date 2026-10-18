"""Paths and the disk floor. Every large output goes under ``LOOPKIT_DATA``.

No data-drive path is written literally in loopkit's code. The data root comes
from, in order: the process environment, the repo ``.env``, then the documented
default in the committed ``.env.example``. Everything else asks this module for
its directory, so moving the data root is one line in ``.env``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ENV_VAR = "LOOPKIT_DATA"
DISK_FLOOR_GB = 10.0
DATA_SUBDIRS = ("runs", "bakeoff", "videos", "cache")


class DiskFloorError(RuntimeError):
    """Free space on a drive loopkit writes to fell below the floor."""


def _documented_default() -> str | None:
    """``LOOPKIT_DATA`` as written in ``.env.example`` (the documented default)."""
    example = REPO_ROOT / ".env.example"
    if not example.is_file():
        return None
    for line in example.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(DATA_ENV_VAR + "="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def data_root(create: bool = False) -> Path:
    """The external data root (``LOOPKIT_DATA``)."""
    from .config import env

    raw = env(DATA_ENV_VAR) or _documented_default()
    if not raw:
        raise RuntimeError(
            f"{DATA_ENV_VAR} is not set. Add {DATA_ENV_VAR}=... to {REPO_ROOT / '.env'} or the environment."
        )
    root = Path(raw.strip())
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _sub(name: str, create: bool = True) -> Path:
    path = data_root() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir(create: bool = True) -> Path:
    return _sub("runs", create)


def bakeoff_dir(create: bool = True) -> Path:
    return _sub("bakeoff", create)


def videos_dir(create: bool = True) -> Path:
    return _sub("videos", create)


def cache_dir(create: bool = True) -> Path:
    return _sub("cache", create)


def run_dir(run_id: str, create: bool = True) -> Path:
    """``LOOPKIT_DATA/runs/<run_id>``: one directory per traced run."""
    if not run_id or any(c in run_id for c in '\\/:*?"<>|'):
        raise ValueError(f"run_id {run_id!r} is not a plain directory name")
    path = runs_dir(create) / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_path() -> Path:
    """The spend ledger, shared by every session and the bake-off."""
    return data_root(create=True) / "spend_ledger.jsonl"


def ensure_data_dirs() -> list[Path]:
    """Create ``LOOPKIT_DATA`` and its standard subfolders."""
    root = data_root(create=True)
    made = []
    for name in DATA_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        made.append(path)
    return made


def menagerie_dir() -> Path:
    """The Menagerie clone (gitignored, under ``third_party/``)."""
    return REPO_ROOT / "third_party" / "mujoco_menagerie"


def panda_dir() -> Path:
    return menagerie_dir() / "franka_emika_panda"


def free_space_gb(paths: list[Path] | None = None) -> dict[str, float]:
    """Free gigabytes on the drives holding the repo and the data root."""
    if paths is None:
        paths = [REPO_ROOT]
        try:
            paths.append(data_root())
        except RuntimeError:
            pass
    out: dict[str, float] = {}
    for path in paths:
        anchor = Path(path).resolve().anchor
        if not anchor or anchor in out or not Path(anchor).exists():
            continue
        out[anchor] = round(shutil.disk_usage(anchor).free / 1024**3, 2)
    return out


def check_disk_floor(min_gb: float = DISK_FLOOR_GB, paths: list[Path] | None = None) -> dict[str, float]:
    """Free space per drive. Raises DiskFloorError when any drive is under ``min_gb``."""
    free = free_space_gb(paths)
    low = {drive: gb for drive, gb in free.items() if gb < float(min_gb)}
    if low:
        detail = ", ".join(f"{d} {gb:.1f} GB" for d, gb in sorted(low.items()))
        raise DiskFloorError(f"disk floor {min_gb:.0f} GB crossed: {detail} free. Stop and report.")
    return free
