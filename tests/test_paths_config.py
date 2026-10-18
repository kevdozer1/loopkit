from __future__ import annotations

import re
from pathlib import Path

import pytest

from loopkit import config, paths


def test_data_root_comes_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    run = paths.run_dir("abc123")
    assert run == tmp_path / "data" / "runs" / "abc123" and run.is_dir()
    made = paths.ensure_data_dirs()
    assert {p.name for p in made} == {"runs", "bakeoff", "videos", "cache"}
    assert paths.ledger_path().parent == tmp_path / "data"


def test_data_root_documented_default(monkeypatch):
    monkeypatch.delenv("LOOPKIT_DATA", raising=False)
    monkeypatch.setattr(config, "_DOTENV", {})
    default = paths._documented_default()
    assert default, ".env.example must document LOOPKIT_DATA"
    assert paths.data_root() == Path(default)


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "x:y", "../up"])
def test_run_dir_rejects_paths(monkeypatch, tmp_path, bad):
    monkeypatch.setenv("LOOPKIT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        paths.run_dir(bad)


def test_disk_floor(tmp_path):
    free = paths.check_disk_floor(min_gb=0.0, paths=[tmp_path])
    assert free and all(gb > 0 for gb in free.values())
    with pytest.raises(paths.DiskFloorError):
        paths.check_disk_floor(min_gb=10**9, paths=[tmp_path])


def test_parse_dotenv_is_crlf_safe(tmp_path):
    text = "# comment\r\nA=1\r\nB = two words \r\n\r\nC=\"quoted\"\r\nexport D='x'\r\nEMPTY=\r\nbad line\r\n"
    path = tmp_path / ".env"
    path.write_bytes(text.encode("utf-8"))
    values = config.dotenv_values(path)
    assert values == {"A": "1", "B": "two words", "C": "quoted", "D": "x", "EMPTY": ""}
    assert not any("\r" in v for v in values.values())


def test_env_precedence_and_empty(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV", {"LK_TEST_A": "from_file", "LK_TEST_B": "from_file", "LK_TEST_E": ""})
    monkeypatch.setenv("LK_TEST_A", "from_process")
    monkeypatch.setenv("LK_TEST_B", "")
    assert config.env("LK_TEST_A") == "from_process"
    assert config.env("LK_TEST_B") == "from_file"  # empty process value counts as unset
    assert config.env("LK_TEST_E", "dflt") == "dflt"
    assert config.env("LK_TEST_MISSING") is None


def test_load_secret_names_the_variable(monkeypatch):
    monkeypatch.setattr(config, "_DOTENV", {})
    monkeypatch.delenv("LK_TEST_KEY", raising=False)
    with pytest.raises(config.MissingCredentialError) as err:
        config.load_secret(["LK_TEST_KEY", "LK_TEST_KEY_ALT"], "Test provider")
    assert "LK_TEST_KEY" in str(err.value)
    secret = "sk-test-" + "9" * 20
    monkeypatch.setenv("LK_TEST_KEY_ALT", secret)
    assert config.load_secret(["LK_TEST_KEY", "LK_TEST_KEY_ALT"], "Test provider") == secret


def test_models_yaml_parses_with_required_fields():
    cfg = config.load_models()
    assert cfg.currency == "USD" and cfg.unit == "usd_per_million_tokens"
    required = {"display", "adapter", "id", "price_in", "price_out", "vision", "roles", "verified", "source", "accessed"}
    import yaml

    raw = yaml.safe_load(config.MODELS_YAML.read_text(encoding="utf-8"))["models"]
    for key, fields in raw.items():
        missing = required - set(fields)
        assert not missing, f"models.yaml entry {key} is missing {sorted(missing)}"
    for key, entry in cfg.models.items():
        assert entry.key == key
        assert entry.adapter in config.ADAPTERS
        assert entry.roles
        if entry.verified:
            assert entry.id and entry.priced, key
    assert cfg.get("fake").adapter == "fake"
    with pytest.raises(KeyError):
        cfg.get("no_such_model")


def test_bakeoff_yaml_matches_models_and_seed_rules():
    models = config.load_models()
    bake = config.load_bakeoff()
    for key in bake.models + bake.optional_models:
        assert key in models.models, key
        assert bake.caps_usd[key] > 0
    # CLAUDE.md rule 11: bake-off training 1000-1009, held-out 2000-2019
    assert bake.train_seeds.seeds() == list(range(1000, 1010))
    assert bake.heldout_seeds.seeds() == list(range(2000, 2020))
    assert sum(bake.caps_usd[m] for m in bake.models) <= bake.total_cap_usd + 1e-9


def test_no_literal_data_drive_in_code():
    """CLAUDE.md rule 4: no data-drive path is written literally in committed code."""
    needle = re.compile(re.escape("D" + ":") + r"[\\/]")
    offenders = []
    for root in (paths.REPO_ROOT / "loopkit", paths.REPO_ROOT / "tests"):
        for py in root.rglob("*.py"):
            for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if needle.search(line):
                    offenders.append(f"{py.relative_to(paths.REPO_ROOT)}:{n}")
    assert not offenders, offenders
