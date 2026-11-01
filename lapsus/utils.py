# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: configuration, logging, deterministic RNGs and on-disk caches."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_SOURCES = ROOT / "sources.yaml"

# Characters that must never appear in any file we write (em dash, en dash).
FORBIDDEN_DASHES = (chr(0x2014), chr(0x2013))

log = logging.getLogger("lapsus")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("stanza", "urllib3", "httpx", "filelock", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, {k: v for k, v in overrides.items() if v is not None})
    # Resolve paths relative to the repository root unless absolute.
    for k, p in list(cfg["paths"].items()):
        cfg["paths"][k] = str(p if os.path.isabs(p) else (ROOT / p))
    return cfg


def load_sources(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_SOURCES, encoding="utf-8") as f:
        return yaml.safe_load(f)


def languages_for(lang: str) -> list[str]:
    return ["en", "it"] if lang == "both" else [lang]


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def rng_for(*parts: Any) -> random.Random:
    """Deterministic RNG independent of PYTHONHASHSEED (seeded from a sha1 digest)."""
    return random.Random(int(stable_hash(*parts)[:16], 16))


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)
    assert_no_dashes(text, str(path))
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_text(path: str | os.PathLike, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    assert_no_dashes(text, str(path))
    Path(path).write_text(text, encoding="utf-8")


def assert_no_dashes(text: str, where: str = "") -> None:
    for d in FORBIDDEN_DASHES:
        if d in text:
            raise ValueError(f"Forbidden dash U+{ord(d):04X} found in {where or 'text'}")


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class KVCache:
    """A tiny thread-safe sqlite key-value store with JSON values.

    Used for LanguageTool results, Stanza parses and LM scores so that the slow
    steps are computed once per (tool version, text).
    """

    def __init__(self, path: str | os.PathLike):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(keys)
        out: dict[str, Any] = {}
        with self._lock:
            for i in range(0, len(keys), 900):
                part = keys[i : i + 900]
                q = "SELECT k, v FROM kv WHERE k IN (%s)" % ",".join("?" * len(part))
                for k, v in self._conn.execute(q, part):
                    out[k] = json.loads(v)
        return out

    def set_many(self, items: dict[str, Any]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()],
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def package_version(dist: str) -> str:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return "not installed"


def ensure_java_on_path() -> None:
    """LanguageTool needs Java 17+. Use the conda-provided JDK if java is not on PATH."""
    import shutil
    import subprocess
    import sys

    java = shutil.which("java")
    if java:
        try:  # macOS ships a /usr/bin/java stub that fails when no JDK is installed
            subprocess.run([java, "-version"], check=True, capture_output=True, timeout=30)
            return
        except Exception:
            pass
    candidates = [Path(sys.prefix) / "lib" / "jvm" / "bin", Path(sys.prefix) / "bin"]
    if os.environ.get("JAVA_HOME"):
        candidates.insert(0, Path(os.environ["JAVA_HOME"]) / "bin")
    for c in candidates:
        if (c / "java").exists():
            os.environ["PATH"] = f"{c}{os.pathsep}{os.environ.get('PATH', '')}"
            return
