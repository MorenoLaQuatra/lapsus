# SPDX-License-Identifier: Apache-2.0
"""Offline fixtures: hand-written sentences pre-parsed into JSON, a small
Morph-it! subset, no network, no model downloads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))

from lapsus.lang.en import English  # noqa: E402
from lapsus.lang.it import Italian  # noqa: E402
from lapsus.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def en():
    return English()


@pytest.fixture(scope="session")
def it():
    return Italian(FIX / "morphit_subset.txt")


@pytest.fixture(scope="session")
def parses():
    return {lg: json.loads((FIX / f"parses_{lg}.json").read_text(encoding="utf-8")) for lg in ("en", "it")}


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    for k in c["paths"]:
        c["paths"][k] = str(tmp_path / k)
    return c


def make_items(parses: list[dict], lang: str, copies: int = 1) -> list[dict]:
    """Pool-like rows built from fixture parses (copies get distinct source ids)."""
    items = []
    for c in range(copies):
        for k, p in enumerate(parses):
            items.append({"lang": lang, "text": p["text"], "source": "fixture", "source_id": f"{c}-{k}",
                          "source_author": "tester", "license": "CC0-1.0", "parse": p, "tokens": p["tokens"]})
    return items
