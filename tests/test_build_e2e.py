# SPDX-License-Identifier: Apache-2.0
"""End-to-end build on fixture pools with stub tools (no LanguageTool, Stanza or LM)."""

from __future__ import annotations

import pytest
from datasets import DatasetDict

import lapsus.build as build
import lapsus.lang as lang_registry
from lapsus.corruptions import views_consistent
from lapsus.export import write_config
from lapsus.licensing import LicenseError
from tests.conftest import make_items


CLEAN: set[str] = set()


class _Scorer:
    name = "stub"

    def __init__(self, clean):
        self.clean = set(clean)

    def score(self, texts):
        return [(0.0 if t in self.clean or t in CLEAN else -10.0, 1) for t in texts]


class _LT:
    def flags(self, texts):
        return [True] * len(texts)

    def close(self):
        pass


def _unique_items(parses, lg, copies):
    """Copies of the fixture sentences with a distinct last word, so clean texts never repeat."""
    import copy

    from lapsus.lang.base import split_core

    items = make_items(parses, lg, copies)
    for k, it in enumerate(items):
        p = copy.deepcopy(it["parse"])
        core, punct = split_core(p["tokens"][-1])
        p["tokens"][-1] = core + "".join(chr(97 + (k // 26 ** j) % 26) for j in range(3)) + punct
        p["text"] = " ".join(p["tokens"]) if lg == "en" else it["parse"]["text"].rsplit(" ", 1)[0] + " " + p["tokens"][-1]
        it.update(parse=p, text=p["text"], tokens=p["tokens"])
    return items


def _pools(parses, lg, copies, permissive_every=1):
    main = _unique_items(parses[lg], lg, copies)
    CLEAN.update(r["text"] for r in main)
    for k, r in enumerate(main):
        r["license"] = "CC0-1.0" if k % permissive_every == 0 else "CC-BY-2.0"
    n_perm = sum(r["license"] == "CC0-1.0" for r in main)
    return {"main": main, "ood": [], "natural": [],
            "report": {"main": {"permissive_pool": n_perm, "total_pool": len(main), "rejected": {}, "per_license": {}},
                       "ood": {}, "natural": {}}}


@pytest.fixture()
def small_cfg(cfg):
    cfg["splits"].update(train=40, validation=10, test=10, test_unseen_lemmas=10, surplus=0.3)
    cfg["verify"].update(calibration_per_op=20, delta=0.5)
    cfg["min_sentences"] = 50
    return cfg


@pytest.fixture()
def stub_tools(monkeypatch, parses, en, it):
    clean = {p["text"] for lg in parses for p in parses[lg]}

    class Tools:
        def __init__(self, cfg, lang):
            self.d = {"lm": _Scorer(clean), "lt": _LT()}

        def __getitem__(self, k):
            return self.d[k]

        def close(self):
            pass

    monkeypatch.setattr(build, "Tools", Tools)
    monkeypatch.setattr(lang_registry, "_INSTANCES", {"en": en, "it": it})


@pytest.mark.parametrize("lg", ["en", "it"])
def test_build_language_end_to_end(lg, parses, small_cfg, stub_tools, tmp_path):
    pools = _pools(parses, lg, copies=12)
    res = build.build_language(small_cfg, {"sources": []}, lg, pools, "auto")
    assert res["decision"]["mode"] == "apache-strict"
    assert res["license"]["hf"] == "apache-2.0"
    rows = res["rows"]
    assert set(rows) == {"train", "validation", "test", "test_unseen_lemmas"}
    for split, rs in rows.items():
        for r in rs:
            assert views_consistent(r) and r["split"] == split
        n = len(rs)
        n_ok = sum(r["binary"] == "ok" for r in rs)
        assert n // 2 <= n_ok < n  # half clean (more only if a corruption role fell short)
    corrupted = [r for rs in rows.values() for r in rs if r["binary"] == "not_ok"]
    assert {r["multiclass"] for r in corrupted} >= {"word_order", "repeated_word", "multiple"}
    counts = write_config(tmp_path, lg, rows)
    dd = DatasetDict.load_from_disk(str(tmp_path / "hf" / lg))
    assert {k: len(v) for k, v in dd.items()} == counts


def test_auto_falls_back_to_source_licenses(parses, small_cfg, stub_tools):
    pools = _pools(parses, "it", copies=12, permissive_every=10)  # tiny permissive pool
    res = build.build_language(small_cfg, {"sources": []}, "it", pools, "auto")
    assert res["decision"]["mode"] == "source-licenses"
    assert res["license"]["hf"] == "cc-by-2.0"
    assert {r["license"] for rs in res["rows"].values() for r in rs} >= {"CC-BY-2.0"}


def test_apache_strict_uses_only_permissive_rows(parses, small_cfg, stub_tools):
    pools = _pools(parses, "en", copies=24, permissive_every=2)
    small_cfg["min_sentences"] = 60
    res = build.build_language(small_cfg, {"sources": []}, "en", pools, "apache-strict")
    assert {r["license"] for rs in res["rows"].values() for r in rs} == {"CC0-1.0"}


def test_apache_strict_fails_with_small_pool(parses, small_cfg, stub_tools):
    pools = _pools(parses, "en", copies=12, permissive_every=10)
    with pytest.raises(LicenseError):
        build.build_language(small_cfg, {"sources": []}, "en", pools, "apache-strict")


def test_build_is_deterministic(parses, small_cfg, stub_tools):
    a = build.build_language(small_cfg, {"sources": []}, "en", _pools(parses, "en", 12), "auto")
    b = build.build_language(small_cfg, {"sources": []}, "en", _pools(parses, "en", 12), "auto")
    assert a["rows"] == b["rows"]
