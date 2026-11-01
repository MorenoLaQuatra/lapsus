# SPDX-License-Identifier: Apache-2.0
import pytest

from lapsus.filters import (dedup, filter_a, filter_b, length_bucket, lm_filter, lt_match_is_error, name_cap,
                            normalize_text, normalized_key)
from lapsus.utils import load_config

FCFG = load_config()["filters"]


def test_normalize_text():
    assert normalize_text("  Don’t   go.  ") == "Don't go."


@pytest.mark.parametrize("text,why", [
    ("Hi.", "length"),
    ("This is one sentence. This is another one here.", "more than one sentence"),
    ("This is a sentence that is far too long because it keeps going on and on and on without any end at all.", "length"),
])
def test_filter_a_rejects(text, why, en):
    t, reason = filter_a(text, en, FCFG)
    assert t is None and reason == why


@pytest.mark.parametrize("text,why", [
    ("I bought 3 apples yesterday.", "digit"),
    ("Visit www.example.com for more.", "url"),
    ("He said \"hello\" to me.", "quote"),
    ("I like this (a lot) really.", "bracket"),
    ("I love it so much \U0001F600 today.", "emoji"),
    ("I work for NASA in the city.", "all-caps"),
    ("Tom and Mary went to Paris.", "proper nouns"),
])
def test_filter_b_rejects(text, why, en):
    assert filter_b(text, en, FCFG, set()) == why


def test_filter_b_accepts(en, it):
    assert filter_b("I would like a cup of tea.", en, FCFG, set()) == ""
    assert filter_b("Tom is reading a book.", en, FCFG, set()) == ""
    assert filter_b("Ho visto l'uomo della stazione.", it, FCFG, set()) == ""


def test_lt_ignores_style_and_casing():
    assert not lt_match_is_error(["UPPERCASE_SENTENCE_START", "CASING", "typographical"])
    assert not lt_match_is_error(["X", "STYLE", "style"])
    assert lt_match_is_error(["HE_VERB_AGR", "GRAMMAR", "grammar"])
    assert lt_match_is_error(["MORFOLOGIK_RULE_EN_US", "TYPOS", "misspelling"])


def test_lm_filter_drops_lowest_fraction_per_bucket():
    rows = [{"tokens": ["w"] * 6, "lm": float(i)} for i in range(100)] + \
           [{"tokens": ["w"] * 16, "lm": float(i)} for i in range(40)]
    keep, drop, thr = lm_filter(rows, FCFG)
    assert len(drop) == 5 + 2
    assert min(r["lm"] for r in keep if len(r["tokens"]) == 6) == 5.0
    assert length_bucket(6, FCFG["length_buckets"]) != length_bucket(16, FCFG["length_buckets"])


def _row(text, lemmas, sid, lic="CC-BY-2.0"):
    return {"text": text, "source": "t", "source_id": sid, "license": lic,
            "parse": {"info": [{"lemma": l, "upos": "NOUN"} for l in lemmas], "tokens": text.split()}}


def test_dedup_exact_normalized_near():
    rows = [
        _row("The cat sleeps on the mat.", "the cat sleep on the mat".split(), "1"),
        _row("The cat sleeps on the mat.", "the cat sleep on the mat".split(), "2", "CC0-1.0"),
        _row("the cat sleeps on the mat", "the cat sleep on the mat".split(), "3"),
        _row("The cat slept on the mat.", "the cat sleep on the mat".split(), "4"),
        _row("A bird sings in the tall green tree.", "a bird sing in the tall green tree".split(), "5"),
    ]
    keep, dropped = dedup(rows, FCFG, 42, "t")
    assert len(dropped["g_dedup_exact"]) == 1
    assert len(dropped["g_dedup_normalized"]) == 1
    assert len(dropped["g_dedup_near"]) == 1
    assert {r["source_id"] for r in keep} == {"2", "5"}  # the permissive duplicate wins
    assert normalized_key("Hello, World!") == "hello world"


def test_name_cap(en):
    rows = []
    for i in range(200):
        name = "tom" if i < 100 else None
        info = [{"upos": "PROPN", "lemma": "Tom"}] if name else [{"upos": "NOUN", "lemma": "cat"}]
        rows.append({"source": "t", "source_id": str(i), "parse": {"info": info, "tokens": ["x"]}})
    keep, drop = name_cap(rows, en, 0.02, 42, "t")
    assert sum(1 for r in keep if r["parse"]["info"][0]["upos"] == "PROPN") == 4  # 2 percent of 200
