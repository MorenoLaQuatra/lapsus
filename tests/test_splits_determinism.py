# SPDX-License-Identifier: Apache-2.0
import random

from lapsus.corruptions import corrupt, corrupt_multi
from lapsus.lang.en import English
from lapsus.splits import assign_splits, check_leakage
from lapsus.utils import rng_for


def _synthetic(n=900):
    rng = random.Random(0)
    vocab = ["w" + "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3)) for i in range(400)]
    rows = []
    for i in range(n):
        lemmas = rng.sample(vocab, 3)
        info = [{"upos": "NOUN", "lemma": l} for l in lemmas]
        rows.append({"lang": "en", "source": "syn", "source_id": str(i), "text": f"s{i}", "license": "CC0-1.0",
                     "parse": {"info": info, "tokens": lemmas}})
    return rows


SIZES = {"train": 300, "validation": 60, "test": 60, "test_unseen_lemmas": 60}


def test_unseen_lemmas_and_no_group_leakage():
    lang = English()
    rows = _synthetic()
    out, info = assign_splits(rows, SIZES, lang, 42, 0.0)
    reserved = set(info["reserved_lemmas"])
    assert reserved and len(out["test_unseen_lemmas"]) == 60
    for s in ("train", "validation", "test"):
        for r in out[s]:
            assert not (lang.content_lemmas(r["parse"]) & reserved)
    for r in out["test_unseen_lemmas"]:
        assert lang.content_lemmas(r["parse"]) & reserved
    ds = {s: [{"lang": "en", "source": r["source"], "source_id": r["source_id"], "clean_text": r["text"],
               "id": r["source_id"]} for r in rs] for s, rs in out.items()}
    lookup = {r["source_id"]: lang.content_lemmas(r["parse"]) for r in rows}
    assert check_leakage(ds, reserved, lookup) == []


def test_leakage_detector_catches_leaks():
    ds = {"train": [{"lang": "en", "source": "a", "source_id": "1", "clean_text": "x", "id": "1"}],
          "test": [{"lang": "en", "source": "a", "source_id": "1", "clean_text": "x", "id": "1b"}]}
    problems = check_leakage(ds)
    assert any("source sentence" in p for p in problems) and any("clean text" in p for p in problems)


def test_forced_splits_respected():
    lang = English()
    rows = _synthetic()
    forced = {"5": "test", "6": "validation"}
    out, _ = assign_splits(rows, SIZES, lang, 42, 0.0, forced)
    where = {r["source_id"]: s for s, rs in out.items() for r in rs}
    for sid, s in forced.items():
        assert where.get(sid) in (s, None)


def test_splits_deterministic():
    lang = English()
    a, _ = assign_splits(_synthetic(), SIZES, lang, 42, 0.0)
    b, _ = assign_splits(_synthetic(), SIZES, lang, 42, 0.0)
    c, _ = assign_splits(_synthetic(), SIZES, lang, 7, 0.0)
    ids = lambda o: {s: [r["source_id"] for r in rs] for s, rs in o.items()}  # noqa: E731
    assert ids(a) == ids(b) and ids(a) != ids(c)


def test_corruption_deterministic(parses, en, it):
    for lang, ps in ((en, parses["en"]), (it, parses["it"])):
        for p in ps:
            for op in ("word_order", "agreement", "wrong_function_word"):
                d1 = corrupt(p, lang, [op], rng_for(42, p["text"], op))
                d2 = corrupt(p, lang, [op], rng_for(42, p["text"], op))
                assert (d1 is None and d2 is None) or (d1.tokens == d2.tokens and d1.tags == d2.tags)
            m1 = corrupt_multi(p, lang, 2, rng_for(1, p["text"]))
            m2 = corrupt_multi(p, lang, 2, rng_for(1, p["text"]))
            assert (m1 is None and m2 is None) or m1.tokens == m2.tokens
    outs = {tuple(corrupt(parses["en"][4], en, ["word_order"], rng_for(s)).tokens) for s in range(10)}
    assert len(outs) > 1  # different seeds give different corruptions
