# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from lapsus import OPS
from lapsus.corruptions import Draft, OP_FUNCS, corrupt, corrupt_multi, label_views, views_consistent
from lapsus.lang.base import split_core

EXPECTED_TAGGED = {"word_order": 2, "missing_word": 1, "repeated_word": 1, "agreement": 1, "wrong_function_word": 1}


def _all_results(parse, lang, op, seeds=12):
    out = []
    for s in range(seeds):
        d = corrupt(parse, lang, [op], random.Random(s))
        if d is not None:
            out.append(d)
    return out


@pytest.mark.parametrize("lg", ["en", "it"])
@pytest.mark.parametrize("op", OPS)
def test_each_op_tag_alignment(lg, op, parses, en, it):
    lang = en if lg == "en" else it
    produced = 0
    for p in parses[lg]:
        for d in _all_results(p, lang, op):
            produced += 1
            assert len(d.tokens) == len(d.tags) == len(d.origin) == len(d.src)
            assert set(d.tags) <= {"O", op}
            assert sum(t != "O" for t in d.tags) == EXPECTED_TAGGED[op]
            assert lang.tokenize(lang.detokenize(d.tokens)) == d.tokens
            views = label_views(d.tags, len(d.ops))
            assert views_consistent({"tokens": d.tokens, **views})
            assert views["multiclass"] == op and views["binary"] == "not_ok"
            n_clean = len(p["tokens"])
            if op == "missing_word":
                assert len(d.tokens) == n_clean - 1
            elif op == "repeated_word":
                assert len(d.tokens) == n_clean + 1
            else:
                assert len(d.tokens) == n_clean
    assert produced > 0, f"{op} never applied on {lg} fixtures"


@pytest.mark.parametrize("lg", ["en", "it"])
def test_missing_word_tags_token_after_gap(lg, parses, en, it):
    lang = en if lg == "en" else it
    for p in parses[lg]:
        for s in range(10):
            d = corrupt(p, lang, ["missing_word"], random.Random(s))
            if d is None:
                continue
            k = d.ops[0]["clean_indices"][0]  # deleted clean index
            tagged = d.tags.index("missing_word")
            if k < len(p["tokens"]) - 1:
                # the tagged token is the clean token that followed the gap
                assert d.origin[tagged] == k + 1
            else:
                assert tagged == len(d.tokens) - 1


@pytest.mark.parametrize("lg", ["en", "it"])
def test_repeated_word_duplicate_is_tagged(lg, parses, en, it):
    lang = en if lg == "en" else it
    for p in parses[lg]:
        d = corrupt(p, lang, ["repeated_word"], random.Random(3))
        if d is None:
            continue
        j = d.tags.index("repeated_word")
        assert j > 0 and d.origin[j] is None
        assert split_core(d.tokens[j])[0].lower() == split_core(d.tokens[j - 1])[0].lower()


def test_word_order_keeps_punctuation_in_place(parses, en):
    for p in parses["en"]:
        d = corrupt(p, en, ["word_order"], random.Random(1))
        if d is not None:
            assert d.tokens[-1].endswith(".")


def test_italian_word_order_constraints(parses, it):
    for p in parses["it"]:
        info = p["info"]
        for s in range(15):
            d = corrupt(p, it, ["word_order"], random.Random(s))
            if d is None:
                continue
            i, j = sorted(d.ops[0]["clean_indices"])
            pair = {info[i]["upos"], info[j]["upos"]}
            assert pair != {"ADJ", "NOUN"} and pair != {"NOUN"}
            assert not (info[i]["head"] == info[j]["head"] and info[i]["deprel"].split(":")[0] in
                        {"nsubj", "obj", "iobj", "obl"} and info[j]["deprel"].split(":")[0] in {"nsubj", "obj", "iobj", "obl"})


@pytest.mark.parametrize("lg", ["en", "it"])
def test_multi_corruption_distinct_nonoverlapping(lg, parses, en, it):
    lang = en if lg == "en" else it
    produced = 0
    for p in parses[lg]:
        for k in (2, 3):
            d = corrupt_multi(p, lang, k, random.Random(7 + k))
            if d is None:
                continue
            produced += 1
            ops = [o["op"] for o in d.ops]
            assert len(ops) == k == len(set(ops))
            present = {t for t in d.tags if t != "O"}
            assert present == set(ops)
            views = label_views(d.tags, len(d.ops))
            assert views["multiclass"] == "multiple" and views_consistent({"tokens": d.tokens, **views})
    assert produced > 0


def test_ops_never_touch_tagged_tokens(parses, en):
    p = parses["en"][0]
    d = Draft.from_parse(p)
    d.tags = ["word_order"] * len(d.tags)  # everything already tagged
    for op, fn in OP_FUNCS.items():
        assert fn(d, p, en, random.Random(0)) is None, op
