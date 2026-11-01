# SPDX-License-Identifier: Apache-2.0
"""Verification must reject accidentally valid corruptions."""

import random

from lapsus.build import corrupt_split, mix_targets
from lapsus.corruptions import Draft, corrupt
from lapsus.filters import normalized_key
from lapsus.lm import TableScorer
from lapsus.verify import OpStats, calibrate_delta, lm_diffs, structural_check
from tests.conftest import make_items


def test_rejects_corruption_equal_to_clean_pool_sentence(parses, en):
    p = parses["en"][0]
    d = corrupt(p, en, ["repeated_word"], random.Random(0))
    assert structural_check(d, p, en, set()) == ""
    pool = {normalized_key(en.detokenize(d.tokens))}  # the "corrupted" text is a real clean sentence
    assert structural_check(d, p, en, pool) == "in_clean_pool"


def test_rejects_identical_to_clean(parses, en):
    p = parses["en"][0]
    d = Draft.from_parse(p)
    d.ops = [{"op": "word_order"}]
    d.tags[0] = d.tags[1] = "word_order"
    assert structural_check(d, p, en, set()) == "same_as_clean"


def test_rejects_bad_agreement(parses, en):
    p = next(x for x in parses["en"] if x["text"].startswith("The dog eats"))
    d = corrupt(p, en, ["agreement"], random.Random(0))
    d.ops[0]["edit"] = dict(d.ops[0]["edit"], relation="obj")  # relation not in the clean parse
    assert structural_check(d, p, en, set()) == "agreement_check"


def test_italian_orthography_rejected(parses, it):
    p = next(x for x in parses["it"] if "l'uomo" in x["text"])
    d = Draft.from_parse(p)
    j = d.tokens.index("uomo")
    d.tokens[j] = "ragazzo"  # "l'ragazzo" is a spelling artifact
    d.tags[j] = "agreement"
    d.ops = [{"op": "wrong_function_word"}]
    d.tags[j] = "wrong_function_word"
    assert structural_check(d, p, it, set()) == "orthography"


def test_lm_delta_rejects_preferred_corruptions():
    scorer = TableScorer({"good": -1.0, "bad": -5.0, "clean2": -3.0, "valid2": -1.0})
    diffs = lm_diffs(scorer, [("good", "bad"), ("clean2", "valid2")])
    assert diffs == [4.0, -2.0]
    delta = calibrate_delta({"a": diffs}, target_pass_rate=0.5, min_delta=0.0)
    assert delta >= 0.0
    assert [d >= delta for d in diffs] == [True, False]


def test_calibrate_delta_quantile():
    diffs = {"a": [float(i) for i in range(100)]}
    assert calibrate_delta(diffs, 0.85, 0.0) == 15.0
    assert calibrate_delta({"a": [-5.0, -4.0]}, 0.85, 0.0) == 0.0


class _Scorer:
    """Clean sentences always score higher than anything else."""
    name = "stub"

    def __init__(self, clean):
        self.clean = set(clean)

    def score(self, texts):
        return [(0.0 if t in self.clean else -10.0, 1) for t in texts]


def test_corrupt_split_mix_and_verification(parses, en, cfg):
    items = make_items(parses["en"], "en", copies=12)
    tools = {"lm": _Scorer({i["text"] for i in items})}
    stats = OpStats()
    n = 60
    chosen, info = corrupt_split(items, n, "train", en, tools, cfg, 0.5, set(), stats)
    assert len(chosen) == n
    t = mix_targets(n, cfg["mix"])
    n_corrupted = sum(1 for _, d in chosen if d is not None)
    assert n_corrupted + sum(info["shortfall"].values()) == n - t["clean"]
    assert len({(i["source_id"]) for i, _ in chosen}) == n  # one row per source sentence
