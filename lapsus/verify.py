# SPDX-License-Identifier: Apache-2.0
"""Verification of corrupted rows (spec section 4).

A corrupted candidate is accepted only if
  1. its text differs from the clean sentence,
  2. it equals no sentence of the clean pool (normalized comparison),
  3. every agreement edit passes the independent lexicon re-check (the new form
     exists, differs in the targeted feature, and the clean parse confirms the
     relation),
  4. len(tokens) == len(token_tags) and all label views are consistent,
  5. LM minimal pair: mean_logprob(clean) - mean_logprob(corrupted) >= delta.
LanguageTool's flag rate on accepted rows is reported as a diagnostic only.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .corruptions import Draft, label_views, views_consistent
from .filters import normalized_key
from .lang.base import Language, Parse
from .lm import mean_lp


def structural_check(draft: Draft, parse: Parse, lang: Language, clean_pool_keys: set[str]) -> str:
    """Checks 1 to 4. Returns '' if fine, otherwise the rejection reason."""
    corrupted = lang.detokenize(draft.tokens)
    clean = parse["text"]
    if normalized_key(corrupted) == normalized_key(clean):
        return "same_as_clean"
    if normalized_key(corrupted) in clean_pool_keys:
        return "in_clean_pool"
    for op in draft.ops:
        if op["op"] == "agreement" and not lang.verify_agreement(parse, op["edit"]):
            return "agreement_check"
    if len(draft.tokens) != len(draft.tags):
        return "tag_alignment"
    if lang.tokenize(corrupted) != draft.tokens:
        return "tokenization_roundtrip"
    if not lang.orthography_ok(draft.tokens):
        return "orthography"
    views = label_views(draft.tags, len(draft.ops))
    row = {"tokens": draft.tokens, **views}
    if not views_consistent(row):
        return "label_views"
    return ""


def lm_diffs(scorer: Any, pairs: list[tuple[str, str]]) -> list[float]:
    """mean_lp(clean) - mean_lp(corrupted) for each (clean, corrupted) pair."""
    if not pairs:
        return []
    texts = [t for p in pairs for t in p]
    scores = scorer.score(texts)
    return [mean_lp(scores[2 * i]) - mean_lp(scores[2 * i + 1]) for i in range(len(pairs))]


def calibrate_delta(diffs_by_op: dict[str, list[float]], target_pass_rate: float, min_delta: float) -> float:
    """Largest delta such that `target_pass_rate` of all candidates pass, never below min_delta.

    Pooling over operations keeps "most ops pass"; the per-op pass rates are
    reported so weak operations are visible.
    """
    pooled = sorted(x for v in diffs_by_op.values() for x in v)
    if not pooled:
        return min_delta
    k = int((1.0 - target_pass_rate) * len(pooled))
    return max(min_delta, pooled[min(k, len(pooled) - 1)])


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = sorted(values)

    def q(p: float) -> float:
        return round(v[min(len(v) - 1, int(p * len(v)))], 4)

    return {"n": len(v), "mean": round(statistics.fmean(v), 4), "p05": q(0.05), "p25": q(0.25),
            "p50": q(0.5), "p75": q(0.75), "p95": q(0.95)}


class OpStats:
    def __init__(self):
        self.tried: dict[str, int] = defaultdict(int)
        self.no_target: dict[str, int] = defaultdict(int)
        self.rejected: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.accepted: dict[str, int] = defaultdict(int)
        self.lm_diffs: dict[str, list[float]] = defaultdict(list)
        self.lt_flags: dict[str, list[bool]] = defaultdict(list)

    def as_dict(self) -> dict:
        out = {}
        for op in sorted(set(self.tried) | set(self.accepted)):
            verified = self.accepted.get(op, 0) + sum(self.rejected[op].values())
            out[op] = {
                "attempts": self.tried.get(op, 0),
                "no_target": self.no_target.get(op, 0),
                "verified_candidates": verified,
                "accepted": self.accepted.get(op, 0),
                "acceptance_rate": round(self.accepted.get(op, 0) / verified, 4) if verified else None,
                "rejections": dict(self.rejected[op]),
                "lm_diff": distribution(self.lm_diffs.get(op, [])),
                "languagetool_flag_rate": (round(sum(self.lt_flags[op]) / len(self.lt_flags[op]), 4)
                                           if self.lt_flags.get(op) else None),
            }
        return out
