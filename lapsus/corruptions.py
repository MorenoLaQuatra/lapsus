# SPDX-License-Identifier: Apache-2.0
"""Corruption operations (spec section 3).

A corruption works on a Draft: the current tokens, one tag per token, and the
index of each token in the CLEAN sentence (`origin`, None for inserted tokens).
Every token carries its tag through all later edits. Target selection always
uses the clean parse (UPOS, feats, deprel), reached through `origin`.

Operations and tagging conventions
  word_order           swap two tokens at distance <= 3; both are tagged.
                       Only the word "cores" move: trailing punctuation stays in place.
  missing_word         delete one function word; the token right AFTER the gap is
                       tagged (the last token if the gap is at the end).
  repeated_word        duplicate a token right after itself; the duplicate is tagged.
  agreement            replace a word with another form of the same lemma that
                       differs in one agreement feature; the changed token is tagged.
  wrong_function_word  replace a function word with a member of its confusion set;
                       the replaced token is tagged.

Multi-corruption applies 2 or 3 DIFFERENT operations sequentially on
non-overlapping tokens; an operation may never touch a tagged token.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from . import OPS
from .lang.base import Language, Parse, match_case, split_core


@dataclass
class Draft:
    tokens: list[str]
    tags: list[str]
    origin: list[int | None]
    src: list[int]                      # clean index the token text comes from (for duplicates too)
    ops: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_parse(cls, parse: Parse) -> "Draft":
        n = len(parse["tokens"])
        return cls(list(parse["tokens"]), ["O"] * n, list(range(n)), list(range(n)))

    def copy(self) -> "Draft":
        return Draft(list(self.tokens), list(self.tags), list(self.origin), list(self.src),
                     [dict(o) for o in self.ops])

    def pos_of(self, k: int) -> int | None:
        for p, o in enumerate(self.origin):
            if o == k:
                return p
        return None


def _core(tok: str) -> str:
    return split_core(tok)[0]


def untouched(d: Draft, parse: Parse, p: int | None) -> bool:
    """Token at p is untagged, original (not inserted) and its core is unchanged."""
    if p is None or d.tags[p] != "O" or d.origin[p] is None:
        return False
    return _core(d.tokens[p]) == _core(parse["tokens"][d.origin[p]])


def present_unchanged(d: Draft, parse: Parse, k: int) -> bool:
    p = d.pos_of(k)
    return p is not None and _core(d.tokens[p]) == _core(parse["tokens"][k])


# --------------------------------------------------------------- operations
def op_word_order(d: Draft, parse: Parse, lang: Language, rng: random.Random) -> Draft | None:
    n = len(d.tokens)
    cands = []
    for p in range(n):
        for q in range(p + 1, min(n, p + 4)):
            if not (untouched(d, parse, p) and untouched(d, parse, q)):
                continue
            i, j = d.origin[p], d.origin[q]
            if _core(d.tokens[p]).lower() == _core(d.tokens[q]).lower():
                continue
            if lang.word_order_allowed(parse, min(i, j), max(i, j)):  # type: ignore[type-var]
                cands.append((p, q))
    if not cands:
        return None
    p, q = rng.choice(cands)
    out = d.copy()
    cp, pp = split_core(out.tokens[p])
    cq, pq = split_core(out.tokens[q])
    out.tokens[p], out.tokens[q] = cq + pp, cp + pq
    out.origin[p], out.origin[q] = out.origin[q], out.origin[p]
    out.src[p], out.src[q] = out.src[q], out.src[p]
    out.tags[p] = out.tags[q] = "word_order"
    out.ops.append({"op": "word_order", "clean_indices": [d.origin[p], d.origin[q]]})
    return out


def op_missing_word(d: Draft, parse: Parse, lang: Language, rng: random.Random) -> Draft | None:
    n = len(d.tokens)
    if n < 3:
        return None
    cands = []
    for k in lang.missing_word_targets(parse):
        p = d.pos_of(k)
        if not untouched(d, parse, p):
            continue
        assert p is not None
        gap_tag = p + 1 if p + 1 < n else p - 1      # token right after the gap (index before deletion)
        if d.tags[gap_tag] != "O":
            continue
        punct = split_core(d.tokens[p])[1]
        if punct and (p == 0 or d.tags[p - 1] != "O"):
            continue
        cands.append((p, gap_tag))
    if not cands:
        return None
    p, gap_tag = rng.choice(cands)
    out = d.copy()
    punct = split_core(out.tokens[p])[1]
    if punct:
        out.tokens[p - 1] = _core(out.tokens[p - 1]) + punct
    deleted = out.tokens[p]
    for lst in (out.tokens, out.tags, out.origin, out.src):
        del lst[p]
    new_pos = gap_tag if gap_tag < p else gap_tag - 1
    out.tags[new_pos] = "missing_word"
    out.ops.append({"op": "missing_word", "clean_indices": [d.origin[p]], "deleted": _core(deleted)})
    return out


def op_repeated_word(d: Draft, parse: Parse, lang: Language, rng: random.Random) -> Draft | None:
    cands = [p for p in range(len(d.tokens))
             if untouched(d, parse, p) and lang.repeat_allowed(parse, d.origin[p])]  # type: ignore[arg-type]
    if not cands:
        return None
    p = rng.choice(cands)
    out = d.copy()
    core, punct = split_core(out.tokens[p])
    out.tokens[p] = core
    out.tokens.insert(p + 1, core + punct)
    out.tags.insert(p + 1, "repeated_word")
    out.origin.insert(p + 1, None)
    out.src.insert(p + 1, out.src[p])
    out.ops.append({"op": "repeated_word", "clean_indices": [d.origin[p]]})
    return out


def _replace(d: Draft, p: int, new_core: str, tag: str, info: dict) -> Draft:
    out = d.copy()
    core, punct = split_core(out.tokens[p])
    out.tokens[p] = match_case(core, new_core) + punct
    out.tags[p] = tag
    out.ops.append({"op": tag, **info})
    return out


def op_agreement(d: Draft, parse: Parse, lang: Language, rng: random.Random) -> Draft | None:
    cands = []
    for e in lang.agreement_candidates(parse):
        p = d.pos_of(e["index"])
        if not untouched(d, parse, p):
            continue
        # the other member of the agreement relation must still be there, unchanged
        partner = e["head"] if e["index"] == e["dep"] else e["dep"]
        if partner != e["index"] and not present_unchanged(d, parse, partner):
            continue
        cands.append((p, e))
    if not cands:
        return None
    p, e = rng.choice(cands)
    return _replace(d, p, e["new"], "agreement", {"clean_indices": [e["index"]], "edit": e})


def op_wrong_function_word(d: Draft, parse: Parse, lang: Language, rng: random.Random) -> Draft | None:
    cands = []
    for e in lang.function_word_candidates(parse):
        p = d.pos_of(e["index"])
        if untouched(d, parse, p) and e["new"].lower() != _core(d.tokens[p]).lower():  # type: ignore[index]
            cands.append((p, e))
    if not cands:
        return None
    p, e = rng.choice(cands)
    return _replace(d, p, e["new"], "wrong_function_word", {"clean_indices": [e["index"]], "edit": e})


OP_FUNCS = {
    "word_order": op_word_order,
    "missing_word": op_missing_word,
    "repeated_word": op_repeated_word,
    "agreement": op_agreement,
    "wrong_function_word": op_wrong_function_word,
}


def applicable_ops(parse: Parse, lang: Language, rng: random.Random) -> list[str]:
    d = Draft.from_parse(parse)
    return [op for op in OPS if OP_FUNCS[op](d, parse, lang, random.Random(rng.random())) is not None]


def corrupt(parse: Parse, lang: Language, ops: list[str], rng: random.Random) -> Draft | None:
    """Apply `ops` (distinct) sequentially; None if any op has no valid target."""
    assert len(set(ops)) == len(ops), "operations in a multi-corruption must be different"
    d = Draft.from_parse(parse)
    for op in ops:
        nd = OP_FUNCS[op](d, parse, lang, rng)
        if nd is None:
            return None
        d = nd
    fix_case(d, parse, lang)
    return d


def corrupt_multi(parse: Parse, lang: Language, k: int, rng: random.Random, attempts: int = 6) -> Draft | None:
    combos = list(itertools.permutations(OPS, k))
    rng.shuffle(combos)
    for ops in combos[:attempts * 4]:
        d = corrupt(parse, lang, list(ops), rng)
        if d is not None:
            return d
    return None


def fix_case(d: Draft, parse: Parse, lang: Language) -> None:
    """Cosmetic: keep a capital only at the start of the corrupted sentence."""
    keep_caps = lang.always_capitalized()
    first_is_propn = parse["info"] and parse["info"][0]["upos"] == "PROPN"
    for p, tok in enumerate(d.tokens):
        if p != 0 and d.src[p] == 0 and tok[:1].isupper() and not first_is_propn and _core(tok) not in keep_caps:
            d.tokens[p] = tok[0].lower() + tok[1:]
    if d.tokens and d.tokens[0][:1].islower():
        d.tokens[0] = d.tokens[0][0].upper() + d.tokens[0][1:]


# ---------------------------------------------------------------- label views
def label_views(tags: list[str], n_corruptions: int) -> dict[str, Any]:
    present = [op for op in OPS if op in tags]
    if n_corruptions == 0:
        mc = "none"
    elif n_corruptions == 1:
        mc = present[0] if present else "none"
    else:
        mc = "multiple"
    return {
        "binary": "not_ok" if present else "ok",
        "multiclass": mc,
        "multilabel": [1 if op in tags else 0 for op in OPS],
        "token_tags": list(tags),
        "token_binary": [0 if t == "O" else 1 for t in tags],
        "n_corruptions": n_corruptions,
    }


def views_consistent(row: dict) -> bool:
    """All label views agree with token_tags and n_corruptions."""
    tags = row["token_tags"]
    if len(row["tokens"]) != len(tags) or len(row["token_binary"]) != len(tags):
        return False
    if row["multiclass"] == -1 or row.get("multiclass") is None:  # natural rows: binary + token level only
        return (row["binary"] == "not_ok") == any(row["token_binary"])
    present = [op for op in OPS if op in tags]
    if row["token_binary"] != [0 if t == "O" else 1 for t in tags]:
        return False
    if row["multilabel"] != [1 if op in tags else 0 for op in OPS]:
        return False
    if (row["binary"] == "not_ok") != bool(present):
        return False
    n = row["n_corruptions"]
    if n != len(present):
        return False
    if n == 0:
        return row["multiclass"] == "none"
    if n == 1:
        return row["multiclass"] == present[0]
    return row["multiclass"] == "multiple"
