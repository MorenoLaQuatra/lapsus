# SPDX-License-Identifier: Apache-2.0
"""Split assignment (spec section 6).

* Every row is one source sentence and is used once: splits are assigned per
  (source, source_id), after deduplication, so nothing leaks across splits.
* test_unseen_lemmas: a set of content lemmas (NOUN, VERB, ADJ) is reserved;
  every sentence containing a reserved lemma goes to test_unseen_lemmas or is
  dropped, so reserved lemmas never occur in train, validation or test.
* `forced` maps a source_id to a split (used for Italian Tatoeba sentences whose
  English translation is already in a split, keeping translations split-consistent).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from .lang.base import Language
from .utils import rng_for

log = logging.getLogger("lapsus.splits")

MAIN_SPLITS = ["train", "validation", "test", "test_unseen_lemmas"]


def group_key(row: dict) -> tuple[str, str]:
    return (row["source"], row["source_id"])


def choose_reserved_lemmas(rows: list[dict], lemmas: list[set[str]], target: int, seed: int, tag: str,
                           min_df: int = 2, max_df: int = 25, blocked: set[int] | None = None) -> tuple[set[str], set[int]]:
    """Pick mid-frequency content lemmas until their sentences reach `target`."""
    df: Counter[str] = Counter()
    for ls in lemmas:
        df.update(ls)
    cands = sorted(l for l, c in df.items() if min_df <= c <= max_df and l.isalpha() and len(l) > 2)
    rng = rng_for(seed, tag, "reserved")
    rng.shuffle(cands)
    index: dict[str, list[int]] = {}
    for i, ls in enumerate(lemmas):
        for l in ls:
            index.setdefault(l, []).append(i)
    reserved: set[str] = set()
    covered: set[int] = set()
    blocked = blocked or set()
    for l in cands:
        if len(covered) >= target:
            break
        idxs = index[l]
        if any(i in blocked for i in idxs):
            continue  # would force a sentence pinned to another split out of it
        reserved.add(l)
        covered.update(idxs)
    return reserved, covered


def assign_splits(rows: list[dict], sizes: dict[str, int], lang: Language, seed: int, surplus: float,
                  forced: dict[str, str] | None = None) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    forced = forced or {}
    tag = f"splits-{lang.code}"
    order = sorted(rows, key=lambda r: rng_for(seed, tag, *group_key(r)).random())
    lemmas = [lang.content_lemmas(r["parse"]) for r in order]
    blocked = {i for i, r in enumerate(order)
               if forced.get(r["source_id"]) not in (None, "test_unseen_lemmas")}
    want = {k: int(round(sizes.get(k, 0) * (1 + surplus))) for k in MAIN_SPLITS}
    reserved, covered = choose_reserved_lemmas(order, lemmas, want["test_unseen_lemmas"], seed, tag, blocked=blocked)
    out: dict[str, list[dict]] = {k: [] for k in MAIN_SPLITS}
    dropped = Counter()
    for i in sorted(covered):
        r = order[i]
        f = forced.get(r["source_id"])
        if f not in (None, "test_unseen_lemmas"):
            dropped["reserved_lemma_conflict"] += 1
            continue
        if len(out["test_unseen_lemmas"]) < want["test_unseen_lemmas"]:
            out["test_unseen_lemmas"].append(r)
        else:
            dropped["reserved_lemma_overflow"] += 1
    rest = [r for i, r in enumerate(order) if i not in covered]
    # shrink the surplus first, then scale the split sizes only if the pool is still too small
    base = {k: sizes.get(k, 0) for k in ("train", "validation", "test")}
    need = sum(int(round(v * (1 + surplus))) for v in base.values())
    scale = 1.0
    if len(rest) < need:
        eff = max(0.0, len(rest) / max(1, sum(base.values())) - 1.0)
        if eff > 0:
            log.warning("[%s] pool too small for surplus %.2f: using surplus %.3f", lang.code, surplus, eff)
            for k in base:
                want[k] = int(base[k] * (1 + eff))
        else:
            scale = len(rest) / max(1, sum(base.values()))
            log.warning("[%s] pool too small for requested split sizes: scaling train/validation/test by %.3f",
                        lang.code, scale)
            for k in base:
                want[k] = int(base[k] * scale)
    # forced first
    free = []
    for r in rest:
        f = forced.get(r["source_id"])
        if f in ("train", "validation", "test"):
            if len(out[f]) < want[f]:
                out[f].append(r)
            else:
                dropped["forced_overflow"] += 1
        elif f == "test_unseen_lemmas":
            dropped["forced_unseen_without_reserved_lemma"] += 1
        elif f is not None:
            dropped["forced_conflict"] += 1
        else:
            free.append(r)
    for k in ("validation", "test", "train"):
        while len(out[k]) < want[k] and free:
            out[k].append(free.pop(0))
    dropped["unused"] += len(free)
    info = {
        "reserved_lemmas": sorted(reserved),
        "n_reserved_lemmas": len(reserved),
        "scale": round(scale, 4),
        "dropped": dict(dropped),
        "assigned_with_surplus": {k: len(v) for k, v in out.items()},
    }
    return out, info


def check_leakage(dataset: dict[str, list[dict]], reserved: set[str] | None = None,
                  lemma_lookup: dict[str, set[str]] | None = None) -> list[str]:
    """Return a list of leakage problems (empty if none)."""
    problems = []
    seen_group: dict[tuple[str, str, str], str] = {}
    seen_clean: dict[tuple[str, str], str] = {}
    for split, rows in dataset.items():
        for r in rows:
            g = (r["lang"], r["source"], r["source_id"])
            if g in seen_group and seen_group[g] != split:
                problems.append(f"source sentence {g} in {seen_group[g]} and {split}")
            seen_group[g] = split
            c = (r["lang"], r["clean_text"])
            if c in seen_clean and seen_clean[c] != split:
                problems.append(f"clean text {c[1]!r} in {seen_clean[c]} and {split}")
            seen_clean[c] = split
    if reserved and lemma_lookup is not None:
        for split in ("train", "validation", "test"):
            for r in dataset.get(split, []):
                bad = lemma_lookup.get(r["id"], set()) & reserved
                if bad:
                    problems.append(f"reserved lemma(s) {sorted(bad)[:3]} in {split} row {r['id']}")
        for r in dataset.get("test_unseen_lemmas", []):
            if not (lemma_lookup.get(r["id"], set()) & reserved):
                problems.append(f"test_unseen_lemmas row {r['id']} has no reserved lemma")
    return problems
