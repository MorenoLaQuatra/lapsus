# SPDX-License-Identifier: Apache-2.0
"""Language interface.

Everything language specific (tokenization, closed-class tables, corruption
target rules, morphology lexicons, tool names) lives behind this interface so a
new language only needs a new module in lapsus/lang/ registered in
lapsus/lang/__init__.py.

Parses are plain dicts so they can be cached as JSON:

    parse = {
      "text": raw sentence,
      "tokens": [whitespace tokens],
      "info": [ {"upos", "lemma", "feats": {..}, "head": int (-1 root),
                 "deprel", "words": [[text, lemma, upos, feats_str, deprel], ...],
                 "contracted": bool}, ... one per token ],
      "n_roots": int, "ok": bool, "reason": str,
    }
"""

from __future__ import annotations

import re
from typing import Any

TRAIL_PUNCT = ".,!?;:"
CONTENT_UPOS = {"NOUN", "VERB", "ADJ"}

Parse = dict[str, Any]
Edit = dict[str, Any]


def feats_to_dict(feats: str | None) -> dict[str, str]:
    if not feats:
        return {}
    out = {}
    for kv in feats.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def split_core(token: str) -> tuple[str, str]:
    """Split trailing punctuation: 'dog,' -> ('dog', ',')."""
    i = len(token)
    while i > 0 and token[i - 1] in TRAIL_PUNCT:
        i -= 1
    return token[:i], token[i:]


def match_case(template: str, new: str) -> str:
    if template[:1].isupper() and new:
        return new[0].upper() + new[1:]
    return new


def children(parse: Parse, head: int) -> list[int]:
    return [i for i, t in enumerate(parse["info"]) if t["head"] == head]


def core_lower(parse: Parse, i: int) -> str:
    return split_core(parse["tokens"][i])[0].lower()


class Language:
    code: str = ""
    name: str = ""
    stanza_lang: str = ""
    lingua_name: str = ""          # lingua.Language enum member name
    lt_codes: list[str] = []        # a sentence passes LanguageTool if it passes in ANY variant
    default_lm: str = ""
    tatoeba_code: str = ""

    # ------------------------------------------------------------------ tokens
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans of tokenize(text) inside text."""
        spans = []
        pos = 0
        for tok in self.tokenize(text):
            start = text.index(tok, pos)
            spans.append((start, start + len(tok)))
            pos = start + len(tok)
        return spans

    # ----------------------------------------------------------------- filters
    def proper_noun_count_heuristic(self, tokens: list[str], known_names: set[str]) -> int:
        """Cheap pre-parse proper-noun count (capitalized non-initial tokens)."""
        n = 0
        for i, tok in enumerate(tokens):
            core = split_core(tok)[0]
            core = re.sub(r"'s$", "", core)
            if not core[:1].isupper():
                continue
            if i == 0 and core not in known_names:
                continue
            if core in self.always_capitalized():
                continue
            n += 1
        return n

    def always_capitalized(self) -> set[str]:
        return set()

    # -------------------------------------------------------------- corruption
    def missing_word_targets(self, parse: Parse) -> list[int]:
        raise NotImplementedError

    def word_order_allowed(self, parse: Parse, i: int, j: int) -> bool:
        raise NotImplementedError

    def repeat_allowed(self, parse: Parse, i: int) -> bool:
        raise NotImplementedError

    def agreement_candidates(self, parse: Parse) -> list[Edit]:
        """Each edit: {index, new, feature, old_value, new_value, relation, kind}."""
        raise NotImplementedError

    def function_word_candidates(self, parse: Parse) -> list[Edit]:
        """Each edit: {index, new, set}."""
        raise NotImplementedError

    def verify_agreement(self, parse: Parse, edit: Edit) -> bool:
        """Independent re-check: the new form exists, differs in the targeted feature,
        and the relation held in the clean parse."""
        raise NotImplementedError

    def relation_holds(self, parse: Parse, edit: Edit) -> bool:
        rel = edit.get("relation")
        dep, head = edit.get("dep"), edit.get("head")
        if rel is None or dep is None or head is None:
            return False
        info = parse["info"]
        if not (0 <= dep < len(info) and 0 <= head < len(info)):
            return False
        if info[dep]["head"] != head:
            return False
        # the relation may sit on a word inside a multi-word token (article of "della")
        rels = {info[dep]["deprel"].split(":")[0]} | {w[4].split(":")[0] for w in info[dep].get("words", [])}
        return rel in rels

    def orthography_ok(self, tokens: list[str]) -> bool:
        """Language-specific orthographic sanity of a corrupted token sequence (default: always fine)."""
        return True

    # -------------------------------------------------------------- utilities
    def content_lemmas(self, parse: Parse) -> set[str]:
        out = set()
        for t in parse["info"]:
            if t["upos"] in CONTENT_UPOS and t["lemma"]:
                out.add(t["lemma"].lower())
        return out

    def proper_nouns(self, parse: Parse) -> list[str]:
        return [
            (t["lemma"] or core_lower(parse, i)).lower()
            for i, t in enumerate(parse["info"])
            if t["upos"] == "PROPN"
        ]

    def version_info(self) -> dict[str, str]:
        return {}
