# SPDX-License-Identifier: Apache-2.0
"""Agreement forms must exist in the lexicon and differ in the targeted feature."""

import random

from lapsus.corruptions import corrupt
from lapsus.lang.en import _inflections
from lapsus.lang.it import ARTPRE, DEF_ART, INDEF_ART


def test_italian_agreement_forms_exist_in_lexicon(parses, it):
    n = 0
    for p in parses["it"]:
        for e in it.agreement_candidates(p):
            n += 1
            assert it.verify_agreement(p, e), e
            new = e["new"].lower()
            if e["lex"] in ("morphit", "morphit_verb"):
                analyses = [a for a in it.lex.analyses[new] if a[0] == e["lemma"]]
                assert analyses, f"{new} not in Morph-it! for lemma {e['lemma']}"
            else:
                assert new in DEF_ART or new in INDEF_ART or new in ARTPRE
    assert n >= 10


def test_italian_agreement_examples(parses, it):
    p = next(x for x in parses["it"] if x["text"].startswith("Le ragazze sono stanche"))
    news = {e["new"] for e in it.agreement_candidates(p)}
    assert "stanca" in news or "stanchi" in news  # predicative adjective gender/number
    p = next(x for x in parses["it"] if x["text"].startswith("La ragazza è andata"))
    news = {e["new"] for e in it.agreement_candidates(p)}
    assert "andato" in news or "andate" in news  # participle with essere


def test_english_agreement_forms_exist(parses, en):
    n = 0
    for p in parses["en"]:
        for e in en.agreement_candidates(p):
            n += 1
            assert en.verify_agreement(p, e), e
            if e["lex"] in ("verb", "noun"):
                infl = _inflections(e["lemma"], "VERB" if e["lex"] == "verb" else "NOUN")
                assert e["new"] in infl[e["new_tag"]]
    assert n >= 8


def test_english_agreement_examples(parses, en):
    p = next(x for x in parses["en"] if x["text"].startswith("The dog eats"))
    news = {e["new"] for e in en.agreement_candidates(p)}
    assert "eat" in news and "a" in news  # subject-verb and a/an
    p = next(x for x in parses["en"] if x["text"].startswith("This book"))
    news = {e["new"] for e in en.agreement_candidates(p)}
    assert "these" in news or "books" in news


def test_verify_rejects_tampered_agreement(parses, en, it):
    for lang, ps in ((en, parses["en"]), (it, parses["it"])):
        for p in ps:
            for e in lang.agreement_candidates(p):
                bad = dict(e, relation="obj")
                assert not lang.verify_agreement(p, bad)
                same = dict(e, new=p["tokens"][e["index"]].rstrip(".").lower())
                assert not lang.verify_agreement(p, same)


def test_agreement_op_tags_changed_token(parses, it):
    p = parses["it"][0]
    d = corrupt(p, it, ["agreement"], random.Random(0))
    assert d is not None
    j = d.tags.index("agreement")
    assert d.tokens[j].lower() != p["tokens"][d.origin[j]].lower()
