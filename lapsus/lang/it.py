# SPDX-License-Identifier: Apache-2.0
"""Italian: elision-aware tokenization, Morph-it! morphology, corruption target rules.

Tokenization is whitespace based, but elided forms are split off as their own
token and keep the apostrophe: "l'uomo" -> ["l'", "uomo"], "dell'acqua" ->
["dell'", "acqua"]. Detokenization glues an elided token to the next word.

Open-class word forms (nouns, adjectives, verbs, possessive and demonstrative
determiners) come from Morph-it! (same lemma, different feature). Articles and
articulated prepositions are closed classes and use the tables below, because
Morph-it! lemmatizes them inconsistently (e.g. "la" has lemma "la").
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

from .base import Edit, Language, Parse, children, core_lower, split_core

ELISION_SPLIT = re.compile(r"(?<=')(?=\w)")
APOCOPES = {"po'", "mo'", "be'", "di'", "fa'", "sta'", "va'", "da'", "to'"}
VOWELS = "aeiouàèéìòùh"

DEF_ART = {"il": ("m", "s"), "lo": ("m", "s"), "la": ("f", "s"), "l'": (None, "s"),
           "i": ("m", "p"), "gli": ("m", "p"), "le": ("f", "p")}
INDEF_ART = {"un": ("m", "s"), "uno": ("m", "s"), "una": ("f", "s"), "un'": ("f", "s")}
PREP_STEMS = {"di": "de", "a": "a", "da": "da", "in": "ne", "su": "su"}
ART_SUFFIX = {"il": "l", "lo": "llo", "la": "lla", "l'": "ll'", "i": "i", "gli": "gli", "le": "lle"}
ARTPRE: dict[str, tuple[str, str]] = {
    _stem + _suf: (_p, _a) for _p, _stem in PREP_STEMS.items() for _a, _suf in ART_SUFFIX.items()
}
# Sanity: a+il=al, a+lo=allo, a+la=alla, a+l'=all', a+i=ai, a+gli=agli, a+le=alle,
# di+il=del, di+lo=dello, ..., in+il=nel, su+il=sul, da+il=dal.

SIMPLE_PREPS = ["a", "di", "da", "in", "su", "per", "con"]
ALLOMORPH_SWAP = {"il": "lo", "lo": "il", "i": "gli", "gli": "i", "un": "uno", "uno": "un"}
NO_REPEAT = {"molto", "più", "piano", "pian", "sì", "no", "via", "ciao", "così", "tanto", "poco",
             "bene", "mai", "ora", "presto", "adagio", "forse", "proprio", "quasi"}
WORD_ORDER_PREFERRED = {"DET", "ADP", "AUX", "PRON", "VERB"}
WORD_ORDER_BANNED = {"ADV", "PUNCT", "INTJ", "X", "SYM"}
ARG_RELS = {"nsubj", "obj", "iobj", "obl", "ccomp", "xcomp", "csubj", "expl"}
VERBAL_CATS = {"VER", "AUX", "MOD", "ASP", "CAU"}
MOODS = {"ind", "sub", "cond", "impr", "inf", "part", "ger"}
TENSES = {"pres", "past", "impf", "fut"}
SKIP_DET_LEMMAS = {"quello", "bello", "quel"}  # heavy allomorphy, not handled


def pick_form(cands: list[str]) -> str:
    """Prefer full, vowel-final forms over apocopated ones ("sono" over "son", "hanno" over "han")."""
    return sorted(cands, key=lambda c: (c[-1] not in "aeiouàèéìòù", -len(c), c))[0]


def lo_class(word: str) -> bool:
    w = word.lower()
    return bool(re.match(r"^(s[^aeiouàèéìòù]|z|gn|ps|pn|x|y|i[aeiou])", w))


def starts_vowel(word: str) -> bool:
    return word[:1].lower() in VOWELS


def def_article(g: str, n: str, nxt: str) -> str:
    if n == "s":
        if starts_vowel(nxt):
            return "l'"
        if g == "m":
            return "lo" if lo_class(nxt) else "il"
        return "la"
    if g == "m":
        return "gli" if (starts_vowel(nxt) or lo_class(nxt)) else "i"
    return "le"


def indef_article(g: str, nxt: str) -> str:
    if g == "m":
        return "uno" if lo_class(nxt) else "un"
    return "un'" if starts_vowel(nxt) else "una"


def artpre_form(prep: str, art: str) -> str | None:
    for form, pa in ARTPRE.items():
        if pa == (prep, art):
            return form
    return None


def parse_morphit_tag(tag: str) -> tuple[str, dict[str, str]] | None:
    """'NOUN-M:s' -> ('NOUN', {g: m, n: s}); 'VER:ind+pres+3+s' -> ('VER', {...})."""
    if ":" in tag:
        cat, rest = tag.split(":", 1)
    else:
        cat, rest = tag, ""
    feats: dict[str, str] = {}
    if cat.startswith("NOUN-"):
        g = cat.split("-")[1].lower()
        return "NOUN", {"g": g, "n": rest}
    if cat == "ADJ":
        parts = rest.split("+")
        if len(parts) != 3:
            return None
        return "ADJ", {"deg": parts[0], "g": parts[1], "n": parts[2]}
    if cat.startswith("DET-") and rest:
        parts = rest.split("+")
        if len(parts) != 2:
            return None
        return cat, {"g": parts[0], "n": parts[1]}
    if cat in VERBAL_CATS:
        parts = rest.split("+")
        if not parts or parts[0] not in MOODS:
            return None
        feats["mood"] = parts[0]
        if len(parts) > 1 and parts[1] in TENSES:
            feats["tense"] = parts[1]
        rest_parts = parts[2:]
        if feats["mood"] == "part":
            if len(rest_parts) >= 2:
                feats["n"], feats["g"] = rest_parts[0], rest_parts[1]
        elif feats["mood"] in ("inf", "ger"):
            if rest_parts:
                feats["cli"] = "+".join(rest_parts)
        else:
            if len(rest_parts) >= 2:
                feats["p"], feats["n"] = rest_parts[0], rest_parts[1]
        return cat, feats
    return None


class MorphIt:
    """Morph-it! lexicon: form -> analyses, (lemma, cat) -> forms."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self.analyses: dict[str, list[tuple[str, str, dict[str, str]]]] = defaultdict(list)
        self.by_lemma: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
        raw = Path(path).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        for line in text.splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            form, lemma, tag = parts
            parsed = parse_morphit_tag(tag)
            if not parsed:
                continue
            cat, feats = parsed
            form, lemma = form.lower(), lemma.lower()
            self.analyses[form].append((lemma, cat, feats))
            self.by_lemma[(lemma, cat)].append((form, feats))

    def has_form(self, form: str) -> bool:
        return form.lower() in self.analyses

    def forms(self, lemma: str, cat: str, **want: str) -> list[str]:
        out = []
        for form, feats in self.by_lemma.get((lemma, cat), []):
            if all(feats.get(k) == v for k, v in want.items()) and not form.startswith("'"):
                out.append(form)
        return sorted(set(out))

    def find(self, form: str, cats: set[str], **want: str) -> list[tuple[str, str, dict[str, str]]]:
        return [
            (lem, cat, f)
            for lem, cat, f in self.analyses.get(form.lower(), [])
            if (cat in cats or any(cat.startswith(c) for c in cats if c.endswith("-")))
            and all(f.get(k) == v for k, v in want.items() if v is not None)
        ]


_LEXICON_CACHE: dict[str, MorphIt] = {}


def find_morphit_file(raw_dir: str | os.PathLike) -> Path | None:
    base = Path(raw_dir) / "morphit"
    for cand in sorted(base.rglob("morph-it_*.txt")):
        if "utf8" not in cand.name:
            return cand
    return None


G_MAP = {"Masc": "m", "Fem": "f"}
N_MAP = {"Sing": "s", "Plur": "p"}
FLIP = {"m": "f", "f": "m", "s": "p", "p": "s"}


class Italian(Language):
    code = "it"
    name = "Italian"
    stanza_lang = "it"
    lingua_name = "ITALIAN"
    lt_codes = ["it-IT"]
    default_lm = "Qwen/Qwen2.5-0.5B"
    tatoeba_code = "ita"

    def __init__(self, lexicon_path: str | os.PathLike | None = None):
        self._lexicon_path = str(lexicon_path) if lexicon_path else None

    # ----------------------------------------------------------------- lexicon
    def set_lexicon_path(self, path: str | os.PathLike) -> None:
        self._lexicon_path = str(path)

    @property
    def lex(self) -> MorphIt:
        if not self._lexicon_path:
            raise RuntimeError("Morph-it! lexicon path not set (run `python -m lapsus download --lang it`).")
        if self._lexicon_path not in _LEXICON_CACHE:
            _LEXICON_CACHE[self._lexicon_path] = MorphIt(self._lexicon_path)
        return _LEXICON_CACHE[self._lexicon_path]

    # ------------------------------------------------------------------ tokens
    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for w in text.split():
            out.extend(p for p in ELISION_SPLIT.split(w) if p)
        return out

    def detokenize(self, tokens: list[str]) -> str:
        parts: list[str] = []
        for i, tok in enumerate(tokens):
            parts.append(tok)
            if i + 1 < len(tokens):
                glue = tok.endswith("'") and tok.lower() not in APOCOPES and tokens[i + 1][:1].isalnum()
                if not glue:
                    parts.append(" ")
        return "".join(parts)

    def is_elided(self, token: str) -> bool:
        return token.endswith("'") and token.lower() not in APOCOPES

    def orthography_ok(self, tokens: list[str]) -> bool:
        """An elided token ("l'", "c'", "dell'") must be followed by a vowel- or h-initial word.

        Otherwise detokenization would glue it to a consonant ("c'sono"), which is a
        spelling artifact rather than the intended grammatical error.
        """
        for i, tok in enumerate(tokens):
            if self.is_elided(tok):
                if i + 1 >= len(tokens) or not starts_vowel(tokens[i + 1]):
                    return False
        return True

    def always_capitalized(self) -> set[str]:
        return {"Lei", "La", "Le", "Loro", "Suo", "Sua", "Suoi", "Sue", "Ella"}

    # ------------------------------------------------------------ missing word
    def missing_word_targets(self, parse: Parse) -> list[int]:
        info = parse["info"]
        out = []
        for i, t in enumerate(info):
            head = t["head"]
            core = core_lower(parse, i)
            if t["contracted"] and core in ARTPRE and head > i:
                out.append(i)  # articulated preposition (deletes preposition and article)
                continue
            if len(t["words"]) != 1:
                continue
            if t["upos"] == "DET" and t["feats"].get("PronType") == "Art" and t["deprel"] == "det":
                if head > i and info[head]["upos"] == "NOUN" and info[head]["feats"].get("Number") != "Plur":
                    out.append(i)
            elif t["upos"] == "ADP" and t["deprel"] in ("case", "mark") and head > i:
                out.append(i)
            elif t["upos"] == "AUX" and t["deprel"] in ("aux", "cop", "aux:pass"):
                out.append(i)
            elif t["upos"] == "PRON" and t["feats"].get("Clitic") == "Yes" and t["deprel"].startswith("expl"):
                out.append(i)
        return out

    # -------------------------------------------------------------- word order
    def word_order_allowed(self, parse: Parse, i: int, j: int) -> bool:
        info = parse["info"]
        a, b = info[i], info[j]
        ua, ub = a["upos"], b["upos"]
        if ua in WORD_ORDER_BANNED or ub in WORD_ORDER_BANNED:
            return False
        if ua == ub:  # includes NOUN-NOUN
            return False
        if {ua, ub} == {"ADJ", "NOUN"}:
            return False
        if not ({ua, ub} & WORD_ORDER_PREFERRED):
            return False
        if core_lower(parse, i) == core_lower(parse, j):
            return False
        if self.is_elided(parse["tokens"][i]) or self.is_elided(parse["tokens"][j]):
            return False  # moving an elided form strands the apostrophe
        rel_a, rel_b = a["deprel"].split(":")[0], b["deprel"].split(":")[0]
        if a["head"] == b["head"] and rel_a in ARG_RELS and rel_b in ARG_RELS:
            return False  # two arguments of one verb: free word order may keep it grammatical
        for x, xi, y, yi in ((a, i, b, j), (b, j, a, i)):
            if x["deprel"].startswith("nsubj"):
                # subject inversion is grammatical in Italian
                if x["head"] == yi:
                    return False
                if y["head"] == x["head"] and y["deprel"] in ("aux", "cop", "aux:pass"):
                    return False
        return True

    # ---------------------------------------------------------- repeated word
    def repeat_allowed(self, parse: Parse, i: int) -> bool:
        t = parse["info"][i]
        return t["upos"] not in ("PUNCT", "SYM", "X", "INTJ") and core_lower(parse, i) not in NO_REPEAT

    # --------------------------------------------------------------- agreement
    def _noun_gender_unique(self, parse: Parse, n: int) -> str | None:
        t = parse["info"][n]
        if t["upos"] != "NOUN":
            return None
        form = core_lower(parse, n)
        genders = {f["g"] for _, cat, f in self.lex.analyses.get(form, []) if cat == "NOUN"}
        g = G_MAP.get(t["feats"].get("Gender", ""))
        if len(genders) == 1 and g in genders:
            return g
        return None

    def _subject_features(self, parse: Parse, s: int) -> tuple[str | None, str | None]:
        """(unique gender or None, number or None) of a subject token."""
        t = parse["info"][s]
        n = N_MAP.get(t["feats"].get("Number", ""))
        g = self._noun_gender_unique(parse, s) if t["upos"] == "NOUN" else None
        return g, n

    def _nominal_flip(self, parse: Parse, idx: int, cat: str, feature: str, gender_ok: bool) -> Edit | None:
        """Change gender ('g') or number ('n') of a NOUN/ADJ/DET-* token via Morph-it!."""
        if feature == "g" and not gender_ok:
            return None
        t = parse["info"][idx]
        form = core_lower(parse, idx)
        g = G_MAP.get(t["feats"].get("Gender", ""))
        n = N_MAP.get(t["feats"].get("Number", ""))
        analyses = self.lex.find(form, {cat}, g=g, n=n)
        analyses = [a for a in analyses if a[2].get("deg", "pos") == "pos"]
        if not analyses:
            return None
        same_lemma = [a for a in analyses if a[0] == (t["lemma"] or "").lower()]
        lemma, acat, feats = (same_lemma or analyses)[0]
        if lemma in SKIP_DET_LEMMAS:
            return None
        want = dict(feats)
        want[feature] = FLIP[feats[feature]]
        cands = [f for f in self.lex.forms(lemma, acat, **want) if f != form]
        old_vals = {k: feats[k] for k in ("g", "n")}
        cands = [c for c in cands if not self.lex.find(c, {acat}, **old_vals)]
        if not cands:
            return None
        new = pick_form(cands)
        return dict(index=idx, new=new, feature={"g": "Gender", "n": "Number"}[feature],
                    old_value=feats[feature], new_value=want[feature], lex="morphit",
                    lemma=lemma, cat=acat, old_feats=old_vals)

    def agreement_candidates(self, parse: Parse) -> list[Edit]:
        info = parse["info"]
        toks = parse["tokens"]
        edits: list[Edit] = []
        for d, t in enumerate(info):
            h = t["head"]
            if h < 0:
                continue
            core = core_lower(parse, d)
            head = info[h]
            # ---- determiner / articulated preposition + noun
            if head["upos"] == "NOUN" and (t["deprel"] == "det" or (t["contracted"] and core in ARTPRE)):
                gu = self._noun_gender_unique(parse, h)
                nxt = split_core(toks[d + 1])[0] if d + 1 < len(toks) else ""
                ng = G_MAP.get(head["feats"].get("Gender", ""))
                if core in DEF_ART or (t["contracted"] and core in ARTPRE):
                    prep, art = (None, core) if core in DEF_ART else ARTPRE[core]
                    g0, n0 = DEF_ART[art]
                    g0 = g0 or ng
                    if g0:
                        options = [("n", g0, FLIP[n0])]
                        if gu:
                            options.append(("g", FLIP[g0], n0))
                        for feat, g1, n1 in options:
                            new_art = def_article(g1, n1, nxt)
                            new = new_art if prep is None else artpre_form(prep, new_art)
                            if new and new != core:
                                edits.append(dict(index=d, new=new, feature={"g": "Gender", "n": "Number"}[feat],
                                                  old_value=g0 if feat == "g" else n0,
                                                  new_value=g1 if feat == "g" else n1,
                                                  lex="artpre" if prep else "art", art_feats=[g1, n1],
                                                  dep=d, head=h, relation="det", kind="det_noun"))
                elif core in INDEF_ART and gu:
                    g0, _ = INDEF_ART[core]
                    new = indef_article(FLIP[g0], nxt)
                    if new != core:
                        edits.append(dict(index=d, new=new, feature="Gender", old_value=g0, new_value=FLIP[g0],
                                          lex="art", art_feats=[FLIP[g0], "s"], dep=d, head=h,
                                          relation="det", kind="det_noun"))
                elif t["upos"] == "DET" and len(t["words"]) == 1:
                    for feat in ("n", "g"):
                        e = self._nominal_flip(parse, d, "DET-", feat, gender_ok=gu is not None)
                        if e:
                            e.update(dep=d, head=h, relation="det", kind="det_noun")
                            edits.append(e)
                # noun number change, keeping the determiner
                if t["deprel"] == "det" and len(head["words"]) == 1:
                    e = self._nominal_flip(parse, h, "NOUN", "n", gender_ok=False)
                    if e:
                        e.update(dep=d, head=h, relation="det", kind="det_noun")
                        edits.append(e)
            # ---- attributive adjective
            if t["upos"] == "ADJ" and t["deprel"] == "amod" and head["upos"] == "NOUN" and len(t["words"]) == 1:
                gu = self._noun_gender_unique(parse, h)
                for feat in ("n", "g"):
                    e = self._nominal_flip(parse, d, "ADJ", feat, gender_ok=gu is not None)
                    if e:
                        e.update(dep=d, head=h, relation="amod", kind="amod")
                        edits.append(e)
        # ---- subject-based agreement: verb, predicative adjective, participle
        for s, t in enumerate(info):
            if not t["deprel"].startswith("nsubj") or t["head"] < 0:
                continue
            h = t["head"]
            head = info[h]
            sg, sn = self._subject_features(parse, s)
            kids = children(parse, h)
            # predicative adjective (copula) and participle with essere
            is_pred_adj = head["upos"] == "ADJ" and any(info[c]["deprel"] == "cop" for c in kids)
            is_part = (
                head["upos"] == "VERB" and head["feats"].get("VerbForm") == "Part"
                and any(info[c]["deprel"] in ("aux", "aux:pass") and (info[c]["lemma"] or "") in ("essere", "venire")
                        for c in kids)
            )
            if (is_pred_adj or is_part) and len(head["words"]) == 1 and sn:
                for feat in ("n", "g"):
                    if feat == "g" and not sg:
                        continue
                    if is_pred_adj:
                        e = self._nominal_flip(parse, h, "ADJ", feat, gender_ok=True)
                    else:
                        e = self._participle_flip(parse, h, feat)
                    if e:
                        e.update(dep=s, head=h, relation="nsubj", kind="predicative" if is_pred_adj else "participle")
                        edits.append(e)
            # finite verb
            fin = None
            if head["upos"] in ("VERB", "AUX") and head["feats"].get("VerbForm") == "Fin":
                fin = h
            else:
                for c in sorted(kids):
                    if info[c]["deprel"] in ("aux", "cop", "aux:pass") and info[c]["feats"].get("VerbForm") == "Fin":
                        fin = c
                        break
            if fin is not None and len(info[fin]["words"]) == 1:
                for e in self._verb_flips(parse, fin):
                    e.update(dep=s, head=h, relation="nsubj", kind="subj_verb")
                    edits.append(e)
        return edits

    def _participle_flip(self, parse: Parse, idx: int, feature: str) -> Edit | None:
        t = parse["info"][idx]
        form = core_lower(parse, idx)
        g = G_MAP.get(t["feats"].get("Gender", ""))
        n = N_MAP.get(t["feats"].get("Number", ""))
        analyses = self.lex.find(form, VERBAL_CATS, mood="part", tense="past", g=g, n=n)
        if not analyses:
            return None
        same = [a for a in analyses if a[0] == (t["lemma"] or "").lower()]
        lemma, cat, feats = (same or analyses)[0]
        want = dict(feats)
        want[feature] = FLIP[feats[feature]]
        old_vals = {"mood": "part", "g": feats["g"], "n": feats["n"]}
        cands = [f for f in self.lex.forms(lemma, cat, **want) if f != form]
        cands = [c for c in cands if not self.lex.find(c, VERBAL_CATS, **old_vals)]
        if not cands:
            return None
        return dict(index=idx, new=pick_form(cands), feature={"g": "Gender", "n": "Number"}[feature],
                    old_value=feats[feature], new_value=want[feature], lex="morphit",
                    lemma=lemma, cat=cat, old_feats=old_vals)

    def _verb_flips(self, parse: Parse, idx: int) -> list[Edit]:
        t = parse["info"][idx]
        form = core_lower(parse, idx)
        p = t["feats"].get("Person")
        n = N_MAP.get(t["feats"].get("Number", ""))
        if not p or not n:
            return []
        analyses = [a for a in self.lex.find(form, VERBAL_CATS, p=p, n=n) if a[2].get("mood") in ("ind", "sub", "cond")]
        if not analyses:
            return []
        same = [a for a in analyses if a[0] == (t["lemma"] or "").lower()]
        lemma, cat, feats = (same or analyses)[0]
        old_vals = {"p": p, "n": n}
        out = []
        targets = [("Number", {"p": p, "n": FLIP[n]})] + [
            ("Person", {"p": q, "n": n}) for q in ("1", "2", "3") if q != p
        ]
        for feat_name, tv in targets:
            want = {"mood": feats["mood"], "tense": feats.get("tense", "pres"), **tv}
            cands = [f for f in self.lex.forms(lemma, cat, **want) if f != form]
            # the new form must not also be a valid analysis with the old person/number
            cands = [c for c in cands if not [a for a in self.lex.find(c, VERBAL_CATS, **old_vals) if a[0] == lemma]]
            if cands:
                key = "n" if feat_name == "Number" else "p"
                out.append(dict(index=idx, new=pick_form(cands), feature=feat_name, old_value=old_vals[key],
                                new_value=tv[key], lex="morphit_verb", lemma=lemma, cat=cat,
                                old_feats=old_vals, new_feats=want))
        return out

    def verify_agreement(self, parse: Parse, edit: Edit) -> bool:
        if not self.relation_holds(parse, edit):
            return False
        old = core_lower(parse, edit["index"])
        new = edit["new"].lower()
        if old == new:
            return False
        lex = edit.get("lex")
        key = {"Gender": "g", "Number": "n", "Person": "p"}.get(edit["feature"])
        if lex == "art":
            table = {**DEF_ART, **INDEF_ART}
            if new not in table:
                return False
            g1, n1 = table[new]
            g0, n0 = table.get(old, (None, None))
            if key == "n":
                return n1 != n0
            return g1 is not None and g0 is not None and g0 != g1
        if lex == "artpre":
            if new not in ARTPRE or old not in ARTPRE:
                return False
            (p0, a0), (p1, a1) = ARTPRE[old], ARTPRE[new]
            gn0, gn1 = DEF_ART[a0], DEF_ART[a1]
            if p0 != p1:
                return False
            if key == "n":
                return gn0[1] != gn1[1]
            return gn1[0] is not None and gn0[0] is not None and gn0[0] != gn1[0]
        if lex in ("morphit", "morphit_verb"):
            cats = VERBAL_CATS if lex == "morphit_verb" or edit["cat"] in VERBAL_CATS else {edit["cat"]}
            new_an = [a for a in self.lex.find(new, cats) if a[0] == edit["lemma"]]
            if not new_an:
                return False  # the new form must be a real word form of the same lemma
            if not any(a[2].get(key) == edit["new_value"] for a in new_an):
                return False
            old_feats = edit["old_feats"]
            # must not be analysable with the old feature values (accidental validity)
            return not any(all(a[2].get(k) == v for k, v in old_feats.items()) for a in new_an)
        return False

    # ---------------------------------------------------- wrong function word
    def function_word_candidates(self, parse: Parse) -> list[Edit]:
        info = parse["info"]
        edits: list[Edit] = []
        for i, t in enumerate(info):
            core = core_lower(parse, i)
            if t["contracted"] and core in ARTPRE:
                prep, art = ARTPRE[core]
                for p2 in PREP_STEMS:
                    if p2 != prep:
                        new = artpre_form(p2, art)
                        if new:
                            edits.append(dict(index=i, new=new, set="articulated_preposition"))
                continue
            if len(t["words"]) != 1:
                continue
            if t["upos"] == "ADP" and core in SIMPLE_PREPS:
                for alt in SIMPLE_PREPS:
                    if alt != core:
                        edits.append(dict(index=i, new=alt, set="preposition"))
            elif t["upos"] == "DET" and t["deprel"] == "det" and core in ALLOMORPH_SWAP:
                # il/lo, i/gli, un/uno allomorph confusion; gender/number changes of the
                # article are produced by the agreement operation instead.
                edits.append(dict(index=i, new=ALLOMORPH_SWAP[core], set="article"))
        return edits

    def version_info(self) -> dict[str, str]:
        return {"morph-it": Path(self._lexicon_path).name if self._lexicon_path else "not loaded"}
