# SPDX-License-Identifier: Apache-2.0
"""English: whitespace tokenization, LemmInflect morphology, corruption target rules."""

from __future__ import annotations

from functools import lru_cache

from .base import Edit, Language, Parse, children, core_lower

MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought"}
ARTICLES = {"a", "an", "the"}
VOWEL_SOUND_EXCEPTIONS_A = {"one", "once", "uni", "use", "user", "usual", "usually", "euro", "europe",
                            "european", "university", "unique", "unit", "united", "union", "useful",
                            "uniform", "universe", "unicorn", "utensil", "eulogy", "ewe", "ouija"}
VOWEL_SOUND_EXCEPTIONS_AN = {"hour", "hours", "honest", "honestly", "honor", "honour", "heir", "herb"}

BE_FLIP = {
    # form: (new form, feature, old value, new value)
    "is": ("are", "Number", "Sing", "Plur"),
    "are": ("is", "Number", "Plur", "Sing"),
    "am": ("is", "Person", "1", "3"),
    "was": ("were", "Number", "Sing", "Plur"),
    "were": ("was", "Number", "Plur", "Sing"),
}
CONTRACTION_FLIP = {
    "isn't": ("aren't", "Number", "Sing", "Plur"),
    "aren't": ("isn't", "Number", "Plur", "Sing"),
    "wasn't": ("weren't", "Number", "Sing", "Plur"),
    "weren't": ("wasn't", "Number", "Plur", "Sing"),
    "doesn't": ("don't", "Number", "Sing", "Plur"),
    "don't": ("doesn't", "Number", "Plur", "Sing"),
    "hasn't": ("haven't", "Number", "Sing", "Plur"),
    "haven't": ("hasn't", "Number", "Plur", "Sing"),
}
DEM_FLIP = {"this": "these", "these": "this", "that": "those", "those": "that"}
DEM_NUMBER = {"this": "Sing", "that": "Sing", "these": "Plur", "those": "Plur"}
SING_DETS = {"a", "an", "every", "each", "another"}
PLUR_DETS = {"these", "those", "many", "several", "few", "both"}

FUNCTION_SETS = {
    "in/on/at": ["in", "on", "at"],
    "to/for/of": ["to", "for", "of"],
}
NO_REPEAT = {"that", "had", "do", "very", "so", "really", "much", "many", "far", "more", "no", "yes",
             "bye", "ha", "again", "over", "out", "her", "is", "in", "on", "by", "too", "well"}
WORD_ORDER_PREFERRED = {"DET", "ADP", "AUX", "PRON", "VERB", "PART"}
WORD_ORDER_BANNED = {"ADV", "PUNCT", "INTJ", "X", "SYM"}


@lru_cache(maxsize=None)
def _inflections(lemma: str, upos: str) -> dict[str, tuple[str, ...]]:
    from lemminflect import getAllInflections

    return getAllInflections(lemma, upos=upos)


def a_or_an(next_word: str) -> str:
    w = next_word.lower()
    if w in VOWEL_SOUND_EXCEPTIONS_AN or any(w.startswith(x) for x in ("hour", "honest", "honou", "honor")):
        return "an"
    if any(w.startswith(x) for x in VOWEL_SOUND_EXCEPTIONS_A):
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def _finite_for_subject(parse: Parse, head: int) -> int | None:
    """Index of the finite verb that agrees with the subject of `head`."""
    info = parse["info"]
    h = info[head]
    if h["upos"] in ("VERB", "AUX") and h["feats"].get("VerbForm") == "Fin":
        return head
    for c in sorted(children(parse, head)):
        t = info[c]
        if t["deprel"] in ("aux", "cop", "aux:pass") and t["feats"].get("VerbForm") == "Fin":
            return c
    return None


class English(Language):
    code = "en"
    name = "English"
    stanza_lang = "en"
    lingua_name = "ENGLISH"
    lt_codes = ["en-US", "en-GB"]
    default_lm = "openai-community/gpt2"
    tatoeba_code = "eng"

    def always_capitalized(self) -> set[str]:
        return {"I", "I'm", "I'd", "I'll", "I've"}

    # ------------------------------------------------------------ missing word
    def missing_word_targets(self, parse: Parse) -> list[int]:
        info = parse["info"]
        out = []
        for i, t in enumerate(info):
            if len(t["words"]) != 1:
                continue  # never delete contractions such as "don't"
            core = core_lower(parse, i)
            head = t["head"]
            if t["upos"] == "DET" and core in ARTICLES and t["deprel"] == "det":
                if head > i and info[head]["upos"] == "NOUN" and info[head]["feats"].get("Number") != "Plur":
                    out.append(i)
            elif t["upos"] == "ADP" and t["deprel"] == "case" and head > i:
                out.append(i)
            elif t["upos"] == "AUX" and t["deprel"] in ("aux", "cop", "aux:pass"):
                if t["lemma"] in MODALS or core in MODALS:
                    continue
                subj = [c for c in children(parse, head) if info[c]["deprel"].startswith("nsubj")]
                if subj and all(s < i for s in subj):  # not inverted (questions)
                    out.append(i)
            elif t["upos"] == "PART" and core == "to" and t["deprel"] == "mark":
                out.append(i)
        return out

    # -------------------------------------------------------------- word order
    def word_order_allowed(self, parse: Parse, i: int, j: int) -> bool:
        info = parse["info"]
        a, b = info[i], info[j]
        if a["upos"] in WORD_ORDER_BANNED or b["upos"] in WORD_ORDER_BANNED:
            return False
        if a["upos"] == b["upos"]:
            return False
        if not ({a["upos"], b["upos"]} & WORD_ORDER_PREFERRED):
            return False
        if core_lower(parse, i) == core_lower(parse, j):
            return False
        # Subject-auxiliary swaps produce valid questions ("he is" -> "is he").
        for x, y in ((a, b), (b, a)):
            if x["deprel"].startswith(("nsubj", "expl")) and y["upos"] == "AUX":
                return False
        return True

    # ---------------------------------------------------------- repeated word
    def repeat_allowed(self, parse: Parse, i: int) -> bool:
        t = parse["info"][i]
        return t["upos"] not in ("PUNCT", "SYM", "X", "INTJ") and core_lower(parse, i) not in NO_REPEAT

    # --------------------------------------------------------------- agreement
    def agreement_candidates(self, parse: Parse) -> list[Edit]:
        info = parse["info"]
        edits: list[Edit] = []
        for s, t in enumerate(info):
            if t["deprel"] not in ("nsubj", "nsubj:pass") or t["head"] < 0:
                continue
            fin = _finite_for_subject(parse, t["head"])
            if fin is None:
                continue
            e = self._subject_verb_edit(parse, fin, s)
            if e:
                e.update(dep=s, head=t["head"], relation="nsubj", kind="subj_verb")
                edits.append(e)
        for d, t in enumerate(info):
            if t["deprel"] != "det" or t["head"] < 0 or len(t["words"]) != 1:
                continue
            n = t["head"]
            noun = info[n]
            dform = core_lower(parse, d)
            if dform in ("a", "an") and d + 1 < len(info):
                edits.append(dict(index=d, new="an" if dform == "a" else "a", feature="Phon",
                                  old_value=dform, new_value="an" if dform == "a" else "a",
                                  lex="a_an", dep=d, head=n, relation="det", kind="a_an"))
            if noun["upos"] != "NOUN" or len(noun["words"]) != 1:
                continue
            nnum = noun["feats"].get("Number")
            if dform in DEM_FLIP and nnum == DEM_NUMBER[dform]:
                new = DEM_FLIP[dform]
                edits.append(dict(index=d, new=new, feature="Number", old_value=DEM_NUMBER[dform],
                                  new_value=DEM_NUMBER[new], lex="dem", dep=d, head=n,
                                  relation="det", kind="det_noun"))
            noun_form = core_lower(parse, n)
            lemma = (noun["lemma"] or noun_form).lower()
            infl = _inflections(lemma, "NOUN")
            if (dform in SING_DETS or dform in DEM_FLIP) and nnum == "Sing":
                if noun_form in infl.get("NN", ()) and infl.get("NNS"):
                    new = infl["NNS"][0]
                    if new != noun_form and new not in infl.get("NN", ()):
                        edits.append(dict(index=n, new=new, feature="Number", old_value="Sing",
                                          new_value="Plur", lex="noun", lemma=lemma, old_tag="NN",
                                          new_tag="NNS", dep=d, head=n, relation="det", kind="det_noun"))
            elif (dform in PLUR_DETS) and nnum == "Plur":
                if noun_form in infl.get("NNS", ()) and infl.get("NN"):
                    new = infl["NN"][0]
                    if new != noun_form and new not in infl.get("NNS", ()):
                        edits.append(dict(index=n, new=new, feature="Number", old_value="Plur",
                                          new_value="Sing", lex="noun", lemma=lemma, old_tag="NNS",
                                          new_tag="NN", dep=d, head=n, relation="det", kind="det_noun"))
        return edits

    def _subject_verb_edit(self, parse: Parse, fin: int, subj: int) -> Edit | None:
        t = parse["info"][fin]
        form = core_lower(parse, fin)
        feats = t["feats"]
        if feats.get("Mood") not in (None, "Ind"):
            return None
        if form in CONTRACTION_FLIP:
            new, feat, ov, nv = CONTRACTION_FLIP[form]
            return dict(index=fin, new=new, feature=feat, old_value=ov, new_value=nv, lex="contraction")
        if len(t["words"]) != 1:
            return None
        if form in BE_FLIP:
            new, feat, ov, nv = BE_FLIP[form]
            subj_info = parse["info"][subj]
            if form == "were" and core_lower(parse, subj) != "you" and subj_info["feats"].get("Number") == "Sing":
                return None  # "if I were" subjunctive: "was" may be acceptable
            return dict(index=fin, new=new, feature=feat, old_value=ov, new_value=nv, lex="be")
        lemma = (t["lemma"] or "").lower()
        if not lemma or lemma in MODALS or feats.get("Tense") != "Pres":
            return None
        infl = _inflections(lemma, "VERB")
        vbz, vbp, vbd = infl.get("VBZ", ()), infl.get("VBP", ()), infl.get("VBD", ())
        if form in vbz and vbp and vbp[0] != form:
            return dict(index=fin, new=vbp[0], feature="Number", old_value="Sing", new_value="Plur",
                        lex="verb", lemma=lemma, old_tag="VBZ", new_tag="VBP")
        if form in vbp and form not in vbz and form not in vbd and vbz and vbz[0] != form:
            return dict(index=fin, new=vbz[0], feature="Number", old_value="Plur", new_value="Sing",
                        lex="verb", lemma=lemma, old_tag="VBP", new_tag="VBZ")
        return None

    def verify_agreement(self, parse: Parse, edit: Edit) -> bool:
        if not self.relation_holds(parse, edit):
            return False
        old = core_lower(parse, edit["index"])
        new = edit["new"].lower()
        if old == new:
            return False
        lex = edit.get("lex")
        if lex == "a_an":
            return {old, new} == {"a", "an"}
        if lex == "dem":
            return DEM_FLIP.get(old) == new and DEM_NUMBER[old] != DEM_NUMBER[new]
        if lex == "be":
            return BE_FLIP.get(old, (None,))[0] == new
        if lex == "contraction":
            return CONTRACTION_FLIP.get(old, (None,))[0] == new
        if lex in ("verb", "noun"):
            upos = "VERB" if lex == "verb" else "NOUN"
            infl = _inflections(edit["lemma"], upos)
            return (
                edit["old_tag"] != edit["new_tag"]
                and old in infl.get(edit["old_tag"], ())
                and new in infl.get(edit["new_tag"], ())
                and new not in infl.get(edit["old_tag"], ())
            )
        return False

    # ---------------------------------------------------- wrong function word
    def function_word_candidates(self, parse: Parse) -> list[Edit]:
        info = parse["info"]
        edits: list[Edit] = []
        for i, t in enumerate(info):
            if len(t["words"]) != 1:
                continue
            core = core_lower(parse, i)
            if t["upos"] == "ADP":
                for set_name, members in FUNCTION_SETS.items():
                    if core in members:
                        for alt in members:
                            if alt != core:
                                edits.append(dict(index=i, new=alt, set=set_name))
            elif core == "the" and t["deprel"] == "det" and i + 1 < len(info):
                # Restricted a/the confusion: "a" replacing "the" is only produced where it is
                # ungrammatical (plural head noun or superlative adjective), never "the" for "a".
                head = info[t["head"]] if t["head"] >= 0 else None
                nxt = info[i + 1]
                plural_head = head is not None and head["upos"] == "NOUN" and head["feats"].get("Number") == "Plur"
                superlative = nxt["upos"] == "ADJ" and nxt["feats"].get("Degree") == "Sup"
                if plural_head or superlative:
                    edits.append(dict(index=i, new=a_or_an(core_lower(parse, i + 1)), set="a/the"))
        return edits

    def version_info(self) -> dict[str, str]:
        from ..utils import package_version

        return {"lemminflect": package_version("lemminflect")}
