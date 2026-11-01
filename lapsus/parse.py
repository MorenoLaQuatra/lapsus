# SPDX-License-Identifier: Apache-2.0
"""Stanza parsing with an on-disk cache and alignment to Lapsus tokens.

Stanza tokenizes the raw sentence itself (so multi-word tokens such as Italian
"della" -> "di la" or English "don't" are expanded properly). Its tokens are then
aligned by character offsets to the Lapsus whitespace/elision tokens, and each
Lapsus token gets the analysis of its first non-punctuation word (articulated
prepositions are marked `contracted`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .lang.base import Language, Parse, feats_to_dict
from .utils import KVCache, chunked, package_version, stable_hash

log = logging.getLogger("lapsus.parse")


def align(text: str, lang: Language, sentences: list[list[dict[str, Any]]]) -> Parse:
    """Build a Lapsus parse from Stanza output.

    `sentences` is a list of sentences, each a list of tokens:
        {"start": int, "end": int, "words": [{"id", "text", "lemma", "upos", "feats", "head", "deprel"}]}
    """
    tokens = lang.tokenize(text)
    spans = lang.token_spans(text)
    parse: Parse = {"text": text, "tokens": tokens, "info": [], "n_roots": 0, "ok": False, "reason": ""}
    n_roots = sum(1 for s in sentences for t in s for w in t["words"] if w["head"] == 0)
    parse["n_roots"] = n_roots
    if len(sentences) != 1:
        parse["reason"] = f"stanza found {len(sentences)} sentences"
        return parse
    sent = sentences[0]
    # map every stanza word id -> lapsus token index
    word_to_tok: dict[int, int] = {}
    tok_words: list[list[dict]] = [[] for _ in tokens]
    for st in sent:
        idx = None
        for i, (a, b) in enumerate(spans):
            if a <= st["start"] < b:
                idx = i
                break
        if idx is None or st["end"] > spans[idx][1]:
            parse["reason"] = "token alignment failed"
            return parse
        for w in st["words"]:
            word_to_tok[w["id"]] = idx
            tok_words[idx].append(w)
    info = []
    for i, words in enumerate(tok_words):
        content = [w for w in words if w["upos"] != "PUNCT"]
        if not content:
            parse["reason"] = f"token {i} has no non-punctuation word"
            return parse
        prim = content[0]
        head_tok = -1
        # head of the token: first content word whose head lies outside this token
        for w in content:
            if w["head"] == 0:
                head_tok = -1
                break
            ht = word_to_tok.get(w["head"])
            if ht is not None and ht != i:
                head_tok = ht
                break
        contracted = len(content) > 1 and content[0]["upos"] == "ADP" and any(w["upos"] == "DET" for w in content[1:])
        info.append({
            "upos": prim["upos"],
            "lemma": prim.get("lemma") or "",
            "feats": feats_to_dict(prim.get("feats")),
            "head": head_tok,
            "deprel": prim["deprel"] if not any(w["head"] == 0 for w in content) else "root",
            "words": [[w["text"], w.get("lemma") or "", w["upos"], w.get("feats") or "", w["deprel"]] for w in content],
            "contracted": contracted,
        })
    parse["info"] = info
    if any(w["upos"] == "X" for s in sentences for t in s for w in t["words"]):
        parse["reason"] = "X tag"
        return parse
    if n_roots != 1:
        parse["reason"] = f"{n_roots} roots"
        return parse
    parse["ok"] = True
    return parse


class Parser:
    def __init__(self, lang: Language, cfg: dict):
        self.lang = lang
        self.cfg = cfg
        self.model_dir = str(Path(cfg["paths"]["cache"]) / "stanza_resources")
        self.version = package_version("stanza")
        self.cache = KVCache(Path(cfg["paths"]["cache"]) / f"parse_{lang.code}.sqlite")
        self._nlp = None

    def _pipeline(self):
        if self._nlp is None:
            import stanza
            import torch

            use_gpu = self.cfg["stanza"].get("use_gpu", "auto")
            if use_gpu == "auto":
                use_gpu = torch.cuda.is_available()
            self._nlp = stanza.Pipeline(
                self.lang.stanza_lang, model_dir=self.model_dir,
                processors="tokenize,mwt,pos,lemma,depparse", use_gpu=bool(use_gpu),
                download_method=None, verbose=False,
                pos_batch_size=3000, depparse_batch_size=3000, lemma_batch_size=3000,
            )
        return self._nlp

    def key(self, text: str) -> str:
        return stable_hash("stanza", self.version, self.lang.code, text)

    def parse(self, texts: list[str]) -> list[Parse]:
        keys = [self.key(t) for t in texts]
        cached = self.cache.get_many(keys)
        todo = sorted({t for t, k in zip(texts, keys) if k not in cached})
        if todo:
            import stanza

            nlp = self._pipeline()
            log.info("[%s] parsing %d sentences with Stanza %s", self.lang.code, len(todo), self.version)
            bs = int(self.cfg["stanza"].get("batch_size", 256))
            for batch in chunked(todo, bs * 8):
                docs = nlp.bulk_process([stanza.Document([], text=t) for t in batch])
                new = {}
                for t, doc in zip(batch, docs):
                    try:
                        sents = [[{"start": tok.start_char, "end": tok.end_char,
                                   "words": [{"id": w.id, "text": w.text, "lemma": w.lemma, "upos": w.upos,
                                              "feats": w.feats, "head": w.head, "deprel": w.deprel}
                                             for w in tok.words]} for tok in s.tokens] for s in doc.sentences]
                        p = align(t, self.lang, sents)
                    except Exception as exc:  # parsing failure is a filter rejection
                        p = {"text": t, "tokens": self.lang.tokenize(t), "info": [], "n_roots": 0,
                             "ok": False, "reason": f"failure: {type(exc).__name__}"}
                    new[self.key(t)] = p
                self.cache.set_many(new)
                cached.update(new)
        return [cached[k] for k in keys]
