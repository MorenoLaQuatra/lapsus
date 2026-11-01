# SPDX-License-Identifier: Apache-2.0
"""Causal LM scoring (log-probabilities) with an on-disk cache.

Used by the clean-sentence LM filter (step f) and the minimal-pair verifier.
Deliberately a causal LM, never a BERT-family model, so the verifier is not the
kind of model students fine-tune on the dataset.

score(text) returns (sum_logprob, n_lm_tokens). The verifier compares mean
log-probabilities (sum / n) so that operations which change the length (missing
or repeated words) are compared fairly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from .utils import KVCache, stable_hash

log = logging.getLogger("lapsus.lm")


class Scorer(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[tuple[float, int]]: ...


def mean_lp(s: tuple[float, int]) -> float:
    return s[0] / max(1, s[1])


class CausalLMScorer:
    def __init__(self, model_name: str, cache_dir: str | Path, batch_size: int = 64, device: str = "auto"):
        self.name = model_name
        self.batch_size = batch_size
        self.device_pref = device
        self.cache = KVCache(Path(cache_dir) / "lm_scores.sqlite")
        self._model = None
        self._tok = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self.device_pref
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self._device = dev
        log.info("Loading causal LM %s on %s", self.name, dev)
        self._tok = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.float32).to(dev).eval()
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token

    def key(self, text: str) -> str:
        return stable_hash("lm", self.name, text)

    def score(self, texts: list[str]) -> list[tuple[float, int]]:
        keys = [self.key(t) for t in texts]
        cached = self.cache.get_many(keys)
        todo = sorted({t for t, k in zip(texts, keys) if k not in cached}, key=len)
        if todo:
            self._load()
            new = {}
            for i in range(0, len(todo), self.batch_size):
                batch = todo[i : i + self.batch_size]
                for t, s in zip(batch, self._score_batch(batch)):
                    new[self.key(t)] = list(s)
                if len(new) >= 4096:
                    self.cache.set_many(new)
                    cached.update(new)
                    new = {}
            self.cache.set_many(new)
            cached.update(new)
        return [tuple(cached[k]) for k in keys]  # type: ignore[misc]

    def _score_batch(self, texts: list[str]) -> list[tuple[float, int]]:
        import torch

        tok, model = self._tok, self._model
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        seqs = [[bos] + e for e in enc]
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            att[j, : len(s)] = 1
        ids, att = ids.to(self._device), att.to(self._device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = ids[:, 1:]
        lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * att[:, 1:]
        sums = lp.sum(-1).tolist()
        counts = att[:, 1:].sum(-1).tolist()
        return [(float(s), int(c)) for s, c in zip(sums, counts)]


class TableScorer:
    """Deterministic scorer for tests: looks scores up in a dict, defaults to 0."""

    def __init__(self, table: dict[str, float] | None = None, name: str = "table"):
        self.table = table or {}
        self.name = name

    def score(self, texts: list[str]) -> list[tuple[float, int]]:
        return [(float(self.table.get(t, 0.0)), 1) for t in texts]
