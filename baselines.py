# SPDX-License-Identifier: Apache-2.0
"""TF-IDF + logistic regression baselines (spec section 7).

For each language: train on `train`, evaluate on every other split, for the
binary task and the multilabel task, with unigrams and then 1-2 grams over the
Lapsus tokens. If unigram binary accuracy exceeds 70 percent the script warns
and lists the top features (a sign of surface cues). As a reality check the
binary models are also evaluated on CoLA (English, in-domain validation) and
ItaCoLA (Italian, test split). These sets are used for evaluation only, never
for training.

Run with: python -m lapsus baselines [--lang en|it|both]
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lapsus import OPS  # noqa: E402
from lapsus.lang import get_language  # noqa: E402
from lapsus.utils import write_json, write_text  # noqa: E402

log = logging.getLogger("lapsus.baselines")
WARN_ACC = 0.70


def _load(out_dir: Path, lang: str, split: str) -> list[dict]:
    p = out_dir / "jsonl" / lang / f"{split}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def _normalize(text: str, lang: str, mode: str, cfg: dict) -> str:
    L = get_language(lang, cfg)
    toks = L.tokenize(re.sub(r"\s+", " ", text.strip()))
    if mode != "none":
        toks = [t.lower() for t in toks]
    if mode == "lower_nopunct" and toks:
        toks[-1] = toks[-1].rstrip(".!?") or toks[-1]
    return " ".join(toks)


def _eval_sets(cfg: dict, lang: str) -> dict[str, tuple[list[str], list[int]]]:
    raw = Path(cfg["paths"]["raw"]) / "eval"
    out = {}
    if lang == "en":
        p = raw / "validation-00000-of-00001.parquet"
        if p.exists():
            import pandas as pd

            df = pd.read_parquet(p)
            # CoLA label 1 = acceptable -> Lapsus "ok" (0); 0 = unacceptable -> "not_ok" (1)
            out["CoLA (in-domain validation)"] = (
                [_normalize(s, lang, cfg["normalize"], cfg) for s in df["sentence"]],
                [0 if int(y) == 1 else 1 for y in df["label"]],
            )
    if lang == "it":
        p = raw / "ItaCoLA_dataset.tsv"
        if p.exists():
            rows = [r for r in csv.DictReader(p.open(encoding="utf-8"), delimiter="\t") if r["Split"] == "test"]
            out["ItaCoLA (test)"] = (
                [_normalize(r["Sentence"], lang, cfg["normalize"], cfg) for r in rows],
                [0 if r["Acceptability"].strip() == "1" else 1 for r in rows],
            )
    return out


def _metrics_binary(y, p) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    return {"accuracy": round(accuracy_score(y, p), 4), "f1_not_ok": round(f1_score(y, p, pos_label=1), 4),
            "mcc": round(matthews_corrcoef(y, p), 4), "n": len(y)}


def _metrics_multilabel(Y, P) -> dict:
    from sklearn.metrics import f1_score

    return {"micro_f1": round(f1_score(Y, P, average="micro", zero_division=0), 4),
            "macro_f1": round(f1_score(Y, P, average="macro", zero_division=0), 4),
            "per_label_f1": {op: round(v, 4) for op, v in zip(OPS, f1_score(Y, P, average=None, zero_division=0))},
            "n": len(Y)}


def run_language(cfg: dict, lang: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    out_dir = Path(cfg["paths"]["out"])
    train = _load(out_dir, lang, "train")
    if not train:
        raise SystemExit(f"no train split for {lang}: run `python -m lapsus build` first")
    splits = {s: _load(out_dir, lang, s) for s in
              ("validation", "test", "test_unseen_lemmas", "test_ood_source", "test_natural")}
    splits = {k: v for k, v in splits.items() if v}
    external = _eval_sets(cfg, lang)
    res: dict = {"lang": lang, "train_rows": len(train), "settings": {}}
    for name, ngram in (("unigram", (1, 1)), ("uni+bigram", (1, 2))):
        vec = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, ngram_range=ngram,
                              min_df=2, sublinear_tf=True)
        X = vec.fit_transform([" ".join(r["tokens"]) for r in train])
        y = [1 if r["binary"] == "not_ok" else 0 for r in train]
        Y = [r["multilabel"] for r in train]
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=cfg["seed"]).fit(X, y)
        mclf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0, random_state=cfg["seed"])).fit(X, Y)
        entry: dict = {"binary": {}, "multilabel": {}, "reality_check": {}}
        for s, rows in splits.items():
            Xs = vec.transform([" ".join(r["tokens"]) for r in rows])
            ys = [1 if r["binary"] == "not_ok" else 0 for r in rows]
            entry["binary"][s] = _metrics_binary(ys, clf.predict(Xs))
            if s != "test_natural":
                entry["multilabel"][s] = _metrics_multilabel([r["multilabel"] for r in rows], mclf.predict(Xs))
        for ename, (texts, ys) in external.items():
            entry["reality_check"][ename] = _metrics_binary(ys, clf.predict(vec.transform(texts)))
        acc = entry["binary"].get("test", {}).get("accuracy", 0)
        if name == "unigram":
            coefs = clf.coef_[0]
            names = vec.get_feature_names_out()
            order = coefs.argsort()
            entry["top_features"] = {
                "not_ok": [(names[i], round(float(coefs[i]), 3)) for i in order[::-1][:20]],
                "ok": [(names[i], round(float(coefs[i]), 3)) for i in order[:20]],
            }
            if acc > WARN_ACC:
                entry["warning"] = (f"unigram binary test accuracy {acc} > {WARN_ACC}: the task is partly solvable "
                                    "from surface lexical cues (see top_features)")
                log.warning("[%s] %s", lang, entry["warning"])
        res["settings"][name] = entry
    return res


def format_results(all_res: dict) -> str:
    L = ["# Lapsus baselines (TF-IDF + logistic regression)", ""]
    for lang, res in all_res.items():
        L += [f"## {lang} (train rows: {res['train_rows']})", ""]
        for name, e in res["settings"].items():
            L += [f"### {name}", "", "| split | binary acc | F1 not_ok | MCC | multilabel micro F1 | macro F1 |",
                  "|---|---|---|---|---|---|"]
            for s, m in e["binary"].items():
                ml = e["multilabel"].get(s, {})
                L.append(f"| {s} | {m['accuracy']} | {m['f1_not_ok']} | {m['mcc']} | {ml.get('micro_f1', 'n/a')} | "
                         f"{ml.get('macro_f1', 'n/a')} |")
            for s, m in e["reality_check"].items():
                L.append(f"| {s} (reality check, eval only) | {m['accuracy']} | {m['f1_not_ok']} | {m['mcc']} | n/a | n/a |")
            if e.get("warning"):
                L += ["", f"**Warning:** {e['warning']}"]
            if e.get("top_features"):
                L += ["", "Top not_ok features: " + ", ".join(f"{w} ({c})" for w, c in e["top_features"]["not_ok"][:12]),
                      "", "Top ok features: " + ", ".join(f"{w} ({c})" for w, c in e["top_features"]["ok"][:12])]
            L.append("")
    return "\n".join(L) + "\n"


def main(cfg: dict, langs: list[str]) -> dict:
    all_res = {lg: run_language(cfg, lg) for lg in langs}
    out = Path(cfg["paths"]["out"]) / "reports"
    write_json(out / "baselines.json", all_res)
    text = format_results(all_res)
    write_text(out / "baselines.md", text)
    print(text)
    return all_res


if __name__ == "__main__":
    from lapsus.utils import languages_for, load_config, setup_logging

    setup_logging()
    c = load_config()
    main(c, languages_for(c["lang"]))
