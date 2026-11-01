# SPDX-License-Identifier: Apache-2.0
"""Dataset card and Hugging Face Hub mirror.

`push` is a dry run unless --push-for-real is given. The canonical home of each
dataset version is a GitHub Release (see lapsus/package.py); the Hub is a mirror.
The token comes from HF_TOKEN or a prior `huggingface-cli login`; it is never
stored or printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import MULTICLASS, OPS, TAGS
from .export import ordered_splits
from .utils import write_text

RELATED_WORK = [
    ("BLiMP", "Warstadt et al. (2020), The Benchmark of Linguistic Minimal Pairs for English, TACL."),
    ("CoLA", "Warstadt, Singh and Bowman (2019), Neural Network Acceptability Judgments, TACL."),
    ("ItaCoLA", "Trotta et al. (2021), Monolingual and Cross-Lingual Acceptability Judgments with the Italian CoLA corpus, Findings of EMNLP."),
    ("BLiMP-IT", "Italian minimal-pair benchmarks modelled on BLiMP (see the BLiMP-IT and related Italian minimal-pair releases)."),
    ("AcCompl-it", "Brunato et al. (2020), AcCompl-it at EVALITA 2020: acceptability and complexity prediction for Italian."),
    ("DaLA", "Danish linguistic acceptability dataset built by corrupting correct sentences with controlled error types."),
]


def _fmt_dist(d: dict) -> str:
    if not d:
        return "n/a"
    return f"mean {d['mean']}, p05 {d['p05']}, p50 {d['p50']}, p95 {d['p95']} (n={d['n']})"


def render_card(info: dict, repo_id: str) -> str:
    configs = info["configs"]
    yaml_cfgs = []
    for name, c in configs.items():
        yaml_cfgs.append(f"- config_name: {name}")
        yaml_cfgs.append("  data_files:")
        for split in ordered_splits(c["splits"]):
            yaml_cfgs.append(f"  - split: {split}")
            yaml_cfgs.append(f"    path: {name}/{split}-*")
    langs = sorted({lg for lg in info["languages"]})
    total = sum(sum(c["splits"].values()) for c in configs.values())
    size = "10K<n<100K" if total < 100_000 else "100K<n<1M"
    head = "\n".join([
        "---",
        "language:",
        *[f"- {lg}" for lg in langs],
        f"license: {info['repo_license']}",
        "pretty_name: Lapsus",
        f"size_categories:\n- {size}",
        "task_categories:",
        "- text-classification",
        "- token-classification",
        "tags:",
        "- synthetic",
        "- grammar",
        "- acceptability",
        "configs:",
        *yaml_cfgs,
        "---",
    ])
    L: list[str] = [head, "", "# Lapsus", ""]
    L += ["## Purpose", "",
          "Lapsus is a bilingual (English and Italian) grammatical acceptability dataset for teaching NLP with "
          "transformers (BERT, RoBERTa, XLM-R). Clean sentences come from open corpora, pass a chain of filters, "
          "and half of them are corrupted with named operations whose output is verified by independent checks. "
          "The same rows support four tasks: binary acceptability, multiclass error type, multi-label error types "
          "and token-level error tagging.", "",
          f"Canonical releases live on GitHub (https://github.com/MorenoLaQuatra/lapsus/releases); this Hub repository "
          f"({repo_id}) is a mirror. Dataset version: {info['version']}, seed {info['seed']}.", ""]
    L += ["## Configs and splits", "", "| config | licence | " + " | ".join(
        ["train", "validation", "test", "test_unseen_lemmas", "test_ood_source", "test_natural"]) + " |",
          "|---|---|" + "---|" * 6]
    for name, c in configs.items():
        cells = [str(c["splits"].get(s, "")) for s in
                 ["train", "validation", "test", "test_unseen_lemmas", "test_ood_source", "test_natural"]]
        L.append(f"| {name} | {c['license']} | " + " | ".join(cells) + " |")
    L += ["", "- `both` concatenates `en` and `it` (use the `lang` column).",
          "- `parallel` holds Tatoeba-linked English/Italian translation pairs (`pair_id`), both members in the same "
          "split, for cross-lingual demos. Members can be clean or corrupted independently.", ""]
    L += ["## Schema", "",
          "One row per source sentence; no source sentence is reused.", "",
          "| column | type | description |", "|---|---|---|",
          "| id | string | `<lang>-<source>-<source_id>` |",
          "| lang | string | `en` or `it` |",
          "| text | string | the (possibly corrupted) sentence after normalization |",
          "| tokens | list of string | tokens of `text` (see Tokenization) |",
          "| clean_text | string | the clean sentence, same normalization |",
          "| clean_text_raw | string | the clean sentence as in the source (NFC, trimmed) |",
          "| binary | ClassLabel | ok, not_ok |",
          f"| multiclass | ClassLabel | {', '.join(MULTICLASS)} |",
          f"| multilabel | 5 x int8 | fixed order: {', '.join(OPS)} |",
          f"| token_tags | list of ClassLabel | {', '.join(TAGS)} |",
          "| token_binary | list of int8 | 1 if the token carries an error tag |",
          "| n_corruptions | int8 | 0, 1, 2 or 3 |",
          "| source, source_id, source_author | string | provenance (Tatoeba sentence id and contributor username) |",
          "| license | string | SPDX licence of the sentence |",
          "| split | string | split name |", "",
          "In `test_natural` only `binary` and `token_binary` are defined (real errors have no operation type): "
          "`multiclass` and `token_tags` are -1 (null in csv/jsonl) and `multilabel`, `n_corruptions` are -1.", "",
          "CSV files store list columns as JSON strings and labels as names; JSONL stores labels as names; the "
          "Hub/DatasetDict version stores ClassLabel integers.", "",
          f"Mix per split: {int(info['mix']['clean'] * 100)} percent clean, {int(info['mix']['single'] * 100)} percent "
          f"one corruption (balanced over the 5 operations), {int(info['mix']['multi'] * 100)} percent two or three "
          "different corruptions.", ""]
    L += ["## Corruption operations", "",
          "- **word_order**: swap two tokens at distance up to 3; both tokens are tagged. Swaps must involve a "
          "determiner, preposition, auxiliary, pronoun or verb; adverbs never move; English never swaps a subject with "
          "an auxiliary (that makes a valid question); Italian never swaps ADJ and NOUN, two NOUNs, two arguments of one "
          "verb, or a subject with its verb.",
          "- **missing_word**: delete one function word (article, preposition, non-modal auxiliary, infinitival "
          "\"to\"; Italian articles, simple and articulated prepositions, auxiliaries and reflexive clitics).",
          "- **repeated_word**: duplicate a token right after itself; the duplicate is tagged. Words whose doubling can "
          "be grammatical (\"that that\", \"had had\", \"molto molto\") are excluded.",
          "- **agreement**: EN subject-verb (LemmInflect VBZ/VBP, be, negative contractions), determiner-noun number "
          "(this/these, a + plural noun), a/an. IT gender and number across DET (articles, articulated prepositions, "
          "possessives, demonstratives), NOUN and ADJ; subject-verb person and number; predicative adjective and "
          "participle (with essere) agreement. Forms come from Morph-it! (same lemma, different feature) and must be "
          "real word forms that cannot also carry the original feature value. Only relations present in the clean "
          "parse (nsubj, det, amod) are targeted.",
          "- **wrong_function_word**: confusion sets. EN: in/on/at, to/for/of, and a restricted a/the confusion "
          "(\"a\" for \"the\" only before plural nouns or superlatives). IT: a/di/da/in/su/per/con, articulated "
          "prepositions (preposition part changed), article allomorphs il/lo, i/gli, un/uno (article gender/number "
          "changes belong to agreement).",
          "- **multiple**: 2 or 3 different operations applied sequentially on non-overlapping tokens.", ""]
    L += ["## Missing-word tag convention", "",
          "A deleted word leaves no token to tag, so the token right after the gap is tagged `missing_word`; if the "
          "gap is at the end of the sentence, the last token is tagged.", ""]
    L += ["## Tokenization and normalization", "",
          "Tokens are whitespace separated. In Italian, elided forms are split off and keep their apostrophe "
          "(\"l'uomo\" -> \"l'\", \"uomo\"; \"dell'acqua\" -> \"dell'\", \"acqua\") and `text` glues them back. "
          "Punctuation stays attached to words. "
          f"Normalization: `{info['normalize']}` (lowercase and strip the final punctuation by default); "
          "`clean_text_raw` keeps the original form.", ""]
    L += ["## Sources and attribution", ""]
    for s in info["sources"]:
        if s["enabled"]:
            dates = ", ".join(s["export_dates"]) or "n/a"
            L.append(f"- **{s['name']}** ({s['url']}), licence {s['license']}, export date(s): {dates}. {s['attribution']}")
    L += ["", "Full attribution, including every Tatoeba contributor, is in ATTRIBUTION.md.", ""]
    L += ["## Licence mode and pool sizes", "",
          f"Requested mode: `{info['license_mode_requested']}`. Labels, corruptions and annotations are Apache-2.0.", "",
          "| lang | mode used | reason | permissive pool | total pool | threshold | config licence |",
          "|---|---|---|---|---|---|---|"]
    for lg, li in info["languages"].items():
        d = li["decision"]
        L.append(f"| {lg} | {d['mode']} | {d['reason']} | {d['permissive_pool']} | {d['total_pool']} | "
                 f"{d['threshold']} | {li['license']['hf']} |")
    L += ["", f"Hub repository licence (most restrictive across configs): `{info['repo_license']}`.", ""]
    for lg, li in info["languages"].items():
        if li.get("extras_excluded"):
            L.append(f"- {lg}: not built because of the licence mode: {'; '.join(li['extras_excluded'])}")
    L += [""]
    L += ["## Filters and verification", "",
          "Clean-sentence filters: (a) NFC, one sentence, 5 to 20 tokens; (b) no digits, URLs, emoji, quotes, "
          "brackets, symbols, all-caps, at most one proper noun, any single proper noun capped at 2 percent of the "
          "pool; (c) language ID (lingua); (d) LanguageTool with no grammar or typo match (style, casing and "
          "punctuation rules ignored); (e) Stanza parse with exactly one root and no X tags; (f) causal LM filter "
          "dropping the lowest 5 percent mean log-prob per length bucket; (g) exact, normalized and MinHash "
          "near-duplicate removal on lemma shingles.", ""]
    for lg, li in info["languages"].items():
        rej = li["filter_rejections"]
        L.append(f"- {lg} rejections: " + ", ".join(f"{k} {v}" for k, v in rej.items()))
    L += ["", "Verification of corrupted rows: differs from the clean sentence; equals no clean-pool sentence; "
          "agreement re-checked against the lexicon and the clean parse; tags aligned and label views consistent; "
          "causal-LM minimal pair mean_logprob(clean) - mean_logprob(corrupted) >= delta. LanguageTool flag rates "
          "are diagnostic only.", "",
          "| lang | op | acceptance rate | LM diff (clean minus corrupted) | LanguageTool flag rate |",
          "|---|---|---|---|---|"]
    for lg, li in info["languages"].items():
        L.append(f"| {lg} | delta | {li['calibration']['delta']} | | |")
        for op, s in li["op_stats"].items():
            L.append(f"| {lg} | {op} | {s['acceptance_rate']} | {_fmt_dist(s['lm_diff'])} | {s['languagetool_flag_rate']} |")
    L += [""]
    audit = info.get("audit") or {}
    if audit.get("filled"):
        L += ["Human audit results:", ""]
        for lg, a in audit["languages"].items():
            L.append(f"- {lg}: {json.dumps(a, ensure_ascii=False)}")
    else:
        L += ["Human audit: audit/clean_{lang}.csv (200 clean rows) and audit/corrupted_{lang}.csv (200 corrupted "
              "rows, stratified by type) are provided for annotation; results are not filled in yet."]
    L += [""]
    L += ["## Splits", "",
          "Splits are built per language and grouped by source sentence, with automated leakage tests. "
          "`test_unseen_lemmas` uses reserved content lemmas that never occur in train, validation or test. "
          "`test_ood_source` uses a different source when enabled and allowed by the licence mode. `test_natural` "
          "contains real learner errors (CoEdIT src vs tgt aligned with difflib for English; Italian only from "
          "user-supplied files). The `parallel` config keeps translation pairs in the same split, and Italian "
          "translations of English sentences are forced into the English sentence's split.", ""]
    for lg, li in info["languages"].items():
        L.append(f"- {lg}: {li['split_info']['n_reserved_lemmas']} reserved lemmas; dropped: "
                 f"{json.dumps(li['split_info']['dropped'])}")
    L += [""]
    L += ["## Tool and model licences", "", "| tool or model | version | licence | notes |", "|---|---|---|---|"]
    for t in info["tools"]:
        flag = " **NON-COMMERCIAL**" if t.get("noncommercial") else ""
        L.append(f"| {t['name']} | {t.get('version', '')} | {t['license']}{flag} | {t.get('notes', '')} |")
    L += ["", "Tool outputs (parses, LM scores) are not redistributed; they only drive filtering, target selection "
          "and verification.", ""]
    L += ["## Related work", ""] + [f"- **{n}**: {d}" for n, d in RELATED_WORK] + [""]
    L += ["## Limitations", "",
          "- Corruptions are synthetic and rule based: they cover five error families, not the full range of learner "
          "errors (see test_natural for real errors).",
          "- Verification is automatic. A corrupted sentence can occasionally still be grammatical (for example a "
          "preposition swap that yields another valid reading), and filters can let through awkward clean sentences. "
          "The audit files estimate both error rates.",
          "- Tatoeba sentences are short, conversational and dominated by a few contributors and recurring names; "
          "the proper-noun cap reduces but does not remove this.",
          "- Default normalization lowercases text and strips final punctuation, so capitalization and punctuation "
          "cues are not available to models.",
          "- Surface cues: repeated words and some word-order swaps are detectable with shallow features; see the "
          "TF-IDF baselines before claiming a model learned syntax.",
          "- The Italian LanguageTool grammar coverage is limited, so the Italian clean filter relies more on the "
          "parser and LM filters.", ""]
    return "\n".join(L) + "\n"


def write_card(info: dict, out_dir: str | Path, repo_id: str) -> Path:
    path = Path(out_dir) / "hub" / "README.md"
    write_text(path, render_card(info, repo_id))
    return path


def show_examples(dd_by_config: dict[str, Any]) -> None:
    """Print one row per task for every config."""
    for name, dd in dd_by_config.items():
        split = "train" if "train" in dd else list(dd.keys())[0]
        ds = dd[split]
        feats = ds.features
        bad = next((r for r in ds if r["binary"] == 1 and r["n_corruptions"] >= 1), ds[0])
        multi = next((r for r in ds if r["n_corruptions"] >= 2), bad)
        print(f"\n=== config {name} ({split}) ===")
        print(f"[binary]      {bad['text']!r} -> {feats['binary'].int2str(bad['binary'])}")
        print(f"[multiclass]  {bad['text']!r} -> {feats['multiclass'].int2str(bad['multiclass'])}")
        print(f"[multilabel]  {multi['text']!r} -> {dict(zip(OPS, multi['multilabel']))}")
        tagger = feats["token_tags"].feature
        print("[token]       " + " ".join(t if g == 0 else f"{t}/{tagger.int2str(g)}"
                                         for t, g in zip(bad["tokens"], bad["token_tags"])))


def push(cfg: dict, repo_id: str, private: bool, for_real: bool) -> None:
    from datasets import DatasetDict, load_dataset

    out = Path(cfg["paths"]["out"])
    info = json.loads((out / "build_info.json").read_text())
    card = write_card(info, out, repo_id)
    configs = list(info["configs"])
    if not for_real:
        print(f"DRY RUN: would push configs {configs} to https://huggingface.co/datasets/{repo_id} "
              f"(private={private}), upload out/csv and out/jsonl under raw/, ATTRIBUTION.md and the card {card}.")
        print("Pass --push-for-real to actually push. Local preview of one row per task:")
        show_examples({c: DatasetDict.load_from_disk(str(out / "hf" / c)) for c in configs})
        return
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None  # falls back to a prior `huggingface-cli login`
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for c in configs:
        DatasetDict.load_from_disk(str(out / "hf" / c)).push_to_hub(repo_id, config_name=c, private=private, token=token)
    for sub in ("csv", "jsonl"):
        api.upload_folder(folder_path=str(out / sub), path_in_repo=f"raw/{sub}", repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(out / "ATTRIBUTION.md"), path_in_repo="ATTRIBUTION.md", repo_id=repo_id,
                    repo_type="dataset")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    loaded: dict[str, Any] = {}
    for c in configs:
        loaded[c] = load_dataset(repo_id, c, token=token)
    show_examples(loaded)
