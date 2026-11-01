# SPDX-License-Identifier: Apache-2.0
"""Export: out/csv/{config}/{split}.csv, out/jsonl/{config}/{split}.jsonl and a
saved DatasetDict per config (out/hf/{config}) with proper Features.

CSV and JSONL store label NAMES ("ok", "agreement", ...). In CSV the list
columns (tokens, multilabel, token_tags, token_binary) are JSON strings. In the
DatasetDict, labels are ClassLabel integers. In test_natural only `binary` and
`token_binary` are defined: multiclass, token_tags (null / -1), multilabel and
n_corruptions (-1) are undefined there.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import BINARY, MULTICLASS, OPS, TAGS
from .utils import assert_no_dashes, write_text

COLUMNS = ["id", "lang", "text", "tokens", "clean_text", "clean_text_raw", "binary", "multiclass", "multilabel",
           "token_tags", "token_binary", "n_corruptions", "source", "source_id", "source_author", "license", "split"]
LIST_COLUMNS = {"tokens", "multilabel", "token_tags", "token_binary"}
SPLIT_ORDER = ["train", "validation", "test", "test_unseen_lemmas", "test_ood_source", "test_natural"]


def features(parallel: bool = False):
    from datasets import ClassLabel, Features, Sequence, Value

    f = {
        "id": Value("string"),
        "lang": Value("string"),
        "text": Value("string"),
        "tokens": Sequence(Value("string")),
        "clean_text": Value("string"),
        "clean_text_raw": Value("string"),
        "binary": ClassLabel(names=BINARY),
        "multiclass": ClassLabel(names=MULTICLASS),
        "multilabel": Sequence(Value("int8"), length=len(OPS)),
        "token_tags": Sequence(ClassLabel(names=TAGS)),
        "token_binary": Sequence(Value("int8")),
        "n_corruptions": Value("int8"),
        "source": Value("string"),
        "source_id": Value("string"),
        "source_author": Value("string"),
        "license": Value("string"),
        "split": Value("string"),
    }
    if parallel:
        f["pair_id"] = Value("string")
    return Features(f)


def encode_row(r: dict) -> dict:
    out = dict(r)
    out["binary"] = BINARY.index(r["binary"])
    out["multiclass"] = -1 if r["multiclass"] is None else MULTICLASS.index(r["multiclass"])
    out["token_tags"] = [-1 if t is None else TAGS.index(t) for t in r["token_tags"]]
    return out


def ordered_splits(splits: dict[str, Any]) -> list[str]:
    return [s for s in SPLIT_ORDER if s in splits] + sorted(s for s in splits if s not in SPLIT_ORDER)


def write_config(out_dir: str | Path, config: str, splits: dict[str, list[dict]], parallel: bool = False) -> dict:
    from datasets import Dataset, DatasetDict

    out_dir = Path(out_dir)
    cols = COLUMNS + (["pair_id"] if parallel else [])
    counts = {}
    dd = {}
    for split in ordered_splits(splits):
        rows = splits[split]
        if not rows:
            continue
        counts[split] = len(rows)
        cpath = out_dir / "csv" / config / f"{split}.csv"
        jpath = out_dir / "jsonl" / config / f"{split}.jsonl"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([json.dumps(r[c], ensure_ascii=False) if c in LIST_COLUMNS
                            else ("" if r[c] is None else r[c]) for c in cols])
        with open(jpath, "w", encoding="utf-8") as f:
            for r in rows:
                line = json.dumps({c: r[c] for c in cols}, ensure_ascii=False)
                assert_no_dashes(line, str(jpath))
                f.write(line + "\n")
        dd[split] = Dataset.from_list([{c: encode_row(r)[c] for c in cols} for r in rows],
                                      features=features(parallel))
    DatasetDict(dd).save_to_disk(str(out_dir / "hf" / config))
    return counts


def write_attribution(path: str | Path, source_recs: list[dict], configs_rows: dict[str, list[dict]],
                      license_by_config: dict[str, dict]) -> None:
    lines = ["# Attribution", "",
             "Lapsus labels, corruptions and annotations are released under Apache-2.0.",
             "The underlying sentences keep the licence of their source, recorded per row in the `license` column.",
             ""]
    lines.append("## Dataset licence per config")
    lines.append("")
    for cfg_name, lic in license_by_config.items():
        lines.append(f"- {cfg_name}: {lic['hf']} (row licences: {', '.join(lic['components'])})")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    for s in source_recs:
        if not s["enabled"]:
            continue
        lines.append(f"### {s['name']}")
        lines.append("")
        lines.append(f"- URL: {s['url']}")
        lines.append(f"- Licence: {s['license']}")
        if s["export_dates"]:
            lines.append(f"- Export date(s): {', '.join(s['export_dates'])}")
        lines.append(f"- Attribution: {s['attribution']}")
        lines.append("")
    authors: Counter[str] = Counter()
    for rows in configs_rows.values():
        for r in rows:
            if r["source"] == "tatoeba":
                authors[r["source_author"] or "(unknown)"] += 1
    if authors:
        lines.append("## Tatoeba credit")
        lines.append("")
        lines.append("Tatoeba sentences are the work of the Tatoeba community (https://tatoeba.org). "
                     "Each row keeps the sentence id (`source_id`, see https://tatoeba.org/en/sentences/show/<id>) "
                     "and the contributor username (`source_author`). Contributors whose sentences appear in this "
                     "dataset, with row counts:")
        lines.append("")
        for a, n in sorted(authors.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {a}: {n}")
        lines.append("")
    write_text(path, "\n".join(lines) + "\n")
