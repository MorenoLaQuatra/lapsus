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

from . import OPS
from .export import ordered_splits
from .utils import write_text

SPLITS = ["train", "validation", "test", "test_unseen_lemmas", "test_ood_source", "test_natural"]


def _yaml_head(info: dict) -> str:
    configs = info["configs"]
    yaml_cfgs = []
    for name, c in configs.items():
        yaml_cfgs.append(f"- config_name: {name}")
        yaml_cfgs.append("  data_files:")
        for split in ordered_splits(c["splits"]):
            yaml_cfgs.append(f"  - split: {split}")
            yaml_cfgs.append(f"    path: {name}/{split}-*")
    total = sum(sum(c["splits"].values()) for c in configs.values())
    size = "10K<n<100K" if total < 100_000 else "100K<n<1M"
    return "\n".join([
        "---",
        "language:",
        *[f"- {lg}" for lg in sorted(info["languages"])],
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


def render_card(info: dict, repo_id: str) -> str:
    configs = info["configs"]
    used = [s for s in SPLITS if any(s in c["splits"] for c in configs.values())]
    mix = info["mix"]
    L: list[str] = [_yaml_head(info), "", "# Lapsus", ""]
    L += ["Lapsus is a small dataset of grammatical and ungrammatical sentences in English and Italian. "
          "It was made for the course **Elaborazione del Linguaggio Naturale** at "
          "[Kore University of Enna](https://www.uke.it), for hands-on exercises with BERT-style models. "
          "The code that builds it is on [GitHub](https://github.com/MorenoLaQuatra/lapsus).", ""]
    L += ["## How it works", "",
          "We take clean sentences from [Tatoeba](https://tatoeba.org) and break some of them on purpose, "
          "keeping track of what we broke and where. There are five kinds of errors:", "",
          "- `word_order`: two nearby words are swapped (*\"i never your cut hair\"*)",
          "- `missing_word`: a function word is deleted (*\"she went the station\"*)",
          "- `repeated_word`: a word is doubled (*\"he is is tired\"*)",
          "- `agreement`: a form no longer agrees (*\"the dogs runs\"*, *\"la ragazzi\"*)",
          "- `wrong_function_word`: the wrong preposition or article (*\"in\"* instead of *\"on\"*, "
          "*\"di\"* instead of *\"da\"*)", "",
          f"In every split {int(mix['clean'] * 100)}% of the sentences are clean, "
          f"{int(mix['single'] * 100)}% have one error and {int(mix['multi'] * 100)}% have two or three "
          "(labelled `multiple`).", "",
          "Each broken sentence is checked before it is kept: the new word must be a real form, the result "
          "must not be another clean sentence, and a small language model must like it less than the original, "
          "that is $\\log p(\\text{clean}) - \\log p(\\text{corrupted}) \\geq \\delta$ "
          "(average per token). The checks are automatic, so a few mistakes are still in there.", ""]
    L += ["## Tasks", "",
          "The same rows support four tasks. For *\"the dogs runs fast\"*:", "",
          "| task | column | label |", "|---|---|---|",
          "| binary | `binary` | `not_ok` |",
          "| error type | `multiclass` | `agreement` |",
          "| error types (multi-label) | `multilabel` | $[0, 0, 0, 1, 0]$ |",
          "| token tagging | `token_tags` | *runs* is tagged `agreement`, the rest `O` |", "",
          f"The `multilabel` order is {', '.join(f'`{o}`' for o in OPS)}. "
          "When a word is missing, the token right after the gap gets the tag.", ""]
    L += ["## Configs and splits", "",
          "| config | " + " | ".join(f"`{s}`" for s in used) + " |",
          "|---|" + "---|" * len(used)]
    for name, c in configs.items():
        L.append(f"| `{name}` | " + " | ".join(str(c["splits"].get(s, "")) for s in used) + " |")
    L += ["", "`both` is `en` plus `it` with a `lang` column. `parallel` is a small set of English-Italian "
          "translation pairs (`pair_id`). In `test_unseen_lemmas` the content words never appear in the other "
          "splits, so it tests generalization rather than memorization.", ""]
    L += ["```python", "from datasets import load_dataset", "", f'ds = load_dataset("{repo_id}", "it")', "```", ""]
    L += ["Other useful columns: `tokens`, `clean_text` (the original sentence), `n_corruptions`, "
          "`source_id`, `source_author` and `license`. Text is lowercased and the final punctuation is "
          "removed; Italian elisions are split (*l\'uomo* becomes *l\'* + *uomo*).", ""]
    names = {"en": "English", "it": "Italian"}
    srcs = {"apache-2.0": "CC0 (public domain)", "cc-by-2.0": "CC BY 2.0 FR"}
    parts = [f"{names.get(lg, lg)} sentences are {srcs.get(li['license']['hf'], li['license']['hf'])}"
             for lg, li in info["languages"].items()]
    L += ["## Licence", "",
          f"The dataset is released as `{info['repo_license']}` because it follows its Tatoeba sources: "
          + ", ".join(parts) + ". Each row has its own `license` column and every contributor is listed in "
          "`ATTRIBUTION.md`. Labels and code are Apache-2.0."]
    L += ["", "## Limitations", "",
          "- The errors are synthetic and cover only five families, not everything learners do.",
          "- Tatoeba sentences are short and conversational.",
          "- Some errors (like repeated words) are easy to spot with shallow features, so try a TF-IDF "
          "baseline before claiming a model learned syntax.", ""]
    L += ["## Citation", "", "```bibtex", "@misc{lapsus,",
          "  title  = {Lapsus: a bilingual grammatical acceptability dataset for teaching NLP},",
          "  howpublished = {\\url{https://github.com/MorenoLaQuatra/lapsus}},",
          "  year   = {2026}", "}", "```"]
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
