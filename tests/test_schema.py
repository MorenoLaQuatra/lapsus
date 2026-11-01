# SPDX-License-Identifier: Apache-2.0
import json
import random

from datasets import DatasetDict

from lapsus import BINARY, MULTICLASS, OPS, TAGS
from lapsus.build import make_row, normalize_tokens
from lapsus.corruptions import corrupt, views_consistent
from lapsus.export import COLUMNS, features, write_config
from tests.conftest import make_items


def _rows(parses, en, it):
    rows = []
    for lang, lg in ((en, "en"), (it, "it")):
        for k, item in enumerate(make_items(parses[lg], lg)):
            d = None if k % 2 == 0 else corrupt(item["parse"], lang, [OPS[k % len(OPS)]], random.Random(k))
            row, _ = make_row(lang, item, d, "train", "lower_nopunct")
            rows.append(row)
    return rows


def test_rows_have_schema_and_consistent_views(parses, en, it):
    for r in _rows(parses, en, it):
        assert list(r.keys()) == COLUMNS
        assert len(r["tokens"]) == len(r["token_tags"]) == len(r["token_binary"])
        assert len(r["multilabel"]) == 5
        assert r["binary"] in BINARY and r["multiclass"] in MULTICLASS
        assert all(t in TAGS for t in r["token_tags"])
        assert views_consistent(r)
        assert r["text"] == r["text"].lower() and not r["text"].endswith(".")


def test_normalization_modes():
    toks = ["Tom", "is", "here."]
    assert normalize_tokens(toks, "none") == toks
    assert normalize_tokens(toks, "lower") == ["tom", "is", "here."]
    assert normalize_tokens(toks, "lower_nopunct") == ["tom", "is", "here"]


def test_export_roundtrip_with_natural_rows(tmp_path, parses, en, it):
    rows = _rows(parses, en, it)
    natural = dict(rows[0], id="en-x-1", split="test_natural", binary="not_ok", multiclass=None,
                   multilabel=[-1] * 5, token_tags=[None] * len(rows[0]["tokens"]),
                   token_binary=[1] + [0] * (len(rows[0]["tokens"]) - 1), n_corruptions=-1)
    counts = write_config(tmp_path, "en", {"train": rows, "test_natural": [natural]})
    assert counts == {"train": len(rows), "test_natural": 1}
    dd = DatasetDict.load_from_disk(str(tmp_path / "hf" / "en"))
    assert dd["train"].features == features()
    r0 = dd["train"][0]
    assert dd["train"].features["binary"].int2str(r0["binary"]) == rows[0]["binary"]
    nat = dd["test_natural"][0]
    assert nat["multiclass"] == -1 and set(nat["token_tags"]) == {-1}
    line = (tmp_path / "csv" / "en" / "train.csv").read_text().splitlines()[1]
    assert "[" in line  # list columns are JSON strings
    j = json.loads((tmp_path / "jsonl" / "en" / "train.jsonl").read_text().splitlines()[0])
    assert isinstance(j["tokens"], list)
