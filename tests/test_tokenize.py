# SPDX-License-Identifier: Apache-2.0
def test_italian_elision_tokens(it):
    assert it.tokenize("Ho visto l'uomo dell'anno.") == ["Ho", "visto", "l'", "uomo", "dell'", "anno."]
    assert it.tokenize("un po' di pane") == ["un", "po'", "di", "pane"]
    for text in ["Ho visto l'uomo dell'anno.", "un po' di pane", "C'è un'amica."]:
        assert it.detokenize(it.tokenize(text)) == text


def test_italian_repeated_elision_roundtrip(it):
    toks = ["l'", "l'", "uomo"]
    assert it.tokenize(it.detokenize(toks)) == toks


def test_english_whitespace(en):
    assert en.tokenize("I don't know Tom's car.") == ["I", "don't", "know", "Tom's", "car."]
    assert en.token_spans("I am here.") == [(0, 1), (2, 4), (5, 10)]


def test_no_forbidden_dashes_in_repo():
    from pathlib import Path

    from lapsus.utils import FORBIDDEN_DASHES

    root = Path(__file__).resolve().parent.parent
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".cff", ".txt", ".sh", ".json"} \
                and not any(part in {"data", "out", "release", ".git"} for part in p.parts):
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert not any(d in text for d in FORBIDDEN_DASHES), p
