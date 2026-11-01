# SPDX-License-Identifier: Apache-2.0
"""Lapsus: a bilingual grammatical acceptability dataset builder for teaching NLP."""

from pathlib import Path

__version__ = (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip() if (
    Path(__file__).resolve().parent.parent / "VERSION"
).exists() else "0.0.0"

OPS = ["word_order", "missing_word", "repeated_word", "agreement", "wrong_function_word"]
TAGS = ["O"] + OPS
MULTICLASS = ["none"] + OPS + ["multiple"]
BINARY = ["ok", "not_ok"]
