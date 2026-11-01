# Contributing to Lapsus

Thanks for helping. Lapsus is a teaching resource, so clarity and reproducibility
matter more than cleverness.

## Ground rules

- Python 3.10+, type hints, and an SPDX header (`# SPDX-License-Identifier: Apache-2.0`) on every source file.
- Never use em dashes or en dashes anywhere (code, docs, cards). Use hyphens, commas or colons.
  `tests/test_tokenize.py::test_no_forbidden_dashes_in_repo` enforces this.
- No secrets in code. The Hub token comes from `HF_TOKEN` or a prior `huggingface-cli login`.
- Never commit raw data, caches or build outputs (`data/`, `out/`, `release/` are git-ignored).
- Keep the build deterministic: derive every random choice from `lapsus.utils.rng_for(seed, ...)`.

## Development setup

```bash
pip install -e ".[en,it,hub,test]"
python -m pytest
```

The tests run offline on small fixtures (no network, no model downloads); CI runs
exactly `pytest` (see `.github/workflows/tests.yml`).

## Adding a language

1. Implement `lapsus.lang.base.Language` in `lapsus/lang/<code>.py`: tokenization,
   closed-class tables, `missing_word_targets`, `word_order_allowed`, `repeat_allowed`,
   `agreement_candidates`, `verify_agreement`, `function_word_candidates`, tool names
   (Stanza language, lingua language, LanguageTool codes, default causal LM).
2. Register it in `lapsus/lang/__init__.py`.
3. Add sources for it in `sources.yaml` and fixture parses plus tests in `tests/`.

Nothing outside `lapsus/lang/` should need to change.

## Adding a source

Add an entry to `sources.yaml` with `name`, `url`, `license` (SPDX), `attribution`,
`languages`, `enabled`, `role` and `permissive_for_apache` (true ONLY for Apache-2.0,
CC0 and public domain), and a loader in `lapsus/sources.py`.

## Releases

Only the repository owner publishes releases. `python -m lapsus package --version X.Y.Z`
builds the release folder; tagging and uploading are done by hand.
