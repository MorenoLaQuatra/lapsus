# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and the project uses semantic versioning for dataset releases.

## [0.1.0] - 2026-11-01

### Added

- Builder pipeline: Tatoeba sources with per-row licence (CC0 vs CC BY 2.0 FR), optional
  CoEdIT GEC and UD Italian PUD sources, licence modes (auto, apache-strict, source-licenses).
- Clean-sentence filters (normalization and length, lexical, language ID, LanguageTool,
  Stanza parse, causal-LM filter, exact/normalized/MinHash deduplication, proper-noun cap).
- Five corruption operations (word_order, missing_word, repeated_word, agreement,
  wrong_function_word) and multi-corruption, with language modules for English and Italian.
- Verification of corrupted rows (lexicon re-check, clean-pool check, LM minimal pairs) and
  per-operation acceptance reports.
- Splits with reserved-lemma and out-of-domain test sets, leakage tests, configs en, it,
  both and parallel.
- CSV, JSONL and DatasetDict export, dataset card, Hub mirror (dry run by default), release
  packaging, audit CSVs and TF-IDF baselines.
