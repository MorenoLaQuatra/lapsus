# Lapsus

Lapsus is a small dataset of grammatical and ungrammatical sentences in English and Italian, plus the code that builds it. It was made for the course **Elaborazione del Linguaggio Naturale** at [Kore University of Enna](https://www.uke.it), as material for hands-on exercises with BERT-style models.

## The idea

We take clean sentences from [Tatoeba](https://tatoeba.org), break about half of them on purpose, and keep track of exactly what we broke and where. Each sentence can then be used for four tasks:

- **binary**: is the sentence acceptable or not?
- **multiclass**: which kind of error is it?
- **multi-label**: which errors are present (a sentence can have more than one)?
- **token tagging**: which words are wrong?

The errors are of five kinds: word order, missing word, repeated word, agreement (for example *"the dogs runs"*, *"la ragazzi"*) and wrong function word (*"in"* instead of *"on"*, *"di"* instead of *"da"*).

## What is in it

For each language there are about 12000 training sentences, 1500 for validation, 1500 for test, and 1500 in a harder test set whose content words never appear in training. There are four configs: `en`, `it`, `both`, and a small `parallel` set of English-Italian translation pairs.

Every corrupted sentence is checked before it is kept: the change must be a real word form, must not produce another valid sentence, and a small language model must find it less likely than the original. You can still find mistakes, so please report them.

## Building it yourself

You need Python 3.10+ and Java 17+ (for LanguageTool).

```bash
pip install -e ".[en,it,hub,test]"
python -m lapsus download     # get the raw sources
python -m lapsus licenses     # filter and print the licence report
python -m lapsus build        # build the dataset into out/
python -m lapsus check        # sanity checks
python -m lapsus baselines    # simple TF-IDF baselines
```

The build is deterministic (`--seed 42`) and caches everything under `data/`. Parsing takes a while the first time. Use `--lang en` or `--lang it` to build one language only, and see `configs/default.yaml` for all the options.

## Licence

The code is Apache-2.0. The data follows its sources: English sentences are CC0, Italian sentences are CC BY 2.0 FR (Tatoeba), so each row has a `license` column and the authors are listed in `ATTRIBUTION.md`. Run `python -m lapsus licenses` for the details.

## Citation

```bibtex
@misc{lapsus,
  title = {{Lapsus}: A bilingual grammatical acceptability dataset for teaching {NLP}},
  year  = {2026},
  url   = {https://github.com/MorenoLaQuatra/lapsus}
}
```
