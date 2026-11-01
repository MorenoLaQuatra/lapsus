# SPDX-License-Identifier: Apache-2.0
"""Source download and loading.

Every loaded row is a dict:
    {lang, text, source, source_id, source_author, license, role}

Tatoeba: the per-language "sentences_detailed" export gives id, lang, text and
the author username; the per-language "sentences_CC0" export lists the ids
released under CC0 1.0. Every other sentence is CC BY 2.0 FR. Export dates are
taken from the HTTP Last-Modified header and stored in a manifest next to the
files (data/raw/<source>/manifest.json).
"""

from __future__ import annotations

import bz2
import csv
import datetime as dt
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("lapsus.sources")

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language/"
MORPHIT_URL = "https://docs.sslmit.unibo.it/lib/exe/fetch.php?media=resources:morph-it.tgz"
COEDIT_BASE = "https://huggingface.co/datasets/grammarly/coedit/resolve/main/"
UD_RAW = "https://raw.githubusercontent.com/{repo}/master/{file}"
EVAL_SETS = {
    "cola_validation": "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/cola/validation-00000-of-00001.parquet",
    "itacola": "https://raw.githubusercontent.com/dhfbk/ItaCoLA-dataset/main/ItaCoLA_dataset.tsv",
}
NC_SOURCES_WARNING = (
    "\n" + "!" * 78 + "\n"
    "!! WARNING: source {name} is licensed {license} (NON-COMMERCIAL, SHARE-ALIKE).\n"
    "!! Every config that includes it becomes non-commercial and share-alike.\n"
    + "!" * 78
)


# --------------------------------------------------------------------- utils
def _download(url: str, dest: Path, force: bool = False) -> dict[str, Any]:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return {}
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        h = hashlib.sha256()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
        tmp.rename(dest)
        return {
            "url": url,
            "file": dest.name,
            "last_modified": r.headers.get("Last-Modified", ""),
            "sha256": h.hexdigest(),
            "bytes": dest.stat().st_size,
            "downloaded_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


def _update_manifest(folder: Path, entries: list[dict[str, Any]]) -> None:
    mf = folder / "manifest.json"
    data = json.loads(mf.read_text()) if mf.exists() else {"files": {}}
    for e in entries:
        if e:
            data["files"][e["file"]] = e
    folder.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(data, indent=2) + "\n")


def read_manifest(raw_dir: str | Path, source: str) -> dict[str, Any]:
    mf = Path(raw_dir) / source / "manifest.json"
    return json.loads(mf.read_text()) if mf.exists() else {"files": {}}


def enabled_sources(sources_cfg: dict, lang: str) -> list[dict]:
    out = []
    for s in sources_cfg["sources"]:
        if s.get("enabled") and lang in s.get("languages", []):
            if "NC" in s.get("license", ""):
                warnings.warn(NC_SOURCES_WARNING.format(name=s["name"], license=s["license"]))
                log.warning(NC_SOURCES_WARNING.format(name=s["name"], license=s["license"]))
            out.append(s)
    return out


def roles_of(src: dict) -> list[str]:
    r = src.get("role", "main")
    return list(r) if isinstance(r, list) else [r]


# ------------------------------------------------------------------ download
def download_all(cfg: dict, sources_cfg: dict, langs: list[str], force: bool = False,
                 with_eval: bool = True) -> None:
    raw = Path(cfg["paths"]["raw"])
    for src in sources_cfg["sources"]:
        if not src.get("enabled") or not set(src.get("languages", [])) & set(langs):
            continue
        if src["loader"] == "tatoeba":
            folder = raw / "tatoeba"
            files = [f for lg in langs for f in src["files"].get(lg, [])]
            if set(langs) >= {"en", "it"} and src["files"].get("links"):
                files.append(src["files"]["links"])
            entries = [_download(TATOEBA_BASE + f, folder / Path(f).name, force) for f in files]
            _update_manifest(folder, entries)
        elif src["loader"] == "coedit":
            folder = raw / "coedit"
            entries = [_download(COEDIT_BASE + f, folder / f, force) for f in ("train.jsonl", "validation.jsonl")]
            _update_manifest(folder, entries)
        elif src["loader"] == "ud":
            folder = raw / src["name"]
            entries = [_download(UD_RAW.format(repo=src["repo"], file=f), folder / f, force) for f in src["ud_files"]]
            _update_manifest(folder, entries)
    if "it" in langs:
        folder = raw / "morphit"
        try:
            e = _download(MORPHIT_URL, folder / "morph-it.tgz", force)
        except Exception as exc:
            raise SystemExit(
                f"Morph-it! download failed ({exc}).\nThe server may refuse scripted clients. Download it manually "
                f"(for example in a browser, or `curl -L -o {folder / 'morph-it.tgz'} '{MORPHIT_URL}'`) and rerun "
                "`python -m lapsus download --lang it`."
            )
        if not e and (folder / "morph-it.tgz").exists() and "morph-it.tgz" not in read_manifest(raw, "morphit")["files"]:
            p = folder / "morph-it.tgz"
            e = {"url": MORPHIT_URL, "file": p.name, "last_modified": "", "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "downloaded_at": "manual"}
        _update_manifest(folder, [e])
        if not list(folder.rglob("morph-it_*.txt")):
            import tarfile

            with tarfile.open(folder / "morph-it.tgz") as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".txt") and ".." not in m.name and not m.name.startswith("/"):
                        tf.extract(m, folder)
    if with_eval:
        folder = raw / "eval"
        entries = []
        for name, url in EVAL_SETS.items():
            if (name == "itacola" and "it" in langs) or (name.startswith("cola") and "en" in langs):
                entries.append(_download(url, folder / Path(url).name, force))
        _update_manifest(folder, entries)
    download_stanza_models(cfg, langs)


def download_stanza_models(cfg: dict, langs: list[str]) -> None:
    try:
        import stanza
    except ImportError:
        log.warning("stanza not installed; skipping model download")
        return
    from .lang import get_language

    model_dir = str(Path(cfg["paths"]["cache"]) / "stanza_resources")
    for lg in langs:
        stanza.download(get_language(lg).stanza_lang, model_dir=model_dir,
                        processors="tokenize,mwt,pos,lemma,depparse", verbose=False)


# ------------------------------------------------------------------- loaders
def _read_bz2_tsv(path: Path) -> Iterable[list[str]]:
    with bz2.open(path, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def load_tatoeba(raw_dir: str | Path, lang_code: str, tatoeba_code: str, src: dict) -> list[dict]:
    folder = Path(raw_dir) / "tatoeba"
    detailed = folder / f"{tatoeba_code}_sentences_detailed.tsv.bz2"
    cc0 = folder / f"{tatoeba_code}_sentences_CC0.tsv.bz2"
    if not detailed.exists():
        raise FileNotFoundError(f"{detailed} missing: run `python -m lapsus download`")
    cc0_ids = set()
    if cc0.exists():
        for parts in _read_bz2_tsv(cc0):
            if parts and parts[0].isdigit():
                cc0_ids.add(parts[0])
    rows = []
    for parts in _read_bz2_tsv(detailed):
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        sid, _, text, user = parts[0], parts[1], parts[2], parts[3]
        rows.append({
            "lang": lang_code,
            "text": text,
            "source": "tatoeba",
            "source_id": sid,
            "source_author": "" if user in ("\\N", "") else user,
            "license": "CC0-1.0" if sid in cc0_ids else "CC-BY-2.0",
            "role": "main",
        })
    return rows


def load_tatoeba_links(raw_dir: str | Path) -> list[tuple[str, str]]:
    path = Path(raw_dir) / "tatoeba" / "eng-ita_links.tsv.bz2"
    if not path.exists():
        return []
    return [(p[0], p[1]) for p in _read_bz2_tsv(path) if len(p) >= 2]


def _strip_instruction(src: str) -> str:
    return src.split(": ", 1)[1] if ": " in src else src


def load_coedit(raw_dir: str | Path, src: dict) -> tuple[list[dict], list[dict]]:
    """Returns (ood clean rows from tgt, natural pairs from src/tgt)."""
    folder = Path(raw_dir) / "coedit"
    ood, natural = [], []
    for fname in ("train.jsonl", "validation.jsonl"):
        path = folder / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m lapsus download`")
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("task") != "gec":
                    continue
                sid = str(ex.get("_id", ""))
                base = {"lang": "en", "source": "coedit_gec", "source_id": sid, "source_author": "",
                        "license": src["license"]}
                ood.append({**base, "text": ex["tgt"].strip(), "role": "ood"})
                natural.append({**base, "text": _strip_instruction(ex["src"]).strip(),
                                "correction": ex["tgt"].strip(), "role": "natural"})
    return ood, natural


def load_ud(raw_dir: str | Path, src: dict, lang: str) -> list[dict]:
    folder = Path(raw_dir) / src["name"]
    rows = []
    for fname in src["ud_files"]:
        path = folder / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m lapsus download`")
        sid, text = None, None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
            elif not line.strip() and text:
                rows.append({"lang": lang, "text": text, "source": src["name"], "source_id": sid or "",
                             "source_author": "", "license": src["license"], "role": roles_of(src)[0]})
                sid, text = None, None
    return rows


def load_user_natural(sources_cfg: dict, lang: str) -> list[dict]:
    spec = (sources_cfg.get("user_natural") or {}).get(lang) or {}
    if not spec.get("path"):
        return []
    rows = []
    with open(spec["path"], encoding="utf-8") as f:
        for k, line in enumerate(csv.reader(f, delimiter="\t")):
            if len(line) >= 2:
                rows.append({"lang": lang, "text": line[0], "correction": line[1], "source": "user_natural",
                             "source_id": str(k), "source_author": "",
                             "license": spec.get("license") or "LicenseRef-unverified", "role": "natural"})
    return rows


def load_language_rows(cfg: dict, sources_cfg: dict, lang: str) -> dict[str, list[dict]]:
    """All enabled rows for one language, grouped by role (main, ood, natural)."""
    from .lang import get_language

    raw = cfg["paths"]["raw"]
    L = get_language(lang, cfg)
    out: dict[str, list[dict]] = {"main": [], "ood": [], "natural": []}
    for src in enabled_sources(sources_cfg, lang):
        if src["loader"] == "tatoeba":
            out["main"].extend(load_tatoeba(raw, lang, L.tatoeba_code, src))
        elif src["loader"] == "coedit":
            ood, nat = load_coedit(raw, src)
            if "ood" in roles_of(src):
                out["ood"].extend(ood)
            if "natural" in roles_of(src):
                out["natural"].extend(nat)
        elif src["loader"] == "ud":
            out["ood"].extend(load_ud(raw, src, lang))
    out["natural"].extend(load_user_natural(sources_cfg, lang))
    return out


def source_records(cfg: dict, sources_cfg: dict, langs: list[str]) -> list[dict]:
    """Source metadata (with export dates) for cards and release notes."""
    recs = []
    for src in sources_cfg["sources"]:
        if not set(src.get("languages", [])) & set(langs):
            continue
        folder = src["name"] if src["loader"] != "tatoeba" else "tatoeba"
        if src["loader"] == "coedit":
            folder = "coedit"
        mf = read_manifest(cfg["paths"]["raw"], folder)
        dates = sorted({e.get("last_modified", "") for e in mf["files"].values() if e.get("last_modified")})
        recs.append({
            "name": src["name"], "url": src["url"], "license": src["license"],
            "languages": src["languages"], "enabled": bool(src.get("enabled")),
            "permissive_for_apache": bool(src.get("permissive_for_apache")),
            "attribution": " ".join(src.get("attribution", "").split()),
            "export_dates": dates,
            "files": {k: {kk: v.get(kk) for kk in ("url", "last_modified", "sha256")} for k, v in mf["files"].items()},
        })
    return recs
