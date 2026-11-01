# SPDX-License-Identifier: Apache-2.0
"""Build orchestration: pools -> licence mode -> splits -> corruptions -> verified rows."""

from __future__ import annotations

import csv
import difflib
import json
import logging
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import OPS, __version__
from .corruptions import Draft, corrupt, corrupt_multi, label_views
from .filters import LangID, LTChecker, filter_natural, normalized_key, run_filters
from .lang import get_language
from .lang.base import Language, split_core
from .licensing import combine_licenses, decide_mode, format_license_report, is_permissive, row_allowed
from .splits import MAIN_SPLITS, assign_splits, check_leakage
from .utils import package_version, rng_for, stable_hash, write_json, write_text
from .verify import OpStats, calibrate_delta, distribution, lm_diffs, structural_check

log = logging.getLogger("lapsus.build")

ROLE_ORDER = ["agreement", "wrong_function_word", "multi3", "missing_word", "multi2", "word_order", "repeated_word"]


# ------------------------------------------------------------------- tools
class Tools:
    """Lazily created heavy tools for one language."""

    def __init__(self, cfg: dict, lang: Language):
        self.cfg, self.lang = cfg, lang
        self._d: dict[str, Any] = {}

    def __getitem__(self, k: str):
        if k not in self._d:
            if k == "langid":
                self._d[k] = LangID(self.lang)
            elif k == "lt":
                self._d[k] = LTChecker(self.lang, self.cfg)
            elif k == "parser":
                from .parse import Parser

                self._d[k] = Parser(self.lang, self.cfg)
            elif k == "lm":
                from .lm import CausalLMScorer

                name = self.cfg["lm"].get(self.lang.code) or self.lang.default_lm
                self._d[k] = CausalLMScorer(name, self.cfg["paths"]["cache"], self.cfg["lm"]["batch_size"],
                                            self.cfg["lm"]["device"])
            else:
                raise KeyError(k)
        return self._d[k]

    def close(self):
        if "lt" in self._d:
            self._d["lt"].close()


def tool_versions(cfg: dict, langs: list[str]) -> dict[str, Any]:
    v = {
        "lapsus": __version__,
        "stanza": package_version("stanza"),
        "languagetool": str(cfg["languagetool"]["version"]),
        "language_tool_python": package_version("language-tool-python"),
        "lemminflect": package_version("lemminflect"),
        "lingua-language-detector": package_version("lingua-language-detector"),
        "datasketch": package_version("datasketch"),
        "transformers": package_version("transformers"),
        "torch": package_version("torch"),
        "datasets": package_version("datasets"),
        "lm": {lg: cfg["lm"].get(lg) or get_language(lg).default_lm for lg in langs},
    }
    if "it" in langs:
        v["morph-it"] = get_language("it", cfg).version_info().get("morph-it")
    return v


# ------------------------------------------------------------------- pools
def pool_cache_key(cfg: dict, sources_cfg: dict, lang: str) -> str:
    from .sources import enabled_sources, read_manifest

    srcs = [(s["name"], s.get("license")) for s in enabled_sources(sources_cfg, lang)]
    manifests = {s: read_manifest(cfg["paths"]["raw"], "tatoeba" if s == "tatoeba" else s) for s, _ in srcs}
    return stable_hash("pool-v3", lang, cfg["seed"], json.dumps(cfg["filters"], sort_keys=True), srcs,
                       json.dumps(manifests, sort_keys=True), cfg["lm"].get(lang), cfg["languagetool"]["version"],
                       package_version("stanza"))[:16]


def get_pools(cfg: dict, sources_cfg: dict, lang_code: str, force: bool = False) -> dict[str, Any]:
    from .sources import load_language_rows

    key = pool_cache_key(cfg, sources_cfg, lang_code)
    path = Path(cfg["paths"]["cache"]) / "pools" / f"{lang_code}-{key}.pkl"
    if path.exists() and not force:
        log.info("[%s] loading filtered pools from cache %s", lang_code, path.name)
        with open(path, "rb") as f:
            return pickle.load(f)
    lang = get_language(lang_code, cfg)
    tools = Tools(cfg, lang)
    rows = load_language_rows(cfg, sources_cfg, lang_code)
    log.info("[%s] loaded rows: main=%d ood=%d natural=%d", lang_code, len(rows["main"]), len(rows["ood"]),
             len(rows["natural"]))
    main, main_rep = run_filters(rows["main"], lang, cfg, tools, f"{lang_code}-main", cfg["filters"]["pool_target"],
                                 progress=log.info)
    ood, ood_rep = ([], {})
    if rows["ood"]:
        ood, ood_rep = run_filters(rows["ood"], lang, cfg, tools, f"{lang_code}-ood",
                                   int(cfg["splits"]["test_ood_source"] * 3), progress=log.info)
    natural, nat_rep = ([], {})
    if rows["natural"]:
        natural, nat_rep = filter_natural(rows["natural"], lang, cfg, tools)
    tools.close()
    out = {"lang": lang_code, "key": key, "main": main, "ood": ood, "natural": natural,
           "report": {"main": main_rep, "ood": ood_rep, "natural": nat_rep}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def license_report(cfg: dict, pools: dict[str, dict], requested: str) -> dict:
    """Pool sizes per language and licence, plus the mode each language will use."""
    from .licensing import LicenseError

    rep: dict[str, Any] = {"requested_mode": requested, "min_sentences": cfg["min_sentences"], "languages": {}}
    for lg, p in pools.items():
        m = p["report"]["main"]
        try:
            decision = decide_mode(lg, requested, m["permissive_pool"], m["total_pool"], cfg["min_sentences"]).as_dict()
        except LicenseError as exc:
            decision = {"mode": "FAIL", "requested": requested, "reason": str(exc)}
        ood = p["report"]["ood"]
        rep["languages"][lg] = {
            "per_license": m["per_license"],
            "permissive_pool": m["permissive_pool"],
            "total_pool": m["total_pool"],
            "threshold": cfg["min_sentences"],
            "stopped_early": m.get("stopped_early", False),
            "filter_rejections": m["rejected"],
            "filter_reasons": m.get("reasons", {}),
            "lm_thresholds": m.get("lm_thresholds", {}),
            "ood": {"total": ood.get("total_pool", 0), "permissive": ood.get("permissive_pool", 0)} if ood else None,
            "natural_pairs": len(p["natural"]),
            "decision": decision,
        }
    return rep


# --------------------------------------------------------------- utilities
def normalize_tokens(tokens: list[str], mode: str) -> list[str]:
    if mode == "none":
        return list(tokens)
    out = [t.lower() for t in tokens]
    if mode == "lower_nopunct" and out:
        core = out[-1].rstrip(".!?")
        if core:
            out[-1] = core
    return out


def mix_targets(n: int, mix: dict) -> dict[str, int]:
    n_single = int(round(n * mix["single"]))
    n_multi = int(round(n * mix["multi"]))
    per_op = [n_single // len(OPS)] * len(OPS)
    for i in range(n_single - sum(per_op)):
        per_op[i] += 1
    t = {op: c for op, c in zip(OPS, per_op)}
    t["multi3"] = int(round(n_multi * mix["multi_three_fraction"]))
    t["multi2"] = n_multi - t["multi3"]
    t["clean"] = n - n_single - n_multi
    return t


def make_row(lang: Language, item: dict, draft: Draft | None, split: str, mode: str) -> tuple[dict, dict]:
    parse = item["parse"]
    tokens_raw = draft.tokens if draft else list(parse["tokens"])
    tags = draft.tags if draft else ["O"] * len(tokens_raw)
    n_ops = len(draft.ops) if draft else 0
    norm = normalize_tokens(tokens_raw, mode)
    row = {
        "id": f"{lang.code}-{item['source']}-{item['source_id']}",
        "lang": lang.code,
        "text": lang.detokenize(norm),
        "tokens": norm,
        "clean_text": lang.detokenize(normalize_tokens(parse["tokens"], mode)),
        "clean_text_raw": item["text"],
        **label_views(tags, n_ops),
        "source": item["source"],
        "source_id": item["source_id"],
        "source_author": item.get("source_author", ""),
        "license": item["license"],
        "split": split,
    }
    meta = {
        "id": row["id"],
        "text_raw": lang.detokenize(tokens_raw),
        "ops": [{k: v for k, v in o.items() if k != "edit"} | ({"edit": {k: o["edit"][k] for k in ("new", "feature", "old_value", "new_value", "kind", "set") if k in o["edit"]}} if "edit" in o else {}) for o in (draft.ops if draft else [])],
        "content_lemmas": sorted(lang.content_lemmas(parse)),
    }
    return row, meta


# --------------------------------------------------------------- corruption
def candidate(item: dict, role: str, lang: Language, seed: int, attempt: int, clean_keys: set[str]) -> tuple[Draft | None, str]:
    rng = rng_for(seed, "corrupt", lang.code, item["source"], item["source_id"], role, attempt)
    parse = item["parse"]
    if role in OPS:
        d = corrupt(parse, lang, [role], rng)
    else:
        d = corrupt_multi(parse, lang, 3 if role == "multi3" else 2, rng)
    if d is None:
        return None, "no_target"
    why = structural_check(d, parse, lang, clean_keys)
    return (None, why) if why else (d, "")


def role_stat_key(role: str) -> str:
    return "multiple" if role.startswith("multi") else role


def calibrate(items: list[dict], lang: Language, tools: Tools, cfg: dict, clean_keys: set[str]) -> tuple[float, dict]:
    vcfg = cfg["verify"]
    diffs: dict[str, list[float]] = {}
    rng = rng_for(cfg["seed"], "calibration", lang.code)
    sample = list(items)
    rng.shuffle(sample)
    for op in OPS:
        pairs = []
        for it in sample:
            if len(pairs) >= vcfg["calibration_per_op"]:
                break
            d, _ = candidate(it, op, lang, cfg["seed"], 1000, clean_keys)
            if d is not None:
                pairs.append((it["text"], lang.detokenize(d.tokens)))
        diffs[op] = lm_diffs(tools["lm"], pairs)
    if vcfg.get("delta") is not None:
        delta = float(vcfg["delta"])
    else:
        delta = calibrate_delta(diffs, vcfg["target_pass_rate"], vcfg["min_delta"])
    rep = {
        "delta": round(delta, 4),
        "target_pass_rate": vcfg["target_pass_rate"],
        "per_op_distribution": {op: distribution(v) for op, v in diffs.items()},
        "per_op_pass_rate_at_delta": {op: (round(sum(x >= delta for x in v) / len(v), 4) if v else None)
                                      for op, v in diffs.items()},
    }
    return delta, rep


def corrupt_split(items: list[dict], n_target: int, split: str, lang: Language, tools: Tools, cfg: dict,
                  delta: float, clean_keys: set[str], stats: OpStats) -> tuple[list[tuple[dict, Draft | None]], dict]:
    seed = cfg["seed"]
    order = list(items)
    rng_for(seed, "roles", lang.code, split).shuffle(order)
    n_rows = min(n_target, len(order))
    targets = mix_targets(n_rows, cfg["mix"])
    # Clean rows are drawn FIRST, uniformly at random, so the clean set is an unbiased sample
    # of the split (drawing them last would leave the sentences where few operations apply).
    clean_items = order[: targets["clean"]]
    available = order[targets["clean"]:]
    chosen: list[tuple[dict, Draft | None]] = []
    shortfall = {}
    max_att = cfg["verify"]["max_attempts"]
    for role in ROLE_ORDER:
        need = targets.get(role, 0)
        sk = role_stat_key(role)
        accepted = 0
        pos = 0
        rejected_items: list[dict] = []
        while accepted < need and pos < len(available):
            want = int((need - accepted) * 1.5) + 8
            batch: list[tuple[dict, Draft]] = []
            while len(batch) < want and pos < len(available):
                it = available[pos]
                pos += 1
                got = None
                for att in range(max_att):
                    stats.tried[sk] += 1
                    d, why = candidate(it, role, lang, seed, att, clean_keys)
                    if d is not None:
                        got = d
                        break
                    if why == "no_target":
                        stats.no_target[sk] += 1
                        break
                    stats.rejected[sk][why] += 1
                if got is None:
                    rejected_items.append(it)
                else:
                    batch.append((it, got))
            diffs = lm_diffs(tools["lm"], [(it["text"], lang.detokenize(d.tokens)) for it, d in batch])
            for (it, d), df in zip(batch, diffs):
                stats.lm_diffs[sk].append(df)
                if accepted < need and df >= delta:
                    chosen.append((it, d))
                    stats.accepted[sk] += 1
                    accepted += 1
                elif df < delta:
                    stats.rejected[sk]["lm_delta"] += 1
                    rejected_items.append(it)
                else:
                    rejected_items.append(it)
        used = {id(it) for it, _ in chosen}
        available = [it for it in available if id(it) not in used]
        if accepted < need:
            shortfall[role] = need - accepted
    n_extra_clean = n_rows - len(chosen) - len(clean_items)
    clean_items = clean_items + available[: max(0, n_extra_clean)]  # fill corruption shortfalls
    chosen.extend((it, None) for it in clean_items)
    info = {"targets": targets, "shortfall": shortfall, "rows": len(chosen),
            "short_rows": max(0, n_target - len(chosen))}
    return chosen, info


# ---------------------------------------------------------- natural errors
def natural_rows(pairs: list[dict], lang: Language, n: int, mode: str, seed: int, main_keys: set[str]) -> list[dict]:
    rng = rng_for(seed, "natural", lang.code)
    pairs = [p for p in pairs if normalized_key(p["correction"]) not in main_keys]
    rng.shuffle(pairs)
    usable = []
    for p in pairs:
        src = lang.tokenize(p["text"])
        tgt = lang.tokenize(p["correction"])
        a = [split_core(t)[0].lower() for t in src]
        b = [split_core(t)[0].lower() for t in tgt]
        usable.append((p, src, tgt, a, b))
    rows = []
    n_bad = n // 2
    n_ok = n - n_bad
    for p, src, tgt, a, b in usable:
        if len(rows) >= n:
            break
        want_bad = sum(1 for r in rows if r["binary"] == "not_ok") < n_bad
        want_ok = sum(1 for r in rows if r["binary"] == "ok") < n_ok
        if a != b and want_bad:
            flags = [0] * len(src)
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
                if op in ("replace", "delete"):
                    for i in range(i1, i2):
                        flags[i] = 1
                elif op == "insert":
                    flags[min(i1, len(src) - 1)] = 1  # missing-word convention: token after the gap
            toks, raw, clean_tokens = src, p["text"], tgt
            binary = "not_ok"
        elif want_ok:
            flags = [0] * len(tgt)
            toks, raw, clean_tokens = tgt, p["correction"], tgt
            binary = "ok"
        else:
            continue
        norm = normalize_tokens(toks, mode)
        rows.append({
            "id": f"{lang.code}-{p['source']}-{p['source_id']}",
            "lang": lang.code, "text": lang.detokenize(norm), "tokens": norm,
            "clean_text": lang.detokenize(normalize_tokens(clean_tokens, mode)),
            "clean_text_raw": p["correction"],
            "binary": binary, "multiclass": None, "multilabel": [-1] * len(OPS),
            "token_tags": [None] * len(norm), "token_binary": flags, "n_corruptions": -1,
            "source": p["source"], "source_id": p["source_id"], "source_author": p.get("source_author", ""),
            "license": p["license"], "split": "test_natural",
        })
    return rows


# ------------------------------------------------------------------- build
def build_language(cfg: dict, sources_cfg: dict, lang_code: str, pools: dict, requested_mode: str,
                   forced: dict[str, str] | None = None) -> dict[str, Any]:
    lang = get_language(lang_code, cfg)
    tools = Tools(cfg, lang)
    mode_rep = pools["report"]["main"]
    decision = decide_mode(lang_code, requested_mode, mode_rep["permissive_pool"], mode_rep["total_pool"],
                           cfg["min_sentences"])
    log.info("[%s] licence mode: %s (%s)", lang_code, decision.mode, decision.reason)
    main = [r for r in pools["main"] if row_allowed(r["license"], decision.mode)]
    clean_keys = {normalized_key(r["text"]) for r in pools["main"]} | {normalized_key(r["text"]) for r in pools["ood"]}
    main_keys = {normalized_key(r["text"]) for r in main}
    ood = [r for r in pools["ood"] if row_allowed(r["license"], decision.mode) and normalized_key(r["text"]) not in main_keys]
    natural = [r for r in pools["natural"] if row_allowed(r["license"], decision.mode)]
    extras_excluded = []
    if pools["ood"] and not ood:
        extras_excluded.append("test_ood_source (licence mode excludes the OOD source)")
    if pools["natural"] and not natural:
        extras_excluded.append("test_natural (licence mode excludes the natural-error source)")
    sizes = cfg["splits"]
    split_items, split_info = assign_splits(main, sizes, lang, cfg["seed"], sizes["surplus"], forced)
    delta, calib = calibrate(split_items["train"], lang, tools, cfg, clean_keys)
    log.info("[%s] LM delta = %.4f (pass rates at delta: %s)", lang_code, delta, calib["per_op_pass_rate_at_delta"])
    stats = OpStats()
    rows: dict[str, list[dict]] = {}
    metas: dict[str, dict] = {}
    split_reports = {}
    plan = [(s, split_items[s], sizes[s]) for s in MAIN_SPLITS]
    if ood:
        ood_sorted = sorted(ood, key=lambda r: rng_for(cfg["seed"], "ood", r["source"], r["source_id"]).random())
        plan.append(("test_ood_source", ood_sorted, sizes["test_ood_source"]))
    for split, items, n in plan:
        chosen, info = corrupt_split(items, n, split, lang, tools, cfg, delta, clean_keys, stats)
        split_reports[split] = info
        out = []
        for it, d in chosen:
            row, meta = make_row(lang, it, d, split, cfg["normalize"])
            out.append(row)
            metas[row["id"]] = meta
        out.sort(key=lambda r: r["id"])
        rows[split] = out
        log.info("[%s] %s: %d rows (shortfall %s)", lang_code, split, len(out), info["shortfall"] or "none")
    if natural:
        nat = natural_rows(natural, lang, sizes["test_natural"], cfg["normalize"], cfg["seed"], main_keys)
        if nat:
            rows["test_natural"] = sorted(nat, key=lambda r: r["id"])
            for r in nat:
                metas[r["id"]] = {"id": r["id"], "text_raw": r["clean_text_raw"], "ops": [], "content_lemmas": []}
    # LanguageTool diagnostic on accepted corrupted rows
    try:
        lt = tools["lt"]
        for split in rows:
            if split == "test_natural":
                continue
            corrupted = [r for r in rows[split] if r["binary"] == "not_ok"]
            flags = lt.flags([metas[r["id"]]["text_raw"] for r in corrupted])
            for r, fl in zip(corrupted, flags):
                stats.lt_flags[r["multiclass"]].append(fl)
    except Exception as exc:  # LanguageTool missing: diagnostic only
        log.warning("[%s] LanguageTool diagnostic skipped: %s", lang_code, exc)
    tools.close()
    lemma_lookup = {rid: set(m["content_lemmas"]) for rid, m in metas.items()}
    leaks = check_leakage(rows, set(split_info["reserved_lemmas"]), lemma_lookup)
    if leaks:
        raise RuntimeError(f"[{lang_code}] split leakage detected: {leaks[:5]}")
    licenses = {r["license"] for rs in rows.values() for r in rs}
    if decision.mode == "apache-strict":
        lic = {"hf": "apache-2.0", "spdx": "Apache-2.0", "family": "permissive", "noncommercial": False,
               "share_alike": False, "attribution": False, "components": sorted(licenses)}
    else:
        lic = combine_licenses(licenses | {"Apache-2.0"})
    return {
        "lang": lang_code, "rows": rows, "metas": metas, "decision": decision.as_dict(), "license": lic,
        "split_info": {k: v for k, v in split_info.items() if k != "reserved_lemmas"},
        "reserved_lemmas": split_info["reserved_lemmas"], "split_reports": split_reports,
        "calibration": calib, "op_stats": stats.as_dict(), "extras_excluded": extras_excluded,
        "pool_report": pools["report"],
    }


def forced_from_links(en_result: dict, links: list[tuple[str, str]]) -> dict[str, str]:
    en_split = {r["source_id"]: s for s, rs in en_result["rows"].items() for r in rs if r["source"] == "tatoeba"}
    forced: dict[str, str] = {}
    conflict = set()
    for e, i in links:
        s = en_split.get(e)
        if s is None or s not in MAIN_SPLITS:
            continue
        if i in forced and forced[i] != s:
            conflict.add(i)
        forced[i] = s
    for i in conflict:
        forced[i] = "__conflict__"
    return forced


def parallel_rows(en: dict, it: dict, links: list[tuple[str, str]], seed: int) -> dict[str, list[dict]]:
    by_id = {}
    for res in (en, it):
        for s, rs in res["rows"].items():
            if s not in MAIN_SPLITS:
                continue
            for r in rs:
                if r["source"] == "tatoeba":
                    by_id[(r["lang"], r["source_id"])] = r
    pairs = sorted(set(links), key=lambda p: rng_for(seed, "parallel", *p).random())
    used: set[tuple[str, str]] = set()
    out: dict[str, list[dict]] = defaultdict(list)
    for e, i in pairs:
        re_, ri = by_id.get(("en", e)), by_id.get(("it", i))
        if not re_ or not ri or re_["split"] != ri["split"]:
            continue
        if ("en", e) in used or ("it", i) in used:
            continue
        used.update({("en", e), ("it", i)})
        pid = f"tatoeba-{e}-{i}"
        out[re_["split"]].append({**re_, "pair_id": pid})
        out[ri["split"]].append({**ri, "pair_id": pid})
    return {s: sorted(rs, key=lambda r: (r["pair_id"], r["lang"])) for s, rs in out.items()}


# -------------------------------------------------------------- stats/audit
def split_stats(rows: list[dict]) -> dict:
    natural = rows and rows[0]["split"] == "test_natural"
    lens = [len(r["tokens"]) for r in rows]
    st = {
        "rows": len(rows),
        "binary": dict(Counter(r["binary"] for r in rows)),
        "length": distribution([float(x) for x in lens]),
        "length_histogram": dict(sorted(Counter(lens).items())),
        "licenses": dict(Counter(r["license"] for r in rows)),
    }
    if not natural:
        st["multiclass"] = dict(Counter(r["multiclass"] for r in rows))
        st["multilabel_totals"] = {op: sum(r["multilabel"][k] for r in rows) for k, op in enumerate(OPS)}
        st["n_corruptions"] = dict(Counter(r["n_corruptions"] for r in rows))
        st["token_tags"] = dict(Counter(t for r in rows for t in r["token_tags"]))
    st["token_binary_positive_rate"] = round(
        sum(sum(r["token_binary"]) for r in rows) / max(1, sum(len(r["token_binary"]) for r in rows)), 4)
    return st


def export_audits(cfg: dict, result: dict) -> dict[str, str]:
    lang = result["lang"]
    adir = Path(cfg["paths"]["audit"])
    adir.mkdir(parents=True, exist_ok=True)
    rng = rng_for(cfg["seed"], "audit", lang)
    pool = [r for s, rs in result["rows"].items() if s != "test_natural" for r in rs]
    clean = sorted([r for r in pool if r["binary"] == "ok"], key=lambda r: r["id"])
    rng.shuffle(clean)
    clean = clean[: cfg["audit"]["n_clean"]]
    p1 = adir / f"clean_{lang}.csv"
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "split", "source", "license", "clean_text_raw", "is_acceptable"])
        for r in clean:
            w.writerow([r["id"], lang, r["split"], r["source"], r["license"], r["clean_text_raw"], ""])
    corrupted = sorted([r for r in pool if r["binary"] == "not_ok"], key=lambda r: r["id"])
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in corrupted:
        by_class[r["multiclass"]].append(r)
    classes = OPS + ["multiple"]
    per = cfg["audit"]["n_corrupted"] // len(classes)
    extra = cfg["audit"]["n_corrupted"] - per * len(classes)
    picked = []
    for k, c in enumerate(classes):
        rs = list(by_class.get(c, []))
        rng.shuffle(rs)
        picked.extend(rs[: per + (1 if k < extra else 0)])
    p2 = adir / f"corrupted_{lang}.csv"
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "split", "multiclass", "operations", "clean_text_raw", "corrupted_text_raw",
                    "tagged_tokens", "is_really_wrong"])
        for r in picked:
            meta = result["metas"][r["id"]]
            ops = "; ".join(
                o["op"] + (f"({o['edit'].get('feature') or o['edit'].get('set')}: {o['edit'].get('new')})" if "edit" in o else "")
                for o in meta["ops"])
            tagged = " ".join(t if g == "O" else f"[{t}|{g}]" for t, g in zip(r["tokens"], r["token_tags"]))
            w.writerow([r["id"], lang, r["split"], r["multiclass"], ops, r["clean_text_raw"], meta["text_raw"],
                        tagged, ""])
    return {"clean": str(p1), "corrupted": str(p2)}


def build_all(cfg: dict, sources_cfg: dict, langs: list[str], requested_mode: str) -> dict[str, Any]:
    from .sources import load_tatoeba_links

    pools = {lg: get_pools(cfg, sources_cfg, lg) for lg in langs}
    lic_rep = license_report(cfg, pools, requested_mode)
    out_dir = Path(cfg["paths"]["out"])
    write_json(out_dir / "reports" / "licenses.json", lic_rep)
    write_text(out_dir / "reports" / "licenses.md", format_license_report(lic_rep))
    results: dict[str, dict] = {}
    links = load_tatoeba_links(cfg["paths"]["raw"]) if set(langs) == {"en", "it"} else []
    for lg in langs:
        forced = None
        if lg == "it" and "en" in results and links:
            forced = forced_from_links(results["en"], links)
        results[lg] = build_language(cfg, sources_cfg, lg, pools[lg], requested_mode, forced)
        # update the licence report with the final decision
        lic_rep["languages"][lg]["decision"] = results[lg]["decision"]
    write_json(out_dir / "reports" / "licenses.json", lic_rep)
    write_text(out_dir / "reports" / "licenses.md", format_license_report(lic_rep))
    parallel = parallel_rows(results["en"], results["it"], links, cfg["seed"]) if links and len(results) == 2 else {}
    return {"results": results, "parallel": parallel, "license_report": lic_rep, "links": len(links)}
