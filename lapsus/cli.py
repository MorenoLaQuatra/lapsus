# SPDX-License-Identifier: Apache-2.0
"""Command line interface: python -m lapsus {download,licenses,build,check,audit,baselines,package,push}."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import OPS, __version__
from .utils import (FORBIDDEN_DASHES, ROOT, languages_for, load_config, load_sources, setup_logging, write_json,
                    write_text)

log = logging.getLogger("lapsus")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config overriding configs/default.yaml")
    p.add_argument("--sources", default=None, help="sources registry (default: sources.yaml)")
    p.add_argument("--seed", type=int, default=None, help="global seed (default 42)")
    p.add_argument("--lang", choices=["en", "it", "both"], default=None)
    p.add_argument("--license-mode", choices=["auto", "apache-strict", "source-licenses"], default=None)
    p.add_argument("--normalize", choices=["none", "lower", "lower_nopunct"], default=None)
    p.add_argument("--min-sentences", type=int, default=None)
    p.add_argument("--enable-source", action="append", default=[], help="enable a source from sources.yaml by name")
    p.add_argument("-v", "--verbose", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="lapsus", description=__doc__)
    ap.add_argument("--version", action="version", version=f"lapsus {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, hlp in [("download", "download corpora, lexicons, Stanza models and evaluation sets"),
                      ("licenses", "run the filter pipeline on the raw pools and print the licence report"),
                      ("build", "build, verify and export all configs"),
                      ("check", "validate the exported dataset (schema, labels, leakage, dashes)"),
                      ("audit", "report error rate / precision from the filled audit CSVs"),
                      ("baselines", "TF-IDF + logistic regression baselines and reality checks"),
                      ("package", "write release/lapsus-vX.Y.Z/"),
                      ("push", "mirror the dataset to the Hugging Face Hub (dry run by default)")]:
        p = sub.add_parser(name, help=hlp)
        _common(p)
        if name == "download":
            p.add_argument("--force", action="store_true")
        if name == "licenses":
            p.add_argument("--force", action="store_true", help="recompute pools ignoring the cache")
        if name == "package":
            p.add_argument("--version", dest="release_version", required=True, help="X.Y.Z")
        if name == "push":
            p.add_argument("--repo-id", required=True, help="morenolq/lapsus")
            p.add_argument("--private", action="store_true")
            p.add_argument("--push-for-real", action="store_true")
    return ap.parse_args(argv)


def _setup(args: argparse.Namespace) -> tuple[dict, dict, list[str]]:
    setup_logging(args.verbose)
    overrides: dict[str, Any] = {"seed": args.seed, "lang": args.lang, "license_mode": args.license_mode,
                                 "normalize": args.normalize, "min_sentences": args.min_sentences}
    cfg = load_config(args.config, overrides)
    sources = load_sources(args.sources)
    for name in args.enable_source:
        hits = [s for s in sources["sources"] if s["name"] == name]
        if not hits:
            raise SystemExit(f"unknown source {name!r}")
        hits[0]["enabled"] = True
    return cfg, sources, languages_for(cfg["lang"])


# ------------------------------------------------------------------ commands
def cmd_download(args) -> None:
    from .sources import download_all

    cfg, sources, langs = _setup(args)
    download_all(cfg, sources, langs, force=args.force)
    print("Downloads complete. Raw data lives in", cfg["paths"]["raw"], "(never committed).")


def cmd_licenses(args) -> None:
    from .build import get_pools, license_report
    from .licensing import format_license_report

    cfg, sources, langs = _setup(args)
    pools = {lg: get_pools(cfg, sources, lg, force=args.force) for lg in langs}
    rep = license_report(cfg, pools, cfg["license_mode"])
    out = Path(cfg["paths"]["out"]) / "reports"
    write_json(out / "licenses.json", rep)
    text = format_license_report(rep)
    write_text(out / "licenses.md", text)
    print(text)


def _config_license(langs: list[str], results: dict) -> dict:
    from .licensing import most_restrictive

    hf = most_restrictive([results[lg]["license"]["hf"] for lg in langs])
    comps = sorted({c for lg in langs for c in results[lg]["license"]["components"]})
    return {"hf": hf, "components": comps}


def _stats_markdown(stats: dict) -> str:
    L = ["# Lapsus stats report", ""]
    for cfg_name, splits in stats["configs"].items():
        L += [f"## config {cfg_name}", "", "| split | rows | ok | not_ok | mean len | multiclass | token positive rate |",
              "|---|---|---|---|---|---|---|"]
        for s, st in splits.items():
            mc = ", ".join(f"{k} {v}" for k, v in sorted((st.get("multiclass") or {}).items(), key=lambda x: str(x[0])))
            L.append(f"| {s} | {st['rows']} | {st['binary'].get('ok', 0)} | {st['binary'].get('not_ok', 0)} | "
                     f"{st['length'].get('mean')} | {mc or 'n/a'} | {st['token_binary_positive_rate']} |")
        L.append("")
    for lg, li in stats["languages"].items():
        L += [f"## language {lg}", "", f"- licence mode: {li['decision']['mode']} ({li['decision']['reason']})",
              f"- LM delta: {li['calibration']['delta']}", "", "### Filter rejections", "",
              "| filter | rejected | top reasons |", "|---|---|---|"]
        for k, v in li["filter_rejections"].items():
            reasons = li["filter_reasons"].get(k, {})
            L.append(f"| {k} | {v} | " + ", ".join(f"{r} {n}" for r, n in list(reasons.items())[:5]) + " |")
        L += ["", "### Operations", "", "| op | attempts | no target | verified | accepted | acceptance | lm diff p50 | LT flag rate |",
              "|---|---|---|---|---|---|---|---|"]
        for op, s in li["op_stats"].items():
            L.append(f"| {op} | {s['attempts']} | {s['no_target']} | {s['verified_candidates']} | {s['accepted']} | "
                     f"{s['acceptance_rate']} | {s['lm_diff'].get('p50')} | {s['languagetool_flag_rate']} |")
        L += ["", "Rejection reasons per op: " + json.dumps({op: s["rejections"] for op, s in li["op_stats"].items()}), ""]
        L += ["Calibration pass rates at delta: " + json.dumps(li["calibration"]["per_op_pass_rate_at_delta"]), ""]
        L += ["Split shortfalls: " + json.dumps({s: r["shortfall"] for s, r in li["split_reports"].items()}), ""]
    return "\n".join(L) + "\n"


def cmd_build(args) -> None:
    from .audit import audit as run_audit
    from .build import build_all, export_audits, split_stats, tool_versions
    from .export import write_attribution, write_config
    from .hub import write_card
    from .licensing import most_restrictive
    from .sources import source_records

    cfg, sources, langs = _setup(args)
    bundle = build_all(cfg, sources, langs, cfg["license_mode"])
    results = bundle["results"]
    out = Path(cfg["paths"]["out"])
    configs: dict[str, dict[str, list[dict]]] = {lg: results[lg]["rows"] for lg in langs}
    config_langs: dict[str, list[str]] = {lg: [lg] for lg in langs}
    if len(langs) == 2:
        both: dict[str, list[dict]] = {}
        for lg in langs:
            for s, rs in results[lg]["rows"].items():
                both.setdefault(s, []).extend(rs)
        configs["both"] = both
        config_langs["both"] = langs
        if bundle["parallel"]:
            configs["parallel"] = bundle["parallel"]
            config_langs["parallel"] = langs
    config_info = {}
    for name, splits in configs.items():
        counts = write_config(out, name, splits, parallel=(name == "parallel"))
        lic = _config_license(config_langs[name], results)
        config_info[name] = {"license": lic["hf"], "license_components": lic["components"], "splits": counts}
        log.info("exported config %s: %s (%s)", name, counts, lic["hf"])
    repo_license = most_restrictive([c["license"] for c in config_info.values()])
    # stats
    stats = {"configs": {n: {s: split_stats(rs) for s, rs in sp.items()} for n, sp in configs.items()},
             "languages": {}}
    for lg in langs:
        r = results[lg]
        m = r["pool_report"]["main"]
        stats["languages"][lg] = {
            "decision": r["decision"], "license": r["license"], "calibration": r["calibration"],
            "op_stats": r["op_stats"], "split_reports": r["split_reports"], "split_info": r["split_info"],
            "filter_rejections": m["rejected"], "filter_reasons": m.get("reasons", {}),
            "filter_per_license": m["per_license"], "lm_thresholds": m.get("lm_thresholds", {}),
            "ood_filter": r["pool_report"]["ood"], "natural_filter": r["pool_report"]["natural"],
        }
    write_json(out / "reports" / "stats.json", stats)
    write_text(out / "reports" / "stats.md", _stats_markdown(stats))
    audits = {lg: export_audits(cfg, results[lg]) for lg in langs}
    for lg in langs:
        write_json(Path(cfg["paths"]["cache"]) / "build" / f"meta_{lg}.json",
                   {"reserved_lemmas": results[lg]["reserved_lemmas"],
                    "content_lemmas": {k: v["content_lemmas"] for k, v in results[lg]["metas"].items()}})
    src_recs = source_records(cfg, sources, langs)
    all_rows = {n: [r for rs in sp.values() for r in rs] for n, sp in configs.items()}
    write_attribution(out / "ATTRIBUTION.md", src_recs, {lg: all_rows[lg] for lg in langs},
                      {n: {"hf": c["license"], "components": c["license_components"]} for n, c in config_info.items()})
    tools = []
    versions = tool_versions(cfg, langs)
    used_lms = set(versions["lm"].values())
    for t in sources.get("tools", []):
        t = dict(t)
        if t.get("version_from"):
            t["version"] = versions.get(t.pop("version_from"), "")
        if "/" in t["name"] and t["name"] not in used_lms:
            continue  # an HF model id that this build did not use
        tools.append(t)
    for m in sorted(used_lms - {t["name"] for t in tools}):
        tools.append({"name": m, "license": "see the model card", "notes": "causal LM (filter and verifier)"})
    deviations = collect_deviations(cfg, results, langs)
    info = {
        "version": __version__, "seed": cfg["seed"], "lang": cfg["lang"], "normalize": cfg["normalize"],
        "license_mode_requested": cfg["license_mode"], "repo_license": repo_license, "mix": cfg["mix"],
        "configs": config_info,
        "languages": {lg: {"decision": results[lg]["decision"], "license": results[lg]["license"],
                           "calibration": results[lg]["calibration"], "op_stats": results[lg]["op_stats"],
                           "split_info": results[lg]["split_info"],
                           "filter_rejections": results[lg]["pool_report"]["main"]["rejected"],
                           "extras_excluded": results[lg]["extras_excluded"]} for lg in langs},
        "tool_versions": versions, "tools": tools, "sources": src_recs, "audit_files": audits,
        "audit": run_audit(cfg["paths"]["audit"], langs), "deviations": deviations,
        "parallel_links": bundle["links"],
    }
    write_json(out / "build_info.json", info)
    write_card(info, out, "morenolq/lapsus")
    print_summary(info)


def collect_deviations(cfg: dict, results: dict, langs: list[str]) -> list[str]:
    dev = []
    for lg in langs:
        r = results[lg]
        if r["split_info"]["scale"] < 1:
            dev.append(f"{lg}: pool too small, train/validation/test scaled by {r['split_info']['scale']}")
        for s, rep in r["split_reports"].items():
            if rep["shortfall"]:
                dev.append(f"{lg}/{s}: corruption shortfall {rep['shortfall']} (filled with clean rows)")
            if rep["short_rows"]:
                dev.append(f"{lg}/{s}: {rep['short_rows']} rows short of the requested size")
        for e in r["extras_excluded"]:
            dev.append(f"{lg}: {e}")
    return dev


def print_summary(info: dict) -> None:
    print("\n==================== Lapsus build summary ====================")
    print(f"requested licence mode: {info['license_mode_requested']}")
    for lg, li in info["languages"].items():
        d = li["decision"]
        print(f"  [{lg}] mode used: {d['mode']} because {d['reason']} -> config licence {li['license']['hf']}")
        for e in li["extras_excluded"]:
            print(f"  [{lg}] not built: {e}")
    print(f"Hub repository licence (most restrictive across configs): {info['repo_license']}")
    for name, c in info["configs"].items():
        print(f"  config {name}: {c['license']} {c['splits']}")
    if info["deviations"]:
        print("deviations:")
        for d in info["deviations"]:
            print("  -", d)
    print("===============================================================\n")


def cmd_check(args) -> int:
    from datasets import DatasetDict

    from .corruptions import views_consistent
    from .export import features
    from .lang import get_language
    from .splits import check_leakage

    cfg, _, langs = _setup(args)
    out = Path(cfg["paths"]["out"])
    info = json.loads((out / "build_info.json").read_text())
    problems: list[str] = []
    for name in info["configs"]:
        dd = DatasetDict.load_from_disk(str(out / "hf" / name))
        expected = features(parallel=(name == "parallel"))
        dataset: dict[str, list[dict]] = {}
        for split, ds in dd.items():
            if ds.features != expected:
                problems.append(f"{name}/{split}: features differ from the schema")
            f = ds.features
            rows = []
            for r in ds:
                r = dict(r)
                r["binary"] = f["binary"].int2str(r["binary"])
                r["multiclass"] = None if r["multiclass"] == -1 else f["multiclass"].int2str(r["multiclass"])
                r["token_tags"] = [None if t == -1 else f["token_tags"].feature.int2str(t) for t in r["token_tags"]]
                if not views_consistent(r):
                    problems.append(f"{name}/{split}/{r['id']}: inconsistent label views")
                L = get_language(r["lang"], cfg)
                if L.tokenize(r["text"]) != r["tokens"]:
                    problems.append(f"{name}/{split}/{r['id']}: tokens do not match tokenize(text)")
                if r["split"] != split:
                    problems.append(f"{name}/{split}/{r['id']}: split column mismatch")
                rows.append(r)
            dataset[split] = rows
            ids = Counter(r["id"] for r in rows)
            if name != "parallel" and any(v > 1 for v in ids.values()):
                problems.append(f"{name}/{split}: duplicate ids")
        if name in langs:
            meta_p = Path(cfg["paths"]["cache"]) / "build" / f"meta_{name}.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
            problems += check_leakage(dataset, set(meta["reserved_lemmas"]) if meta else None,
                                      {k: set(v) for k, v in meta["content_lemmas"].items()} if meta else None)
        elif name != "parallel":
            problems += check_leakage(dataset)
    for p in list(out.rglob("*.csv")) + list(out.rglob("*.jsonl")) + list(out.rglob("*.md")) + list(out.rglob("*.json")):
        text = p.read_text(encoding="utf-8")
        if any(d in text for d in FORBIDDEN_DASHES):
            problems.append(f"forbidden dash in {p}")
    if problems:
        print(f"check FAILED with {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print(f"check passed: {len(info['configs'])} configs, schema, label views, tokenization, leakage and dashes OK")
    return 0


def cmd_audit(args) -> None:
    from .audit import audit, format_audit

    cfg, _, langs = _setup(args)
    rep = audit(cfg["paths"]["audit"], langs)
    write_json(Path(cfg["paths"]["out"]) / "reports" / "audit.json", rep)
    print(format_audit(rep))


def cmd_baselines(args) -> None:
    cfg, _, langs = _setup(args)
    spec = importlib.util.spec_from_file_location("lapsus_baselines", ROOT / "baselines.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.main(cfg, langs)


def cmd_package(args) -> None:
    from .package import package

    cfg, _, _ = _setup(args)
    rel = package(cfg, args.release_version)
    print(f"Release folder written: {rel}")


def cmd_push(args) -> None:
    from .hub import push

    cfg, _, _ = _setup(args)
    push(cfg, args.repo_id, args.private, args.push_for_real)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handlers = {"download": cmd_download, "licenses": cmd_licenses, "build": cmd_build, "check": cmd_check,
                "audit": cmd_audit, "baselines": cmd_baselines, "package": cmd_package, "push": cmd_push}
    rc = handlers[args.command](args)
    sys.exit(rc or 0)
