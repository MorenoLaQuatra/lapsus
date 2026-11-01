# SPDX-License-Identifier: Apache-2.0
"""Human audit: read the filled audit CSVs and report error rate and precision.

audit/clean_{lang}.csv      column is_acceptable  (1/0, yes/no, y/n, true/false)
audit/corrupted_{lang}.csv  column is_really_wrong
Empty cells are ignored, so partially filled files are fine.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

YES = {"1", "yes", "y", "true", "t", "si", "sì"}
NO = {"0", "no", "n", "false", "f"}


def _value(cell: str) -> bool | None:
    c = (cell or "").strip().lower()
    if c in YES:
        return True
    if c in NO:
        return False
    return None


def audit_language(audit_dir: str | Path, lang: str) -> dict:
    out: dict = {"lang": lang}
    p = Path(audit_dir) / f"clean_{lang}.csv"
    if p.exists():
        vals = [_value(r.get("is_acceptable", "")) for r in csv.DictReader(open(p, encoding="utf-8"))]
        filled = [v for v in vals if v is not None]
        out["clean"] = {"rows": len(vals), "annotated": len(filled),
                        "error_rate": round(sum(1 for v in filled if not v) / len(filled), 4) if filled else None}
    p = Path(audit_dir) / f"corrupted_{lang}.csv"
    if p.exists():
        per: dict[str, list[bool]] = defaultdict(list)
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n += 1
            v = _value(r.get("is_really_wrong", ""))
            if v is not None:
                per[r["multiclass"]].append(v)
        allv = [x for v in per.values() for x in v]
        out["corrupted"] = {
            "rows": n, "annotated": len(allv),
            "precision": round(sum(allv) / len(allv), 4) if allv else None,
            "precision_per_operation": {k: {"n": len(v), "precision": round(sum(v) / len(v), 4)} for k, v in sorted(per.items())},
        }
    return out


def audit(audit_dir: str | Path, langs: list[str]) -> dict:
    res = {lg: audit_language(audit_dir, lg) for lg in langs}
    filled = any((r.get("clean", {}).get("annotated") or 0) + (r.get("corrupted", {}).get("annotated") or 0) > 0
                 for r in res.values())
    return {"filled": filled, "languages": res}


def format_audit(rep: dict) -> str:
    lines = []
    for lg, r in rep["languages"].items():
        c = r.get("clean")
        if c:
            lines.append(f"[{lg}] clean audit: {c['annotated']}/{c['rows']} annotated, error rate {c['error_rate']}")
        k = r.get("corrupted")
        if k:
            lines.append(f"[{lg}] corrupted audit: {k['annotated']}/{k['rows']} annotated, precision {k['precision']}")
            for op, s in k["precision_per_operation"].items():
                lines.append(f"    {op}: precision {s['precision']} (n={s['n']})")
    if not rep["filled"]:
        lines.append("No annotations yet: fill is_acceptable / is_really_wrong in the audit CSVs and rerun.")
    return "\n".join(lines)
