# SPDX-License-Identifier: Apache-2.0
"""Licence logic: compatibility table, mode selection and reports.

Modes
  apache-strict    build only from the permissive pool (Apache-2.0, CC0, public
                   domain); the dataset licence is apache-2.0. Fails if the pool
                   is below min_sentences.
  source-licenses  use all enabled sources; labels/corruptions/annotations are
                   Apache-2.0 and the dataset licence is the most restrictive one
                   resulting from the row licences (per-row `license` column and
                   ATTRIBUTION.md).
  auto             per language: apache-strict if the permissive pool reaches the
                   threshold, otherwise source-licenses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

PERMISSIVE = {"Apache-2.0", "CC0-1.0", "LicenseRef-public-domain", "PDDL-1.0", "Unlicense"}

# family rank: higher = more restrictive
FAMILY_RANK = {"permissive": 0, "by": 1, "by-sa": 2, "by-nc": 3, "by-nc-sa": 4, "unverified": 5}


@dataclass(frozen=True)
class LicenseInfo:
    spdx: str
    family: str
    version: str | None
    attribution: bool
    share_alike: bool
    noncommercial: bool


def license_info(spdx: str) -> LicenseInfo:
    s = spdx.strip()
    if s in PERMISSIVE:
        return LicenseInfo(s, "permissive", None, False, False, False)
    if s.startswith("CC-BY"):
        parts = s.split("-")
        version = parts[-1] if parts[-1][:1].isdigit() else (parts[-2] if parts[-2][:1].isdigit() else None)
        sa = "SA" in parts
        nc = "NC" in parts
        nd = "ND" in parts
        if nd:
            return LicenseInfo(s, "unverified", version, True, False, nc)
        fam = "by-nc-sa" if nc and sa else "by-nc" if nc else "by-sa" if sa else "by"
        return LicenseInfo(s, fam, version, True, sa, nc)
    return LicenseInfo(s, "unverified", None, True, False, False)


def is_permissive(spdx: str) -> bool:
    return license_info(spdx).family == "permissive"


def _vkey(v: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in (v or "0").split("."))
    except ValueError:
        return (0,)


def combine_licenses(row_licenses: Iterable[str]) -> dict:
    """Most restrictive resulting licence for a set of row licences.

    Labels, corruptions and annotations are Apache-2.0, so a fully permissive
    set yields apache-2.0. Any CC BY gives cc-by-<v>; any SA gives share-alike;
    any NC gives non-commercial; anything unverified gives "other".
    """
    infos = [license_info(x) for x in set(row_licenses)]
    nc = any(i.noncommercial for i in infos)
    sa = any(i.share_alike for i in infos)
    by = any(i.attribution for i in infos)
    unverified = any(i.family == "unverified" for i in infos)
    if unverified:
        return {"hf": "other", "spdx": "LicenseRef-mixed-unverified", "family": "unverified",
                "noncommercial": nc, "share_alike": sa, "attribution": True,
                "components": sorted({i.spdx for i in infos})}
    if not by:
        return {"hf": "apache-2.0", "spdx": "Apache-2.0", "family": "permissive", "noncommercial": False,
                "share_alike": False, "attribution": False, "components": sorted({i.spdx for i in infos})}
    fam = "by-nc-sa" if nc and sa else "by-nc" if nc else "by-sa" if sa else "by"
    # version: the highest version among the components that impose the chosen family's
    # restrictions (e.g. SA/NC sources); otherwise the highest CC BY version.
    relevant = [i for i in infos if i.family == fam] or [i for i in infos if i.attribution]
    version = max((i.version for i in relevant), key=_vkey)
    spdx = f"CC-{fam.upper()}-{version}"
    return {"hf": f"cc-{fam}-{version}", "spdx": spdx, "family": fam, "noncommercial": nc,
            "share_alike": sa, "attribution": True, "components": sorted({i.spdx for i in infos})}


def most_restrictive(hf_ids: Iterable[str]) -> str:
    """Pick the most restrictive HF licence id across configs."""
    def rank(h: str) -> tuple[int, tuple[int, ...]]:
        if h == "other":
            return (FAMILY_RANK["unverified"], (0,))
        if h in ("apache-2.0", "cc0-1.0"):
            return (0, (0,))
        fam = h[3:].rsplit("-", 1)[0]
        return (FAMILY_RANK.get(fam, 5), _vkey(h.rsplit("-", 1)[-1]))

    ids = list(hf_ids)
    return max(ids, key=rank) if ids else "apache-2.0"


class LicenseError(RuntimeError):
    pass


@dataclass
class ModeDecision:
    lang: str
    requested: str
    mode: str
    permissive_pool: int
    total_pool: int
    threshold: int
    reason: str
    extras_allowed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def decide_mode(lang: str, requested: str, permissive_pool: int, total_pool: int, threshold: int) -> ModeDecision:
    if requested not in ("auto", "apache-strict", "source-licenses"):
        raise ValueError(f"unknown licence mode {requested!r}")
    if requested == "apache-strict":
        if permissive_pool < threshold:
            raise LicenseError(
                f"[{lang}] apache-strict requested but the permissive pool has {permissive_pool} sentences "
                f"after filtering (< min_sentences={threshold}). Use --license-mode source-licenses or auto."
            )
        return ModeDecision(lang, requested, "apache-strict", permissive_pool, total_pool, threshold,
                            f"requested; permissive pool {permissive_pool} >= {threshold}")
    if requested == "source-licenses":
        return ModeDecision(lang, requested, "source-licenses", permissive_pool, total_pool, threshold,
                            "requested")
    if permissive_pool >= threshold:
        return ModeDecision(lang, requested, "apache-strict", permissive_pool, total_pool, threshold,
                            f"auto: permissive pool {permissive_pool} >= threshold {threshold}")
    return ModeDecision(lang, requested, "source-licenses", permissive_pool, total_pool, threshold,
                        f"auto: permissive pool {permissive_pool} < threshold {threshold}, falling back to source licences")


def row_allowed(row_license: str, mode: str) -> bool:
    return mode == "source-licenses" or is_permissive(row_license)


def format_license_report(report: dict) -> str:
    """Human readable licence report (markdown-ish, no long dashes)."""
    lines = ["# Lapsus licence report", ""]
    for lang, r in report["languages"].items():
        lines.append(f"## {lang}")
        lines.append("")
        lines.append("| licence | permissive | raw | after a | after b | processed c-e | survived c-e | final pool |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for lic, c in sorted(r["per_license"].items()):
            lines.append(
                f"| {lic} | {'yes' if is_permissive(lic) else 'no'} | {c.get('raw', 0)} | {c.get('a', 0)} | "
                f"{c.get('b', 0)} | {c.get('processed', 0)} | {c.get('e', 0)} | {c.get('final', 0)} |"
            )
        lines.append("")
        lines.append(f"- permissive pool: {r['permissive_pool']} (threshold min_sentences = {r['threshold']}): "
                     f"{'REACHED' if r['permissive_pool'] >= r['threshold'] else 'NOT reached'}")
        lines.append(f"- total pool: {r['total_pool']}")
        if r.get("stopped_early"):
            lines.append("- note: non-permissive rows were processed only until pool_target was reached; "
                         "'processed c-e' shows how many were examined.")
        if r.get("decision"):
            d = r["decision"]
            lines.append(f"- mode: **{d['mode']}** (requested {d['requested']}): {d['reason']}")
        if r.get("ood"):
            lines.append(f"- out-of-domain pool: {r['ood']}")
        lines.append("")
    return "\n".join(lines) + "\n"
