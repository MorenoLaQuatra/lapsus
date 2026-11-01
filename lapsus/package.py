# SPDX-License-Identifier: Apache-2.0
"""Release packaging: release/lapsus-vX.Y.Z/ with one zip per config (csv, jsonl
and the saved DatasetDict, plus card, attribution and licence files),
SHA256SUMS and RELEASE_NOTES.md. Nothing is uploaded: the owner tags and
publishes the release by hand."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from .utils import ROOT, write_text


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_tree(z: zipfile.ZipFile, folder: Path, arc_prefix: str) -> None:
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            zi = zipfile.ZipInfo(f"{arc_prefix}/{p.relative_to(folder).as_posix()}", date_time=(2020, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, p.read_bytes())


def release_notes(info: dict, version: str, sums: dict[str, str]) -> str:
    L = [f"# Lapsus v{version}", "",
         f"Seed {info['seed']}, normalization `{info['normalize']}`, requested licence mode "
         f"`{info['license_mode_requested']}`, Hub repository licence `{info['repo_license']}`.", ""]
    L += ["## Configs", "", "| config | licence | splits |", "|---|---|---|"]
    for name, c in info["configs"].items():
        L.append(f"| {name} | {c['license']} | " + ", ".join(f"{k} {v}" for k, v in c["splits"].items()) + " |")
    L += ["", "## Licence report", ""]
    for lg, li in info["languages"].items():
        d = li["decision"]
        L.append(f"- {lg}: mode **{d['mode']}** ({d['reason']}); permissive pool {d['permissive_pool']}, total pool "
                 f"{d['total_pool']}, threshold {d['threshold']}; config licence {li['license']['hf']}.")
        for e in li.get("extras_excluded", []):
            L.append(f"  - not built: {e}")
    L += ["", "## Filters (rejections per filter)", ""]
    for lg, li in info["languages"].items():
        L.append(f"- {lg}: " + ", ".join(f"{k} {v}" for k, v in li["filter_rejections"].items()))
    L += ["", "## Verification", ""]
    for lg, li in info["languages"].items():
        L.append(f"- {lg}: LM delta {li['calibration']['delta']}; acceptance per operation: "
                 + ", ".join(f"{op} {s['acceptance_rate']}" for op, s in li["op_stats"].items())
                 + "; LanguageTool flag rate: "
                 + ", ".join(f"{op} {s['languagetool_flag_rate']}" for op, s in li["op_stats"].items()))
    L += ["", "## Audit", ""]
    audit = info.get("audit") or {}
    if audit.get("filled"):
        for lg, a in audit["languages"].items():
            L.append(f"- {lg}: {json.dumps(a, ensure_ascii=False)}")
    else:
        L.append("- Audit CSVs exported (audit/clean_{lang}.csv, audit/corrupted_{lang}.csv); not yet annotated.")
    L += ["", "## Sources", ""]
    for s in info["sources"]:
        if s["enabled"]:
            L.append(f"- {s['name']}: {s['url']} ({s['license']}), export date(s) {', '.join(s['export_dates']) or 'n/a'}")
    L += ["", "## Tool versions", ""]
    for k, v in info["tool_versions"].items():
        L.append(f"- {k}: {v}")
    if info.get("deviations"):
        L += ["", "## Deviations from the specification", ""] + [f"- {d}" for d in info["deviations"]]
    L += ["", "## Files", "", "```"] + [f"{h}  {n}" for n, h in sums.items()] + ["```", ""]
    return "\n".join(L)


def package(cfg: dict, version: str) -> Path:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("version must look like X.Y.Z")
    out = Path(cfg["paths"]["out"])
    info = json.loads((out / "build_info.json").read_text())
    rel = Path(cfg["paths"]["release"]) / f"lapsus-v{version}"
    rel.mkdir(parents=True, exist_ok=True)
    sums: dict[str, str] = {}
    card = out / "hub" / "README.md"
    for name in info["configs"]:
        zpath = rel / f"lapsus-v{version}-{name}.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            prefix = f"lapsus-v{version}-{name}"
            _add_tree(z, out / "csv" / name, f"{prefix}/csv")
            _add_tree(z, out / "jsonl" / name, f"{prefix}/jsonl")
            _add_tree(z, out / "hf" / name, f"{prefix}/hf")
            for f, arc in ((card, "README.md"), (out / "ATTRIBUTION.md", "ATTRIBUTION.md"),
                           (ROOT / "LICENSE", "LICENSE"), (ROOT / "NOTICE", "NOTICE")):
                if f.exists():
                    zi = zipfile.ZipInfo(f"{prefix}/{arc}", date_time=(2020, 1, 1, 0, 0, 0))
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(zi, f.read_bytes())
        sums[zpath.name] = _sha256(zpath)
    write_text(rel / "SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in sums.items()))
    write_text(rel / "RELEASE_NOTES.md", release_notes(info, version, sums))
    return rel
