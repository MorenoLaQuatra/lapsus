# SPDX-License-Identifier: Apache-2.0
"""Clean-sentence filters (spec section 2). Each filter logs its rejection count.

a  NFC, trim, one sentence, min..max tokens
b  no digits, URLs, emoji, quotes, brackets, symbols, all-caps, at most one proper noun
c  language ID must match (lingua)
d  LanguageTool: no grammar/typo matches (style, casing, punctuation ignored)
e  Stanza: exactly one root, no X tags, no failures (also re-checks proper nouns)
f  causal LM: drop the lowest 5 percent mean log-prob per length bucket
g  deduplicate: exact, normalized, near-duplicates (MinHash on lemma shingles)
b2 cap any single proper noun (first name) at 2 percent of the pool. It runs
   after parsing because it uses the PROPN tags, which are more reliable than the
   capitalization heuristic used in b.

c, d and e are the slow steps: rows are processed in chunks (permissive rows
first, then the others in a seeded random order) until `pool_target` rows have
survived. All permissive rows are always processed, so the licence report is
exact for the permissive pool.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .lang.base import Language, split_core
from .licensing import is_permissive
from .utils import KVCache, chunked, ensure_java_on_path, rng_for, stable_hash

log = logging.getLogger("lapsus.filters")

FILTER_ORDER = ["a_normalize_length", "b_lexical", "c_langid", "d_languagetool", "e_parse", "f_lm",
                "g_dedup_exact", "g_dedup_normalized", "g_dedup_near", "b2_name_cap"]

RE_DIGIT = re.compile(r"\d")
RE_URL = re.compile(r"(https?://|www\.|\.(com|org|net|it|uk)\b|@|#)", re.I)
RE_QUOTES = re.compile(r"[\"“”„«»‹›‘`´]")
RE_BRACKETS = re.compile(r"[()\[\]{}<>]")
RE_SYMBOLS = re.compile("[/\\\\|*%&+=~^_$" + "".join(chr(c) for c in (0x20AC, 0xA3, 0xA7, 0xB0, 0x2026, 0x2013, 0x2014, 0x2012, 0x2015, 0xB7, 0x2022)) + "]")
RE_MULTI_SENT = re.compile(r"[.!?]+\s+\S")
RE_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF️‍]"
)
LT_IGNORE_CATEGORIES = {"STYLE", "CASING", "PUNCTUATION", "TYPOGRAPHY", "REDUNDANCY", "PLAIN_ENGLISH",
                        "WIKIPEDIA", "GENDER_NEUTRALITY", "MISC_STYLE"}
LT_IGNORE_ISSUES = {"style", "typographical", "whitespace", "register"}
LT_IGNORE_RULES = {"UPPERCASE_SENTENCE_START", "EN_QUOTES", "WHITESPACE_RULE", "PUNCTUATION_PARAGRAPH_END",
                   "COMMA_PARENTHESIS_WHITESPACE", "DOUBLE_PUNCTUATION", "ITALIAN_PUNCTUATION_PARAGRAPH_END"}


# --------------------------------------------------------------------- a + b
def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"(?<=\w)[’ʼ](?=\w|\s|$)", "'", t)  # typographic apostrophe in words
    t = re.sub(r"\s+", " ", t).strip()
    return t


def filter_a(text: str, lang: Language, fcfg: dict) -> tuple[str | None, str]:
    t = normalize_text(text)
    if not t:
        return None, "empty"
    if RE_MULTI_SENT.search(t):
        return None, "more than one sentence"
    if re.search(r"[.!?]{2,}$", t) and not t.endswith("?!"):
        return None, "ellipsis or repeated end punctuation"
    n = len(lang.tokenize(t))
    if n < fcfg["min_tokens"] or n > fcfg["max_tokens"]:
        return None, "length"
    return t, ""


def filter_b(text: str, lang: Language, fcfg: dict, known_names: set[str]) -> str:
    if RE_DIGIT.search(text):
        return "digit"
    if RE_URL.search(text):
        return "url"
    if RE_EMOJI.search(text):
        return "emoji"
    if RE_QUOTES.search(text):
        return "quote"
    if RE_BRACKETS.search(text):
        return "bracket"
    if RE_SYMBOLS.search(text):
        return "symbol"
    tokens = lang.tokenize(text)
    for tok in tokens:
        core = split_core(tok)[0]
        if not core or not re.search(r"\w", core):
            return "punctuation-only token"
        if tok.startswith("'") or tok.startswith("-") or core.endswith("-"):
            return "quote"
        letters = [c for c in core if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            return "all-caps"
        if not all(c.isalpha() or c in "'-" for c in core):
            return "non-letter character"
    if lang.proper_noun_count_heuristic(tokens, known_names) > fcfg["max_proper_nouns"]:
        return "proper nouns"
    return ""


def known_names_from(texts: list[str], lang: Language) -> set[str]:
    """Capitalized words seen mid-sentence (likely names) to catch sentence-initial names."""
    c: Counter[str] = Counter()
    for t in texts:
        toks = lang.tokenize(t)
        for tok in toks[1:]:
            core = re.sub(r"'s$", "", split_core(tok)[0])
            if core[:1].isupper():
                c[core] += 1
    return {w for w, n in c.items() if n >= 3}


# ------------------------------------------------------------------------- c
class LangID:
    def __init__(self, lang: Language):
        from lingua import Language as LL, LanguageDetectorBuilder

        names = ["ENGLISH", "ITALIAN", "FRENCH", "SPANISH", "GERMAN", "PORTUGUESE", "DUTCH", "LATIN",
                 "ROMANIAN", "CATALAN", "ESPERANTO"]
        if lang.lingua_name not in names:
            names.append(lang.lingua_name)
        self.target = getattr(LL, lang.lingua_name)
        self.det = LanguageDetectorBuilder.from_languages(*[getattr(LL, n) for n in names]).build()

    def matches(self, texts: list[str]) -> list[bool]:
        res = self.det.detect_languages_in_parallel_of(texts)
        return [r == self.target for r in res]


# ------------------------------------------------------------------------- d
def lt_match_is_error(m: list[str]) -> bool:
    rule, cat, issue = m
    if rule in LT_IGNORE_RULES or cat in LT_IGNORE_CATEGORIES or (issue or "") in LT_IGNORE_ISSUES:
        return False
    return True


class LTChecker:
    """LanguageTool on a local server, cached on disk.

    language_tool_python is used only to download the pinned LanguageTool release. The
    server is started here with its output sent to a log file: the library pipes stderr
    without draining it, which blocks the server after a few thousand requests.
    """

    def __init__(self, lang: Language, cfg: dict):
        self.lang = lang
        self.version = str(cfg["languagetool"]["version"])
        self.threads = int(cfg["languagetool"].get("threads", 4))
        self.cache = KVCache(Path(cfg["paths"]["cache"]) / f"languagetool_{lang.code}.sqlite")
        self.log_path = Path(cfg["paths"]["cache"]) / "languagetool_server.log"
        self._proc = None
        self._url: str | None = None

    def _server(self) -> str:
        if self._url:
            return self._url
        import socket
        import subprocess
        import time

        import requests
        from language_tool_python.download_lt import download_lt
        from language_tool_python.utils import get_language_tool_download_path

        ensure_java_on_path()
        download_lt(self.version)
        jar = Path(get_language_tool_download_path()) / f"LanguageTool-{self.version}" / "languagetool-server.jar"
        if not jar.exists():
            raise FileNotFoundError(f"LanguageTool server jar not found: {jar}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "ab")
        self._proc = subprocess.Popen(
            ["java", "-cp", str(jar), "org.languagetool.server.HTTPServer", "--port", str(port),
             "--allow-origin", "*"], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}/v2/"
        for _ in range(240):
            try:
                if requests.get(url + "languages", timeout=5).ok:
                    self._url = url
                    return url
            except Exception:
                pass
            if self._proc.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError(f"LanguageTool server did not start (see {self.log_path})")

    def matches(self, texts: list[str], code: str) -> list[list[list[str]]]:
        keys = [stable_hash("lt", self.version, code, t) for t in texts]
        cached = self.cache.get_many(keys)
        todo = [(t, k) for t, k in zip(texts, keys) if k not in cached]
        if todo:
            import threading
            import time

            import requests

            url = self._server() + "check"
            local = threading.local()

            def run(tk):
                t, k = tk
                if not hasattr(local, "s"):
                    local.s = requests.Session()
                for attempt in range(5):
                    try:
                        r = local.s.post(url, data={"language": code, "text": t}, timeout=120)
                        r.raise_for_status()
                        ms = r.json()["matches"]
                        return k, [[m["rule"]["id"], m["rule"]["category"]["id"], m["rule"].get("issueType", "") or ""]
                                   for m in ms]
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(2 * (attempt + 1))
                return k, []

            new = {}
            with ThreadPoolExecutor(self.threads) as ex:
                for k, ms in ex.map(run, todo):
                    new[k] = ms
                    if len(new) >= 2000:
                        self.cache.set_many(new)
                        cached.update(new)
                        new = {}
            self.cache.set_many(new)
            cached.update(new)
        return [cached[k] for k in keys]

    def passes(self, texts: list[str]) -> list[bool]:
        ok = [False] * len(texts)
        remaining = list(range(len(texts)))
        for code in self.lang.lt_codes:
            if not remaining:
                break
            res = self.matches([texts[i] for i in remaining], code)
            nxt = []
            for i, ms in zip(remaining, res):
                if any(lt_match_is_error(m) for m in ms):
                    nxt.append(i)
                else:
                    ok[i] = True
            remaining = nxt
        return ok

    def flags(self, texts: list[str]) -> list[bool]:
        """Diagnostic for corrupted rows: flagged in the primary variant."""
        res = self.matches(texts, self.lang.lt_codes[0])
        return [any(lt_match_is_error(m) for m in ms) for ms in res]

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc, self._url = None, None


# ---------------------------------------------------------------------- f, g
def length_bucket(n_tokens: int, edges: list[int]) -> int:
    for b in range(len(edges) - 1):
        if edges[b] <= n_tokens < edges[b + 1]:
            return b
    return len(edges) - 2


def lm_filter(rows: list[dict], fcfg: dict) -> tuple[list[dict], list[dict], dict]:
    """Drop the lowest `lm_drop_fraction` mean log-prob per length bucket."""
    edges = fcfg["length_buckets"]
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[length_bucket(len(r["tokens"]), edges)].append(r)
    keep, drop, thresholds = [], [], {}
    for b, rs in sorted(by_bucket.items()):
        vals = sorted(r["lm"] for r in rs)
        k = int(len(vals) * fcfg["lm_drop_fraction"])
        thr = vals[k] if k < len(vals) and k > 0 else float("-inf")
        thresholds[f"{edges[b]}-{edges[b + 1] - 1}"] = thr
        for r in rs:
            (keep if r["lm"] >= thr else drop).append(r)
    return keep, drop, thresholds


def normalized_key(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s']", "", t)
    return re.sub(r"\s+", " ", t).strip()


def lemma_shingles(row: dict) -> set[str]:
    lem = [(t["lemma"] or "").lower() for t in row["parse"]["info"]]
    sh = set(lem)
    sh.update(f"{a} {b}" for a, b in zip(lem, lem[1:]))
    return sh


def dedup(rows: list[dict], fcfg: dict, seed: int, tag: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Exact, normalized and MinHash near-duplicate removal. Permissive rows win ties."""
    from datasketch import MinHash, MinHashLSH

    order = sorted(rows, key=lambda r: (0 if is_permissive(r["license"]) else 1,
                                        stable_hash(seed, tag, r["source"], r["source_id"])))
    dropped: dict[str, list[dict]] = {"g_dedup_exact": [], "g_dedup_normalized": [], "g_dedup_near": []}
    seen_exact, seen_norm = set(), set()
    stage1 = []
    for r in order:
        if r["text"] in seen_exact:
            dropped["g_dedup_exact"].append(r)
            continue
        seen_exact.add(r["text"])
        nk = normalized_key(r["text"])
        if nk in seen_norm:
            dropped["g_dedup_normalized"].append(r)
            continue
        seen_norm.add(nk)
        stage1.append(r)
    lsh = MinHashLSH(threshold=fcfg["near_dup_threshold"], num_perm=fcfg["minhash_num_perm"])
    keep = []
    for i, r in enumerate(stage1):
        m = MinHash(num_perm=fcfg["minhash_num_perm"], seed=1)
        for s in lemma_shingles(r):
            m.update(s.encode("utf-8"))
        if lsh.query(m):
            dropped["g_dedup_near"].append(r)
            continue
        lsh.insert(str(i), m)
        keep.append(r)
    return keep, dropped


def name_cap(rows: list[dict], lang: Language, share: float, seed: int, tag: str) -> tuple[list[dict], list[dict]]:
    cap = max(1, int(share * len(rows)))
    counts: Counter[str] = Counter()
    keep, drop = [], []
    for r in sorted(rows, key=lambda r: stable_hash(seed, tag, "cap", r["source"], r["source_id"])):
        names = lang.proper_nouns(r["parse"])
        if any(counts[n] >= cap for n in names):
            drop.append(r)
            continue
        counts.update(names)
        keep.append(r)
    return keep, drop


# ------------------------------------------------------------------ pipeline
class FilterStats:
    def __init__(self):
        self.rejected: Counter[str] = Counter()
        self.reasons: dict[str, Counter[str]] = defaultdict(Counter)
        self.per_license: dict[str, Counter[str]] = defaultdict(Counter)

    def reject(self, filt: str, row: dict, reason: str = "") -> None:
        self.rejected[filt] += 1
        if reason:
            self.reasons[filt][reason] += 1

    def as_dict(self) -> dict:
        return {
            "rejected": {k: self.rejected.get(k, 0) for k in FILTER_ORDER},
            "reasons": {k: dict(v.most_common(15)) for k, v in self.reasons.items()},
            "per_license": {k: dict(v) for k, v in self.per_license.items()},
        }


def run_filters(rows: list[dict], lang: Language, cfg: dict, tools: dict[str, Any], tag: str,
                pool_target: int | None, use_lt: bool = True, use_parse: bool = True,
                use_lm: bool = True, progress: Callable[[str], None] | None = None) -> tuple[list[dict], dict]:
    """Run the filter pipeline on rows of one language and role. Returns (pool, report)."""
    fcfg = cfg["filters"]
    seed = cfg["seed"]
    st = FilterStats()
    for r in rows:
        st.per_license[r["license"]]["raw"] += 1
    # ---- a, b (cheap, on everything)
    stage = []
    for r in rows:
        t, why = filter_a(r["text"], lang, fcfg)
        if t is None:
            st.reject("a_normalize_length", r, why)
            continue
        r = {**r, "text": t}
        st.per_license[r["license"]]["a"] += 1
        stage.append(r)
    names = known_names_from([r["text"] for r in stage], lang)
    after_b = []
    for r in stage:
        why = filter_b(r["text"], lang, fcfg, names)
        if why:
            st.reject("b_lexical", r, why)
            continue
        st.per_license[r["license"]]["b"] += 1
        after_b.append(r)
    # ---- c, d, e (expensive, chunked, permissive first)
    perm = [r for r in after_b if is_permissive(r["license"])]
    rest = [r for r in after_b if not is_permissive(r["license"])]
    rng = rng_for(seed, tag, "order")
    rng.shuffle(perm)
    rng.shuffle(rest)
    survivors: list[dict] = []
    stopped_early = False
    queue = [("perm", c) for c in chunked(perm, fcfg["chunk_size"])] + [("rest", c) for c in chunked(rest, fcfg["chunk_size"])]
    for group, chunk in queue:
        if group == "rest" and pool_target is not None and len(survivors) >= pool_target:
            stopped_early = True
            break
        for r in chunk:
            st.per_license[r["license"]]["processed"] += 1
        ok = tools["langid"].matches([r["text"] for r in chunk])
        chunk2 = []
        for r, good in zip(chunk, ok):
            if good:
                chunk2.append(r)
                st.per_license[r["license"]]["c"] += 1
            else:
                st.reject("c_langid", r)
        if use_lt:
            ok = tools["lt"].passes([r["text"] for r in chunk2])
            chunk3 = []
            for r, good in zip(chunk2, ok):
                if good:
                    chunk3.append(r)
                    st.per_license[r["license"]]["d"] += 1
                else:
                    st.reject("d_languagetool", r)
        else:
            chunk3 = chunk2
        if use_parse:
            parses = tools["parser"].parse([r["text"] for r in chunk3])
            for r, p in zip(chunk3, parses):
                if not p["ok"]:
                    st.reject("e_parse", r, p["reason"].split(":")[0])
                    continue
                if sum(1 for t in p["info"] if t["upos"] == "PROPN") > fcfg["max_proper_nouns"]:
                    st.reject("e_parse", r, "proper nouns (parse)")
                    continue
                r["parse"] = p
                r["tokens"] = p["tokens"]
                st.per_license[r["license"]]["e"] += 1
                survivors.append(r)
        else:
            for r in chunk3:
                r["tokens"] = lang.tokenize(r["text"])
                survivors.append(r)
        if progress:
            progress(f"[{tag}] processed chunk: {len(survivors)} survivors so far")
    # ---- f
    thresholds = {}
    if use_lm and survivors:
        scores = tools["lm"].score([r["text"] for r in survivors])
        for r, s in zip(survivors, scores):
            r["lm"] = s[0] / max(1, s[1])
        survivors, dropped, thresholds = lm_filter(survivors, fcfg)
        for r in dropped:
            st.reject("f_lm", r)
    for r in survivors:
        st.per_license[r["license"]]["f"] += 1
    # ---- g
    if use_parse:
        survivors, dropped_g = dedup(survivors, fcfg, seed, tag)
        for k, rs in dropped_g.items():
            for r in rs:
                st.reject(k, r)
        survivors, dropped_cap = name_cap(survivors, lang, fcfg["name_cap"], seed, tag)
        for r in dropped_cap:
            st.reject("b2_name_cap", r)
    for r in survivors:
        st.per_license[r["license"]]["final"] += 1
    report = st.as_dict()
    report["lm_thresholds"] = thresholds
    report["stopped_early"] = stopped_early
    report["permissive_pool"] = sum(1 for r in survivors if is_permissive(r["license"]))
    report["total_pool"] = len(survivors)
    return survivors, report


def filter_natural(rows: list[dict], lang: Language, cfg: dict, tools: dict[str, Any]) -> tuple[list[dict], dict]:
    """Natural-error pairs: a, b and c on both the erroneous text and its correction."""
    fcfg = cfg["filters"]
    st = FilterStats()
    stage = []
    for r in rows:
        src, why1 = filter_a(r["text"], lang, fcfg)
        tgt, why2 = filter_a(r["correction"], lang, fcfg)
        if src is None or tgt is None:
            st.reject("a_normalize_length", r, why1 or why2)
            continue
        stage.append({**r, "text": src, "correction": tgt})
    names = known_names_from([r["correction"] for r in stage], lang)
    stage2 = []
    for r in stage:
        why = filter_b(r["text"], lang, fcfg, names) or filter_b(r["correction"], lang, fcfg, names)
        if why:
            st.reject("b_lexical", r, why)
            continue
        stage2.append(r)
    ok1 = tools["langid"].matches([r["text"] for r in stage2]) if stage2 else []
    ok2 = tools["langid"].matches([r["correction"] for r in stage2]) if stage2 else []
    out = []
    for r, a, b in zip(stage2, ok1, ok2):
        if a and b:
            out.append(r)
        else:
            st.reject("c_langid", r)
    return out, st.as_dict()
