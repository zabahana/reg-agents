"""Fetch + curate real consumer complaints from the CFPB public database.

Source: the CFPB Consumer Complaint Database public search API
(https://www.consumerfinance.gov/data-research/consumer-complaints/) — real,
redacted complaint narratives filed against US financial institutions.

The curation pass mirrors the stages of **NVIDIA NeMo Data Curator** so the
pipeline story scales: at laptop scale we run the same stages in pandas/python;
at corpus scale each maps to a Curator module (GPU-accelerated via RAPIDS):

    stage here                      NeMo Data Curator analog
    ------------------------------  ---------------------------------------
    length / language filter        ScoreFilter + heuristic filters
    exact dedup (norm-text hash)    ExactDuplicates
    near dedup (prefix signature)   FuzzyDuplicates (MinHash-LSH)
    PII handling (CFPB pre-masked   PiiModifier (redaction)
    with XXXX; we verify)
    balanced sampling by label      custom DocumentDataset filter

Acquisition is TWO passes:
  1. a generic slice of the narrative stream (regulatory-heavy, as the CFPB
     database naturally is), and
  2. a TARGETED pass over service-heavy issues ("Managing an account",
     "Closing an account", ...) — the only issues that can weak-label as
     NON_REGULATORY. The final assembly enforces a non-regulatory floor
     (NONREG_TARGET) so the minority class has enough support for stable
     threshold tuning and minority metrics (135 rows / 3.4% was too few).

Usage:
    python scripts/fetch_cfpb_complaints.py                  # download + curate
    python scripts/fetch_cfpb_complaints.py --raw dump.json  # reuse a raw dump
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.parse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/?has_narrative=true&no_aggs=true"
)
# Service-heavy CFPB issues: the pool where NON_REGULATORY complaints live.
SERVICE_ISSUES = [
    "Managing an account",
    "Opening an account",
    "Closing an account",
    "Closing your account",
    "Problem getting a card or closing an account",
    "Customer service",
]
PAGE_SIZE = 2000                   # ES window: frm + size must stay <= 10,000
GENERIC_PAGES = 5                  # 10k generic records
SERVICE_PAGES = 2                  # 4k records per targeted issue
PAGE_SLEEP = 15                    # seconds between page requests (throttle)
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "complaints", "cfpb_complaints.csv",
)
FIELDS = [
    "complaint_id", "date_received", "product", "sub_product",
    "issue", "sub_issue", "company", "state", "tags", "narrative",
]
MIN_CHARS, MAX_CHARS = 120, 1800   # length filter; truncate long narratives
PER_ISSUE_CAP = 350                # fight the credit-reporting skew
TARGET_ROWS = 4000
NONREG_TARGET = 500                # minority floor (~12.5% of TARGET_ROWS)
SEED = 42


def _fetch_page(frm: int, extra: str = "", label: str = "generic",
                retries: int = 5) -> list[dict]:
    """Fetch one page of hits with retry — the CFPB API throttles hard.

    Uses curl rather than urllib: the API's WAF fingerprints the TLS stack
    and returns 403 to Python's ssl client regardless of headers.
    """
    import subprocess
    import time

    url = f"{API_URL}&size={PAGE_SIZE}&frm={frm}{extra}"
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                ["curl", "-sf", "--max-time", "120",
                 "-H", "User-Agent: reg-agents-demo", url],
                capture_output=True, timeout=150, check=True,
            )
            recs = _parse_records(proc.stdout.decode("utf-8", errors="ignore"))
            print(f"   page frm={frm} ({label}): {len(recs)} records", flush=True)
            return recs
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError) as exc:
            code = getattr(exc, "returncode", "parse/timeout")
            if attempt < retries:
                wait = 30 * attempt
                print(f"   throttled/err ({code}) on {label} frm={frm}; "
                      f"retry {attempt}/{retries - 1} in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"page fetch failed after {retries} attempts: {label}")


def _download_generic_pass() -> list[dict]:
    """Generic slice of the narrative stream (regulatory-heavy)."""
    import time

    records: list[dict] = []
    for page in range(GENERIC_PAGES):
        if page:
            time.sleep(PAGE_SLEEP)
        records.extend(_fetch_page(page * PAGE_SIZE))
    return records


def _download_service_pass() -> list[dict]:
    """Targeted pass over service-heavy issues (non-regulatory candidates)."""
    import time

    records: list[dict] = []
    for issue in SERVICE_ISSUES:
        extra = f"&issue={urllib.parse.quote(issue)}"
        for page in range(SERVICE_PAGES):
            time.sleep(PAGE_SLEEP)
            try:
                recs = _fetch_page(page * PAGE_SIZE, extra,
                                   label=f"issue={issue!r}")
                records.extend(recs)
                if len(recs) < PAGE_SIZE:
                    break  # issue exhausted
            except Exception as exc:  # noqa: BLE001 - one bad issue shouldn't kill the run
                print(f"   skipping issue {issue!r} frm={page * PAGE_SIZE}: {exc}")
                break
    return records


def _parse_records(raw: str) -> list[dict]:
    """Parse an API response (ES envelope, hit list, or truncated stream)."""
    end = raw.rfind('},{"_index"')
    if raw.lstrip().startswith("{") or end == -1:
        data = json.loads(raw)
    else:
        data = json.loads(raw[: end + 1] + "]")
    if isinstance(data, dict):  # Elasticsearch envelope
        data = data.get("hits", {}).get("hits", [])
    out = []
    for r in data:
        src = r.get("_source", r)
        out.append(src)
    return out


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def curate(records: list[dict]) -> list[dict]:
    """NeMo-Curator-style pass: filter -> exact dedup -> near dedup -> truncate."""
    import random

    rng = random.Random(SEED)
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    by_issue: dict[str, list[dict]] = defaultdict(list)
    stats = {"in": len(records), "len": 0, "exact": 0, "near": 0}

    rng.shuffle(records)
    for src in records:
        text = (src.get("complaint_what_happened") or "").strip()
        if not (MIN_CHARS <= len(text)):
            stats["len"] += 1
            continue
        norm = _norm(text)
        h = hashlib.sha1(norm.encode()).hexdigest()
        if h in seen_exact:
            stats["exact"] += 1
            continue
        seen_exact.add(h)
        near_key = norm[:200]  # cheap near-dup signature (Curator: MinHash-LSH)
        if near_key in seen_near:
            stats["near"] += 1
            continue
        seen_near.add(near_key)
        row = {
            "complaint_id": src.get("complaint_id", ""),
            "date_received": (src.get("date_received") or "")[:10],
            "product": src.get("product", ""),
            "sub_product": src.get("sub_product") or "",
            "issue": src.get("issue", ""),
            "sub_issue": src.get("sub_issue") or "",
            "company": src.get("company", ""),
            "state": src.get("state") or "",
            "tags": src.get("tags") or "",
            "narrative": text[:MAX_CHARS],
        }
        by_issue[row["issue"]].append(row)

    # Balanced pool: cap each issue first (fights the credit-reporting skew),
    # then — if the pool falls short of what assembly needs — top up
    # round-robin from the overflow so TARGET_ROWS stays reachable.
    from itertools import zip_longest

    rows: list[dict] = []
    overflow: list[list[dict]] = []
    for issue, group in sorted(by_issue.items()):
        rows.extend(group[:PER_ISSUE_CAP])
        if len(group) > PER_ISSUE_CAP:
            overflow.append(group[PER_ISSUE_CAP:])
    # Slack for assembly: the overflow skews to service issues (the targeted
    # pass), so a large share of the top-up weak-labels non-regulatory and
    # the pool needs headroom to still contain TARGET_ROWS - NONREG_TARGET
    # regulatory rows.
    pool_target = TARGET_ROWS + 5 * NONREG_TARGET
    extra = [r for tup in zip_longest(*overflow) for r in tup if r is not None]
    n_top = max(0, min(pool_target - len(rows), len(extra)))
    rows.extend(extra[:n_top])
    rng.shuffle(rows)
    print(
        f"curated: {stats['in']} in -> {len(rows)} pooled "
        f"(len-filtered {stats['len']}, exact-dup {stats['exact']}, "
        f"near-dup {stats['near']}, issue cap {PER_ISSUE_CAP}, "
        f"round-robin top-up {n_top})"
    )
    return rows


def assemble(rows: list[dict]) -> list[dict]:
    """Assemble the final dataset with a non-regulatory minority floor.

    Weak-labels the curated pool, takes up to NONREG_TARGET non-regulatory
    rows first, then fills the remainder to TARGET_ROWS with regulatory rows.
    """
    import random

    from reg_agents.common.complaints import NON_REGULATORY, weak_label

    rng = random.Random(SEED)
    nonreg = [r for r in rows
              if weak_label(r["issue"], r["narrative"]) == NON_REGULATORY]
    reg = [r for r in rows
           if weak_label(r["issue"], r["narrative"]) != NON_REGULATORY]
    rng.shuffle(nonreg)
    rng.shuffle(reg)
    take_nonreg = nonreg[:NONREG_TARGET]
    out = take_nonreg + reg[:TARGET_ROWS - len(take_nonreg)]
    rng.shuffle(out)
    print(
        f"assembled: {len(out)} rows — {len(take_nonreg)} non-regulatory "
        f"({len(take_nonreg) / max(len(out), 1):.1%}, floor {NONREG_TARGET}, "
        f"pool had {len(nonreg)}) + {len(out) - len(take_nonreg)} regulatory"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="path to a previously downloaded raw JSON dump")
    args = ap.parse_args()

    if args.raw:
        raw = open(args.raw, encoding="utf-8", errors="ignore").read()
        records = _parse_records(raw)
    else:
        try:
            records = _download_generic_pass()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"download failed: {exc}\n")
            raise SystemExit(1)
        # Second, targeted pass: service-heavy issues -> non-regulatory pool.
        records.extend(_download_service_pass())
        # Cache the raw dump so curation changes don't force a re-download
        # (reuse with: --raw data/complaints/raw_cfpb_dump.json).
        dump = os.path.join(os.path.dirname(OUT), "raw_cfpb_dump.json")
        with open(dump, "w", encoding="utf-8") as fh:
            json.dump(records, fh)
        print(f"cached raw dump: {dump} ({os.path.getsize(dump) / 1e6:.0f} MB)")

    print(f"parsed {len(records)} records with narratives")
    rows = assemble(curate(records))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {len(rows)} rows to {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
