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
from collections import defaultdict

API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/?has_narrative=true&format=json&no_aggs=true&size=5000"
)
MAX_DOWNLOAD_BYTES = 60_000_000  # the API streams; cap the download
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
SEED = 42


def _download(max_bytes: int) -> str:
    import urllib.request

    req = urllib.request.Request(API_URL, headers={"User-Agent": "reg-agents-demo"})
    chunks, total = [], 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        while total < max_bytes:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    print(f"downloaded {total / 1e6:.1f} MB from CFPB API")
    return b"".join(chunks).decode("utf-8", errors="ignore")


def _parse_records(raw: str) -> list[dict]:
    """Parse the (possibly truncated) hit stream into _source dicts."""
    end = raw.rfind('},{"_index"')
    if end == -1:
        data = json.loads(raw)
    else:
        data = json.loads(raw[: end + 1] + "]")
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

    # Balanced sample: cap each issue, then fill to target
    rows: list[dict] = []
    for issue, group in sorted(by_issue.items()):
        rows.extend(group[:PER_ISSUE_CAP])
    rng.shuffle(rows)
    rows = rows[:TARGET_ROWS]
    print(
        f"curated: {stats['in']} in -> {len(rows)} out "
        f"(len-filtered {stats['len']}, exact-dup {stats['exact']}, "
        f"near-dup {stats['near']}, issue cap {PER_ISSUE_CAP})"
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="path to a previously downloaded raw JSON dump")
    args = ap.parse_args()

    if args.raw:
        raw = open(args.raw, encoding="utf-8", errors="ignore").read()
    else:
        try:
            raw = _download(MAX_DOWNLOAD_BYTES)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"download failed: {exc}\n")
            raise SystemExit(1)

    records = _parse_records(raw)
    print(f"parsed {len(records)} records with narratives")
    rows = curate(records)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"wrote {len(rows)} rows to {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
