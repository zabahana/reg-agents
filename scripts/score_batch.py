#!/usr/bin/env python3
"""Batch-scoring trigger for the complaint pipeline (ingestion layer).

Feeds a dataset through the two-stage complaint model (stage-1 regulatory
gate -> stage-2 RAG + LLM regulation label) and writes a scored CSV with
complaint_id, complaint, score and LLM reasoning per row.

Default input is the reserved 5% scoring holdout — carved out of the CFPB
dataset BEFORE the 80/10/10 modeling split, so it is genuinely unseen data.

  # score the reserved holdout (200 rows)
  python scripts/score_batch.py

  # score an arbitrary CSV (needs a narrative/complaint/text column)
  python scripts/score_batch.py --input my_complaints.csv --output scored.csv

  # cheap smoke run: 20 rows, no LLM (keyword fallback for stage 2)
  python scripts/score_batch.py --limit 20 --no-llm
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from reg_agents.common import complaints as C  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "scoring",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="CSV to score (default: the reserved 5%% "
                                    "scoring holdout)")
    ap.add_argument("--output", help="output CSV path (default: "
                                     "data/scoring/scored_<timestamp>.csv)")
    ap.add_argument("--limit", type=int, help="score only the first N rows")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM (keyword fallback for stage 2)")
    args = ap.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        source = args.input
    else:
        df = C.scoring_holdout()
        source = f"reserved {C.HOLDOUT_FRAC:.0%} scoring holdout"
    if args.limit:
        df = df.head(args.limit)

    print(f"== scoring {len(df)} complaints from {source} "
          f"(llm={not args.no_llm}) ==")
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"   {done}/{total} scored ({time.time() - t0:.0f}s)")

    scored = C.score_batch(df, use_llm=not args.no_llm, progress=progress)

    out = args.output
    if not out:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR,
                           f"scored_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    scored.to_csv(out, index=False)

    n_reg = int(scored["is_regulatory"].sum())
    print(f"== done in {time.time() - t0:.0f}s ==")
    print(f"   {n_reg}/{len(scored)} regulatory · "
          f"top labels: {scored['label'].value_counts().head(5).to_dict()}")
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
