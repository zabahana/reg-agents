#!/usr/bin/env python3
"""Dual-judge agreement study: NIM + OpenAI vs the stage-1 logistic gate.

Both LLM providers are FIRST-CLASS judges (OpenAI is not fallback logic):
each independently answers the same binary question the production
logistic-regression gate answers — is this complaint regulatory? — on the
held-out test fold. The study documents, per judge and pairwise:

  - agreement / disagreement counts with the logistic-regression gate
  - inter-judge agreement (NIM vs OpenAI) and Cohen's kappa
  - agreement of every party with the weak labels (the noisy reference)
  - the disputed set: rows where BOTH judges disagree with the gate
    (candidate gate errors or weak-label errors — adjudication queue)

Artifacts (docs/complaint_model/):
  03_judge_agreement.md · judge_agreement.json · figures/judge_agreement.png

Run:
  python scripts/judge_agreement_study.py             # full test fold (380)
  python scripts/judge_agreement_study.py --limit 50  # smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reg_agents.common import complaints as C  # noqa: E402
from reg_agents.common import llm  # noqa: E402
from reg_agents.config import get_settings  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "complaint_model")
FIG_DIR = os.path.join(OUT_DIR, "figures")


# The hosted NIM endpoint (integrate.api.nvidia.com) rate-limits aggressively;
# keep its concurrency low and retry 429s with backoff. Abstention = the row
# still failed after retries — never replaced by a heuristic verdict.
PROVIDER_WORKERS = {"nim": 2, "openai": 8}


def judge_rows(texts, provider, workers=8):
    """Run one judge over all rows concurrently with rate-limit retries."""
    import random

    def one(text):
        for attempt in range(6):
            try:
                return C.llm_judge_regulatory(text, provider)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                retryable = "429" in msg or "rate" in msg.lower() or "503" in msg
                if attempt == 5 or not retryable:
                    return {"provider": provider, "error": msg[:200]}
                time.sleep(min(2 ** attempt, 20) + random.random())

    workers = min(workers, PROVIDER_WORKERS.get(provider, workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, texts))


def pair_stats(a, b):
    """Agreement stats between two verdict vectors (with abstention masks)."""
    from sklearn.metrics import cohen_kappa_score

    mask = np.array([x is not None and y is not None for x, y in zip(a, b)])
    aa = np.array([bool(x) for x, m in zip(a, mask) if m])
    bb = np.array([bool(x) for x, m in zip(b, mask) if m])
    n = int(mask.sum())
    agree = int((aa == bb).sum())
    kappa = float(cohen_kappa_score(aa, bb)) if 0 < aa.sum() < n and 0 < bb.sum() < n else float("nan")
    return {"n": n, "agree": agree, "disagree": n - agree,
            "rate": round(agree / max(n, 1), 4),
            "kappa": round(kappa, 4) if kappa == kappa else None,
            # confusion cells: (first verdict, second verdict)
            "cells": {"both_regulatory": int((aa & bb).sum()),
                      "both_non_regulatory": int((~aa & ~bb).sum()),
                      "first_only_regulatory": int((aa & ~bb).sum()),
                      "second_only_regulatory": int((~aa & bb).sum())}}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="judge only the first N rows of the split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", choices=("test", "val", "train"), default="test",
                    help="which fold to judge (test=agreement study; val/train="
                         "RLAIF preference corpus that does not touch the test fold)")
    args = ap.parse_args()

    s = get_settings()
    judges = llm.available_judges()
    if set(judges) != {"nim", "openai"}:
        print(f"!! need both judges configured, found {judges}", file=sys.stderr)
        sys.exit(1)
    models = {"nim": s.nim_model, "openai": s.openai_model}

    df = C.load_complaints()
    s1 = C.train_stage1(df)
    x_tr, x_va, x_te, y_tr, y_va, y_te = C.split_stage1(df)
    fold = {
        "test": (x_te, y_te),
        "val": (x_va, y_va),
        "train": (x_tr, y_tr),
    }[args.split]
    x_fold, y_fold = fold
    if args.limit:
        x_fold, y_fold = x_fold.iloc[: args.limit], y_fold.iloc[: args.limit]
    texts = list(x_fold)

    champ = s1["models"][s1["champion"]]
    proba = champ.predict_proba(s1["vectorizer"].transform(texts))[:, 1]
    gate = [bool(p >= s1["threshold"]) for p in proba]
    weak = [bool(v) for v in y_fold]

    print(f"== dual-judge study: {len(texts)} {args.split} rows ==")
    print(f"   gate: {s1['champion']} @ {s1['threshold']} · "
          f"judges: nim={models['nim']}, openai={models['openai']}")
    t0 = time.time()
    judge_raw, verdicts, errors = {}, {}, {}
    for provider in ("nim", "openai"):
        rows = judge_rows(texts, provider, workers=args.workers)
        judge_raw[provider] = rows
        verdicts[provider] = [r.get("is_regulatory") for r in rows]
        errors[provider] = sum(1 for r in rows if "error" in r)
        n_ok = len(rows) - errors[provider]
        print(f"   {provider}: {n_ok}/{len(rows)} verdicts "
              f"({errors[provider]} abstentions) [{time.time() - t0:.0f}s]")

    pairs = {
        "nim_vs_gate": pair_stats(verdicts["nim"], gate),
        "openai_vs_gate": pair_stats(verdicts["openai"], gate),
        "nim_vs_openai": pair_stats(verdicts["nim"], verdicts["openai"]),
        "gate_vs_weak": pair_stats(gate, weak),
        "nim_vs_weak": pair_stats(verdicts["nim"], weak),
        "openai_vs_weak": pair_stats(verdicts["openai"], weak),
    }

    # Full per-row panel — input to the DPO / RLAIF preference pipeline.
    panel_rows = []
    for i, text in enumerate(texts):
        vn, vo = verdicts["nim"][i], verdicts["openai"][i]
        rn = judge_raw["nim"][i]
        ro = judge_raw["openai"][i]
        panel_rows.append({
            "idx": int(i),
            "split": args.split,
            "narrative": text,
            "gate": gate[i],
            "gate_proba": float(proba[i]),
            "weak": weak[i],
            "nim": vn,
            "openai": vo,
            "nim_reason": rn.get("reason", ""),
            "openai_reason": ro.get("reason", ""),
            "nim_confidence": rn.get("confidence"),
            "openai_confidence": ro.get("confidence"),
            "nim_error": rn.get("error"),
            "openai_error": ro.get("error"),
        })
    # test → agreement study artifact; val/train → RLAIF preference corpus
    rows_name = ("judge_agreement_rows.jsonl" if args.split == "test"
                 else f"dpo_panel_{args.split}_rows.jsonl")
    rows_path = os.path.join(OUT_DIR, rows_name)
    with open(rows_path, "w", encoding="utf-8") as fh:
        for row in panel_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"   wrote {rows_path} ({len(panel_rows)} rows)")

    # Disputed set: both judges returned a verdict and both disagree with gate.
    disputed = []
    for row in panel_rows:
        vn, vo = row["nim"], row["openai"]
        if vn is None or vo is None:
            continue
        if vn != row["gate"] and vo != row["gate"]:
            disputed.append({
                "gate": row["gate"], "nim": vn, "openai": vo, "weak": row["weak"],
                "judges_match_weak": vn == row["weak"] and vo == row["weak"],
                "narrative": row["narrative"][:220],
            })
    n_disp = len(disputed)
    n_judges_right = sum(1 for d in disputed if d["judges_match_weak"])

    for name, p in pairs.items():
        print(f"   {name}: {p['agree']}/{p['n']} agree "
              f"({p['rate']:.1%}, kappa={p['kappa']})")
    print(f"   disputed (both judges vs gate): {n_disp} rows, "
          f"judges side with weak label on {n_judges_right}")

    if args.split != "test":
        # RLAIF corpus only — do not overwrite the committed test-fold study.
        meta = {
            "split": args.split, "n_rows": len(texts),
            "gate": {"model": s1["champion"], "threshold": s1["threshold"]},
            "judges": {p: {"model": models[p], "abstentions": errors[p]}
                       for p in ("nim", "openai")},
            "pairs": pairs,
            "disputed": {"n": n_disp, "judges_match_weak": n_judges_right},
            "rows_path": rows_path,
        }
        meta_path = os.path.join(OUT_DIR, f"dpo_panel_{args.split}.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"== done in {time.time() - t0:.0f}s (RLAIF panel, split={args.split}) ==")
        print(f"   wrote {rows_path}, {meta_path}")
        return

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["NIM vs LR gate", "OpenAI vs LR gate", "NIM vs OpenAI"]
    keys = ["nim_vs_gate", "openai_vs_gate", "nim_vs_openai"]
    ag = [pairs[k]["agree"] for k in keys]
    dis = [pairs[k]["disagree"] for k in keys]
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, ag, 0.4, label="agree", color="#76b900")
    axes[0].bar(x + 0.2, dis, 0.4, label="disagree", color="#d62728")
    for i, (a, d) in enumerate(zip(ag, dis)):
        axes[0].text(i - 0.2, a, str(a), ha="center", va="bottom", fontsize=9)
        axes[0].text(i + 0.2, d, str(d), ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x, names, fontsize=9)
    axes[0].set_ylabel("test rows")
    axes[0].set_title("Verdict agreement (held-out test fold)")
    axes[0].legend()

    parties = ["LR gate", "NIM judge", "OpenAI judge"]
    rates = [pairs["gate_vs_weak"]["rate"], pairs["nim_vs_weak"]["rate"],
             pairs["openai_vs_weak"]["rate"]]
    bars = axes[1].bar(parties, rates, color=["#1f77b4", "#76b900", "#10a37f"])
    for b, r in zip(bars, rates):
        axes[1].text(b.get_x() + b.get_width() / 2, r, f"{r:.1%}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("agreement with weak label")
    axes[1].set_title("Each party vs the weak reference")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "judge_agreement.png"), dpi=150)
    plt.close(fig)

    # ---- JSON ---------------------------------------------------------------
    results = {
        "n_test_rows": len(texts),
        "gate": {"model": s1["champion"], "threshold": s1["threshold"]},
        "judges": {p: {"model": models[p], "abstentions": errors[p]}
                   for p in ("nim", "openai")},
        "pairs": pairs,
        "disputed": {"n": n_disp, "judges_match_weak": n_judges_right,
                     "rows": disputed[:25]},
    }
    with open(os.path.join(OUT_DIR, "judge_agreement.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ---- markdown -----------------------------------------------------------
    def pair_row(label, key):
        p = pairs[key]
        return [label, p["n"], p["agree"], p["disagree"],
                f"{p['rate']:.1%}", p["kappa"] if p["kappa"] is not None else "—"]

    pair_rows = [
        pair_row("**NIM judge vs LR gate**", "nim_vs_gate"),
        pair_row("**OpenAI judge vs LR gate**", "openai_vs_gate"),
        pair_row("NIM judge vs OpenAI judge", "nim_vs_openai"),
        pair_row("LR gate vs weak label", "gate_vs_weak"),
        pair_row("NIM judge vs weak label", "nim_vs_weak"),
        pair_row("OpenAI judge vs weak label", "openai_vs_weak"),
    ]
    cell_rows = []
    for label, key, first, second in [
            ("NIM vs LR gate", "nim_vs_gate", "judge", "gate"),
            ("OpenAI vs LR gate", "openai_vs_gate", "judge", "gate"),
            ("NIM vs OpenAI", "nim_vs_openai", "NIM", "OpenAI")]:
        c = pairs[key]["cells"]
        cell_rows.append([label, c["both_regulatory"], c["both_non_regulatory"],
                          c["first_only_regulatory"], c["second_only_regulatory"]])

    md = f"""# Dual-judge agreement study — NIM + OpenAI vs the stage-1 logistic gate

*Generated by `scripts/judge_agreement_study.py` — regenerate after any
retrain. Committed run: **{len(texts)} held-out test rows**, gate =
`{s1['champion']}` at its validation-tuned cut-off {s1['threshold']}.*

Both LLM providers act as **independent, first-class judges** — OpenAI is no
longer fallback logic. Each judge answers the same binary question as the
production gate (is this complaint regulatory?) from the complaint text
alone, with a strict-JSON verdict. API failures are recorded as abstentions,
never silently replaced by a keyword heuristic.

| party | role | model |
|---|---|---|
| LR gate | production stage-1 classifier | `{s1['champion']}` (TF-IDF, cut-off {s1['threshold']}) |
| NIM judge | NVIDIA NIM inference (OpenAI-compatible API) | `{models['nim']}` |
| OpenAI judge | OpenAI API | `{models['openai']}` |

## 1 · Agreement / disagreement counts

Abstentions — NIM: {errors['nim']}, OpenAI: {errors['openai']} (excluded pairwise).

{md_table(["pair", "n", "agree", "disagree", "rate", "Cohen's kappa"], pair_rows)}

![judge agreement](figures/judge_agreement.png)

## 2 · Where they disagree (verdict cells)

{md_table(["pair", "both say regulatory", "both say non-regulatory",
           "first only regulatory", "second only regulatory"], cell_rows)}

## 3 · The disputed set — both judges vs the gate

On **{n_disp}** rows both judges return the same verdict AND it contradicts
the LR gate. On **{n_judges_right}/{n_disp}** of those, the judges side with
the weak label — i.e., likely gate errors; the remainder are candidate
weak-label errors. Either way this set is the natural **human-adjudication
queue**: it is where model, judges, and weak supervision cannot all be right.

## 4 · Reading the numbers

- Judge-vs-gate agreement bounds how much an LLM second opinion would change
  gate decisions in production; the disputed set sizes the review queue.
- Inter-judge agreement (NIM vs OpenAI, kappa above) measures judge
  reliability independent of the gate — low kappa would mean the judges are
  not a stable reference and neither can arbitrate the gate.
- No party is ground truth: weak labels are regex/taxonomy-derived (see the
  MDD leakage audit), the gate is trained on those labels, and LLM judges
  have their own biases. Agreement statistics here are triangulation, not
  accuracy claims — the adjudicated golden set remains the standing fix.
"""
    with open(os.path.join(OUT_DIR, "03_judge_agreement.md"), "w",
              encoding="utf-8") as fh:
        fh.write(md)

    print(f"== done in {time.time() - t0:.0f}s ==")
    print(f"   wrote {OUT_DIR}/03_judge_agreement.md, judge_agreement.json, "
          f"figures/judge_agreement.png")


if __name__ == "__main__":
    main()
