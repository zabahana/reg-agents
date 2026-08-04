#!/usr/bin/env python3
"""DPO / RLAIF preference training from the dual-judge panel.

Builds preference pairs where NIM + OpenAI consensus is *chosen* and the
opposite regulatory label is *rejected*, then optimizes a stage-1 policy
with the Bradley-Terry / DPO objective:

    L = -log σ(β · (log πθ(y_w|x) - log πθ(y_l|x)))

Two methods:

  classifier  (default, CPU-friendly) — DistilBERT binary policy; log π(y|x)
              from the class logit. Demonstrates DPO math on the production
              gate's task without a generative LM.
  generative  — LoRA + TRL DPOTrainer on a small instruct model (requires
              `peft` and `trl`; use on GPU / Brev for a serious run).

Artifacts (docs/complaint_model/):
  dpo_preferences.jsonl · 04_dpo_rlaif.md · dpo_results.json
  figures/dpo_comparison.png · artifacts/dpo_policy/ (classifier weights)

Run:
  python scripts/train_dpo_from_judges.py                  # classifier DPO
  python scripts/train_dpo_from_judges.py --method generative --max-steps 100
  python scripts/judge_agreement_study.py                  # (re)build rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reg_agents.common import complaints as C  # noqa: E402
from reg_agents.common import dpo_preferences as P  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "complaint_model")
FIG_DIR = os.path.join(OUT_DIR, "figures")
ART_DIR = os.path.join(OUT_DIR, "artifacts", "dpo_policy")
# RLAIF panels must come from val/train — never the test-fold agreement study
# (that would leak labels into the policy we evaluate on test).
PANEL_CANDIDATES = [
    os.path.join(OUT_DIR, "dpo_panel_val_rows.jsonl"),
    os.path.join(OUT_DIR, "dpo_panel_train_rows.jsonl"),
]
PAIRS_PATH = os.path.join(OUT_DIR, "dpo_preferences.jsonl")
SEED = C.SEED


def _hide_triton_path():
    """Avoid local triton/ model-repo shadowing the torch triton package."""
    saved = list(sys.path)
    sys.path = [p for p in sys.path
                if os.path.abspath(p or os.getcwd()) != ROOT]
    return saved


def collect_pairs(weak_bootstrap: int) -> list:
    """RLAIF from val/train judge panel + weak contrastive from the train fold.

    The committed test-fold agreement study is intentionally *not* used as
    training signal — those rows are the evaluation holdout.
    """
    pairs = []
    panel_path = next((p for p in PANEL_CANDIDATES if os.path.exists(p)), None)
    if panel_path:
        panel = P.load_panel_rows(panel_path)
        # belt-and-suspenders: drop any row tagged as test
        panel = [r for r in panel if r.get("split", "val") != "test"]
        pairs = P.build_rlaif_pairs(panel)
        print(f"   RLAIF from {panel_path}: {len(pairs)} pairs "
              f"({sum(1 for p in pairs if p['high_value'])} high-value/disputed)")
    else:
        print("   no val/train judge panel yet — RLAIF pairs = 0 "
              "(run: python scripts/judge_agreement_study.py --split val)")

    if weak_bootstrap != 0:
        df = C.load_complaints()
        x_tr, _x_va, _x_te, y_tr, _y_va, _y_te = C.split_stage1(df)
        limit = None if weak_bootstrap < 0 else weak_bootstrap
        boot = P.build_weak_contrastive(list(x_tr), list(y_tr.astype(bool)),
                                        limit=limit)
        pairs.extend(boot)
        print(f"   + weak contrastive from train fold: {len(boot)} pairs")
    return pairs


def train_classifier_dpo(pairs, epochs=3, beta=0.1, batch_size=8, lr=2e-5,
                         max_len=256):
    """DistilBERT policy on RLAIF preferred labels.

    For binary pairs where rejected = ¬chosen, the reference-free DPO /
    Bradley-Terry loss reduces to logistic CE on the preferred label
    (log π(y_w) - log π(y_l) = ±logit). We therefore train class-weighted CE
    on ``chosen_label`` — same optimum, stable under 87% regulatory skew —
    and keep β / pairwise framing in the docs for the generative DPO path.
    High-value disputed pairs are upsampled 3× so the judge-vs-gate signal
    is not drowned by the weak-label bootstrap.
    """
    saved = _hide_triton_path()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    finally:
        sys.path = saved

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print(f"   device: {dev}")

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)

    # Upsample disputed (high-value) pairs 3×.
    train_prefs = list(pairs)
    hv = [p for p in pairs if p.get("high_value")]
    train_prefs.extend(hv * 2)

    class PrefDS(Dataset):
        def __init__(self, prefs):
            self.enc = tok([p["narrative"] for p in prefs], truncation=True,
                           padding="max_length", max_length=max_len,
                           return_tensors="pt")
            self.y = torch.tensor([int(p["chosen_label"]) for p in prefs])

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    ds = PrefDS(train_prefs)
    y_np = ds.y.numpy()
    n_pos, n_neg = int((y_np == 1).sum()), int((y_np == 0).sum())
    # Sample inversely to class frequency so minority preferred labels are seen.
    class_w = {0: len(y_np) / (2 * max(n_neg, 1)),
               1: len(y_np) / (2 * max(n_pos, 1))}
    sample_w = [class_w[int(y)] for y in y_np]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                    replacement=True)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    w = torch.tensor([class_w[0], class_w[1]], dtype=torch.float).to(dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    history = []
    for ep in range(epochs):
        tot, n = 0.0, 0
        for batch in dl:
            labels = batch.pop("labels").to(dev)
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**batch).logits
            # Equivalent DPO margin when rejected = ¬chosen: CE(y_chosen).
            # Explicit BT form (for logging parity with generative DPO):
            logp = torch.log_softmax(logits, dim=-1)
            yw = labels
            yl = 1 - labels
            bt = -torch.nn.functional.logsigmoid(
                beta * (logp.gather(1, yw.view(-1, 1)).squeeze(1)
                        - logp.gather(1, yl.view(-1, 1)).squeeze(1))
            ).mean()
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(labels)
            n += len(labels)
        mean = tot / max(n, 1)
        history.append({"epoch": ep + 1, "ce_loss": round(mean, 4),
                        "bt_loss_last": round(float(bt.detach()), 4)})
        print(f"   epoch {ep + 1}/{epochs} ce_loss={mean:.4f} "
              f"bt_loss≈{float(bt.detach()):.4f}", flush=True)

    @torch.no_grad()
    def predict_proba(texts):
        model.eval()
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(list(texts[i:i + 32]), truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
            out.append(probs.cpu().numpy())
        return np.concatenate(out)

    os.makedirs(ART_DIR, exist_ok=True)
    model.save_pretrained(ART_DIR)
    tok.save_pretrained(ART_DIR)
    return predict_proba, history, {
        "base": name, "beta": beta, "device": dev, "epochs": epochs,
        "method": "classifier_rlaif_dpo",
        "note": "binary opposite-class DPO ≡ class-weighted CE on preferred label",
        "n_train_rows": len(train_prefs), "n_pos": n_pos, "n_neg": n_neg,
    }


def train_generative_dpo(pairs, max_steps=100, beta=0.1,
                         model_name="HuggingFaceTB/SmolLM2-360M-Instruct"):
    """LoRA + TRL DPOTrainer on a small instruct model."""
    try:
        from peft import LoraConfig, TaskType
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("!! generative method needs peft + trl: "
              "pip install peft trl", file=sys.stderr)
        sys.exit(1)

    saved = _hide_triton_path()
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    finally:
        sys.path = saved

    if torch.cuda.is_available():
        dev_map = "auto"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dev_map = None
        dtype = torch.float32
        print("   warning: generative DPO on CPU is slow; prefer --method classifier "
              "or a Brev GPU")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=dev_map)
    if dev_map is None:
        model = model.to("cpu")

    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    args = DPOConfig(
        output_dir=os.path.join(ART_DIR, "generative"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        max_steps=max_steps,
        beta=beta,
        logging_steps=10,
        save_steps=max_steps,
        remove_unused_columns=False,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(ART_DIR, "generative"))
    # Generative eval is left to a separate generation pass; return a stub.
    return None, [{"max_steps": max_steps}], {
        "base": model_name, "beta": beta, "method": "generative_dpo_lora",
        "max_steps": max_steps,
    }


def evaluate_policy(predict_proba, pairs_summary):
    """Compare DPO policy vs LR gate on the held-out test fold."""
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 average_precision_score)

    df = C.load_complaints()
    s1 = C.train_stage1(df)
    x_tr, x_va, x_te, y_tr, y_va, y_te = C.split_stage1(df)
    texts = list(x_te)
    y = np.asarray(y_te).astype(int)

    champ = s1["models"][s1["champion"]]
    gate_p = champ.predict_proba(s1["vectorizer"].transform(texts))[:, 1]
    thr = s1["threshold"]
    gate_pred = (gate_p >= thr).astype(int)

    dpo_p = predict_proba(texts)
    # Validation-style cut-off on the DPO scores using the val fold.
    dpo_p_va = predict_proba(list(x_va))
    dpo_thr = C.optimal_threshold(y_va, dpo_p_va)
    dpo_pred = (dpo_p >= dpo_thr).astype(int)

    src = np.array([C.label_source(df.loc[i, "issue"], df.loc[i, "narrative"])
                    for i in x_te.index])
    meta = src == "metadata"

    def pack(name, proba, pred, thr_):
        out = {
            "name": name,
            "threshold": round(float(thr_), 3),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "agree_with_gate": round(float((pred == gate_pred).mean()), 4),
        }
        if 0 < y[meta].sum() < meta.sum():
            out["leakage_free_roc"] = round(
                float(roc_auc_score(y[meta], proba[meta])), 4)
            out["leakage_free_n"] = int(meta.sum())
        return out

    lr = pack("logistic_regression_gate", gate_p, gate_pred, thr)
    dpo = pack("dpo_policy", dpo_p, dpo_pred, dpo_thr)
    return {"gate": lr, "dpo": dpo, "pairs": pairs_summary}


def write_figure(eval_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    metrics = ["roc_auc", "pr_auc", "f1", "leakage_free_roc"]
    labels = ["ROC-AUC", "PR-AUC", "F1", "leakage-free ROC"]
    g = [eval_res["gate"].get(m, 0) for m in metrics]
    d = [eval_res["dpo"].get(m, 0) for m in metrics]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, g, 0.4, label="LR gate", color="#1f77b4")
    ax.bar(x + 0.2, d, 0.4, label="DPO policy (RLAIF)", color="#76b900")
    for i, (a, b) in enumerate(zip(g, d)):
        ax.text(i - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_title("Stage-1 gate vs DPO policy (held-out test)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dpo_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_markdown(eval_res, train_meta, history, elapsed):
    g, d = eval_res["gate"], eval_res["dpo"]
    ps = eval_res["pairs"]
    md = f"""# DPO / RLAIF from the dual-judge panel

*Generated by `scripts/train_dpo_from_judges.py` in {elapsed:.0f}s.*

Preference data is built from the **NIM + OpenAI judge panel**
(`scripts/judge_agreement_study.py`). When both first-class judges agree,
that consensus is the *chosen* completion and the opposite regulatory label
is *rejected*. Rows where the consensus also contradicts the logistic-
regression gate are the **high-value disputed set** — the same adjudication
queue documented in [`03_judge_agreement.md`](03_judge_agreement.md).

## Preference corpus

| | |
|---|---|
| total pairs | {ps['n_pairs']} |
| by source | {ps['by_source']} |
| high-value (judges vs gate) | {ps['high_value']} |
| chosen regulatory rate | {ps['chosen_regulatory_rate']:.1%} |

Pairs: [`dpo_preferences.jsonl`](dpo_preferences.jsonl).

## Training

| | |
|---|---|
| method | `{train_meta['method']}` |
| base model | `{train_meta['base']}` |
| β (DPO temperature) | {train_meta['beta']} |
| note | {train_meta.get('note', '—')} |
| loss history | {history} |

For binary pairs where *rejected* = ¬*chosen*, reference-free DPO reduces to
logistic CE on the preferred label; we train **class-weighted CE** (same
optimum, stable under the regulatory skew) and keep the generative
`--method generative` path for full TRL `DPOTrainer` + LoRA.

## Held-out test comparison

![dpo comparison](figures/dpo_comparison.png)

| party | thr | ROC-AUC | PR-AUC | F1 | leakage-free ROC | agree w/ LR gate |
|---|---|---|---|---|---|---|
| LR gate | {g['threshold']} | {g['roc_auc']} | {g['pr_auc']} | {g['f1']} | {g.get('leakage_free_roc', '—')} | 1.0 |
| DPO policy | {d['threshold']} | {d['roc_auc']} | {d['pr_auc']} | {d['f1']} | {d.get('leakage_free_roc', '—')} | {d['agree_with_gate']} |

## How to read this

- This is **RLAIF → DPO**: AI judges (NIM + OpenAI) supply the preference
  signal; no human ranking labels were required to start the loop.
- The LR gate remains the production millisecond CPU path. The DPO policy is
  the post-training challenger — promote it only after the golden-set
  adjudication of the disputed rows confirms the judges were right.
- Re-run `judge_agreement_study.py` whenever the panel or dataset changes;
  it now persists `judge_agreement_rows.jsonl` for this pipeline.
"""
    path = os.path.join(OUT_DIR, "04_dpo_rlaif.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=("classifier", "generative"),
                    default="classifier")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--weak-bootstrap", type=int, default=-1,
                    help="N weak-label contrastive pairs from the train fold "
                         "(-1 = all train rows, 0 = disable)")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="generative DPO only")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    print("== build preference pairs ==")
    t0 = time.time()
    pairs = collect_pairs(weak_bootstrap=args.weak_bootstrap)
    if len(pairs) < 10:
        print(f"!! too few pairs ({len(pairs)}); need the judge panel or a larger "
              f"--weak-bootstrap", file=sys.stderr)
        sys.exit(1)
    P.write_pairs(pairs, PAIRS_PATH)
    summary = P.summarize_pairs(pairs)
    print(f"   wrote {PAIRS_PATH}: {summary}")

    print(f"== train ({args.method}) ==")
    if args.method == "classifier":
        predict, history, meta = train_classifier_dpo(
            pairs, epochs=args.epochs, beta=args.beta, batch_size=args.batch_size)
    else:
        predict, history, meta = train_generative_dpo(
            pairs, max_steps=args.max_steps, beta=args.beta)
        if predict is None:
            # Generative path saves adapters; skip discriminative eval.
            results = {"pairs": summary, "train": meta, "history": history,
                       "note": "generative adapters saved; run a generation eval separately"}
            with open(os.path.join(OUT_DIR, "dpo_results.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)
            print(f"== done in {time.time() - t0:.0f}s (generative adapters only) ==")
            return

    print("== evaluate vs LR gate ==")
    eval_res = evaluate_policy(predict, summary)
    write_figure(eval_res)
    md_path = write_markdown(eval_res, meta, history, time.time() - t0)
    results = {"pairs": summary, "train": meta, "history": history,
               "eval": eval_res}
    with open(os.path.join(OUT_DIR, "dpo_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"   gate ROC={eval_res['gate']['roc_auc']}  "
          f"dpo ROC={eval_res['dpo']['roc_auc']}  "
          f"leakage-free dpo={eval_res['dpo'].get('leakage_free_roc')}")
    print(f"== done in {time.time() - t0:.0f}s ==")
    print(f"   wrote {md_path}, dpo_results.json, figures/dpo_comparison.png")


if __name__ == "__main__":
    main()
