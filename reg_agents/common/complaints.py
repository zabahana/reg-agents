"""Complaint → regulation classification (the third model in the system).

Two-stage architecture, mirroring a production complaint-intelligence platform:

  Stage 1 (binary, fast):    is this complaint REGULATORY or not?
                             TF-IDF + {logistic regression, XGBoost} bake-off,
                             champion selected on PR-AUC. (BERT/NeMo is the GPU
                             upgrade path — same interface.)
  Stage 2 (multi-class):     if regulatory, WHICH of the 24 regulation
                             categories? RAG over the policy/regulation corpus
                             + LLM reasoning with few-shot examples, returning
                             a label, rationale, and a cited excerpt. Falls
                             back to a keyword scorer when no LLM is available.

Data: real, redacted narratives from the CFPB Consumer Complaint Database
(data/complaints/cfpb_complaints.csv, see scripts/fetch_cfpb_complaints.py).
Ground-truth labels are *weak labels* derived from the CFPB product/issue
taxonomy plus narrative keyword rules — the standard bootstrap when a bank
first builds this model, and a documented limitation in the validation report.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

_DATA_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "complaints", "cfpb_complaints.csv",
)
SEED = 42
NON_REGULATORY = "NON_REGULATORY"


# --------------------------------------------------------------------------- #
# The 24-category regulation taxonomy
# --------------------------------------------------------------------------- #
@dataclass
class Regulation:
    label: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)


REGULATIONS: Dict[str, Regulation] = {r.label: r for r in [
    Regulation("FCRA_ACCURACY", "FCRA — Accuracy of Reported Information",
               "Inaccurate tradelines, balances, statuses or identity data on a consumer report (FCRA 623/611).",
               ["credit report", "inaccurate", "wrong information", "not mine", "reporting incorrectly"]),
    Regulation("FCRA_INVESTIGATION", "FCRA — Reinvestigation of Disputes",
               "Failure to reasonably reinvestigate a disputed item within 30 days (FCRA 611).",
               ["dispute", "reinvestigat", "investigated", "30 days", "verified without"]),
    Regulation("FCRA_PERMISSIBLE_PURPOSE", "FCRA — Permissible Purpose / Improper Use",
               "Credit report pulled or used without a permissible purpose; unauthorized hard inquiries (FCRA 604).",
               ["hard inquiry", "did not authorize the inquiry", "pulled my credit", "without my permission", "permissible purpose"]),
    Regulation("FDCPA_DEBT_VALIDATION", "FDCPA — Debt Validation / Not Owed",
               "Collection of a debt not owed, already paid, or without validation notice (FDCPA 809).",
               ["debt is not mine", "not owed", "validation", "never received notice", "paid this debt"]),
    Regulation("FDCPA_COMMUNICATION", "FDCPA — Communication Tactics",
               "Harassing, excessive, or improper collector contact; calls after cease request (FDCPA 805/806).",
               ["keep calling", "harass", "called my work", "stop calling", "multiple times a day"]),
    Regulation("FDCPA_THREATS", "FDCPA — False Statements or Threats",
               "False representations, threatened suits or actions a collector cannot take (FDCPA 807).",
               ["threaten", "lawsuit", "sue me", "garnish", "arrest", "false statement"]),
    Regulation("REG_E_UNAUTHORIZED", "Reg E / EFTA — Unauthorized Transfers",
               "Unauthorized electronic transactions, fraud or scams on deposit/prepaid accounts (Reg E 1005.6).",
               ["unauthorized transaction", "scam", "fraudulent charge", "zelle", "stolen", "did not make"]),
    Regulation("REG_E_ERROR_RESOLUTION", "Reg E / EFTA — Error Resolution",
               "Errors in electronic transfers not investigated/credited within required timelines (Reg E 1005.11).",
               ["provisional credit", "error resolution", "transfer failed", "money not received", "wrong amount"]),
    Regulation("REG_Z_BILLING", "Reg Z / FCBA — Billing Error Disputes",
               "Credit card billing error disputes, unauthorized card charges, chargeback handling (Reg Z 1026.13).",
               ["billing error", "dispute the charge", "chargeback", "charge on my statement", "double charged"]),
    Regulation("REG_Z_DISCLOSURE", "Reg Z / TILA — Fees, Interest & Disclosures",
               "Undisclosed or miscalculated card/loan fees, APR or interest terms (Reg Z 1026.6/.7).",
               ["apr", "interest rate", "annual fee", "late fee", "promotional rate", "terms changed"]),
    Regulation("TILA_ORIGINATION", "TILA — Credit Origination & Underwriting Disclosures",
               "Problems applying for or being approved for credit; disclosure failures at origination.",
               ["applied for", "application was", "approval", "pre-approved", "denied the card"]),
    Regulation("ECOA_DISCRIMINATION", "ECOA / Reg B — Credit Discrimination",
               "Discrimination in any credit transaction on a prohibited basis (race, sex, age, etc.).",
               ["discriminat", "because of my race", "because of my age", "redlin", "protected class"]),
    Regulation("ECOA_ADVERSE_ACTION", "ECOA / Reg B — Adverse Action Notices",
               "Missing, late, or unexplained adverse-action / credit-denial notification (Reg B 1002.9).",
               ["adverse action", "no reason for denial", "denied without", "notice of denial"]),
    Regulation("RESPA_SERVICING", "RESPA — Mortgage Servicing & Escrow",
               "Mortgage payment application, escrow analysis, servicing transfers, QWR handling (Reg X).",
               ["escrow", "mortgage servicer", "payment was not applied", "servicing transfer", "qualified written request"]),
    Regulation("RESPA_LOSS_MITIGATION", "RESPA — Loss Mitigation & Foreclosure",
               "Loss-mitigation application handling, dual tracking, foreclosure process issues (Reg X 1024.41).",
               ["loan modification", "foreclosure", "loss mitigation", "forbearance", "short sale"]),
    Regulation("TISA_REG_DD", "TISA / Reg DD — Deposit Account Disclosures",
               "Deposit account fee and rate disclosures, overdraft fee practices on checking/savings.",
               ["overdraft fee", "maintenance fee", "account fees", "interest on savings", "fee disclosure"]),
    Regulation("REG_CC_FUNDS", "Reg CC — Funds Availability",
               "Deposit holds and delayed availability of deposited funds (Reg CC subpart B).",
               ["deposit hold", "funds availability", "check hold", "hold on my deposit", "funds on hold"]),
    Regulation("BSA_AML", "BSA/AML — Account Freezes & Closures",
               "Accounts frozen/closed for unexplained risk or suspicious-activity reasons; blocked funds.",
               ["account was frozen", "account was closed without", "closed my account", "under review", "released my funds"]),
    Regulation("GLBA_PRIVACY", "GLBA / Privacy — Information Sharing & Safeguards",
               "Improper sharing, sale, or safeguarding of nonpublic personal information; opt-out failures.",
               ["privacy", "shared my information", "sold my data", "data breach", "personal information"]),
    Regulation("UDAAP", "UDAAP — Unfair, Deceptive, or Abusive Acts",
               "Deceptive marketing, bait-and-switch, unfair practices not covered by a specific regulation.",
               ["deceptive", "misleading", "false advertis", "bait", "unfair", "trick"]),
    Regulation("SALES_PRACTICES", "Sales Practices — Unauthorized Accounts/Products",
               "Accounts, cards, or add-on products opened or enrolled without consumer consent.",
               ["opened without my", "never authorized the account", "enrolled me", "did not sign up", "without my consent"]),
    Regulation("SCRA_MLA", "SCRA / MLA — Servicemember Protections",
               "Servicemembers Civil Relief Act / Military Lending Act rate caps and protections.",
               ["servicemember", "military", "deployed", "scra", "active duty"]),
    Regulation("LOAN_SERVICING", "Loan Servicing (Auto/Student/Personal) — UDAAP & Servicing Rules",
               "Payment application, payoff, credit reporting and hardship handling on non-mortgage loans.",
               ["auto loan", "student loan", "payoff amount", "repossess", "deferment", "servicer"]),
    Regulation(NON_REGULATORY, "Non-Regulatory — General Service",
               "Customer-service friction without a specific regulatory nexus.",
               ["customer service", "rude", "long wait", "branch", "inconvenient"]),
]}

assert len(REGULATIONS) == 24, "taxonomy must have exactly 24 categories"


# Few-shot examples for the stage-2 LLM prompt (curated, short, distinct).
FEW_SHOTS: List[Tuple[str, str]] = [
    ("There is an account on my credit report that does not belong to me. I disputed it "
     "with the bureau and they said it was verified but never showed me any proof.",
     "FCRA_INVESTIGATION"),
    ("A collection agency keeps calling me at work five times a day about a debt I already "
     "paid two years ago, even after I asked them in writing to stop.",
     "FDCPA_COMMUNICATION"),
    ("Someone made three Zelle transfers from my checking account that I never authorized. "
     "The bank refuses to give me provisional credit while they investigate.",
     "REG_E_UNAUTHORIZED"),
    ("My mortgage servicer lost my loan modification paperwork twice and started foreclosure "
     "while my application was still under review.",
     "RESPA_LOSS_MITIGATION"),
    ("The bank opened a savings account and a credit card in my name that I never asked for. "
     "I only found out when I saw the fees.",
     "SALES_PRACTICES"),
    ("I was denied a personal loan and the bank never told me why — no letter, no explanation, "
     "nothing about which factors caused the denial.",
     "ECOA_ADVERSE_ACTION"),
    ("The card advertised 0% APR for 18 months but I was charged interest from month one, and "
     "the fee schedule was never disclosed when I signed up.",
     "REG_Z_DISCLOSURE"),
    ("The teller was rude and the branch lobby wait was over an hour. I want someone to "
     "apologize for how I was treated.",
     NON_REGULATORY),
    ("The bank suddenly froze my checking account and then closed it without any explanation, "
     "and they are still holding my money three weeks later saying it is under review.",
     "BSA_AML"),
    ("There is a charge on my credit card statement for a purchase I returned. I disputed the "
     "billing error in writing but the card company rebilled me without explanation.",
     "REG_Z_BILLING"),
    ("My checking account was charged four overdraft fees in one day and the fee schedule "
     "they gave me when I opened the account never disclosed this could happen.",
     "TISA_REG_DD"),
    ("I sent my wire transfer a week ago and the money never arrived. The bank has not "
     "investigated the transfer error or given me provisional credit.",
     "REG_E_ERROR_RESOLUTION"),
    ("My mortgage servicer misapplied my monthly payment to principal only and my escrow "
     "analysis doubled my payment with no explanation.",
     "RESPA_SERVICING"),
    ("I deposited a check on Monday and the bank put a ten-day hold on the funds without "
     "telling me when the money would be available.",
     "REG_CC_FUNDS"),
]


# --------------------------------------------------------------------------- #
# Weak labeling: CFPB product/issue taxonomy + narrative keyword overrides
# --------------------------------------------------------------------------- #
_ISSUE_MAP: List[Tuple[str, str]] = [
    ("incorrect information on your report", "FCRA_ACCURACY"),
    ("problem with a credit reporting company's investigation", "FCRA_INVESTIGATION"),
    ("problem with a company's investigation into an existing", "FCRA_INVESTIGATION"),
    ("improper use of your report", "FCRA_PERMISSIBLE_PURPOSE"),
    ("unable to get your credit report", "FCRA_ACCURACY"),
    ("attempts to collect debt not owed", "FDCPA_DEBT_VALIDATION"),
    ("written notification about debt", "FDCPA_DEBT_VALIDATION"),
    ("communication tactics", "FDCPA_COMMUNICATION"),
    ("took or threatened to take negative", "FDCPA_THREATS"),
    ("false statements or representation", "FDCPA_THREATS"),
    ("threatened to contact someone", "FDCPA_COMMUNICATION"),
    ("fraud or scam", "REG_E_UNAUTHORIZED"),
    ("unauthorized transactions", "REG_E_UNAUTHORIZED"),
    ("problem with a lender or other company charging your account", "REG_E_ERROR_RESOLUTION"),
    ("money was not available when promised", "REG_E_ERROR_RESOLUTION"),
    ("other transaction problem", "REG_E_ERROR_RESOLUTION"),
    ("problem with a purchase shown on your statement", "REG_Z_BILLING"),
    ("problem with a purchase or transfer", "REG_Z_BILLING"),
    ("fees or interest", "REG_Z_DISCLOSURE"),
    ("other features, terms, or problems", "REG_Z_DISCLOSURE"),
    ("advertising and marketing", "UDAAP"),
    ("getting a credit card", "TILA_ORIGINATION"),
    ("getting a loan or lease", "TILA_ORIGINATION"),
    ("applying for a mortgage", "TILA_ORIGINATION"),
    ("getting a line of credit", "TILA_ORIGINATION"),
    ("trouble during payment process", "RESPA_SERVICING"),
    ("struggling to pay mortgage", "RESPA_LOSS_MITIGATION"),
    ("closing on a mortgage", "RESPA_SERVICING"),
    ("managing the loan or lease", "LOAN_SERVICING"),
    ("dealing with your lender or servicer", "LOAN_SERVICING"),
    ("struggling to pay your loan", "LOAN_SERVICING"),
    ("problem when making payments", "LOAN_SERVICING"),
    ("struggling to repay your loan", "LOAN_SERVICING"),
    ("repossession", "LOAN_SERVICING"),
    ("deposits and withdrawals", "REG_CC_FUNDS"),
    ("funds availability", "REG_CC_FUNDS"),
]

_KEYWORD_OVERRIDES: List[Tuple[str, str]] = [
    (r"discriminat|because of my (race|age|gender|religion|national)", "ECOA_DISCRIMINATION"),
    (r"servicemember|active duty|\bscra\b|military lending", "SCRA_MLA"),
    (r"privacy|shared my (personal )?information|sold my (data|information)|data breach",
     "GLBA_PRIVACY"),
    (r"(opened|enrolled).{0,40}without my (permission|consent|knowledge|authorization)",
     "SALES_PRACTICES"),
    (r"(frozen|froze|closed).{0,30}(account|funds)|account.{0,30}(frozen|closed) without",
     "BSA_AML"),
    (r"adverse action|denied .{0,40}(no|without) (reason|explanation|notice)",
     "ECOA_ADVERSE_ACTION"),
]

_SERVICE_ISSUES = ("managing an account", "opening an account", "closing an account",
                   "closing your account", "problem getting a card", "customer service")


def weak_label(issue: str, narrative: str) -> str:
    """Map a CFPB complaint to one of the 24 categories (weak supervision)."""
    text = (narrative or "").lower()
    for pattern, label in _KEYWORD_OVERRIDES:
        if re.search(pattern, text):
            return label
    issue_l = (issue or "").lower()
    for prefix, label in _ISSUE_MAP:
        if issue_l.startswith(prefix):
            return label
    if any(issue_l.startswith(s) for s in _SERVICE_ISSUES):
        # Account-management complaints: regulatory only with a specific hook.
        # Priority matters: fraud/unauthorized beats fee mentions.
        if re.search(r"unauthorized|stolen|fraud|scam|did not make", text):
            return "REG_E_UNAUTHORIZED"
        if re.search(r"overdraft|fee", text):
            return "TISA_REG_DD"
        if re.search(r"hold|availability", text):
            return "REG_CC_FUNDS"
        return NON_REGULATORY
    return NON_REGULATORY if not issue_l else "UDAAP"


@lru_cache
def load_complaints(csv_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path or _DATA_CSV)
    df["narrative"] = df["narrative"].fillna("").astype(str)
    df["label"] = [weak_label(i, n) for i, n in zip(df["issue"], df["narrative"])]
    df["is_regulatory"] = (df["label"] != NON_REGULATORY).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Stage 1: binary REGULATORY / NON-REGULATORY bake-off
# --------------------------------------------------------------------------- #
def train_stage1(df: Optional[pd.DataFrame] = None, test_size: float = 0.25) -> Dict:
    """Train logistic + XGBoost on TF-IDF; return models, split and metrics."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    df = df if df is not None else load_complaints()
    x_tr, x_te, y_tr, y_te = train_test_split(
        df["narrative"], df["is_regulatory"], test_size=test_size,
        random_state=SEED, stratify=df["is_regulatory"],
    )
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    xt_tr, xt_te = vec.fit_transform(x_tr), vec.transform(x_te)

    candidates: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED),
    }
    try:
        import xgboost as xgb
        candidates["xgboost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
            tree_method="hist", eval_metric="aucpr", random_state=SEED)
    except Exception:  # noqa: BLE001 - xgboost optional (e.g. macOS w/o libomp)
        pass

    leaderboard, fitted, curves = [], {}, {}
    for name, model in candidates.items():
        model.fit(xt_tr, y_tr)
        proba = model.predict_proba(xt_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        fitted[name] = model
        curves[name] = {"y_true": y_te.to_numpy(), "y_score": proba}
        leaderboard.append({
            "model": name,
            "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "f1": round(float(f1_score(y_te, preds)), 4),
            "precision": round(float(precision_score(y_te, preds)), 4),
            "recall": round(float(recall_score(y_te, preds)), 4),
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        })
    leaderboard.sort(key=lambda r: (r["pr_auc"], r["roc_auc"]), reverse=True)
    champion = leaderboard[0]["model"]
    champ_preds = (curves[champion]["y_score"] >= 0.5).astype(int)
    cm = confusion_matrix(curves[champion]["y_true"], champ_preds).tolist()

    return {
        "vectorizer": vec,
        "models": fitted,
        "champion": champion,
        "leaderboard": leaderboard,
        "confusion_matrix": cm,
        "curves": curves,
        "dataset": {
            "n_rows": int(len(df)),
            "n_train": int(len(x_tr)),
            "n_test": int(len(x_te)),
            "regulatory_rate": round(float(df["is_regulatory"].mean()), 4),
            "source": "CFPB Consumer Complaint Database (public, narratives redacted)",
        },
    }


@lru_cache
def _stage1_cached():
    return train_stage1()


def classify_binary(text: str) -> Dict:
    """Stage 1: probability the complaint is regulatory."""
    s1 = _stage1_cached()
    xt = s1["vectorizer"].transform([text])
    prob = float(s1["models"][s1["champion"]].predict_proba(xt)[0, 1])
    return {
        "is_regulatory": prob >= 0.5,
        "probability": round(prob, 4),
        "model": s1["champion"],
    }


# --------------------------------------------------------------------------- #
# Stage 2: RAG + LLM multi-class regulation labeling (with citations)
# --------------------------------------------------------------------------- #
_STAGE2_SYS = (
    "You are a bank compliance analyst. Classify the consumer complaint into "
    "EXACTLY ONE of the regulation categories provided. Use the retrieved "
    "policy/regulation excerpts to ground your decision.\n"
    "Disambiguation rules:\n"
    "- Unauthorized/fraudulent transactions on DEPOSIT/debit/prepaid accounts "
    "-> REG_E_UNAUTHORIZED; disputed charges on CREDIT CARDS -> REG_Z_BILLING.\n"
    "- Fees on DEPOSIT accounts (overdraft, maintenance) -> TISA_REG_DD; fees/"
    "interest/APR on CREDIT products -> REG_Z_DISCLOSURE.\n"
    "- Bank freezes/closes an account for risk or 'review' reasons -> BSA_AML. "
    "SALES_PRACTICES only when a product was OPENED without consent.\n"
    "- Mortgage payment/escrow/servicing problems -> RESPA_SERVICING; "
    "modification/foreclosure -> RESPA_LOSS_MITIGATION. Non-mortgage loan "
    "servicing -> LOAN_SERVICING.\n"
    "- Credit-report content -> the FCRA_* categories; third-party debt "
    "collectors -> the FDCPA_* categories.\n"
    "Respond ONLY with a JSON object: {\"label\": <CATEGORY_CODE>, "
    "\"confidence\": <0..1>, \"rationale\": <1-2 sentences>, "
    "\"citation_source\": <source doc of the most relevant excerpt>}."
)


def _taxonomy_block() -> str:
    return "\n".join(f"- {r.label}: {r.description}" for r in REGULATIONS.values())


def _few_shot_block() -> str:
    return "\n\n".join(
        f"COMPLAINT: {t}\nANSWER: {{\"label\": \"{lbl}\"}}" for t, lbl in FEW_SHOTS
    )


def keyword_classify(text: str) -> Tuple[str, float]:
    """LLM-free fallback: keyword score over the taxonomy."""
    t = text.lower()
    scores = {
        label: sum(1 for kw in reg.keywords if kw in t)
        for label, reg in REGULATIONS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "UDAAP", 0.2
    conf = min(0.4 + 0.15 * scores[best], 0.9)
    return best, round(conf, 2)


def classify_regulation(text: str, retriever=None, use_llm: bool = True) -> Dict:
    """Stage 2: label the regulation category with a cited excerpt."""
    from reg_agents.common.corpus import RegulationRetriever

    retriever = retriever or _default_retriever()
    if retriever is None:
        retriever = RegulationRetriever()

    hits = retriever.search(text[:600], k=4)
    excerpts = [
        {
            "source": h.document.metadata.get("source", ""),
            "heading": h.document.metadata.get("heading", ""),
            "text": h.document.text[:600],
        }
        for h in hits
    ]

    result: Dict = {"mode": "fallback"}
    if use_llm:
        try:
            from reg_agents.common import llm

            excerpt_block = "\n\n".join(
                f"[{e['source']} — {e['heading']}]\n{e['text']}" for e in excerpts
            )
            user = (
                f"REGULATION CATEGORIES:\n{_taxonomy_block()}\n\n"
                f"EXAMPLES:\n{_few_shot_block()}\n\n"
                f"RETRIEVED POLICY EXCERPTS:\n{excerpt_block}\n\n"
                f"COMPLAINT:\n{text[:1800]}\n\nJSON ANSWER:"
            )
            raw = llm.system_user(_STAGE2_SYS, user, temperature=0.0, max_tokens=250)
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {}
            label = str(parsed.get("label", "")).strip().upper()
            if label in REGULATIONS:
                result = {
                    "mode": "rag_llm",
                    "label": label,
                    "confidence": float(parsed.get("confidence", 0.7)),
                    "rationale": str(parsed.get("rationale", "")).strip(),
                    "citation_source": str(parsed.get("citation_source", "")),
                }
        except Exception as exc:  # noqa: BLE001 - degrade to keyword fallback
            result = {"mode": "fallback", "llm_error": str(exc)[:200]}

    if result.get("mode") != "rag_llm":
        label, conf = keyword_classify(text)
        result.update({"label": label, "confidence": conf,
                       "rationale": "Keyword-based fallback classification (no LLM)."})

    reg = REGULATIONS[result["label"]]
    cite_ref = str(result.get("citation_source", "")).lower()
    cited = next((e for e in excerpts if e["source"].lower() in cite_ref),
                 excerpts[0] if excerpts else None)
    result.update({
        "regulation_name": reg.name,
        "regulation_description": reg.description,
        "citation": cited,
    })
    return result


_RETRIEVER = None


def _default_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        try:
            from reg_agents.common.corpus import RegulationRetriever

            _RETRIEVER = RegulationRetriever()
        except Exception:  # noqa: BLE001
            _RETRIEVER = None
    return _RETRIEVER


def classify_complaint(text: str, use_llm: bool = True) -> Dict:
    """Full two-stage pipeline for one complaint."""
    stage1 = classify_binary(text)
    out: Dict = {"stage1": stage1}
    if stage1["is_regulatory"]:
        out["stage2"] = classify_regulation(text, use_llm=use_llm)
    else:
        reg = REGULATIONS[NON_REGULATORY]
        out["stage2"] = {
            "mode": "stage1_gate", "label": NON_REGULATORY,
            "confidence": round(1 - stage1["probability"], 4),
            "rationale": "Stage-1 classifier gated this complaint as non-regulatory.",
            "regulation_name": reg.name,
            "regulation_description": reg.description,
            "citation": None,
        }
    return out


# --------------------------------------------------------------------------- #
# Stage-2 evaluation vs weak labels (stratified sample)
# --------------------------------------------------------------------------- #
FAMILY: Dict[str, str] = {
    "FCRA_ACCURACY": "FCRA", "FCRA_INVESTIGATION": "FCRA",
    "FCRA_PERMISSIBLE_PURPOSE": "FCRA",
    "FDCPA_DEBT_VALIDATION": "FDCPA", "FDCPA_COMMUNICATION": "FDCPA",
    "FDCPA_THREATS": "FDCPA",
    "REG_E_UNAUTHORIZED": "REG_E", "REG_E_ERROR_RESOLUTION": "REG_E",
    "REG_Z_BILLING": "REG_Z_TILA", "REG_Z_DISCLOSURE": "REG_Z_TILA",
    "TILA_ORIGINATION": "REG_Z_TILA",
    "ECOA_DISCRIMINATION": "ECOA", "ECOA_ADVERSE_ACTION": "ECOA",
    "RESPA_SERVICING": "RESPA", "RESPA_LOSS_MITIGATION": "RESPA",
    "TISA_REG_DD": "DEPOSITS", "REG_CC_FUNDS": "DEPOSITS",
    "BSA_AML": "BSA_AML", "GLBA_PRIVACY": "GLBA", "UDAAP": "UDAAP",
    "SALES_PRACTICES": "SALES_PRACTICES", "SCRA_MLA": "SCRA_MLA",
    "LOAN_SERVICING": "LOAN_SERVICING", NON_REGULATORY: "NON_REGULATORY",
}


def evaluate_stage2(n: int = 150, use_llm: bool = True,
                    df: Optional[pd.DataFrame] = None) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score

    df = df if df is not None else load_complaints()
    reg_df = df[df["label"] != NON_REGULATORY]
    per_class = max(2, n // reg_df["label"].nunique())
    sample = (reg_df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_class), random_state=SEED))
              .sample(frac=1.0, random_state=SEED))
    sample = sample.head(n)

    y_true, y_pred, modes = [], [], []
    for _, row in sample.iterrows():
        res = classify_regulation(row["narrative"], use_llm=use_llm)
        y_true.append(row["label"])
        y_pred.append(res["label"])
        modes.append(res["mode"])

    labels = sorted(set(y_true) | set(y_pred))
    per_label = []
    for lbl in labels:
        idx = [i for i, t in enumerate(y_true) if t == lbl]
        if not idx:
            continue
        correct = sum(1 for i in idx if y_pred[i] == lbl)
        per_label.append({"label": lbl, "support": len(idx),
                          "recall": round(correct / len(idx), 3)})
    fam_true = [FAMILY.get(t, t) for t in y_true]
    fam_pred = [FAMILY.get(p, p) for p in y_pred]
    return {
        "n": len(y_true),
        "mode": max(set(modes), key=modes.count) if modes else "none",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # Confusions concentrate WITHIN a regulation family (e.g. FCRA accuracy
        # vs reinvestigation) where weak labels are noisiest, so family-level
        # agreement is reported alongside exact-match.
        "family_accuracy": round(float(accuracy_score(fam_true, fam_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                            average="weighted", zero_division=0)), 4),
        "per_label": per_label,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def label_distribution(df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    df = df if df is not None else load_complaints()
    return df["label"].value_counts().to_dict()
