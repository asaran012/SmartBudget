"""
pipeline/anomaly_detector.py
-----------------------------
Detect unusual transactions using per-category IQR statistics.
GPT is used only to write human-readable explanations — not to detect.

AI Assistant Usage:
    Asked ChatGPT: "Standard deviation vs IQR for anomaly detection on
    personal finance data?" It recommended IQR because spending data is
    skewed — one big purchase distorts the mean. Also suggested the
    separation of concerns: deterministic stats for detection, GPT only
    for plain-English explanation. That principle shaped the whole design.
"""

import os
import json
import uuid
import logging
from collections import defaultdict
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

MIN_SAMPLES        = 5      # min transactions per category before flagging
IQR_MULTIPLIER     = 4.0    # how far above Q3 counts as anomalous
LARGE_ONE_OFF_INR  = 3500   # flag single large charges from new merchants


RECURRING_KEYWORDS = [
    "rent", "landlord", "emi", "loan", "insurance", "electricity", "msedcl", "bescom", "broadband", "internet", "recharge"]

def _is_recurring(description: str) -> bool:
    desc = description.lower()
    return any(kw in desc for kw in RECURRING_KEYWORDS)


def detect_anomalies(transactions: list[dict]) -> list[dict]:
    """
    Returns a list of anomaly dicts with AI-generated explanations.
    """
    expenses = [t for t in transactions if t["amount"] < 0]
    if not expenses:
        logger.info("No expenses found — skipping anomaly detection")
        return []

    thresholds = compute_thresholds(expenses)
    
    flagged = flag_transactions(expenses, thresholds)
    logger.info(f"Flagged {len(flagged)} anomalous transactions in categories = {[t['category'] for t in flagged]}")

    if not flagged:
        return []

    logger.info("Sending flagged transactions to GPT for explanations...")
    results = _attach_explanations(flagged)
    logger.info(f"Anomaly detection complete — {len(results)} anomalies with explanations")
    return results


# ── detection (pure logic, no AI) ────────────────────────────────────────────

def compute_thresholds(expenses) -> dict[str, float]:
    """IQR threshold per spending category."""
    by_cat = defaultdict(list)
    for t in expenses:
        by_cat[t.get("category", "Other")].append(abs(t["amount"]))

    thresholds = {}
    for cat, amounts in by_cat.items():
        if len(amounts) >= MIN_SAMPLES:
            thresholds[cat] = _iqr_threshold(amounts)
    return thresholds


def _iqr_threshold(amounts: list[float]) -> float:
    sorted_vals = sorted(amounts)
    q3 = sorted_vals[int(len(sorted_vals) * 0.75)]
    return q3 * IQR_MULTIPLIER


def flag_transactions(expenses, thresholds) -> list[dict]:
    flagged = []
    merchant_freq: dict[str, int] = defaultdict(int)
    for t in expenses:
        merchant_freq[t["description"].lower()[:25]] += 1

    seen_ids = set()

    for t in expenses:
        cat    = t.get("category", "Other")
        amount = abs(t["amount"])

        # Rule 1 — statistically high for its category
        if cat in thresholds and amount > thresholds[cat]:
            flagged.append({
                **t,
                "reason":    "above_category_threshold",
                "threshold": round(thresholds[cat], 2),
                "ratio":     round(amount / thresholds[cat], 1),
            })
            seen_ids.add(t["id"])

        # Rule 2 — large charge from a merchant seen only once (skip known recurring bills)
        key = t["description"].lower()[:25]
        if (t["id"] not in seen_ids
                and merchant_freq[key] == 1
                and amount > LARGE_ONE_OFF_INR
                and not _is_recurring(t["description"])):
            flagged.append({
                **t,
                "reason":    "large_one_off",
                "threshold": None,
                "ratio":     None,
            })

    return flagged


# ── explanation (GPT) ─────────────────────────────────────────────────────────

def _attach_explanations(flagged: list[dict]) -> list[dict]:
    """Ask GPT to write one plain-English sentence per anomaly."""
    summaries = [
        {
            "description": t["description"],
            "amount":      abs(t["amount"]),
            "date":        t["date"],
            "category":    t.get("category"),
            "reason":      t["reason"],
            "ratio":       t.get("ratio"),
        }
        for t in flagged[:10]   # cap to avoid token overflow
    ]

    prompt = (
        "These bank transactions were flagged as unusual. All amounts are in Indian Rupees (₹).\n"
        f"{json.dumps(summaries, indent=2)}\n\n"
        "Write one short sentence (plain English) explaining why each is unusual. "
        "Use ₹ for amounts.\n"
        "Return ONLY a JSON array of strings in the same order. No markdown."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        explanations = json.loads(raw)
    except Exception:
        explanations = ["Flagged as unusual transaction."] * len(flagged)

    results = []
    for i, t in enumerate(flagged):
        results.append({
            "id":             str(uuid.uuid4()),
            "transaction_id": t["id"],
            "date":           t["date"],
            "description":    t["description"],
            "amount":         abs(t["amount"]),
            "category":       t.get("category"),
            "reason":         t["reason"],
            "ratio":          t.get("ratio"),
            "explanation":    explanations[i] if i < len(explanations) else "Flagged as unusual.",
        })

    return results