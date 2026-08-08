"""
pipeline/categorizer.py
------------------------
Two-pass categorization: keyword rules first, GPT only for unknowns.

AI Assistant Usage:
    Started with full AI categorization but switched back to rule-based
    first pass after observing GPT occasionally miscounting batch responses.
    Rules handle ~90% of transactions instantly and for free. GPT only
    runs on genuinely ambiguous merchants — reducing cost and improving
    reliability.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from database.db import save_labeled_transaction
from ml.ml_classifier import MLCategorizer

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing-api-key")
logger = logging.getLogger(__name__)
ML_MODEL_PATH = "ml/model.joblib"

CATEGORIES: dict[str, list[str]] = {
    "Rent":                 ["Rent"],
    "Income":              ["salary credit", "salary", "direct deposit", "stipend",
                            "refund", "cashback", "interest", "dividend"],
    "Dining":              ["swiggy", "zomato", "kfc", "mcdonalds", "mcdonald",
                            "dominos", "domino", "pizza hut", "burger king", "subway",
                            "starbucks", "chai point", "dunkin", "biryani", "chinese wok",
                            "punjabi tadka", "roll hub", "maggie cafe", "vada pav",
                            "pasta place", "olive bistro"],
    "Groceries":           ["zepto", "blinkit", "swiggy instamart", "bigbasket",
                            "big basket", "dmart", "d-mart", "nature's basket",
                            "reliance fresh", "more supermarket"],
    "Transport":           ["ola", "uber", "rapido", "metro", "irctc", "redbus",
                            "makemytrip", "indigo", "spicejet", "air india"],
    "Subscriptions":       ["netflix", "spotify", "hotstar", "amazon prime",
                            "disney", "youtube premium", "zee5", "sonyliv",
                            "apple music", "gaana", "jiocinema"],
    "Shopping":            ["amazon", "flipkart", "myntra", "ajio", "nykaa",
                            "decathlon", "meesho", "snapdeal", "tata cliq"],
    "Health & Fitness":    ["cult fit", "cultfit", "gym", "pharmacy", "medplus",
                            "apollo", "1mg", "netmeds", "practo", "clinikk"],
    "Utilities & Bills":   ["msedcl", "bescom", "tata power", "jio", "airtel",
                            "vi ", "vodafone", "bsnl", "paytm electricity",
                            "water bill", "electricity", "recharge"],
    "Entertainment":       ["pvr", "inox", "bookmyshow", "steam", "playstation",
                            "xbox", "concert", "event"],
    "Transfers & Payments":["phonepe upi", "gpay upi", "paytm upi", "upi",
                            "neft", "imps", "cred", "rent", "transfer"],
}

BATCH_SIZE = 30
unclassified_log: list[dict] = []


def _load_ml_model(model_path: str = ML_MODEL_PATH) -> Optional[MLCategorizer]:
    if not Path(model_path).exists():
        logger.warning("ML classifier model not found at %s; falling back to rules -> LLM", model_path)
        return None

    try:
        return MLCategorizer(model_path, threshold=0.85)
    except Exception as e:
        logger.warning("ML classifier failed to load from %s; falling back to rules -> LLM: %s", model_path, e)
        return None


ml_model = _load_ml_model()


def categorize_transactions(transactions: list[dict]) -> list[dict]:
    global unclassified_log
    unclassified_log = []

    result = []
    needs_ai = []

    for txn in transactions:
        cat = _keyword_match(txn["description"])
        if cat:
            result.append({**txn, "category": cat, "categorized_by": "rule"})
            continue

        cat, conf = _ml_classify(txn)
        if cat:
            result.append({**txn, "category": cat, "categorized_by": "ml"})
            logger.debug("ML classified '%s' as %s (%.2f)", txn["description"], cat, conf)
            continue

        result.append({**txn, "category": None, "categorized_by": None})
        needs_ai.append(len(result) - 1)

    logger.info(
        "Classified %s/%s transactions before GPT.",
        len(result) - len(needs_ai),
        len(transactions),
    )

    if needs_ai:
        logger.info(f"Sending {len(needs_ai)} unknown transactions to GPT...")
        descs = [result[i]["description"] for i in needs_ai]
        ai_results = _gpt_categorize_all(descs)

        for idx, (cat, from_llm) in zip(needs_ai, ai_results):
            result[idx]["category"] = cat
            result[idx]["categorized_by"] = "llm"
            channel = _infer_channel(result[idx])
            if from_llm:
                save_labeled_transaction(result[idx], cat, channel)
            if cat == "Other":
                unclassified_log.append({
                    "description": result[idx]["description"],
                    "date":        result[idx]["date"],
                    "amount":      result[idx]["amount"],
                })
                logger.warning(f"Unclassified: '{result[idx]['description']}'")

    for txn in result:
        if not txn["category"]:
            txn["category"] = "Other"

    if unclassified_log:
        logger.warning(f"{len(unclassified_log)} transaction(s) could not be classified → 'Other'")

    return result


def get_unclassified() -> list[dict]:
    return unclassified_log


# ── helpers ───────────────────────────────────────────────────────────────────

def _keyword_match(description: str) -> Optional[str]:
    desc_lower = description.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return None


def _ml_classify(txn: dict) -> tuple[Optional[str], float]:
    if ml_model is None:
        return None, 0.0

    channel = _infer_channel(txn)
    try:
        return ml_model.classify(
            txn["description"],
            abs(float(txn["amount"])),
            txn["date"],
            channel,
        )
    except Exception as e:
        logger.warning("ML classifier deferred after error for '%s': %s", txn["description"], e)
        return None, 0.0


def _infer_channel(txn: dict) -> str:
    channel = txn.get("channel")
    if channel:
        return str(channel).upper()

    desc = txn.get("description", "").upper()
    if "NEFT" in desc:
        return "NEFT"
    if "IMPS" in desc:
        return "IMPS"
    if "ACH" in desc:
        return "ACH"
    if "POS" in desc:
        return "POS"
    if "UPI" in desc or "@" in desc:
        return "UPI"
    return "UPI"


def _gpt_categorize_all(descriptions: list[str]) -> list[tuple[str, bool]]:
    all_categories = []
    total_batches  = (len(descriptions) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(descriptions), BATCH_SIZE):
        batch     = descriptions[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        logger.info(f"GPT batch {batch_num}/{total_batches} ({len(batch)} transactions)...")

        result, from_llm = _gpt_categorize_batch(batch, batch_num)
        all_categories.extend(zip(result, from_llm))

        other_count = result.count("Other")
        if other_count:
            logger.warning(f"Batch {batch_num}: {other_count}/{len(batch)} still unclassified after GPT")
        else:
            logger.info(f"Batch {batch_num}: all {len(batch)} classified")

    return all_categories


def _gpt_categorize_batch(descriptions: list[str], batch_num: int = 1) -> tuple[list[str], list[bool]]:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Batch %s: OPENAI_API_KEY is not set; marking GPT fallback rows as Other", batch_num)
        return ["Other"] * len(descriptions), [False] * len(descriptions)

    labels = [c for c in CATEGORIES.keys()] + ["Other"]
    prompt = f"""You are a bank transaction categorizer for an Indian user.
            Categorize each transaction into exactly one of: {', '.join(labels)}

            Transactions: {json.dumps(descriptions)}

            Return ONLY a JSON array of category strings in the same order. No markdown, no explanation."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        cats = json.loads(raw)

        if len(cats) != len(descriptions):
            logger.warning(
                f"Batch {batch_num}: GPT returned {len(cats)} for {len(descriptions)} — padding with 'Other'"
            )
            cats = cats[:len(descriptions)]
            cats += ["Other"] * (len(descriptions) - len(cats))

        validated = [c if c in labels else "Other" for c in cats]
        from_llm = [raw_cat in labels for raw_cat in cats]

        for desc, raw_cat, val in zip(descriptions, cats, validated):
            if val == "Other" and raw_cat not in labels:
                logger.warning(f"Batch {batch_num}: invalid category '{raw_cat}' for '{desc}' → 'Other'")

        return validated, from_llm

    except json.JSONDecodeError as e:
        logger.error(f"Batch {batch_num}: JSON parse failed — {e}")
        return ["Other"] * len(descriptions), [False] * len(descriptions)
    except Exception as e:
        logger.error(f"Batch {batch_num}: GPT call failed — {e}")
        return ["Other"] * len(descriptions), [False] * len(descriptions)
