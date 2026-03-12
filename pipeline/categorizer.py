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

import os
import json
import logging
from typing import Optional
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

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


def categorize_transactions(transactions: list[dict]) -> list[dict]:
    global unclassified_log
    unclassified_log = []

    result     = []
    needs_ai   = []

    for txn in transactions:
        cat = _keyword_match(txn["description"])
        if cat:
            result.append({**txn, "category": cat, "categorized_by": "rules"})
        else:
            result.append({**txn, "category": None, "categorized_by": None})
            needs_ai.append(len(result) - 1)

    logger.info(f"Classified {len(result) - len(needs_ai)}/{len(transactions)} transactions, based on rules provided.")

    if needs_ai:
        logger.info(f"Sending {len(needs_ai)} unknown transactions to GPT...")
        descs      = [result[i]["description"] for i in needs_ai]
        ai_cats    = _gpt_categorize_all(descs)

        for idx, cat in zip(needs_ai, ai_cats):
            result[idx]["category"]       = cat
            result[idx]["categorized_by"] = "ai"
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


def _gpt_categorize_all(descriptions: list[str]) -> list[str]:
    all_categories = []
    total_batches  = (len(descriptions) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(descriptions), BATCH_SIZE):
        batch     = descriptions[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        logger.info(f"GPT batch {batch_num}/{total_batches} ({len(batch)} transactions)...")

        result = _gpt_categorize_batch(batch, batch_num)
        all_categories.extend(result)

        other_count = result.count("Other")
        if other_count:
            logger.warning(f"Batch {batch_num}: {other_count}/{len(batch)} still unclassified after GPT")
        else:
            logger.info(f"Batch {batch_num}: all {len(batch)} classified")

    return all_categories


def _gpt_categorize_batch(descriptions: list[str], batch_num: int = 1) -> list[str]:
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

        for desc, raw_cat, val in zip(descriptions, cats, validated):
            if val == "Other" and raw_cat not in labels:
                logger.warning(f"Batch {batch_num}: invalid category '{raw_cat}' for '{desc}' → 'Other'")

        return validated

    except json.JSONDecodeError as e:
        logger.error(f"Batch {batch_num}: JSON parse failed — {e}")
        return ["Other"] * len(descriptions)
    except Exception as e:
        logger.error(f"Batch {batch_num}: GPT call failed — {e}")
        return ["Other"] * len(descriptions)