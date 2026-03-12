"""
pipeline/csv_parser.py
-----------------------
Normalize bank CSV exports into a standard transaction format.
Handles Chase, Bank of America, Wells Fargo, Capital One, and generic formats.
"""

import csv
import io
import uuid
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DATE_COLS   = ["date", "transaction date", "posted date", "trans date"]
DESC_COLS   = ["description", "transaction description", "memo", "payee", "name"]
AMOUNT_COLS = ["amount", "transaction amount", "amount_credit"]
DEBIT_COLS  = ["debit", "withdrawal", "withdrawal amt."]
CREDIT_COLS = ["credit", "deposit", "deposit amt."]
TYPE_COLS   = ["transaction_type", "type"]


def parse_bank_csv(csv_text: str) -> list[dict]:

    reader  = csv.DictReader(io.StringIO(csv_text.strip()))
    headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}

    if not headers:
        return []

    date_col   = _find_col(headers, DATE_COLS)
    desc_col   = _find_col(headers, DESC_COLS)
    amount_col = _find_col(headers, AMOUNT_COLS)
    # debit_col  = _find_col(headers, DEBIT_COLS)
    # credit_col = _find_col(headers, CREDIT_COLS)
    type_col   = _find_col(headers, TYPE_COLS)

    # print(f"Identified columns - Date: {date_col}, Desc: {desc_col}, Amount: {amount_col}, Debit: {debit_col}, Credit: {credit_col}, Type: {type_col}")
    
    if not date_col or not desc_col:
        logger.warning("CSV missing required columns.")
        return []

    transactions = []

    for row in reader:

        date   = _parse_date(row.get(date_col, "").strip())
        desc   = row.get(desc_col, "").strip()
        amount = _resolve_amount(row, amount_col, type_col)

        if not date or not desc or amount is None:
            continue

        transactions.append({
            "id": str(uuid.uuid4()),
            "date": date,
            "description": desc,
            "amount": amount,
            "category": None,
            "categorized_by": None,
        })

    return sorted(transactions, key=lambda t: t["date"], reverse=True)


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def _find_col(headers: dict, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in headers:
            return headers[c]
    return None


def _parse_date(raw: str) -> Optional[str]:
    """
    Try multiple bank date formats and normalize to YYYY-MM-DD
    """

    formats = [
        "%d/%m/%Y",  # India / EU
        "%d/%m/%y",
        "%m/%d/%Y",  # US
        "%m/%d/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            logger.debug(f"Date parsed: '{raw}' → '{parsed}' (fmt: {fmt})")
            return parsed
        except ValueError:
            continue

    logger.warning(f"Could not parse date: '{raw}'")
    return None


def _to_float(val) -> Optional[float]:
    if not val:
        return None

    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def _resolve_amount(row, amount_col, type_col=None) -> Optional[float]:

    if amount_col:
        v = _to_float(row.get(amount_col))
        if v is not None:
            if type_col:
                transaction_type = row.get(type_col, "").strip().lower()
                if transaction_type == "debit":
                    return -abs(v)
                elif transaction_type == "credit":
                    return abs(v)
            return v

    # debit  = _to_float(row.get(debit_col))  if debit_col  else None
    # credit = _to_float(row.get(credit_col)) if credit_col else None

    # if debit is not None and debit > 0:
    #     return -abs(debit)

    # if credit is not None and credit > 0:
    #     return abs(credit)

    return None