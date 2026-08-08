"""
Tier 2 of SmartBudget's categorizer: the distilled ML classifier.

Sits between keyword rules (Tier 1) and the LLM (Tier 3).
Returns a category only when confident; otherwise defers to the LLM.

Usage inside pipeline/categorizer.py:

    from ml.ml_classifier import MLCategorizer
    ml_model = MLCategorizer("ml/model.joblib", threshold=0.85)

    def categorize(txn):
        cat = keyword_rules(txn.description)            # Tier 1
        if cat:
            return cat, "rule"
        cat, conf = ml_model.classify(txn.description, txn.amount,
                                      txn.date, txn.channel)
        if cat is not None:                             # Tier 2
            return cat, f"ml({conf:.2f})"
        cat = llm_classify(txn)                         # Tier 3
        log_training_example(txn, cat)   # feeds future retraining
        return cat, "llm"
"""

import numpy as np
import pandas as pd
import joblib

NUMERIC_COLS = ["log_amount", "day_of_month", "day_of_week", "is_month_start",
                "ch_UPI", "ch_POS", "ch_NEFT", "ch_IMPS", "ch_ACH"]


class MLCategorizer:
    def __init__(self, model_path: str = "ml/model.joblib", threshold: float = 0.85):
        self.pipe = joblib.load(model_path)
        self.threshold = threshold

    def _featurize(self, description, amount, txn_date, channel) -> pd.DataFrame:
        dt = pd.to_datetime(txn_date)
        row = {
            "raw_description": description,
            "log_amount": np.log1p(abs(float(amount))),
            "day_of_month": dt.day,
            "day_of_week": dt.dayofweek,
            "is_month_start": int(dt.day <= 7),
        }
        for ch in ["UPI", "POS", "NEFT", "IMPS", "ACH"]:
            row[f"ch_{ch}"] = int(channel == ch)
        return pd.DataFrame([row])

    def classify(self, description: str, amount: float,
                 txn_date: str, channel: str):
        """Returns (category, confidence) if confidence >= threshold,
        else (None, confidence) meaning: defer to the LLM."""
        X = self._featurize(description, amount, txn_date, channel)
        proba = self.pipe.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
        if conf >= self.threshold:
            return str(self.pipe.classes_[idx]), conf
        return None, conf


if __name__ == "__main__":
    m = MLCategorizer()
    tests = [
        ("UPI-ZOMATO LTD-48213@paytm", 430, "2026-06-14", "UPI"),
        ("POS 99213 BLINKIT NOIDA", 640, "2026-06-15", "POS"),
        ("NEFT-HDFC HOME LOAN EMI-N4821", 24500, "2026-06-03", "NEFT"),
        ("UPI-RANDOM NEW MERCHANT-99@ybl", 999, "2026-06-20", "UPI"),
    ]
    for d, a, dt, ch in tests:
        cat, conf = m.classify(d, a, dt, ch)
        verdict = f"{cat} ({conf:.2f})" if cat else f"DEFER TO LLM (best guess conf {conf:.2f})"
        print(f"{d:45s} -> {verdict}")
