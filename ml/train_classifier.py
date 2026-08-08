"""
Train SmartBudget's distilled transaction classifier (Tier 2).

Pipeline:
    raw_description --> TF-IDF word 1-2 grams  ┐
    raw_description --> TF-IDF char_wb 3-5 grams ├─ hstack ─> LogisticRegression
    amount, timing, channel --> numeric features ┘

Why these choices (interview defense lives in INTERVIEW_DEFENSE.md):
- char_wb n-grams handle bank-feed mangling (ZMT*ZOMATO ~ ZOMATO share "omat")
- LogisticRegression: fast, well-calibrated-enough probabilities for
  confidence thresholding, interpretable per-class weights
- Threshold sweep quantifies the coverage/accuracy tradeoff that decides
  how much LLM traffic we eliminate

Run:  python ml/train_classifier.py
Outputs: ml/model.joblib, ml/metrics.json, console report
"""

import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive numeric/timing features. Kept as a plain function so the exact
    same transformation is reused at inference time (ml_classifier.py)."""
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    out["log_amount"] = np.log1p(out["amount"].abs())
    out["day_of_month"] = dt.dt.day
    out["day_of_week"] = dt.dt.dayofweek
    out["is_month_start"] = (dt.dt.day <= 7).astype(int)
    for ch in ["UPI", "POS", "NEFT", "IMPS", "ACH"]:
        out[f"ch_{ch}"] = (out["channel"] == ch).astype(int)
    return out


NUMERIC_COLS = ["log_amount", "day_of_month", "day_of_week", "is_month_start",
                "ch_UPI", "ch_POS", "ch_NEFT", "ch_IMPS", "ch_ACH"]


def build_pipeline() -> Pipeline:
    features = ColumnTransformer([
        ("word_tfidf",
         TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                         token_pattern=r"[A-Za-z]{2,}"),   # drop pure ref numbers
         "raw_description"),
        ("char_tfidf",
         TfidfVectorizer(lowercase=True, analyzer="char_wb",
                         ngram_range=(3, 5), min_df=2, max_features=30000),
         "raw_description"),
        ("numeric", StandardScaler(), NUMERIC_COLS),
    ])
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def threshold_sweep(pipe, X_test, y_test):
    """Coverage vs accuracy at each confidence threshold. This table decides
    the production threshold: below it -> defer to LLM."""
    proba = pipe.predict_proba(X_test)
    pred = pipe.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    rows = []
    for t in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        mask = conf >= t
        coverage = mask.mean()
        acc_on_covered = accuracy_score(y_test[mask], pred[mask]) if mask.any() else 0.0
        rows.append({"threshold": t,
                     "coverage": round(float(coverage), 4),
                     "accuracy_on_covered": round(float(acc_on_covered), 4),
                     "deferred_to_llm": round(float(1 - coverage), 4)})
    return rows


def main():
    df = add_numeric_features(pd.read_csv("ml/labeled_transactions.csv"))
    X = df[["raw_description"] + NUMERIC_COLS]
    y = df["category"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline()
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    print(f"train time: {train_s:.1f}s | held-out accuracy: {acc:.4f} | macro F1: {f1:.4f}\n")
    print(classification_report(y_te, pred))

    sweep = threshold_sweep(pipe, X_te, y_te.to_numpy())
    print("threshold  coverage  acc_on_covered  deferred_to_llm")
    for r in sweep:
        print(f"  {r['threshold']:.2f}     {r['coverage']:.3f}     "
              f"{r['accuracy_on_covered']:.4f}          {r['deferred_to_llm']:.3f}")

    # Generalization to merchants never seen in training (production scenario)
    unseen = add_numeric_features(pd.read_csv("ml/unseen_merchants.csv"))
    Xu = unseen[["raw_description"] + NUMERIC_COLS]
    up = pipe.predict(Xu)
    uconf = pipe.predict_proba(Xu).max(axis=1)
    uacc = accuracy_score(unseen["category"], up)
    umask = uconf >= 0.85
    uacc_conf = accuracy_score(unseen["category"][umask], up[umask]) if umask.any() else 0.0
    print(f"\nUNSEEN merchants: raw accuracy {uacc:.4f} | "
          f"coverage@0.85 {umask.mean():.3f} | accuracy on covered {uacc_conf:.4f}")

    # Latency: single-row predict (what the API path pays)
    row = X_te.iloc[[0]]
    pipe.predict_proba(row)  # warm up
    t0 = time.perf_counter()
    N = 500
    for _ in range(N):
        pipe.predict_proba(row)
    lat_us = (time.perf_counter() - t0) / N * 1e6
    print(f"single-transaction inference latency: {lat_us:.0f} µs "
          f"(vs ~1,000,000 µs for an LLM round trip)")

    # Interview artifact: top word features per class
    ct = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    word_names = ct.named_transformers_["word_tfidf"].get_feature_names_out()
    print("\ntop word features per category:")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i][: len(word_names)]
        top = np.argsort(coefs)[-5:][::-1]
        print(f"  {cls:14s} -> {', '.join(word_names[j] for j in top)}")

    joblib.dump(pipe, "ml/model.joblib")
    json.dump({"held_out_accuracy": round(float(acc), 4),
               "macro_f1": round(float(f1), 4),
               "train_seconds": round(train_s, 2),
               "single_pred_latency_us": round(lat_us),
               "unseen_merchant_accuracy": round(float(uacc), 4),
               "threshold_sweep": sweep},
              open("ml/metrics.json", "w"), indent=2)
    print("\nsaved ml/model.joblib and ml/metrics.json")


if __name__ == "__main__":
    main()
