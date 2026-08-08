import logging
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.ml_classifier import MLCategorizer
from database import db
from pipeline import categorizer


def _txn(description: str, amount: float = -430.0) -> dict:
    return {
        "id": "txn-1",
        "date": "2026-06-14",
        "description": description,
        "amount": amount,
        "category": None,
        "categorized_by": None,
    }


def test_ml_classifier_known_merchant_classifies_dining():
    model = MLCategorizer("ml/model.joblib", threshold=0.85)

    category, confidence = model.classify(
        "UPI-ZOMATO LTD-48213@paytm",
        430,
        "2026-06-14",
        "UPI",
    )

    assert category == "Dining"
    assert confidence >= 0.85


def test_ml_classifier_negative_amount_matches_positive_amount():
    model = MLCategorizer("ml/model.joblib", threshold=0.85)

    positive_category, positive_confidence = model.classify(
        "UPI-ZOMATO LTD-48213@paytm",
        430,
        "2026-06-14",
        "UPI",
    )
    negative_category, negative_confidence = model.classify(
        "UPI-ZOMATO LTD-48213@paytm",
        -430,
        "2026-06-14",
        "UPI",
    )

    assert positive_category == negative_category == "Dining"
    assert negative_confidence >= 0.85
    assert abs(positive_confidence - negative_confidence) < 1e-12


def test_ml_classifier_nonsense_merchant_defers():
    model = MLCategorizer("ml/model.joblib", threshold=0.85)

    category, confidence = model.classify(
        "RANDOM NEW MERCHANT XYZ",
        999,
        "2026-06-20",
        "UPI",
    )

    assert category is None
    assert confidence < 0.85


def test_categorizer_does_not_call_llm_when_rule_resolves(mocker):
    gpt = mocker.patch.object(categorizer, "_gpt_categorize_all", return_value=[("Other", False)])

    result = categorizer.categorize_transactions([
        _txn("SWIGGY ORDER MCDONALDS"),
    ])

    assert result[0]["category"] == "Dining"
    assert result[0]["categorized_by"] == "rule"
    gpt.assert_not_called()


def test_categorizer_does_not_call_llm_when_ml_resolves(mocker):
    gpt = mocker.patch.object(categorizer, "_gpt_categorize_all", return_value=[("Other", False)])

    result = categorizer.categorize_transactions([
        _txn("POS 123 CROMA RETAIL DEL", -45000.0),
    ])

    assert result[0]["category"] == "Shopping"
    assert result[0]["categorized_by"] == "ml"
    gpt.assert_not_called()


def test_categorizer_calls_llm_when_ml_defers(mocker):
    gpt = mocker.patch.object(categorizer, "_gpt_categorize_all", return_value=[("Shopping", True)])
    save = mocker.patch.object(categorizer, "save_labeled_transaction")

    result = categorizer.categorize_transactions([
        _txn("RANDOM NEW MERCHANT XYZ"),
    ])

    assert result[0]["category"] == "Shopping"
    assert result[0]["categorized_by"] == "llm"
    gpt.assert_called_once_with(["RANDOM NEW MERCHANT XYZ"])
    save.assert_called_once()


def test_missing_model_falls_back_to_llm(mocker):
    old_model = categorizer.ml_model
    categorizer.ml_model = None
    gpt = mocker.patch.object(categorizer, "_gpt_categorize_all", return_value=[("Other", True)])
    save = mocker.patch.object(categorizer, "save_labeled_transaction")

    try:
        result = categorizer.categorize_transactions([
            _txn("RANDOM NEW MERCHANT XYZ"),
        ])
    finally:
        categorizer.ml_model = old_model

    assert result[0]["category"] == "Other"
    assert result[0]["categorized_by"] == "llm"
    gpt.assert_called_once_with(["RANDOM NEW MERCHANT XYZ"])
    save.assert_called_once()


def test_missing_model_loader_logs_warning(caplog):
    caplog.set_level(logging.WARNING)

    model = categorizer._load_ml_model("ml/does-not-exist.joblib")

    assert model is None
    assert "ML classifier model not found" in caplog.text


def test_openai_key_fallback_does_not_write_labeled_transaction(mocker, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    old_model = categorizer.ml_model
    categorizer.ml_model = None
    save = mocker.patch.object(categorizer, "save_labeled_transaction")

    try:
        result = categorizer.categorize_transactions([
            _txn("RANDOM NEW MERCHANT XYZ", -999.0),
        ])
    finally:
        categorizer.ml_model = old_model

    assert result[0]["category"] == "Other"
    assert result[0]["categorized_by"] == "llm"
    save.assert_not_called()


def test_successful_llm_path_writes_labeled_transaction_with_positive_amount(mocker, monkeypatch, tmp_path):
    old_db_path = db.DB_PATH
    old_model = categorizer.ml_model
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    db.DB_PATH = str(tmp_path / "smartbudget-test.db")
    db.init_db()
    categorizer.ml_model = None
    mocker.patch.object(categorizer, "_gpt_categorize_all", return_value=[("Shopping", True)])

    try:
        result = categorizer.categorize_transactions([
            _txn("RANDOM NEW MERCHANT XYZ", -999.0),
        ])
        with sqlite3.connect(db.DB_PATH) as conn:
            row = conn.execute(
                "SELECT raw_description, amount, channel, category FROM labeled_transactions"
            ).fetchone()
    finally:
        categorizer.ml_model = old_model
        db.DB_PATH = old_db_path

    assert result[0]["category"] == "Shopping"
    assert result[0]["categorized_by"] == "llm"
    assert row == ("RANDOM NEW MERCHANT XYZ", 999.0, "UPI", "Shopping")
