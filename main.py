"""
SmartBudget Pipeline
---------------------
A personal finance backend that ingests bank CSV exports, categorizes
transactions using AI, detects spending anomalies, and generates insights.

Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
load_dotenv()
# Get the root logger and add our formatter to uvicorn's existing handler
logging.getLogger("pipeline.csv_parser").setLevel(logging.DEBUG)
logging.getLogger("pipeline.categorizer").setLevel(logging.DEBUG)
logging.getLogger("pipeline.anomaly_detector").setLevel(logging.DEBUG)
logging.getLogger("pipeline.insights").setLevel(logging.DEBUG)

# Attach a handler so module logs actually print
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger("pipeline").addHandler(_handler)
logging.getLogger("pipeline").setLevel(logging.DEBUG)

from pipeline.csv_parser import parse_bank_csv
from pipeline.categorizer import categorize_transactions, get_unclassified
from pipeline.anomaly_detector import detect_anomalies
from pipeline.insights import generate_insights
from database.db import (
    init_db,
    save_transactions,
    save_anomalies,
    get_all_transactions,
    get_transactions_by_month,
    get_anomalies,
)

app = FastAPI(
    title="SmartBudget Pipeline",
    description="Upload your bank CSV → AI categorizes transactions → anomalies detected → spending insights generated.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    print("✅ Database initialized")

# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "project": "SmartBudget Pipeline",
        "endpoints": {
            "POST /transactions/upload": "Upload a bank CSV export",
            "GET  /transactions":        "List all transactions (filter: ?month=YYYY-MM)",
            "GET  /anomalies":           "Flagged unusual transactions",
            "GET  /insights":            "-generated spending insights",
        },
    }



@app.post("/transactions/upload", summary="Upload and process a bank CSV")
async def upload_csv(file: UploadFile = File(...)):
    """
    Pipeline:
      1. Parse CSV  →  normalize rows across bank formats
      2. Categorize →  keyword rules, GPT-4o-mini for the rest
      3. Anomalies  →  IQR statistics per category, GPT explains each flag
      4. Persist    →  save everything to SQLite
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    text = (await file.read()).decode("utf-8", errors="ignore")

    # Stage 1 — parse
    transactions = parse_bank_csv(text)
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions found — check your CSV format.")

    # Stage 2 — categorize
    transactions = categorize_transactions(transactions)

    # Stage 3 — anomalies
    anomalies = detect_anomalies(transactions)

    # Stage 4 — persist
    save_transactions(transactions)
    save_anomalies(anomalies)

    return {
        "status": "success",
        "transactions_imported": len(transactions),
        "anomalies_flagged": len(anomalies),
        "tip": "Call GET /insights for an AI narrative of your spending.",
    }


@app.get("/transactions", summary="List transactions")
def list_transactions(month: str = None):
    """Optional query param: ?month=YYYY-MM"""
    txns = get_transactions_by_month(month) if month else get_all_transactions()
    total_spent = round(abs(sum(t["amount"] for t in txns if t["amount"] < 0)), 2)
    return {"count": len(txns), "total_spent": total_spent, "transactions": txns}


@app.get("/anomalies", summary="Unusual transactions flagged by anomaly detector")
def list_anomalies():
    rows = get_anomalies()
    return {"count": len(rows), "anomalies": rows}


@app.get("/unclassified", summary="Transactions GPT could not classify")
def list_unclassified():
    """
    Returns transactions that fell back to 'Other' during categorization.
    Useful for identifying gaps in the AI's knowledge or ambiguous descriptions.
    """
    rows = get_unclassified()
    return {
        "count": len(rows),
        "note": "These transactions were categorized as 'Other' — GPT could not confidently classify them.",
        "transactions": rows,
    }



@app.get("/insights", summary="AI-generated spending insights")
def get_insights():
    transactions = get_all_transactions()
    if not transactions:
        raise HTTPException(status_code=404, detail="No transactions found. Upload a CSV first.")
    return generate_insights(transactions)