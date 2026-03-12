# 💸 SmartBudget Pipeline

A personal finance backend that ingests bank CSV exports, categorizes transactions with AI, detects anomalous spending, and generates narrative insights.

> See [`DEV_LOG.md`](./DEV_LOG.md) for how ChatGPT was used as a development assistant.

---

## Pipeline

```
POST /transactions/upload  (bank CSV)
           │
           ▼
   csv_parser       →  normalize multi-bank CSV formats
           │
           ▼
   categorizer      →  keyword rules + GPT-4o-mini batch fallback
           │
           ▼
   anomaly_detector →  IQR stats per category + GPT explanations
           │
           ▼
   SQLite DB        →  persists everything
           │
           ▼
   GET /anomalies   →  flagged transactions with plain-English explanations
   GET /insights    →  GPT-4o narrative spending summary
```

---

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
# → http://localhost:8000/docs
```

**Upload the sample CSV:**
```bash
curl -X POST http://localhost:8000/transactions/upload \
  -F "file=@sample_data/sample_transactions.csv"
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transactions/upload` | Upload bank CSV |
| `GET`  | `/transactions`        | All transactions (`?month=YYYY-MM`) |
| `GET`  | `/anomalies`           | Unusual charges with AI explanations |
| `GET`  | `/insights`            | AI spending narrative |

---

## Running Tests

```bash
pytest
```

Tests cover CSV parsing (multi-format, edge cases), categorizer keyword matching and GPT mock, anomaly detection IQR logic, and the full database layer using a temp SQLite file.

```
tests/
├── test_csv_parser.py       # 14 tests — parsing, date formats, edge cases
├── test_categorizer.py      # 8 tests  — keyword matching, GPT mock
├── test_anomaly_detector.py # 10 tests — IQR logic, GPT mock, failure handling
└── test_db.py               # 10 tests — save/retrieve, isolation via tmp_path
```

---

## Tech Stack

- **FastAPI** — REST API
- **OpenAI GPT-4o** — Spending insights
- **OpenAI GPT-4o-mini** — Batch categorization + anomaly explanations
- **SQLite** — Local storage
- **pytest** — Unit tests with mocking
- **Python 3.11+**

---

## Project Structure

```
smartbudget/
├── main.py
├── pipeline/
│   ├── csv_parser.py          # Multi-bank CSV normalization
│   ├── categorizer.py         # AI transaction categorization
│   ├── anomaly_detector.py    # IQR anomaly detection + GPT explanations
│   └── insights.py            # AI narrative generation
├── database/
│   └── db.py                  # SQLite schema + CRUD
├── tests/
│   ├── test_csv_parser.py
│   ├── test_categorizer.py
│   ├── test_anomaly_detector.py
│   └── test_db.py
├── sample_data/
│   └── sample_transactions.csv
├── DEV_LOG.md
├── pytest.ini
└── requirements.txt
```
