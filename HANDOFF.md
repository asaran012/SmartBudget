# SmartBudget Project Handoff

## 1. Project Goal

SmartBudget is a personal finance backend project.

The goal is to ingest a bank CSV export, normalize the transactions, categorize them intelligently, detect unusual spending, store everything in SQLite, and expose the results through a FastAPI API.

The project is also being prepared as an interview/demo project. It should show:

- A clean backend pipeline.
- Practical AI usage, not AI for everything.
- A cost-conscious categorization system.
- A local ML classifier that reduces LLM usage.
- Explainable anomaly detection using deterministic statistics.
- A retraining feedback loop using real LLM labels.

The high-level product flow is:

```text
Bank CSV upload
  -> CSV parser
  -> transaction categorizer
       Tier 1: keyword rules
       Tier 2: local ML classifier
       Tier 3: OpenAI LLM fallback
  -> anomaly detector
  -> SQLite persistence
  -> API endpoints for transactions, anomalies, unclassified rows, and insights
```

## 2. Current Progress

The backend runs successfully.

We verified the project end-to-end manually:

- Created/activated the virtual environment.
- Installed dependencies.
- Ran the full test suite.
- Ran the ML classifier directly.
- Checked `ml/metrics.json`.
- Started the FastAPI server.
- Uploaded `sample_data/sample_transactions.csv`.
- Verified `/transactions`.
- Verified `/anomalies`.
- Verified `/unclassified`.
- Created and uploaded an ML-specific test CSV.
- Verified that the ML tier works through the API.
- Verified SQLite contents directly.
- Cleared `labeled_transactions`.
- Fixed `.gitignore`.
- Stopped the server cleanly.

Current test status:

```text
10 passed, 32 warnings
```

The warnings are from `joblib`/NumPy while loading the saved model. They are not currently blocking the project.

Current ML metrics from `ml/metrics.json`:

```json
{
  "held_out_accuracy": 0.9967,
  "macro_f1": 0.9963,
  "train_seconds": 0.87,
  "single_pred_latency_us": 891,
  "unseen_merchant_accuracy": 0.5667,
  "threshold_sweep": [
    {
      "threshold": 0.5,
      "coverage": 0.9925,
      "accuracy_on_covered": 0.9983,
      "deferred_to_llm": 0.0075
    },
    {
      "threshold": 0.6,
      "coverage": 0.9867,
      "accuracy_on_covered": 0.9983,
      "deferred_to_llm": 0.0133
    },
    {
      "threshold": 0.7,
      "coverage": 0.965,
      "accuracy_on_covered": 0.9991,
      "deferred_to_llm": 0.035
    },
    {
      "threshold": 0.8,
      "coverage": 0.9208,
      "accuracy_on_covered": 1.0,
      "deferred_to_llm": 0.0792
    },
    {
      "threshold": 0.85,
      "coverage": 0.875,
      "accuracy_on_covered": 1.0,
      "deferred_to_llm": 0.125
    },
    {
      "threshold": 0.9,
      "coverage": 0.7633,
      "accuracy_on_covered": 1.0,
      "deferred_to_llm": 0.2367
    },
    {
      "threshold": 0.95,
      "coverage": 0.3658,
      "accuracy_on_covered": 1.0,
      "deferred_to_llm": 0.6342
    }
  ]
}
```

Interpretation:

- The model performs very well on normal held-out synthetic test data.
- At the production threshold `0.85`, it confidently handles about `87.5%` of transactions.
- It defers about `12.5%` to the LLM.
- Accuracy on confident covered predictions at threshold `0.85` is `1.0` on this synthetic validation split.
- Unseen merchant accuracy is lower at `0.5667`, which is expected because truly new merchant strings are harder.

## 3. Current Code Structure And Important Files

### Root

```text
main.py
frontend.html
requirements.txt
README.md
DEV_LOG.md
.gitignore
smartbudget.db
sample_data/
pipeline/
database/
ml/
tests/
```

### `main.py`

FastAPI entrypoint.

Important endpoints:

```text
GET  /
POST /transactions/upload
GET  /transactions
GET  /anomalies
GET  /unclassified
GET  /insights
```

The upload endpoint does:

```text
read uploaded CSV
-> parse_bank_csv(...)
-> categorize_transactions(...)
-> detect_anomalies(...)
-> save_transactions(...)
-> save_anomalies(...)
```

### `pipeline/csv_parser.py`

Normalizes bank CSV exports into transaction dictionaries.

Important output shape:

```python
{
    "id": "...",
    "date": "YYYY-MM-DD",
    "description": "...",
    "amount": -370.0,
    "category": None,
    "categorized_by": None,
}
```

Amounts:

- Debit rows become negative.
- Credit rows become positive.

Supported header aliases include:

```python
DATE_COLS = ["date", "transaction date", "posted date", "trans date"]
DESC_COLS = ["description", "transaction description", "memo", "payee", "name"]
AMOUNT_COLS = ["amount", "transaction amount", "amount_credit"]
TYPE_COLS = ["transaction_type", "type"]
```

### `pipeline/categorizer.py`

This is the most important business logic file.

The categorizer is now 3-tier:

```text
Tier 1: keyword rules
Tier 2: local ML classifier
Tier 3: OpenAI LLM fallback
```

The public function is:

```python
categorize_transactions(transactions: list[dict]) -> list[dict]
```

Do not change its public interface unless absolutely necessary, because `main.py` imports and uses it.

The source tier is stored in:

```python
categorized_by
```

Allowed values:

```text
rule
ml
llm
```

The canonical category list lives in `CATEGORIES`:

```text
Rent
Income
Dining
Groceries
Transport
Subscriptions
Shopping
Health & Fitness
Utilities & Bills
Entertainment
Transfers & Payments
```

The ML model is loaded once at module level:

```python
ml_model = _load_ml_model()
```

The model path is:

```text
ml/model.joblib
```

Missing model behavior:

- If `ml/model.joblib` is missing, log a warning.
- App should still start.
- Categorizer falls back to rules -> LLM behavior.

LLM feedback logging:

- Only genuine LLM responses are written to `labeled_transactions`.
- Fallback/error paths, including missing `OPENAI_API_KEY`, must not contaminate training data.

### `pipeline/anomaly_detector.py`

Detects unusual transactions using deterministic logic.

Important design:

- Pure statistics decide whether something is anomalous.
- GPT is only used to explain anomalies in plain English.

Rules:

```text
Rule 1: category IQR threshold
Rule 2: large one-off transaction from a merchant seen once
```

Constants:

```python
MIN_SAMPLES = 5
IQR_MULTIPLIER = 4.0
LARGE_ONE_OFF_INR = 3500
```

Large Croma test transaction:

```text
POS 123 CROMA RETAIL DEL
amount: 45000
category: Shopping
reason: large_one_off
```

### `pipeline/insights.py`

Generates a GPT-based spending narrative from saved transactions.

Important:

- This endpoint requires a valid OpenAI API key if used.
- It pre-aggregates spending before sending it to GPT.

### `database/db.py`

SQLite storage.

Database path:

```python
DB_PATH = os.getenv("DB_PATH", "smartbudget.db")
```

Tables:

```text
transactions
anomalies
labeled_transactions
```

`transactions` stores:

```text
id
date
description
amount
category
categorized_by
created_at
```

`anomalies` stores:

```text
id
transaction_id
date
description
amount
category
reason
ratio
explanation
created_at
```

`labeled_transactions` stores future ML training examples:

```text
id
date
raw_description
amount
channel
category
created_at
```

Important decision:

- `labeled_transactions.amount` stores positive magnitude using `abs(float(transaction["amount"]))`.
- This matches ML training data.

### `ml/generate_dataset.py`

Generates synthetic labeled ML training data.

Outputs:

```text
ml/labeled_transactions.csv
ml/unseen_merchants.csv
```

Important:

- Seed is currently `42`.
- Category names must exactly match the canonical category list in `pipeline/categorizer.py`.
- Merchant strings are intentionally mangled to mimic real bank feed descriptions.

Examples of realism:

- UPI/POS/NEFT/IMPS/ACH prefixes.
- Random reference numbers.
- VPA handles like `@paytm`, `@ybl`.
- Truncation.
- Aggregator prefixes like `ZMT*`, `SWG*`, `AMZN*`.

### `ml/train_classifier.py`

Trains the ML classifier.

Pipeline:

```text
raw_description -> word TF-IDF
raw_description -> char_wb TF-IDF
numeric features -> StandardScaler
all features -> LogisticRegression
```

Outputs:

```text
ml/model.joblib
ml/metrics.json
```

Important bug fix already made:

```python
out["log_amount"] = np.log1p(out["amount"].abs())
```

This makes the amount feature sign-invariant.

### `ml/ml_classifier.py`

Loads the saved model and classifies one transaction.

Important method:

```python
classify(description, amount, txn_date, channel)
```

Returns:

```python
(category, confidence)
```

if confidence is above threshold, otherwise:

```python
(None, confidence)
```

Important bug fix already made:

```python
"log_amount": np.log1p(abs(float(amount)))
```

This prevents negative debits from causing invalid `log1p` behavior.

### `tests/test_ml_classifier.py`

Current pytest suite.

It covers:

- Known merchant classification.
- Negative vs positive amount parity.
- Nonsense merchant defers.
- Rule tier prevents LLM call.
- ML tier prevents LLM call.
- ML defer path calls LLM.
- Missing model fallback.
- Missing API key fallback does not write contaminated training data.
- Successful mocked LLM response writes `labeled_transactions` with positive amount.

## 4. Important Design Decisions And Why

### Modular Monolith

The project is a modular monolith:

```text
main.py
pipeline/
database/
ml/
tests/
```

Reason:

- Clean separation of responsibilities.
- Easier to demo and test.
- No need for microservices, queues, or distributed infra for this scale.

### Three-Tier Categorization

The categorizer uses:

```text
rules -> ML -> LLM
```

Reason:

- Rules are free, fast, and reliable for obvious merchants.
- ML is local and fast for learned patterns.
- LLM is powerful but expensive/slow, so it is fallback only.

### Confidence Threshold `0.85`

The ML classifier only emits a category if confidence is at least `0.85`.

Reason:

- At threshold `0.85`, validation showed high precision on covered rows.
- Low-confidence predictions should defer to LLM instead of silently misclassifying.

### Canonical Category Taxonomy

Rules, LLM prompt, and ML model must all emit identical category strings.

Canonical list:

```text
Rent
Income
Dining
Groceries
Transport
Subscriptions
Shopping
Health & Fitness
Utilities & Bills
Entertainment
Transfers & Payments
```

Earlier mismatch:

- Rules used `Dining`.
- Old ML data used `Food`.
- Rules used `Utilities & Bills`.
- Old ML data used `Utilities`.
- Old ML had `Education`, which was not in canonical rules.

Fix:

- Updated `ml/generate_dataset.py` so ML labels exactly match `pipeline/categorizer.py`.

### Feedback Table Only For Real LLM Labels

`labeled_transactions` should store only genuine LLM classifications.

Reason:

- It is future training data.
- If fallback/error paths write rows like `Other`, the training dataset becomes contaminated.

Important behavior:

- Missing `OPENAI_API_KEY` fallback returns `Other` but should not write to `labeled_transactions`.
- JSON parse failures or GPT errors should not write to `labeled_transactions`.

### Sign-Invariant Amount Feature

Bank debits are negative in the app, but ML training data amounts are positive magnitudes.

Bug fixed:

```python
np.log1p(amount)
```

is invalid or wrong for negative debits.

Current behavior:

```python
np.log1p(abs(amount))
```

in both training and serving.

Reason:

- Prevent train/serve skew.
- Avoid NaNs or invalid features.
- Classification for `430` and `-430` should be identical.

### Deterministic Anomaly Detection

Anomalies are detected with Python statistics, not GPT.

Reason:

- Deterministic.
- Auditable.
- Easier to test.
- GPT should explain anomalies, not decide them.

## 5. Algorithms, Libraries, And Technologies

### Backend

- Python
- FastAPI
- Uvicorn
- SQLite
- `python-multipart` for file uploads
- `python-dotenv` for `.env`

### AI

- OpenAI Python SDK
- `gpt-4o-mini` for transaction categorization fallback and anomaly explanations
- `gpt-4o` for spending insights

### ML

- pandas
- numpy
- scikit-learn
- joblib

ML model:

```text
ColumnTransformer
  - word TfidfVectorizer on raw_description
  - char_wb TfidfVectorizer on raw_description
  - StandardScaler on numeric/timing/channel features
LogisticRegression
```

Numeric features:

```text
log_amount
day_of_month
day_of_week
is_month_start
ch_UPI
ch_POS
ch_NEFT
ch_IMPS
ch_ACH
```

`log_amount` is:

```python
np.log1p(abs(amount))
```

### Testing

- pytest
- pytest-mock
- temporary SQLite DB in tests where needed
- mocked LLM calls

## 6. Assumptions And Constraints

### Assumptions

- The project is run from:

```text
/Users/saranay12/SmartBudget
```

- Virtual environment is:

```text
.venv/
```

- Python currently used:

```text
/Users/saranay12/SmartBudget/.venv/bin/python
```

- The app reads `.env` through `load_dotenv()`.
- `.env` should not be committed.
- SQLite database `smartbudget.db` should not be committed.

### Constraints

- Do not commit `.venv/`.
- Do not commit `.env`.
- Do not commit `smartbudget.db`.
- Do not log fake/fallback labels into `labeled_transactions`.
- Do not let ML category names drift from rule/LLM category names.
- Do not modify public `categorize_transactions(...)` interface casually because `main.py` depends on it.

## 7. Bugs, Blockers, And TODOs

### Fixed Bugs

1. Category taxonomy mismatch.

Rules/LLM/ML now use the same category strings.

2. Training-data contamination.

Fallback/error LLM paths no longer write rows into `labeled_transactions`.

3. Negative amount train/serve skew.

Both training and serving use:

```python
np.log1p(abs(amount))
```

4. Missing model crash risk.

If `ml/model.joblib` is missing, app logs a warning and falls back to rules -> LLM.

5. `.gitignore` was missing `.venv/`.

Current `.gitignore` should be:

```text
.env
*.db
__pycache__/
*.pyc
.DS_Store
venv/
.venv/
```

### Current TODOs

1. Update `README.md`.

It is stale. It still describes categorizer as:

```text
keyword rules + GPT fallback
```

It should now describe:

```text
keyword rules -> ML classifier -> LLM fallback
```

It should also document:

- ML training commands.
- ML metrics.
- Three-tier verification.
- `labeled_transactions`.
- Test suite.

2. Update `DEV_LOG.md`.

Add a new section about:

- Adding the ML tier.
- Why local ML sits between rules and LLM.
- Taxonomy mismatch fix.
- Training contamination fix.
- Negative amount bug fix.
- Model threshold decision.

3. Improve `sample_data/sample_transactions.csv`.

Current sample mostly exercises rule and LLM/fallback paths.

Add at least one row that clearly exercises ML:

```csv
06/14/2026,POS 123 CROMA RETAIL DEL,45000.00,debit
```

Potentially include:

```csv
06/14/2026,SWIGGY ORDER MCDONALDS,370.00,debit
06/14/2026,POS 123 CROMA RETAIL DEL,45000.00,debit
06/20/2026,RANDOM NEW MERCHANT XYZ,999.00,debit
```

These demonstrate:

```text
rule
ml
llm/fallback
```

4. Review frontend.

`frontend.html` exists, but it has not been verified recently in this handoff sequence.

Need to check whether:

- It uploads CSV correctly.
- It displays transactions.
- It displays anomalies.
- It handles backend errors clearly.

5. Decide what ML artifacts to commit.

Recommended to commit for reproducibility:

```text
ml/generate_dataset.py
ml/train_classifier.py
ml/ml_classifier.py
ml/metrics.json
ml/model.joblib
ml/INTERVIEW_DEFENSE.md
ml/labeled_transactions.csv
ml/unseen_merchants.csv
tests/test_ml_classifier.py
```

Because the synthetic CSVs and model are part of the demo and allow the project to work immediately.

6. Make a clean commit.

Possible commit message:

```text
Add ML categorization tier with training metrics
```

## 8. Exact Next Task Planned

The exact next task planned was:

```text
Update README.md and DEV_LOG.md to reflect the new ML tier and current verified run flow.
```

Suggested order:

1. Update `README.md`.
2. Update `DEV_LOG.md`.
3. Optionally improve `sample_data/sample_transactions.csv` to include rule + ML + LLM examples.
4. Run:

```bash
python -m pytest
```

5. Run a quick API verification if sample CSV changes.
6. Check:

```bash
git status --short
```

7. Stage and commit once satisfied.

## 9. Important Explanations Already Given

### How To Run The Whole Project

Full run sequence:

```bash
cd /Users/saranay12/SmartBudget
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python ml/ml_classifier.py
cat ml/metrics.json
python -m uvicorn main:app --reload
```

In a second terminal:

```bash
cd /Users/saranay12/SmartBudget
source .venv/bin/activate
curl -s http://127.0.0.1:8000/
curl -s -X POST http://127.0.0.1:8000/transactions/upload \
  -F file=@sample_data/sample_transactions.csv
curl -s http://127.0.0.1:8000/transactions
curl -s http://127.0.0.1:8000/anomalies
curl -s http://127.0.0.1:8000/unclassified
```

Create and upload ML test CSV:

```bash
printf 'Date,Description,Amount_Credit,Transaction_Type\n06/14/2026,POS 123 CROMA RETAIL DEL,45000.00,debit\n' > /tmp/smartbudget_ml_check.csv
curl -s -X POST http://127.0.0.1:8000/transactions/upload \
  -F file=@/tmp/smartbudget_ml_check.csv
curl -s http://127.0.0.1:8000/transactions
curl -s http://127.0.0.1:8000/anomalies
```

Check SQLite:

```bash
sqlite3 smartbudget.db 'SELECT description, amount, category, categorized_by FROM transactions;'
sqlite3 smartbudget.db 'SELECT raw_description, amount, channel, category FROM labeled_transactions;'
```

Clear feedback table:

```bash
sqlite3 smartbudget.db 'DELETE FROM labeled_transactions;'
sqlite3 smartbudget.db 'SELECT raw_description, amount, channel, category FROM labeled_transactions;'
```

Stop server:

```text
Control + C
```

### ML Retraining Sequence

Only when ML data/code changes:

```bash
python ml/generate_dataset.py
python ml/train_classifier.py
python ml/ml_classifier.py
cat ml/metrics.json
python -m pytest
```

### What The Tier Verification Showed

Rule path:

```text
SWIGGY ORDER MCDONALDS
-> Dining
-> categorized_by: rule
```

LLM/fallback path:

```text
Gift
-> Other
-> categorized_by: llm
```

ML path:

```text
POS 123 CROMA RETAIL DEL
-> Shopping
-> categorized_by: ml
```

Anomaly path:

```text
POS 123 CROMA RETAIL DEL
amount: 45000
category: Shopping
reason: large_one_off
```

### Why `Gift` Appeared In `labeled_transactions`

After uploading the sample CSV, `Gift` went through LLM/fallback and appeared as:

```text
Gift|9500.0|UPI|Other
```

The table was later cleared.

Important nuance:

- If it was a genuine LLM response, storing it is okay.
- If it came from missing API key or fallback/error, it is contamination.
- Current code is designed not to log fallback/error rows.

### Why `HDFC HOME LOAN EMI` Deferred In Direct ML Test

Direct ML output:

```text
NEFT-HDFC HOME LOAN EMI-N4821 -> DEFER TO LLM (best guess conf 0.65)
```

This is expected because:

- Confidence `0.65` is below threshold `0.85`.
- The ML tier should defer when uncertain.

### Why `unseen_merchant_accuracy` Is Lower

`unseen_merchant_accuracy` is `0.5667`.

Reason:

- The unseen merchant file intentionally contains brand-new merchant strings.
- This tests generalization to production-like unknown merchants.
- Raw accuracy is harder there.
- The confidence threshold helps: uncertain cases defer to LLM.

## 10. User Preferences While Learning

The user wants to learn by doing.

Preferred teaching style:

- Go from the top.
- Line by line.
- One command at a time.
- Wait for the user to run it and report output.
- Explain what each command proves.
- Avoid dumping too much at once during interactive run-throughs.
- When asked for copy/paste reference, provide all commands in one code cell.
- Be practical and concrete.
- Use exact paths and exact commands.
- Explain why outputs are expected.
- Help interpret results, not just give commands.

The user is learning the project and wants to understand:

- How to run it.
- How to verify it.
- How to explain it.
- How to defend it in interviews.

Tone preference:

- Supportive.
- Step-by-step.
- Beginner-friendly, but not shallow.
- Explain important engineering reasoning.

## 11. Anything Else A New ChatGPT Instance Must Know

### Current Git State

Recent `git status --short` showed:

```text
 M .gitignore
 M database/db.py
 M pipeline/categorizer.py
 M requirements.txt
?? ml/
?? tests/
```

Important:

- `ml/` and `tests/` are untracked.
- `git diff --stat` only showed tracked files, not untracked `ml/` and `tests/`.

Tracked diff summary showed:

```text
.gitignore              | Bin 100 -> 45 bytes
database/db.py          | 44 insertions
pipeline/categorizer.py | 120 changes
requirements.txt        | 5 insertions
```

After fixing `.gitignore`, it should contain:

```text
.env
*.db
__pycache__/
*.pyc
.DS_Store
venv/
.venv/
```

### Server Was Stopped

The server was stopped with:

```text
Control + C
```

### `.env`

`.env` exists and is open in the editor, but its contents were not discussed in detail.

Do not print or expose secrets.

### OpenAI Key Behavior

Some endpoints require a real OpenAI key for full behavior:

- LLM fallback categorization.
- Anomaly explanations.
- Insights.

However:

- The categorizer has safe behavior for missing `OPENAI_API_KEY`.
- Missing-key fallback should not write to `labeled_transactions`.

### API Endpoints Verified

Verified:

```text
GET /
POST /transactions/upload
GET /transactions
GET /anomalies
GET /unclassified
```

Not recently verified:

```text
GET /insights
```

Reason:

- It needs real transaction data and a working OpenAI key.

### Frontend Not Yet Verified

`frontend.html` exists and is open in the editor.

A future task should verify the frontend manually against the backend.

### Keep The Current Learning Rhythm

The user asked:

```text
Lets take it from the top.
Lets go line by line.
Give me one code at a time.
```

Continue that teaching style unless the user asks for a full code block.

### Avoid Accidentally Reverting User Work

There are local changes and untracked files.

Do not run destructive Git commands.

Do not reset `.gitignore`, `ml/`, `tests/`, or modified pipeline files without explicit permission.

