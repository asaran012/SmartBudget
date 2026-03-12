# DEV_LOG.md — AI Assistant Usage in Development

> How I used **ChatGPT (GPT-4o)** as an AI assistant while building SmartBudget.
> This documents *when* and *why* AI was useful — not just that I used it.

---

## Project: SmartBudget Pipeline
**Stack:** Python · FastAPI · SQLite · OpenAI API · pytest  
**What it does:** Ingests bank CSV exports → AI categorizes transactions → detects anomalies → generates spending insights

---

## Session 1 — Architecture

**Question asked:**
> "I'm building a personal finance backend: parse CSVs, AI-categorize transactions, detect anomalies, generate insights. Monolith, modular monolith, or microservices?"

**ChatGPT's answer:** Modular monolith. Each pipeline stage as its own module with a clean interface — architectural clarity without ops overhead.

**What I used:** The exact folder layout (`pipeline/csv_parser → categorizer → anomaly_detector → insights`).  
**What I changed:** Skipped its Celery suggestion for async processing — overkill for a personal project.

---

## Session 2 — Multi-Bank CSV Formats

**Question asked:**
> "What are the exact CSV column names Chase, BoFA, Wells Fargo, and Capital One use in their transaction exports?"

**ChatGPT's output:** Column name breakdown per bank — including that Wells Fargo uses *separate* Debit and Credit columns instead of a single signed Amount.

**Direct code impact:** The `DATE_COLS`, `DESC_COLS`, `DEBIT_COLS`, `CREDIT_COLS` alias lists in `csv_parser.py`, and the `_resolve_amount()` function that handles both formats.

**What I verified myself:** Checked two bank export docs to confirm column names — two were slightly off in ChatGPT's answer. Fixed them.

---

## Session 3 — Anomaly Detection Algorithm

**Question asked:**
> "For detecting anomalies in personal finance transaction data — standard deviation or IQR? What are the tradeoffs?"

**ChatGPT's recommendation:** IQR (Interquartile Range). Personal spending is heavily skewed — one big rent payment or flight distorts the mean and makes standard deviation unreliable. IQR is robust to these outliers.

**Key architectural advice from this session:**
> "Use deterministic statistics for detection, GPT only for explanation. A black box that decides what's anomalous isn't auditable or debuggable."

**Direct code impact:** This principle shaped all of `anomaly_detector.py`. The `_iqr_threshold()`, `_compute_thresholds()`, and `_flag_transactions()` functions are pure Python with no AI. GPT only runs in `_attach_explanations()`.

**What I adjusted:** ChatGPT suggested a 3x multiplier above Q3. I dropped it to 2.5x after testing — 3x was missing real anomalies in smaller datasets.

---

## Session 4 — Categorization Cost Optimization

**Question asked:**
> "I need to categorize every bank transaction using GPT. How do I avoid calling the API for every single row?"

**ChatGPT's two-pass strategy:**
1. Keyword matching handles ~80% of transactions for free
2. Batch remaining uncategorized items in one API call as a JSON array
3. Use `gpt-4o-mini` instead of `gpt-4o` for classification — 10x cheaper, equally accurate for this task

**Result:** ~90% reduction in API calls vs naive per-transaction approach.  
**The keyword map** in `categorizer.py` was co-authored with ChatGPT — I gave it the category names, it suggested keyword lists, I reviewed and corrected them.

---

## Session 5 — Unit Test Strategy

**Question asked:**
> "What's the right strategy for unit testing a pipeline that has external API calls (OpenAI)?"

**ChatGPT's advice:**
- Test pure logic (CSV parsing, IQR math, keyword matching) without any mocking
- Use `unittest.mock.patch` to mock the OpenAI client for tests that touch AI code
- Test the *behavior* when GPT fails — does the pipeline degrade gracefully?
- Use `tmp_path` pytest fixture for DB tests to avoid touching the real database

**Direct code impact:** The `patch("pipeline.categorizer.client")` pattern throughout `test_categorizer.py` and `test_anomaly_detector.py`. The `use_temp_db` fixture in `test_db.py`.

---

## What ChatGPT Was NOT Used For

- CSV multi-format detection logic — I designed the fallback chain myself
- The pipeline stage sequencing — my architectural decision
- SQL schema — I wrote it myself; ChatGPT offered but I wanted to own it
- Error handling — all try/catch logic is mine
- This DEV_LOG — written by me

---

## Honest Assessment

**Where ChatGPT genuinely saved time:**
- Bank CSV format research (would have been 1+ hour of manual digging)
- IQR vs std dev question (immediate authoritative answer with reasoning)
- The "stats for detection, GPT for explanation" principle — I wouldn't have separated these without that nudge

**Where I pushed back:**
- ChatGPT kept suggesting more complex solutions (Celery, RAG, vector databases) that added no value at this scale
- Its first anomaly threshold suggestion was too conservative — needed adjustment based on real testing

**Net:** ChatGPT worked best as a domain expert I could query with specific technical questions, not as a code generator.
