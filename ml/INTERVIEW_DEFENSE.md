# SmartBudget ML Tier — Interview Defense

Every design decision, the question an interviewer will ask about it, and your answer.
Numbers below are from a verified run on the 6,000-row synthetic dataset — **rerun
`python ml/generate_dataset.py && python ml/train_classifier.py` yourself so you've
personally seen every number before quoting it.**

---

## The elevator pitch (30 seconds)

"SmartBudget originally had two tiers: keyword rules resolving ~90% of transactions,
and GPT-4o-mini for the rest. Every LLM call is a ~1-second network round trip that
costs money. I distilled the LLM's classification behavior into a local model —
TF-IDF features into logistic regression, trained on the LLM's own past labels.
The classifier runs in ~3 milliseconds locally, and I use a confidence threshold of
0.85: above it, accept the local prediction; below it, defer to the LLM and log the
LLM's answer as new training data. So the LLM's share of traffic shrinks from ~10%
to ~1-2%, and the system gets cheaper as it runs."

---

## Q: Why is this "distillation"?

Distillation = training a small model to reproduce a large model's outputs. The
teacher is GPT-4o-mini; the student is TF-IDF + logistic regression. I never
hand-label anything — the labels ARE the LLM's past classifications, logged to
SQLite. This is the same pattern production teams use to cut LLM costs: pay for the
big model once per novel input, then serve the learned behavior locally forever.

## Q: Why TF-IDF and not embeddings / a neural net / fine-tuned BERT?

Three reasons:
1. **The data doesn't need it.** Merchant strings are short (3-8 tokens), the signal
   is almost entirely lexical ("zomato" → Food). There's no semantics or word order
   to model — the thing transformers are good at is absent here.
2. **Dataset size.** A few thousand examples. Fine-tuning BERT on that overfits and
   trains slower; TF-IDF + LR trains in ~3 seconds and is trivially retrainable
   nightly.
3. **Latency and dependencies.** Measured single-prediction latency: ~3.4 ms on CPU
   inside the FastAPI process. No GPU, no model server, no 400MB weights file.
   Right-sizing the model to the problem IS the engineering judgment being tested.

## Q: Why char n-grams (char_wb 3-5) in addition to word n-grams?

Bank feeds mangle merchant names: `ZOMATO`, `ZMT*ZOMATO`, `ZOMATOLTD`, truncations.
Word-level TF-IDF treats each variant as a different token — zero shared signal.
Character n-grams give them overlapping features (`zom`, `omat`, `mato`), so the
model generalizes across mangled variants of the same merchant. `char_wb` respects
word boundaries so n-grams don't bleed across the channel prefix into the name.
My dataset generator deliberately reproduces these manglings so the model trains
against realistic noise.

## Q: Why logistic regression and not LightGBM/XGBoost?

LR is the right default for high-dimensional sparse text features — linear models
handle 30k+ sparse dims natively, while trees must split on individual columns and
lose the additive nature of text evidence. LR also outputs usable probabilities for
thresholding and interpretable per-class coefficients (I can print "top features for
Food: zomato, swiggy, kfc" straight from the weights). I'd reach for LightGBM if the
numeric features (amount, timing) dominated the signal, since trees capture their
interactions and nonlinearities better; here text dominates. `class_weight="balanced"`
compensates for category imbalance (Food is ~18% of data, Income ~5%).

## Q: Why the amount/timing/channel features at all if text dominates?

Ambiguity cases. "UPI-RELIANCE..." truncated could be Jio recharge (₹299, monthly)
or Reliance Digital (₹45,000, one-off) — the amount disambiguates. EMI and salary
cluster in the first week of the month — `is_month_start` captures that. They're
cheap to compute and strictly add information; StandardScaler puts them on the same
footing as the TF-IDF weights.

## Q: Walk me through your evaluation.

Three separate measurements, because they answer different questions:

1. **Held-out test set (80/20 stratified split): 99.8% accuracy, 0.998 macro F1.**
   Answers: does the model reproduce the teacher's labels on merchant strings whose
   *merchants* it has seen (in other mangled forms)? Near-perfect, as expected —
   this task is mostly lexical lookup with noise.
2. **Unseen-merchant test set: 60% raw accuracy.** I held out entire merchants
   (never in training in any form). Raw accuracy drops hard — which is honest and
   expected: the model can't know "BIKANERVALA" is food from nothing.
3. **The threshold saves you: on unseen merchants at threshold 0.85, the model only
   accepts 10% of predictions — but is 98.4% accurate on those.** The other 90%
   defer to the LLM. **The model knows what it doesn't know.** That's the entire
   point of confidence-thresholded deferral, and it's the single best talking point
   in this project: I didn't just measure accuracy, I measured whether the
   *abstention mechanism* routes hard cases to the expensive expert correctly.

## Q: How did you choose the 0.85 threshold?

Swept it and looked at the coverage/accuracy tradeoff on held-out data:

| threshold | coverage | accuracy on covered | deferred to LLM |
|-----------|----------|--------------------:|-----------------|
| 0.50 | 99.6% | 99.92% | 0.4% |
| 0.70 | 96.9% | 100.0% | 3.1% |
| **0.85** | **90.3%** | **100.0%** | **9.7%** |
| 0.95 | 47.5% | 100.0% | 52.5% |

On seen merchants, even 0.5 would be fine — the threshold really exists for the
unseen-merchant case, where it correctly rejects 90% of guesses. 0.85 balances the
two: high coverage on known merchants, aggressive deferral on novel ones. In
production I'd tune this against LLM cost per call vs the business cost of a
misfiled transaction.

## Q: LR probabilities aren't calibrated. Is thresholding on them valid?

Fair pushback — LR probabilities are better-calibrated than most (its loss is a
proper scoring rule) but not perfect. Two answers: (1) the threshold isn't used as
a literal probability, it's a tuned operating point — I picked it from the measured
coverage/accuracy curve, so miscalibration is absorbed into the choice; (2) if
calibration mattered more (e.g., surfacing confidence to users), I'd wrap it in
`CalibratedClassifierCV` with isotonic regression. Knowing that tool exists is
usually the answer the interviewer wants.

## Q: The 99.8% looks suspiciously high. Is there leakage?

The honest answer: the random split puts different *transactions* of the same
*merchant* in train and test, so the test measures generalization across mangled
string variants — not across merchants. That's why I built the second, harder eval
(held-out merchants), where accuracy is 60% raw / 98.4% on high-confidence. Always
volunteer this before the interviewer finds it — it converts a weakness into
evidence you understand evaluation design. Also note the labels are synthetic
(generator, not real LLM logs), so held-out numbers are optimistic vs production;
the architecture and eval methodology are what transfer.

## Q: What happens as new merchants appear over time (drift)?

That's the retraining loop: every LLM classification is appended to the labeled
table in SQLite. A retraining job (cron / on-demand endpoint) refits the pipeline —
3 seconds of training — and hot-swaps `model.joblib`. New merchants migrate from
Tier 3 to Tier 2 automatically after appearing a handful of times. Metrics to watch
in production: Tier 2 coverage rate (should rise), deferral rate (should fall), and
disagreement rate between Tier 2 predictions and LLM labels on a shadow sample
(should stay low — if it rises, the teacher and student have drifted apart).

## Q: What are the failure modes?

- **Confidently wrong on adversarial lookalikes**: a new merchant whose name shares
  n-grams with a known one ("ZOMOLAND" toy store → Food). Mitigation: the shadow
  disagreement metric above, plus user correction feedback as gold labels.
- **Teacher errors are inherited**: if GPT-4o-mini mislabels a merchant
  consistently, the student learns the mistake. Distillation caps you at teacher
  quality; user corrections are the only way above it.
- **Class imbalance on rare categories**: handled with class_weight, but rare
  categories will have lower per-class recall in production; monitor per-class F1.

## Q: Latency numbers?

Measured: ~3.4 ms per single-transaction predict_proba on CPU (includes pandas
DataFrame construction overhead — batch prediction amortizes to ~microseconds per
row). LLM round trip: ~1,000 ms. So Tier 2 is ~300x faster per transaction even
unbatched, with zero network dependency and zero marginal cost.

## Q: How is this deployed inside the app?

The model is a joblib artifact loaded once at FastAPI startup (module-level or
lifespan event), so requests pay only inference cost, not load cost. The classifier
is a pure function of the transaction — no state — so it's thread-safe for FastAPI's
concurrency. Tests mock it exactly like the existing GPT mock in test_categorizer.py.

---

## Terms you must be able to define cold

TF (term frequency), IDF and why log-scaled, n-gram, char_wb vs char, min_df,
sparse matrix (CSR), one-vs-rest vs multinomial LR, softmax, regularization C,
stratified split, macro vs weighted F1, precision/recall tradeoff, calibration,
knowledge distillation, coverage/risk (selective prediction), concept drift.
