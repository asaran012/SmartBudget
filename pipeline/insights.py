"""
pipeline/insights.py
---------------------
Generate a narrative spending summary using GPT-4o.

AI Assistant Usage:
    Iterated the prompt ~5 times with ChatGPT feedback. Key improvement:
    pre-aggregate data into a summary dict before sending to GPT instead
    of dumping raw rows — cheaper, faster, and produces better answers.
"""

import os
import json
from collections import defaultdict
import logging
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)


def generate_insights(transactions: list[dict]) -> dict:
    logger.info(f"Generating insights for {len(transactions)} transactions...")
    expenses = [t for t in transactions if t["amount"] < 0]
    income   = [t for t in transactions if t["amount"] > 0]

    by_category: dict[str, float] = defaultdict(float)
    by_month:    dict[str, float] = defaultdict(float)

    for t in expenses:
        by_category[t.get("category", "Other")] += abs(t["amount"])
        by_month[t["date"][:7]]                 += abs(t["amount"])

    dates         = sorted(t["date"] for t in transactions)
    total_spent   = round(sum(abs(t["amount"]) for t in expenses), 2)
    total_income  = round(sum(t["amount"] for t in income), 2)
    top_cat       = max(by_category.items(), key=lambda x: x[1], default=("N/A", 0))
    biggest       = max(expenses, key=lambda t: abs(t["amount"]), default=None)

    # Month-over-month note
    months = sorted(by_month)
    mom    = ""
    if len(months) >= 2:
        prev, last = by_month[months[-2]], by_month[months[-1]]
        pct = ((last - prev) / prev * 100) if prev else 0
        mom = f"Spending {'up' if pct > 0 else 'down'} {abs(pct):.0f}% from {months[-2]} (${prev:.0f}) to {months[-1]} (${last:.0f})."

    summary = {
        "period":            f"{dates[0]} → {dates[-1]}" if dates else "unknown",
        "total_spent":       total_spent,
        "total_income":      total_income,
        "transaction_count": len(transactions),
        "by_category":       {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: -x[1])},
        "monthly_totals":    {k: round(v, 2) for k, v in sorted(by_month.items())},
        "top_category":      {"name": top_cat[0], "amount": round(top_cat[1], 2)},
        "biggest_charge":    {"description": biggest["description"], "amount": round(abs(biggest["amount"]), 2), "date": biggest["date"]} if biggest else None,
        "month_over_month":  mom,
    }

    prompt = prompt = f"""
You are a sharp personal finance analyst reviewing spending data for a young Indian professional. 
All amounts are in Indian Rupees (₹).

Financial summary:
{summary}

Write a short 3–5 line insight report.

Focus on:
• notable spending patterns or trends
• categories that look unusually high
• any single transaction that stands out
• a practical suggestion based on the numbers

Use the actual numbers when possible. Avoid generic advice like "save more money".
Be concise and insightful.
"""
    logger.info("Calling GPT for spending narrative...")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise personal finance coach helping young Indian professionals manage their spending. Give concise, actionable insights."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=400,
        temperature=0.7,
    )

    logger.info("Insights generated successfully")
    return {
        "generated_at": datetime.now().isoformat(),
        "summary":      summary,
        "ai_insight":   resp.choices[0].message.content.strip(),
    }