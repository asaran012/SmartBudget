"""
Generate a synthetic labeled transaction dataset for SmartBudget's ML tier.

Simulates what the real system accumulates over time: transactions that the
LLM (GPT-4o-mini) classified, logged as (raw_description, amount, date,
channel, category). Until months of real LLM logs exist, this bootstrap
dataset lets us train and evaluate the distilled classifier immediately.

Design choices (interview-relevant):
- Merchant strings are deliberately MANGLED the way real bank feeds mangle
  them: channel prefixes (UPI-/POS/NEFT), random VPA handles, ref numbers,
  truncation, aggregator prefixes (ZMT*, SWG*). This is why char n-grams
  matter in the vectorizer.
- Amounts are drawn from per-category log-normal-ish distributions, because
  spend amounts are right-skewed (many small, few large).
- EMI/rent cluster at the start of the month; salary on fixed dates. These
  timing signals are real features the model can learn.
- ~10% of merchants are held out as "unseen" in a separate file to test
  generalization to brand-new merchants (the actual production scenario).
"""

import csv
import random
from datetime import date, timedelta

random.seed(42)

# category -> list of (merchant_display_name, typical_amount_range)
MERCHANTS = {
    "Dining": [
        ("ZOMATO", (120, 900)), ("SWIGGY", (120, 900)), ("DOMINOS PIZZA", (250, 800)),
        ("MCDONALDS", (150, 600)), ("KFC", (200, 700)), ("HALDIRAM", (100, 500)),
        ("BARBEQUE NATION", (800, 3000)), ("CAFE COFFEE DAY", (100, 400)),
        ("WOW MOMO", (100, 350)), ("EATCLUB", (150, 500)), ("FAASOS", (120, 450)),
        ("BURGER KING", (150, 550)), ("PIZZA HUT", (300, 900)),
    ],
    "Groceries": [
        ("SWIGGY INSTAMART", (150, 1200)), ("BLINKIT", (100, 1000)), ("ZEPTO", (100, 900)),
        ("BIGBASKET", (400, 2500)), ("DMART", (500, 3000)), ("RELIANCE FRESH", (200, 1500)),
        ("MORE SUPERMARKET", (200, 1500)), ("NATURES BASKET", (300, 2000)),
        ("JIOMART", (300, 2000)), ("SPENCERS RETAIL", (200, 1500)),
    ],
    "Transport": [
        ("UBER INDIA", (80, 500)), ("OLA CABS", (80, 500)), ("RAPIDO BIKE", (40, 200)),
        ("IRCTC", (300, 2500)), ("DELHI METRO RAIL", (20, 150)), ("BLUSMART", (150, 600)),
        ("REDBUS", (400, 1800)), ("INDIAN OIL PETROL", (500, 3000)),
        ("HP PETROL PUMP", (500, 3000)), ("FASTAG RECHARGE", (200, 1000)),
    ],
    "Shopping": [
        ("AMAZON PAY INDIA", (200, 5000)), ("FLIPKART", (200, 5000)), ("MYNTRA", (400, 4000)),
        ("AJIO", (400, 3500)), ("NYKAA", (300, 2500)), ("DECATHLON", (500, 5000)),
        ("CROMA RETAIL", (1000, 40000)), ("RELIANCE DIGITAL", (1000, 40000)),
        ("MEESHO", (150, 1200)), ("SNITCH", (500, 2500)),
        ("UDEMY ONLINE COURSE", (400, 3000)), ("COURSERA", (2000, 5000)),
        ("LEETCODE PREMIUM", (2500, 13000)), ("INTERVIEWBIT", (1000, 5000)),
        ("SCALER ACADEMY", (5000, 30000)), ("BOOKS KHARIDO STORE", (300, 1500)),
    ],
    "Utilities & Bills": [
        ("JIO RECHARGE", (149, 999)), ("AIRTEL PAYMENTS", (149, 999)),
        ("VODAFONE IDEA", (149, 999)), ("BSES RAJDHANI POWER", (500, 4000)),
        ("TATA POWER DDL", (500, 4000)), ("INDRAPRASTHA GAS", (400, 1500)),
        ("ACT FIBERNET", (700, 1500)), ("TATA PLAY DTH", (300, 800)),
        ("DELHI JAL BOARD", (200, 800)), ("WATER BILL", (200, 800)),
    ],
    "Subscriptions": [
        ("NETFLIX", (199, 649)), ("SPOTIFY INDIA", (119, 199)),
        ("HOTSTAR", (299, 1499)), ("SONY LIV", (299, 999)),
        ("YOUTUBE PREMIUM", (129, 189)), ("APPLE MUSIC", (99, 199)),
        ("AMAZON PRIME", (299, 1499)),
    ],
    "Entertainment": [
        ("BOOKMYSHOW", (200, 1500)), ("PVR CINEMAS", (300, 1200)),
        ("INOX LEISURE", (300, 1200)), ("STEAM PURCHASE", (300, 3000)),
        ("PLAYSTATION STORE", (500, 4000)), ("XBOX STORE", (500, 4000)),
    ],
    "Health & Fitness": [
        ("APOLLO PHARMACY", (100, 1500)), ("PHARMEASY", (150, 1500)), ("TATA 1MG", (150, 1500)),
        ("MAX HOSPITAL", (500, 10000)), ("DR LAL PATHLABS", (400, 3000)),
        ("CULTFIT", (500, 3000)), ("CULT FIT", (500, 3000)),
        ("NETMEDS", (150, 1200)), ("PRACTO", (300, 800)),
    ],
    "Rent": [
        ("RENT TRANSFER LANDLORD", (10000, 30000)),
        ("NOBROKER RENT PAY", (10000, 30000)),
        ("CRED RENT PAYMENT", (10000, 30000)),
    ],
    "Income": [
        ("ASPECT RATIO SALARY", (50000, 90000)), ("NEFT INWARD REMITTANCE", (1000, 20000)),
        ("INTEREST CREDIT SB AC", (100, 2000)), ("IMPS INWARD P2P", (500, 10000)),
        ("FD MATURITY PROCEEDS", (10000, 100000)),
    ],
    "Transfers & Payments": [
        ("HDFC HOME LOAN EMI", (15000, 35000)), ("BAJAJ FINSERV EMI", (2000, 8000)),
        ("HDFC CREDIT CARD AUTOPAY", (3000, 25000)), ("SBI PERSONAL LOAN EMI", (5000, 15000)),
        ("PHONEPE UPI", (100, 20000)), ("GPAY UPI", (100, 20000)),
        ("PAYTM UPI", (100, 20000)), ("CRED PAYMENT", (500, 30000)),
        ("NEFT TRANSFER", (1000, 50000)), ("IMPS TRANSFER", (500, 30000)),
    ],
}

# Merchants excluded from training and used only in the "unseen" test file.
# Same categories, brand-new strings — tests generalization, the production case.
UNSEEN_MERCHANTS = {
    "Dining": [("BIKANERVALA", (150, 700)), ("SUBWAY", (200, 600)), ("THEOBROMA", (300, 1200))],
    "Groceries": [("STAR BAZAAR", (300, 2000)), ("VISHAL MEGA MART", (200, 1500))],
    "Transport": [("MERU CABS", (150, 700)), ("BHARAT PETROLEUM", (500, 3000))],
    "Shopping": [("SHOPPERS STOP", (800, 6000)), ("PANTALOONS", (600, 4000)),
                 ("GEEKSFORGEEKS COURSE", (1000, 6000))],
    "Utilities & Bills": [("MTNL DELHI", (200, 900)), ("ADANI ELECTRICITY", (500, 4000))],
    "Subscriptions": [("AMAZON PRIME VIDEO", (299, 1499)), ("ZEE5", (299, 999))],
    "Entertainment": [("CINEPOLIS", (300, 1100)), ("EVENT TICKET", (500, 2500))],
    "Health & Fitness": [("FORTIS HOSPITAL", (500, 10000)), ("WELLNESS FOREVER", (100, 1200))],
    "Rent": [("BROKER RENT TRANSFER", (10000, 30000))],
    "Income": [("RTGS INWARD CREDIT", (5000, 50000))],
    "Transfers & Payments": [("ICICI HOME LOAN EMI", (15000, 35000)),
                             ("BHIM UPI", (100, 20000)), ("BANK TRANSFER", (1000, 50000))],
}

CATEGORY_WEIGHTS = {
    "Rent": 6,
    "Income": 5,
    "Dining": 18,
    "Groceries": 14,
    "Transport": 12,
    "Subscriptions": 8,
    "Shopping": 12,
    "Health & Fitness": 7,
    "Utilities & Bills": 10,
    "Entertainment": 8,
    "Transfers & Payments": 8,
}

VPA_HANDLES = ["@paytm", "@ybl", "@okaxis", "@oksbi", "@ibl", "@upi", "@apl"]
AGGREGATOR_PREFIX = {"ZOMATO": "ZMT*", "SWIGGY": "SWG*", "AMAZON PAY INDIA": "AMZN*",
                     "NETFLIX": "NFLX*", "FLIPKART": "FKRT*"}


def mangle(merchant: str, channel: str) -> str:
    """Turn a clean merchant name into a realistic bank-feed string."""
    name = merchant
    r = random.random()
    if merchant in AGGREGATOR_PREFIX and r < 0.25:
        name = AGGREGATOR_PREFIX[merchant] + merchant.replace(" ", "")
    elif r < 0.40:
        name = name.replace(" ", "")            # ZOMATOLTD style squashing
    elif r < 0.55:
        name = name[: random.randint(6, max(7, len(name) - 2))]  # truncation

    suffix = random.choice(["", " LTD", " PVT LTD", " INDIA", ""])
    ref = random.randint(1000, 999999)

    if channel == "UPI":
        return f"UPI-{name}{suffix}-{ref}{random.choice(VPA_HANDLES)}"
    if channel == "POS":
        return f"POS {ref} {name}{suffix} {random.choice(['DEL', 'BLR', 'MUM', 'NOIDA', 'GZB'])}"
    if channel == "NEFT":
        return f"NEFT-{name}{suffix}-N{ref}"
    if channel == "IMPS":
        return f"IMPS-{ref}-{name}{suffix}"
    return f"ACH-{name}{suffix}-{ref}"


def pick_channel(category: str) -> str:
    if category == "Income":
        return random.choice(["NEFT", "IMPS", "ACH"])
    if category == "Rent":
        return random.choice(["NEFT", "ACH", "UPI", "IMPS"])
    return random.choices(["UPI", "POS", "NEFT", "IMPS"], weights=[65, 25, 5, 5])[0]


def pick_date(category: str, start: date, days: int) -> date:
    d = start + timedelta(days=random.randint(0, days - 1))
    if category in ("Rent", "Income"):        # cluster near month start
        d = d.replace(day=random.randint(1, 7))
    return d


def sample_amount(lo: float, hi: float) -> float:
    """Right-skewed draw: most spends near the low end, occasional big ones."""
    u = random.random() ** 2
    return round(lo + u * (hi - lo), 2)


def generate(merchant_map, n_rows, out_path):
    cats = list(merchant_map.keys())
    weights = [CATEGORY_WEIGHTS[cat] for cat in cats]
    rows = []
    start = date(2025, 7, 1)
    for _ in range(n_rows):
        cat = random.choices(cats, weights=weights)[0]
        merchant, (lo, hi) = random.choice(merchant_map[cat])
        channel = pick_channel(cat)
        d = pick_date(cat, start, 365)
        rows.append({
            "date": d.isoformat(),
            "raw_description": mangle(merchant, channel),
            "amount": sample_amount(lo, hi),
            "channel": channel,
            "category": cat,   # in production this label comes from the LLM
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    generate(MERCHANTS, 6000, "ml/labeled_transactions.csv")
    generate(UNSEEN_MERCHANTS, 600, "ml/unseen_merchants.csv")
