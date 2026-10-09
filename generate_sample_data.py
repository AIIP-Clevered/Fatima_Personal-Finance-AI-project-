"""
generate_sample_data.py
------------------------
Creates data/personal_transactions.csv with the SAME schema as the Kaggle
dataset (bukolafatunde/personal-finance):

    Date, Description, Amount, Transaction Type, Category, Account Name

Why this exists: the real Kaggle CSV needs a login to download. This script
builds a realistic stand-in with identical columns so the rest of the project
runs today. When you download the real file from Kaggle, just drop it in as
data/personal_transactions.csv and delete this script -- nothing else changes.

The data is deliberately shaped to tell a story: steady paycheck, fixed bills
every month, variable "fun" spending, and a couple of months where spending
creeps up. That gives the insight and recommendation code something real to say.
"""

import os
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)  # reproducible: same file every run

# Each category maps to the merchant names ("Description") you'd actually see
# on a statement. This mirrors how the real dataset links Description -> Category,
# which is exactly what the categorization model will learn.
CATEGORY_MERCHANTS = {
    "Groceries": ["Whole Foods", "Trader Joe's", "Kroger", "Safeway", "Aldi"],
    "Restaurants": ["Olive Garden", "Chipotle", "Local Bistro", "Thai Kitchen"],
    "Fast Food": ["McDonald's", "Burger King", "Wendy's", "Taco Bell"],
    "Coffee Shops": ["Starbucks", "Dunkin", "Blue Bottle Coffee"],
    "Gas & Fuel": ["Shell", "Chevron", "BP", "ExxonMobil"],
    "Shopping": ["Amazon", "Target", "Best Buy", "Macy's"],
    "Utilities": ["City Electric", "Metro Water", "Gas Utility"],
    "Internet": ["Comcast Xfinity"],
    "Mobile Phone": ["Verizon Wireless"],
    "Movies & DVDs": ["Netflix", "AMC Theatres"],
    "Music": ["Spotify"],
    "Alcohol & Bars": ["The Tap Room", "Downtown Bar"],
    "Home Improvement": ["Home Depot", "Lowe's", "IKEA"],
    "Auto Insurance": ["Geico"],
    "Haircut": ["Great Clips"],
    "Mortgage & Rent": ["Rent Payment"],
    "Paycheck": ["Employer Payroll"],
    "Credit Card Payment": ["Credit Card Payment"],
}

# Fixed monthly items: (Category, amount, account, transaction type).
# These repeat every month on a set day -- your recurring life.
FIXED_MONTHLY = [
    ("Paycheck", 4200.00, "Checking", "credit", 1),
    ("Paycheck", 4200.00, "Checking", "credit", 15),
    ("Mortgage & Rent", 1450.00, "Checking", "debit", 3),
    ("Utilities", 140.00, "Checking", "debit", 8),
    ("Internet", 70.00, "Checking", "debit", 8),
    ("Mobile Phone", 85.00, "Checking", "debit", 10),
    ("Auto Insurance", 120.00, "Checking", "debit", 12),
    ("Movies & DVDs", 15.49, "Silver Card", "debit", 5),   # Netflix
    ("Music", 10.99, "Silver Card", "debit", 5),           # Spotify
]

# Discretionary categories: how many times a month, and a (min, max) spend range.
# In "tight" months we bump the counts up to simulate lifestyle creep.
DISCRETIONARY = {
    "Groceries": (8, (25, 90)),
    "Restaurants": (4, (18, 65)),
    "Fast Food": (5, (8, 22)),
    "Coffee Shops": (10, (4, 9)),
    "Gas & Fuel": (4, (30, 60)),
    "Shopping": (3, (20, 180)),
    "Alcohol & Bars": (2, (25, 70)),
    "Home Improvement": (1, (15, 120)),
    "Haircut": (1, (20, 35)),
}

ACCOUNTS_FOR_SPEND = ["Silver Card", "Platinum Card", "Checking"]


def month_starts(start: date, n_months: int):
    y, m = start.year, start.month
    for _ in range(n_months):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m = 1
            y += 1


def build_rows(start: date, n_months: int):
    rows = []
    # months index 3, 4 (0-based) are "tight" -> more discretionary spend
    tight_months = {3, 4}

    for i, first in enumerate(month_starts(start, n_months)):
        # fixed items
        for cat, amt, acct, ttype, day in FIXED_MONTHLY:
            desc = random.choice(CATEGORY_MERCHANTS[cat])
            rows.append([first.replace(day=min(day, 28)), desc, amt, ttype, cat, acct])

        # one credit-card payment per month (money leaving checking)
        rows.append([first.replace(day=20), "Credit Card Payment", 600.00,
                     "debit", "Credit Card Payment", "Checking"])

        # discretionary items
        multiplier = 1.6 if i in tight_months else 1.0
        for cat, (count, (lo, hi)) in DISCRETIONARY.items():
            n = int(round(count * multiplier))
            for _ in range(n):
                day = random.randint(1, 28)
                desc = random.choice(CATEGORY_MERCHANTS[cat])
                amt = round(random.uniform(lo, hi), 2)
                acct = random.choice(ACCOUNTS_FOR_SPEND)
                rows.append([first.replace(day=day), desc, amt, "debit", cat, acct])

    return rows


def main():
    os.makedirs("data", exist_ok=True)
    rows = build_rows(date(2018, 1, 1), n_months=10)
    df = pd.DataFrame(
        rows,
        columns=["Date", "Description", "Amount", "Transaction Type",
                 "Category", "Account Name"],
    )
    df = df.sort_values("Date").reset_index(drop=True)
    # match the real file's date format (M/D/YYYY)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m/%d/%Y")
    out = os.path.join("data", "personal_transactions.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} transactions to {out}")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
