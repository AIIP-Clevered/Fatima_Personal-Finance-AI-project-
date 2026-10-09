"""
insights.py
-----------
Feature 2: turn raw transactions into a few useful facts about the money.

Nothing here is machine learning -- it's honest accounting with pandas. That's
the point: a good "AI finance" tool is mostly clear arithmetic, and only uses ML
where it actually helps (the categorization step). Don't dress up a sum as AI.

What it computes:
  - total income, total spending, net savings, savings rate
  - spending broken down by category (where the money goes)
  - month-by-month income vs spending trend
  - overspending flags: categories where a month runs well above that
    category's own typical month
"""

import numpy as np
import pandas as pd

# Real gotcha in this dataset: BOTH paychecks and credit-card payments are
# recorded as "credit". A credit-card payment is just moving money to pay the
# card, not new income -- counting it would overstate income by ~$30k here.
# So income = credits that are NOT a transfer category. Same exclusion on the
# spending side stops us double-counting the card purchases.
INCOME_TYPE = "credit"
TRANSFER_CATEGORIES = {"Credit Card Payment"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df


def summary(df: pd.DataFrame) -> dict:
    df = prepare(df)

    income_mask = (df["Transaction Type"] == INCOME_TYPE) & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    income = df.loc[income_mask, "Amount"].sum()

    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    spending = df.loc[spend_mask, "Amount"].sum()

    net = income - spending
    savings_rate = (net / income) if income else 0.0

    return {
        "total_income": round(income, 2),
        "total_spending": round(spending, 2),
        "net_savings": round(net, 2),
        "savings_rate": savings_rate,
        "n_months": df["Month"].nunique(),
    }


def spending_by_category(df: pd.DataFrame) -> pd.Series:
    df = prepare(df)
    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    return (
        df.loc[spend_mask]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
    )


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare(df)
    income_mask = (df["Transaction Type"] == INCOME_TYPE) & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    income = df.loc[income_mask].groupby("Month")["Amount"].sum()
    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    spend = df.loc[spend_mask].groupby("Month")["Amount"].sum()

    trend = pd.DataFrame({"income": income, "spending": spend}).fillna(0)
    trend["net"] = (trend["income"] - trend["spending"]).round(2)
    return trend.round(2)


def overspending_flags(df: pd.DataFrame, threshold: float = 1.4):
    """
    Flag (month, category) pairs where spend is >= threshold x that category's
    median month. threshold=1.4 means "40% above a normal month for you".
    Returns a list of dicts, worst first.
    """
    df = prepare(df)
    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    monthly = (
        df.loc[spend_mask]
        .groupby(["Category", "Month"])["Amount"]
        .sum()
        .reset_index()
    )
    typical = monthly.groupby("Category")["Amount"].median()

    flags = []
    for _, row in monthly.iterrows():
        base = typical[row["Category"]]
        if base > 0 and row["Amount"] >= threshold * base:
            flags.append({
                "month": row["Month"],
                "category": row["Category"],
                "spent": round(row["Amount"], 2),
                "typical": round(base, 2),
                "over_by": f"{(row['Amount'] / base - 1):.0%}",
            })
    flags.sort(key=lambda f: f["spent"] - f["typical"], reverse=True)
    return flags


def budget_vs_actual(df: pd.DataFrame, budget_path: str = "data/Budget.csv"):
    """
    Compare average monthly spend per category against the budget in Budget.csv.
    Returns a DataFrame sorted by the biggest dollar overspend. This is more
    honest than a generic 'typical month' because it's the user's own target.
    """
    import os
    if not os.path.exists(budget_path):
        return None

    df = prepare(df)
    n_months = df["Month"].nunique()
    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES)
    )
    avg_monthly = (
        df.loc[spend_mask].groupby("Category")["Amount"].sum() / n_months
    )

    budget = pd.read_csv(budget_path).set_index("Category")["Budget"]
    table = pd.DataFrame({"avg_monthly_spend": avg_monthly.round(2),
                          "budget": budget})
    table = table.dropna(subset=["budget"])
    table["over_budget"] = (table["avg_monthly_spend"] - table["budget"]).round(2)
    # use numpy NaN (a float) for 0-budget categories so the column stays numeric
    # and .round() works; pd.NA would make it an object column and break round()
    safe_budget = table["budget"].replace(0, np.nan)
    table["pct_of_budget"] = (table["avg_monthly_spend"] / safe_budget).round(2)
    return table.sort_values("over_budget", ascending=False)


if __name__ == "__main__":
    df = pd.read_csv("data/personal_transactions.csv")

    s = summary(df)
    print("=== Summary ===")
    print(f"Income:   ${s['total_income']:,.2f}")
    print(f"Spending: ${s['total_spending']:,.2f}")
    print(f"Savings:  ${s['net_savings']:,.2f}  ({s['savings_rate']:.1%} of income)")
    print(f"Over {s['n_months']} months\n")

    print("=== Top spending categories ===")
    print(spending_by_category(df).head(8).to_string(), "\n")

    print("=== Monthly trend ===")
    print(monthly_trend(df).to_string(), "\n")

    print("=== Overspending flags (vs your own typical month) ===")
    for f in overspending_flags(df)[:6]:
        print(f"{f['month']}  {f['category']:16s} ${f['spent']:>8,.2f} "
              f"(typical ${f['typical']:,.2f}, +{f['over_by']})")

    print("\n=== Budget vs actual (avg month) ===")
    bva = budget_vs_actual(df)
    if bva is not None:
        print(bva.head(8).to_string())
