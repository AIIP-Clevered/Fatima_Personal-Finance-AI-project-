"""
forecast.py
-----------
Extra feature: project the next few months of cash flow and savings.

Why not Prophet (which the companion notebook uses)? With only ~21 monthly
data points, Prophet's seasonality fitting is shaky and it's a heavy install.
For this size, a linear trend on *winsorized* monthly spending plus a
recent-average income projection is more robust and, importantly, explainable.
If you ever get years of daily data, Prophet becomes worth it.

The one trick worth understanding: two months here are renovation blowouts
(~$8-9k). If you fit a trend straight through them, the forecast thinks spending
is exploding. So we winsorize -- cap each month's spending at the 90th percentile
before fitting the trend -- so a couple of one-off spikes don't warp the outlook.
We still report the real numbers; we only tame them for the trend fit.
"""

import numpy as np
import pandas as pd

TRANSFER_CATEGORIES = {"Credit Card Payment"}


def _monthly_series(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M")

    income_mask = (df["Transaction Type"] == "credit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES))
    spend_mask = (df["Transaction Type"] == "debit") & (
        ~df["Category"].isin(TRANSFER_CATEGORIES))

    income = df.loc[income_mask].groupby("Month")["Amount"].sum()
    spend = df.loc[spend_mask].groupby("Month")["Amount"].sum()

    idx = pd.period_range(min(income.index.min(), spend.index.min()),
                          max(income.index.max(), spend.index.max()), freq="M")
    income = income.reindex(idx, fill_value=0)
    spend = spend.reindex(idx, fill_value=0)
    return income, spend


def _winsorize(series, upper_pct=90):
    cap = np.percentile(series.values, upper_pct)
    return series.clip(upper=cap)


def forecast_cashflow(df, months_ahead=6, current_cash_savings=5000.0):
    """
    Returns a dict with:
      history:  list of {month, income, spending, net}
      forecast: list of {month, income, spending, net, projected_savings}
      notes:    short strings describing the method / caveats
    """
    income, spend = _monthly_series(df)
    n = len(income)
    x = np.arange(n)

    # income: recent 6-month average (stable, no strong trend)
    recent = min(6, n)
    income_fc = float(income.iloc[-recent:].mean())

    # spending: linear trend on winsorized history
    spend_w = _winsorize(spend).values
    slope, intercept = np.polyfit(x, spend_w, 1)
    # don't let the trend go negative
    spend_fc = [max(0.0, slope * (n + i) + intercept) for i in range(months_ahead)]

    last_month = income.index[-1]
    future_months = pd.period_range(last_month + 1, periods=months_ahead, freq="M")

    history = [
        {"month": str(m), "income": round(float(income[m]), 2),
         "spending": round(float(spend[m]), 2),
         "net": round(float(income[m] - spend[m]), 2)}
        for m in income.index
    ]

    forecast = []
    running = current_cash_savings
    for i, m in enumerate(future_months):
        net = income_fc - spend_fc[i]
        running += net
        forecast.append({
            "month": str(m),
            "income": round(income_fc, 2),
            "spending": round(spend_fc[i], 2),
            "net": round(net, 2),
            "projected_savings": round(running, 2),
        })

    trend_word = "rising" if slope > 5 else "falling" if slope < -5 else "roughly flat"
    notes = [
        f"Income projected as the last {recent}-month average (${income_fc:,.0f}/mo).",
        f"Spending trend is {trend_word} (${slope:+,.0f}/mo) after taming outlier months.",
        f"Starting from ${current_cash_savings:,.0f} cash, projected to "
        f"${forecast[-1]['projected_savings']:,.0f} in {months_ahead} months.",
    ]
    return {"history": history, "forecast": forecast, "notes": notes}


if __name__ == "__main__":
    df = pd.read_csv("data/personal_transactions.csv")
    out = forecast_cashflow(df)
    print("=== Next 6 months (forecast) ===")
    for row in out["forecast"]:
        print(f"  {row['month']}  income ${row['income']:>7,.0f}  "
              f"spend ${row['spending']:>7,.0f}  net ${row['net']:>+8,.0f}  "
              f"savings ${row['projected_savings']:>9,.0f}")
    print("\nNotes:")
    for nnote in out["notes"]:
        print(" -", nnote)
