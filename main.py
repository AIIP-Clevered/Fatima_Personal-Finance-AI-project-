"""
main.py
-------
Runs the whole pipeline end to end and saves charts to outputs/.

    python main.py

Order:
  1. Load transactions (data/personal_transactions.csv)
  2. Train the categorization model and report accuracy
  3. Compute the money summary + insights
  4. Produce the investment recommendation
  5. Save three charts

If data/personal_transactions.csv is missing, run
`python generate_sample_data.py` first (or drop in the real Kaggle CSV).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write image files without needing a display
import matplotlib.pyplot as plt

from categorization import train_categorizer
from insights import (summary, spending_by_category, monthly_trend,
                      overspending_flags, budget_vs_actual)
from recommend import recommend
from features import (detect_subscriptions, detect_anomalies,
                      financial_health_score)
from forecast import forecast_cashflow

# how much cash you currently hold (for emergency-fund + score math).
# In a real app this would come from a linked account balance.
CURRENT_CASH_SAVINGS = 5000

DATA = "data/personal_transactions.csv"
OUT = "outputs"


def hr(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_charts(df, score=None, grade=None, breakdown=None, fc=None):
    os.makedirs(OUT, exist_ok=True)

    # 1. Spending by category (top 10)
    cats = spending_by_category(df).head(10)[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(cats.index, cats.values, color="#3b7ea1")
    plt.title("Where the money goes (top 10 categories)")
    plt.xlabel("Total spent ($)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/spending_by_category.png", dpi=120)
    plt.close()

    # 2. Monthly income vs spending
    trend = monthly_trend(df)
    plt.figure(figsize=(11, 5))
    x = range(len(trend))
    plt.plot(x, trend["income"], marker="o", label="Income", color="#2e7d32")
    plt.plot(x, trend["spending"], marker="o", label="Spending", color="#c62828")
    plt.fill_between(x, trend["income"], trend["spending"],
                     where=(trend["income"] >= trend["spending"]),
                     color="#2e7d32", alpha=0.12)
    plt.fill_between(x, trend["income"], trend["spending"],
                     where=(trend["income"] < trend["spending"]),
                     color="#c62828", alpha=0.12)
    plt.xticks(list(x), trend.index, rotation=45, ha="right")
    plt.title("Monthly income vs spending")
    plt.ylabel("$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/monthly_trend.png", dpi=120)
    plt.close()

    # 3. Budget vs actual
    bva = budget_vs_actual(df)
    if bva is not None:
        top = bva.head(10).sort_values("over_budget")
        colors = ["#c62828" if v > 0 else "#2e7d32" for v in top["over_budget"]]
        plt.figure(figsize=(9, 5))
        plt.barh(top.index, top["over_budget"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Budget vs actual (avg month)  -  red = over budget")
        plt.xlabel("Dollars over / under budget per month")
        plt.tight_layout()
        plt.savefig(f"{OUT}/budget_vs_actual.png", dpi=120)
        plt.close()

    # 4. Financial health score (a simple horizontal gauge + breakdown)
    if score is not None and breakdown is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4),
                                       gridspec_kw={"height_ratios": [1, 2]})
        # gauge
        color = "#2e7d32" if score >= 70 else "#f9a825" if score >= 55 else "#c62828"
        ax1.barh([0], [score], color=color)
        ax1.barh([0], [100], color="#eeeeee", zorder=0)
        ax1.set_xlim(0, 100)
        ax1.set_yticks([])
        ax1.set_title(f"Financial health score: {score}/100  (grade {grade})")
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # breakdown
        parts = {"Savings rate\n(/50)": breakdown["savings_rate_pts"],
                 "Budget\ndiscipline (/30)": breakdown["budget_discipline_pts"],
                 "Emergency\nfund (/20)": breakdown["emergency_fund_pts"]}
        maxes = [50, 30, 20]
        names = list(parts.keys())
        vals = list(parts.values())
        ax2.bar(names, maxes, color="#eeeeee")
        ax2.bar(names, vals, color="#3b7ea1")
        ax2.set_ylabel("points")
        ax2.set_title("Score breakdown")
        plt.tight_layout()
        plt.savefig(f"{OUT}/health_score.png", dpi=120)
        plt.close()

    # 5. Cash-flow forecast: history + projected savings
    if fc is not None:
        hist_m = [h["month"] for h in fc["history"]]
        hist_net = [h["net"] for h in fc["history"]]
        fut_m = [f["month"] for f in fc["forecast"]]
        fut_sav = [f["projected_savings"] for f in fc["forecast"]]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(hist_m)), hist_net, color="#90a4ae",
               label="Actual monthly net")
        off = len(hist_m)
        ax.plot(range(off, off + len(fut_m)), fut_sav, marker="o",
                color="#3b7ea1", label="Projected cumulative savings")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(range(len(hist_m) + len(fut_m)))
        ax.set_xticklabels(hist_m + fut_m, rotation=90, fontsize=7)
        ax.set_title("Cash flow: history (bars) and 6-month savings projection (line)")
        ax.set_ylabel("$")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT}/forecast.png", dpi=120)
        plt.close()

    print(f"Charts saved to {OUT}/")


def main():
    if not os.path.exists(DATA):
        raise SystemExit(
            f"{DATA} not found. Run `python generate_sample_data.py` "
            "or drop in the real Kaggle CSV."
        )

    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} transactions from {DATA}")

    # --- Feature 1: categorization ---
    hr("FEATURE 1  -  Auto-categorization model")
    model, acc, report = train_categorizer(df)
    print(f"Held-out test accuracy: {acc:.1%}")
    for desc in ["Starbucks", "Shell", "Amazon", "Netflix"]:
        print(f"  {desc:12s} -> {model.predict([desc])[0]}")

    # --- Feature 2: insights ---
    hr("FEATURE 2  -  Money insights")
    s = summary(df)
    print(f"Income:   ${s['total_income']:,.2f}")
    print(f"Spending: ${s['total_spending']:,.2f}")
    print(f"Savings:  ${s['net_savings']:,.2f}  ({s['savings_rate']:.1%} of income)")
    print(f"Across {s['n_months']} months\n")
    print("Biggest overspending months (vs your typical month):")
    for f in overspending_flags(df)[:4]:
        print(f"  {f['month']}  {f['category']:16s} ${f['spent']:>9,.2f}  (+{f['over_by']})")

    bva_table = budget_vs_actual(df)
    if bva_table is not None:
        over = bva_table[bva_table["over_budget"] > 0]
        print(f"\nOver budget in {len(over)} categories. Worst:")
        for cat, r in over.head(3).iterrows():
            print(f"  {cat:18s} ${r['avg_monthly_spend']:,.0f}/mo vs "
                  f"${r['budget']:,.0f} budget")

    # --- Feature 3: recommendation ---
    hr("FEATURE 3  -  Investment recommendation")
    rec = recommend(s["total_income"], s["total_spending"], s["n_months"],
                    current_cash_savings=CURRENT_CASH_SAVINGS)
    print(rec.headline)
    print(f"(surplus ${rec.monthly_surplus:,.0f}/mo, "
          f"emergency target ${rec.emergency_target:,.0f})\n")
    for i, step in enumerate(rec.steps, 1):
        print(f"  {i}. {step}")

    # --- Niche feature: subscription audit ---
    hr("EXTRA  -  Subscription audit")
    subs = detect_subscriptions(df)
    total_annual = sum(x["est_annual_cost"] for x in subs)
    print(f"Found {len(subs)} recurring charges, ~${total_annual:,.0f}/yr in total:")
    for x in subs:
        print(f"  {x['merchant']:26s} ${x['typical_amount']:>8,.2f}/mo  "
              f"~${x['est_annual_cost']:,.0f}/yr")

    # --- Niche feature: anomaly / fraud radar ---
    hr("EXTRA  -  Unusual transactions (fraud/oops radar)")
    anoms = detect_anomalies(df)
    print(f"{len(anoms)} transactions stand out from their category's norm:")
    for a in anoms[:6]:
        print(f"  {a['date']}  {a['merchant']:24s} ${a['amount']:>9,.2f}  "
              f"(cat avg ${a['category_avg']:,.2f}, z={a['zscore']})")

    # --- Niche feature: financial health score ---
    hr("EXTRA  -  Financial health score")
    avg_spend = s["total_spending"] / s["n_months"]
    emergency_ratio = CURRENT_CASH_SAVINGS / (avg_spend * 3)
    score, breakdown, grade = financial_health_score(s, bva_table, emergency_ratio)
    print(f"Score: {score}/100  (grade {grade})")
    print(f"  savings rate:      {breakdown['savings_rate_pts']}/50")
    print(f"  budget discipline: {breakdown['budget_discipline_pts']}/30")
    print(f"  emergency fund:    {breakdown['emergency_fund_pts']}/20")

    # --- Niche feature: cash-flow forecast ---
    hr("EXTRA  -  Cash-flow forecast (next 6 months)")
    fc = forecast_cashflow(df, months_ahead=6,
                           current_cash_savings=CURRENT_CASH_SAVINGS)
    for row in fc["forecast"]:
        print(f"  {row['month']}  net ${row['net']:>+8,.0f}  "
              f"-> projected savings ${row['projected_savings']:>9,.0f}")
    print()
    for note in fc["notes"]:
        print(f"  - {note}")

    # --- charts ---
    hr("Charts")
    make_charts(df, score, grade, breakdown, fc)


if __name__ == "__main__":
    main()
