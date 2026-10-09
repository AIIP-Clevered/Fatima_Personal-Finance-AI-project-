"""
recommend.py
------------
Feature 3: a rule-based investment / savings recommendation.

This is deliberately NOT machine learning. Personal-finance advice is a place
where transparent rules beat a black box: you want to be able to explain exactly
why the tool said what it said, and giving investment advice from a model you
can't audit is how people get hurt. So this encodes a few widely-taught rules of
thumb and shows its working.

The logic, in order (a simplified version of the standard "flow chart" advice):
  1. Emergency fund first. Aim for ~3 months of spending in cash before
     investing. If you're short, direct savings there.
  2. Once the emergency fund is covered, split the leftover monthly surplus:
     the bigger your surplus relative to income, the more you can afford to put
     into longer-term investments vs keeping as buffer.
  3. If you're spending more than you earn, no investing -- fix the gap first.

These numbers are defaults you can change. This is educational, not financial
advice.
"""

from dataclasses import dataclass


@dataclass
class Recommendation:
    headline: str
    monthly_surplus: float
    emergency_target: float
    steps: list


def recommend(total_income: float,
              total_spending: float,
              n_months: int,
              current_cash_savings: float = 0.0,
              emergency_months: float = 3.0) -> Recommendation:
    avg_income = total_income / n_months if n_months else 0.0
    avg_spending = total_spending / n_months if n_months else 0.0
    surplus = round(avg_income - avg_spending, 2)
    emergency_target = round(avg_spending * emergency_months, 2)

    steps = []

    # Case 1: spending exceeds income
    if surplus <= 0:
        return Recommendation(
            headline="You're spending more than you earn. Close the gap before investing.",
            monthly_surplus=surplus,
            emergency_target=emergency_target,
            steps=[
                f"Average income ${avg_income:,.0f}/mo vs spending ${avg_spending:,.0f}/mo.",
                "Cut from your largest non-essential categories until surplus is positive.",
                "Revisit investing once you consistently save each month.",
            ],
        )

    # Case 2: positive surplus -> emergency fund, then invest
    savings_rate = surplus / avg_income if avg_income else 0

    if current_cash_savings < emergency_target:
        gap = round(emergency_target - current_cash_savings, 2)
        months_to_fund = round(gap / surplus, 1) if surplus else float("inf")
        steps.append(
            f"Build your emergency fund to ${emergency_target:,.0f} "
            f"(3 months of spending). You have ${current_cash_savings:,.0f}, "
            f"so ${gap:,.0f} to go."
        )
        steps.append(
            f"At ${surplus:,.0f}/mo surplus, that's about {months_to_fund} months. "
            "Keep this money in a high-yield savings account, not investments."
        )
        steps.append(
            "After the fund is full, come back and shift the surplus into investing."
        )
        headline = "Priority: finish your emergency fund, then invest."
    else:
        # emergency fund covered -> suggest an allocation of the surplus
        if savings_rate >= 0.30:
            invest_pct = 0.80
        elif savings_rate >= 0.15:
            invest_pct = 0.60
        else:
            invest_pct = 0.40
        invest_amt = round(surplus * invest_pct, 2)
        buffer_amt = round(surplus - invest_amt, 2)
        steps.append(
            f"Emergency fund is covered (${current_cash_savings:,.0f} >= "
            f"${emergency_target:,.0f})."
        )
        steps.append(
            f"Your savings rate is {savings_rate:.0%}. Of your ${surplus:,.0f}/mo "
            f"surplus, consider ~${invest_amt:,.0f} into long-term investments "
            f"(e.g. a low-cost index fund) and ${buffer_amt:,.0f} kept as buffer."
        )
        steps.append(
            "Automate the transfer on payday so it happens before you can spend it."
        )
        headline = f"You're in good shape -- invest about ${invest_amt:,.0f}/month."

    return Recommendation(
        headline=headline,
        monthly_surplus=surplus,
        emergency_target=emergency_target,
        steps=steps,
    )


if __name__ == "__main__":
    import pandas as pd
    from insights import summary

    df = pd.read_csv("data/personal_transactions.csv")
    s = summary(df)

    # pretend the user currently has $5,000 in cash savings
    rec = recommend(s["total_income"], s["total_spending"], s["n_months"],
                    current_cash_savings=5000)
    print(rec.headline)
    print(f"(surplus ${rec.monthly_surplus:,.0f}/mo, "
          f"emergency target ${rec.emergency_target:,.0f})\n")
    for i, step in enumerate(rec.steps, 1):
        print(f"{i}. {step}")
