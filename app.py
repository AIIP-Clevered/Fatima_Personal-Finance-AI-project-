"""
app.py
------
Interactive Streamlit web app for the AI-Powered Personal Finance system.

It's the same project as `main.py` / `generate_dashboard.py`, but instead of a
static HTML file you get a live app: change your cash balance or the emergency-
fund target in the sidebar and every number, chart, and recommendation updates.
Upload your own bank CSV, or use the bundled sample dataset.

Run it with:

    streamlit run app.py

All the real work still lives in the existing modules -- this file only wires
them to a UI. Nothing is recomputed here that the pipeline doesn't already do.
"""

import os
import pandas as pd
import streamlit as st

# Reuse the exact same logic the command-line pipeline uses.
from categorization import train_categorizer, predict_category
from insights import (summary, spending_by_category, monthly_trend,
                      overspending_flags, budget_vs_actual)
from recommend import recommend
from features import (detect_subscriptions, detect_anomalies,
                      financial_health_score)
from forecast import forecast_cashflow

# Always resolve data files relative to this file, so the app works no matter
# which directory Streamlit is launched from.
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data", "personal_transactions.csv")
BUDGET_PATH = os.path.join(HERE, "data", "Budget.csv")

st.set_page_config(
    page_title="Personal Finance AI",
    page_icon="💰",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading + model training (cached so the app stays snappy)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA)


@st.cache_data(show_spinner=False)
def read_uploaded(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner="Training the categorization model...")
def get_model(df: pd.DataFrame):
    # cache_resource keys on the call; we pass a signature so a new upload
    # retrains. Streamlit hashes the df via the wrapper below.
    return train_categorizer(df)


REQUIRED_COLUMNS = {"Date", "Description", "Amount", "Transaction Type",
                    "Category", "Account Name"}


def money(x: float) -> str:
    return f"${x:,.0f}"


def md(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't read it as LaTeX math."""
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Sidebar: data source + assumptions
# ---------------------------------------------------------------------------
st.sidebar.title("💰 Personal Finance AI")
st.sidebar.caption("Turn a year+ of transactions into explainable money insights.")

st.sidebar.subheader("1. Data")
uploaded = st.sidebar.file_uploader(
    "Upload a transactions CSV", type=["csv"],
    help="Columns: Date, Description, Amount, Transaction Type, Category, "
         "Account Name. Leave empty to use the bundled sample dataset.",
)

if uploaded is not None:
    try:
        df = read_uploaded(uploaded)
        source_label = f"Uploaded file: {uploaded.name}"
    except Exception as e:
        st.sidebar.error(f"Couldn't read that CSV: {e}")
        st.stop()
else:
    df = load_default_data()
    source_label = "Bundled sample dataset (Kaggle personal-finance)"

missing = REQUIRED_COLUMNS - set(df.columns)
if missing:
    st.error(
        "This CSV is missing required columns: "
        + ", ".join(sorted(missing))
        + ".\n\nExpected: " + ", ".join(sorted(REQUIRED_COLUMNS))
    )
    st.stop()

st.sidebar.success(source_label)
st.sidebar.write(f"**{len(df):,}** transactions loaded.")

st.sidebar.subheader("2. Your assumptions")
current_cash = st.sidebar.number_input(
    "Current cash savings ($)", min_value=0, value=5000, step=500,
    help="How much cash you hold now. Drives the emergency-fund and "
         "health-score math. In a real app this would come from a linked "
         "account balance.",
)
emergency_months = st.sidebar.slider(
    "Emergency fund target (months of spending)", 1.0, 12.0, 3.0, 0.5,
)

st.sidebar.divider()
st.sidebar.caption(
    "Educational tool, not financial advice. Every number here is transparent "
    "arithmetic except the categorization model."
)


# ---------------------------------------------------------------------------
# Core computations (all from the existing modules)
# ---------------------------------------------------------------------------
s = summary(df)
bva = budget_vs_actual(df, budget_path=BUDGET_PATH)
avg_spend = s["total_spending"] / s["n_months"] if s["n_months"] else 0
emergency_target = avg_spend * emergency_months
emergency_ratio = current_cash / emergency_target if emergency_target else 0
score, breakdown, grade = financial_health_score(s, bva, emergency_ratio)
rec = recommend(s["total_income"], s["total_spending"], s["n_months"],
                current_cash_savings=current_cash,
                emergency_months=emergency_months)

GRADE_COLOR = {"A": "#2e7d32", "B": "#558b2f", "C": "#f9a825",
               "D": "#ef6c00", "F": "#c62828"}


# ---------------------------------------------------------------------------
# Header + top KPIs
# ---------------------------------------------------------------------------
st.title("Personal Finance AI")
st.caption(
    f"Analysing {len(df):,} transactions across {s['n_months']} months. "
    "Adjust the sidebar to see everything update live."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total income", money(s["total_income"]))
k2.metric("Total spending", money(s["total_spending"]))
k3.metric("Net savings", money(s["net_savings"]))
k4.metric("Savings rate", f"{s['savings_rate']:.0%}")
k5.metric("Health score", f"{score}/100", help=f"Grade {grade}")

st.divider()

tabs = st.tabs([
    "📊 Overview",
    "🧾 Spending",
    "📈 Trends & Budget",
    "🔁 Subscriptions & Anomalies",
    "🔮 Forecast",
    "💡 Recommendation",
    "🏷️ Categorize",
])


# ---------------------------------------------------------------------------
# Tab 1: Overview -- health score + headline recommendation
# ---------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Financial health score")
        color = GRADE_COLOR.get(grade, "#3b7ea1")
        st.markdown(
            f"<div style='font-size:3.5rem;font-weight:700;color:{color};"
            f"line-height:1'>{score}<span style='font-size:1.5rem;color:#888'>"
            f"/100</span> &nbsp;<span style='font-size:2rem'>({grade})</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(score, 100) / 100)

        st.markdown("**Score breakdown**")
        parts = pd.DataFrame({
            "Component": ["Savings rate", "Budget discipline", "Emergency fund"],
            "Points": [breakdown["savings_rate_pts"],
                       breakdown["budget_discipline_pts"],
                       breakdown["emergency_fund_pts"]],
            "Max": [50, 30, 20],
        })
        for _, r in parts.iterrows():
            st.write(f"{r['Component']}: **{r['Points']} / {r['Max']}**")
            st.progress(r["Points"] / r["Max"])

    with right:
        st.subheader("What to do next")
        st.info(md(rec.headline))
        m1, m2 = st.columns(2)
        m1.metric("Monthly surplus", money(rec.monthly_surplus))
        m2.metric("Emergency target", money(rec.emergency_target))
        for i, step in enumerate(rec.steps, 1):
            st.markdown(f"**{i}.** {md(step)}")


# ---------------------------------------------------------------------------
# Tab 2: Spending by category
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Where the money goes")
    cats = spending_by_category(df)
    top_n = st.slider("Show top N categories", 5, min(25, len(cats)),
                      min(10, len(cats)))
    top = cats.head(top_n)
    st.bar_chart(top, horizontal=True, color="#3b7ea1")
    with st.expander("See the full table"):
        st.dataframe(
            cats.rename("Total spent ($)").to_frame().style.format("${:,.2f}"),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab 3: Monthly trend + budget vs actual
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Monthly income vs spending")
    trend = monthly_trend(df)
    st.line_chart(trend[["income", "spending"]],
                  color=["#c62828", "#2e7d32"])

    st.markdown("**Overspending months** (vs your own typical month)")
    flags = overspending_flags(df)
    if flags:
        st.dataframe(pd.DataFrame(flags), width="stretch",
                     hide_index=True)
    else:
        st.write("No months ran notably above your typical spending. Nice.")

    st.divider()
    st.subheader("Budget vs actual (average month)")
    if bva is not None:
        show = bva.copy()
        show.index.name = "Category"
        st.dataframe(
            show.style.format({
                "avg_monthly_spend": "${:,.2f}",
                "budget": "${:,.0f}",
                "over_budget": "${:,.2f}",
                "pct_of_budget": "{:.0%}",
            }).map(
                lambda v: "color:#c62828" if isinstance(v, (int, float))
                and v > 0 else "",
                subset=["over_budget"],
            ),
            width="stretch",
        )
        over = bva[bva["over_budget"] > 0]
        st.caption(
            f"Over budget in {len(over)} of {len(bva)} categories. "
            "Red = spending more than the monthly budget."
        )
    else:
        st.write("No Budget.csv found, so budget comparison is unavailable.")


# ---------------------------------------------------------------------------
# Tab 4: Subscriptions + anomalies
# ---------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Subscription audit")
        subs = detect_subscriptions(df)
        if subs:
            total_annual = sum(x["est_annual_cost"] for x in subs)
            st.metric("Recurring charges found", len(subs),
                      help="Charges that bill about monthly at a near-constant "
                           "amount, in at least half of all months.")
            st.metric("Estimated annual cost", money(total_annual))
            st.dataframe(
                pd.DataFrame(subs)[["merchant", "category", "typical_amount",
                                    "months_seen", "est_annual_cost"]],
                width="stretch", hide_index=True,
                column_config={
                    "typical_amount": st.column_config.NumberColumn(
                        "Typical / mo", format="$%.2f"),
                    "est_annual_cost": st.column_config.NumberColumn(
                        "Est. / yr", format="$%.0f"),
                },
            )
        else:
            st.write("No clear recurring subscriptions detected.")

    with c2:
        st.subheader("Unusual transactions (fraud / oops radar)")
        anoms = detect_anomalies(df)
        if anoms:
            st.metric("Flagged as unusual", len(anoms),
                      help="Transactions far above the norm FOR THEIR OWN "
                           "category (z-score >= 3).")
            st.dataframe(
                pd.DataFrame(anoms)[["date", "merchant", "category", "amount",
                                     "category_avg", "zscore"]],
                width="stretch", hide_index=True,
                column_config={
                    "amount": st.column_config.NumberColumn(
                        "Amount", format="$%.2f"),
                    "category_avg": st.column_config.NumberColumn(
                        "Cat. avg", format="$%.2f"),
                },
            )
        else:
            st.write("Nothing stands out as unusually large.")


# ---------------------------------------------------------------------------
# Tab 5: Cash-flow forecast
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Cash-flow forecast")
    months_ahead = st.slider("Months to project", 3, 12, 6)
    fc = forecast_cashflow(df, months_ahead=months_ahead,
                           current_cash_savings=current_cash)

    hist = pd.DataFrame(fc["history"]).set_index("month")
    fut = pd.DataFrame(fc["forecast"]).set_index("month")

    st.markdown("**Projected cumulative savings**")
    st.line_chart(fut["projected_savings"], color="#3b7ea1")

    st.markdown("**Actual monthly net (history)**")
    st.bar_chart(hist["net"], color="#90a4ae")

    st.markdown("**The forecast, month by month**")
    st.dataframe(
        fut, width="stretch",
        column_config={
            "income": st.column_config.NumberColumn("Income", format="$%.0f"),
            "spending": st.column_config.NumberColumn("Spending", format="$%.0f"),
            "net": st.column_config.NumberColumn("Net", format="$%+.0f"),
            "projected_savings": st.column_config.NumberColumn(
                "Projected savings", format="$%.0f"),
        },
    )
    for note in fc["notes"]:
        st.caption("• " + md(note))


# ---------------------------------------------------------------------------
# Tab 6: Recommendation (full detail)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Investment / savings recommendation")
    st.info(md(rec.headline))
    a, b = st.columns(2)
    a.metric("Monthly surplus", money(rec.monthly_surplus))
    b.metric("Emergency-fund target", money(rec.emergency_target))
    st.markdown("**Step by step**")
    for i, step in enumerate(rec.steps, 1):
        st.markdown(f"**{i}.** {md(step)}")
    st.caption(
        "Rule-based on purpose: personal-finance advice should be explainable, "
        "not a black box. These are widely-taught rules of thumb, not "
        "personalized financial advice."
    )


# ---------------------------------------------------------------------------
# Tab 7: Live categorization (the one ML piece)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Auto-categorize a transaction")
    st.write(
        "The one machine-learning piece: a TF-IDF + logistic-regression model "
        "that reads a transaction description and predicts its category."
    )

    @st.cache_resource(show_spinner="Training the categorization model...")
    def train_cached(sig: str, _df: pd.DataFrame):
        # `sig` makes the cache key change when the dataset changes.
        return train_categorizer(_df)

    sig = f"{source_label}:{len(df)}"
    model, acc, report = train_cached(sig, df)

    st.metric("Held-out test accuracy", f"{acc:.1%}")

    desc = st.text_input("Type a transaction description",
                         value="Starbucks Coffee")
    if desc.strip():
        pred = predict_category(model, desc)
        st.success(f"Predicted category: **{pred}**")

    st.markdown("**Try a few at once**")
    examples = st.text_area(
        "One description per line",
        value="Shell Gas Station\nAmazon Marketplace\nNetflix\nOlive Garden",
    )
    if examples.strip():
        rows = [{"description": line.strip(),
                 "predicted_category": predict_category(model, line.strip())}
                for line in examples.splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(rows), width="stretch",
                     hide_index=True)

    with st.expander("Full classification report (per-category precision/recall)"):
        st.code(report)
