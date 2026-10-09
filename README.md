# AI-Powered Personal Finance Management System

An internship project that turns a year+ of bank transactions into useful,
explainable money insights. It has three core features from the brief plus three
extra "niche" features, and it runs end to end from one command.

The one machine-learning piece (auto-categorization) is used where it genuinely
helps. Everything else is transparent, auditable logic, because for personal
finance you want to be able to explain every number and every recommendation.

## What it does

Core features (from the project brief):

1. **Spending categorization** (`categorization.py`) - a scikit-learn model
   (TF-IDF + logistic regression) that reads a transaction description and
   predicts its category. ~97% accuracy on held-out data.
2. **Savings insights** (`insights.py`) - income vs spending, savings rate,
   spend by category, month-by-month trend, overspending flags, and a
   budget-vs-actual comparison using the dataset's own `Budget.csv`.
3. **Investment recommendation** (`recommend.py`) - rule-based advice that puts
   an emergency fund first, then suggests how much of the monthly surplus to
   invest based on your savings rate.

Extra features (beyond the brief):

4. **Subscription audit** (`features.py`) - finds recurring charges (Netflix,
   insurance, utilities) and estimates their annual cost.
5. **Anomaly / fraud radar** (`features.py`) - flags transactions that are
   unusually large for their own category, the same first-pass idea card
   companies use for fraud alerts.
6. **Financial health score** (`features.py`) - one 0-100 score (with a letter
   grade) built from savings rate, budget discipline, and emergency-fund
   coverage.
7. **Cash-flow forecast** (`forecast.py`) - projects the next 6 months of
   income, spending, and cumulative savings.
8. **Interactive dashboard** (`generate_dashboard.py`) - builds a single
   self-contained `dashboard.html` with all of the above: KPI cards, charts,
   the health score, subscriptions, anomalies, forecast, and recommendation.

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py           # interactive web app (recommended) - opens in your browser
python main.py                 # prints all features + saves 5 charts to outputs/
python generate_dashboard.py   # builds dashboard.html (open it in a browser)
```

### The interactive app (`app.py`)

`streamlit run app.py` launches a live web app version of the whole project.
It reuses the exact same modules as the pipeline - nothing is recomputed - and
wraps them in a UI where every number updates as you change things:

- Upload your own bank CSV, or use the bundled sample dataset.
- Sidebar sliders for your current cash savings and emergency-fund target;
  the health score, recommendation, and forecast all react live.
- Seven tabs: Overview (score + next steps), Spending, Trends & Budget,
  Subscriptions & Anomalies, Forecast, Recommendation, and a live Categorize
  tab where you type a description and the ML model predicts its category.

Each module also runs on its own for testing, e.g. `python insights.py`,
`python features.py`, or `python forecast.py`.

### About the forecast (Prophet vs. what this uses)

The companion notebook uses Facebook Prophet for forecasting. This project
deliberately does **not**: with only ~21 monthly data points, Prophet's
seasonality fitting is unreliable and it's a heavy dependency. Instead
`forecast.py` fits a linear trend on *winsorized* monthly spending (the two
renovation-spike months are capped so they don't warp the trend) and projects
income as a recent average. It's more robust at this data size and every number
is explainable. Prophet becomes worth it once you have years of daily data.

### About the dashboard being self-contained

`dashboard.html` inlines the Chart.js library (vendored at `vendor/chart.umd.js`)
instead of loading it from a CDN, so the file works offline and won't break if a
CDN is down. That's why the generated HTML is ~210 KB.

## Project layout

```
personal_finance_ai/
├── data/
│   ├── personal_transactions.csv   # the transactions (real Kaggle dataset)
│   └── Budget.csv                  # per-category monthly budgets
├── categorization.py               # Feature 1: ML categorization
├── insights.py                     # Feature 2: money insights + budget compare
├── recommend.py                    # Feature 3: investment recommendation
├── features.py                     # Extras: subscriptions, anomalies, score
├── forecast.py                     # Extra: cash-flow forecast
├── generate_dashboard.py           # builds the self-contained dashboard.html
├── vendor/chart.umd.js             # Chart.js, inlined into the dashboard
├── main.py                         # runs everything, saves charts
├── generate_sample_data.py         # makes a stand-in CSV if you don't have the real one
├── personalfinancebudget.ipynb     # exploratory analysis companion (EDA + forecasting)
├── outputs/                        # charts land here
├── requirements.txt
└── README.md
```

## Data notes (worth knowing before you present this)

- Data source: [Kaggle - bukolafatunde/personal-finance](https://www.kaggle.com/datasets/bukolafatunde/personal-finance).
  The real `personal_transactions.csv` and `Budget.csv` are in `data/`.
- **Credit-card payments are not income.** In this dataset both paychecks and
  credit-card payments are recorded as `credit`. The code counts only paychecks
  as income; treating every credit as income would overstate it by ~$30k here.
- **Rare categories.** A few categories (e.g. `Entertainment`) have only one or
  two transactions. The categorization model skips categories with fewer than 3
  rows, because you can't train and test a class from one example.
- **Netflix isn't flagged as a subscription** even though it is one, because its
  price changed several times over the two years, so its amount isn't stable
  enough for the detector's tolerance. This is a real precision/recall trade-off,
  not a bug - loosening the tolerance brings back false positives like
  restaurants. Noted here on purpose.

## Ideas to extend it

- Add cash-flow **forecasting** (the included notebook uses Prophet for this).
- Let the categorization model fall back to a manual rule when its confidence is
  low, instead of always guessing.
- ~~Turn the health score and charts into a small dashboard (Streamlit or a
  static HTML page).~~ Done - see `app.py` (Streamlit) and `dashboard.html`.
- Deploy `app.py` to Streamlit Community Cloud so it's a shareable live URL.
