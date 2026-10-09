"""
generate_dashboard.py
----------------------
Runs the whole pipeline and writes a single self-contained dashboard.html:
KPI cards, interactive charts (Chart.js), the health score, subscriptions,
anomalies, the forecast, and the recommendation. No server needed -- open the
HTML file in any browser.

    python generate_dashboard.py

The data is computed in Python and injected into the page as JSON, so the HTML
is fully offline apart from loading Chart.js from a CDN.
"""

import json
import pandas as pd

from categorization import train_categorizer
from insights import (summary, spending_by_category, monthly_trend,
                      budget_vs_actual)
from recommend import recommend
from features import (detect_subscriptions, detect_anomalies,
                      financial_health_score)
from forecast import forecast_cashflow

DATA = "data/personal_transactions.csv"
CURRENT_CASH_SAVINGS = 5000


def build_data():
    df = pd.read_csv(DATA)

    _, acc, _ = train_categorizer(df)
    s = summary(df)
    cats = spending_by_category(df).head(10)
    trend = monthly_trend(df)
    bva = budget_vs_actual(df)
    rec = recommend(s["total_income"], s["total_spending"], s["n_months"],
                    current_cash_savings=CURRENT_CASH_SAVINGS)
    subs = detect_subscriptions(df)
    anoms = detect_anomalies(df)[:8]
    avg_spend = s["total_spending"] / s["n_months"]
    emergency_ratio = CURRENT_CASH_SAVINGS / (avg_spend * 3)
    score, breakdown, grade = financial_health_score(s, bva, emergency_ratio)
    fc = forecast_cashflow(df, months_ahead=6,
                           current_cash_savings=CURRENT_CASH_SAVINGS)

    bva_rows = []
    if bva is not None:
        for cat, r in bva.head(10).iterrows():
            bva_rows.append({"category": cat,
                             "actual": float(r["avg_monthly_spend"]),
                             "budget": float(r["budget"]),
                             "over": float(r["over_budget"])})

    return {
        "model_accuracy": round(acc * 100, 1),
        "summary": s,
        "categories": {"labels": list(cats.index),
                       "values": [float(v) for v in cats.values]},
        "trend": {"months": list(trend.index),
                  "income": [float(v) for v in trend["income"]],
                  "spending": [float(v) for v in trend["spending"]]},
        "budget": bva_rows,
        "recommendation": {"headline": rec.headline, "steps": rec.steps,
                           "surplus": rec.monthly_surplus,
                           "emergency_target": rec.emergency_target},
        "subscriptions": subs,
        "subs_annual_total": round(sum(x["est_annual_cost"] for x in subs), 0),
        "anomalies": anoms,
        "score": score, "grade": grade, "breakdown": breakdown,
        "forecast": fc,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal Finance Dashboard</title>
<script>__CHARTJS__</script>
<style>
  :root{
    --bg:#0f172a; --card:#1e293b; --card2:#243449; --text:#e2e8f0;
    --muted:#94a3b8; --accent:#38bdf8; --green:#4ade80; --red:#f87171;
    --amber:#fbbf24; --line:#334155;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:28px 32px 8px}
  header h1{margin:0;font-size:24px}
  header p{margin:4px 0 0;color:var(--muted);font-size:14px}
  .wrap{padding:16px 32px 48px;max-width:1200px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
        gap:16px;margin:16px 0 8px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;
       padding:18px 20px}
  .kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase;
              letter-spacing:.5px}
  .kpi .value{font-size:26px;font-weight:650;margin-top:6px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
         padding:18px 20px}
  .panel h2{margin:0 0 12px;font-size:15px;font-weight:600}
  .full{grid-column:1 / -1}
  canvas{max-height:300px}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
  .grade{display:inline-flex;align-items:center;justify-content:center;
         width:54px;height:54px;border-radius:50%;font-size:24px;font-weight:700}
  .rec-steps{margin:8px 0 0;padding-left:18px;color:var(--text);font-size:14px}
  .rec-steps li{margin:6px 0}
  .headline{font-size:15px;font-weight:600;color:var(--accent)}
  .pill{font-size:11px;padding:2px 8px;border-radius:999px;background:var(--card2)}
  .muted{color:var(--muted)}
  .over{color:var(--red)} .under{color:var(--green)}
</style>
</head>
<body>
<header>
  <h1>Personal Finance Dashboard</h1>
  <p>AI-powered analysis of __NMONTHS__ months of transactions &middot;
     categorization model accuracy __ACC__%</p>
</header>
<div class="wrap">

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="panel"><h2>Where the money goes (top categories)</h2>
      <canvas id="catChart"></canvas></div>
    <div class="panel"><h2>Monthly income vs spending</h2>
      <canvas id="trendChart"></canvas></div>
    <div class="panel"><h2>Budget vs actual (avg month)</h2>
      <canvas id="budgetChart"></canvas></div>
    <div class="panel"><h2>Cash-flow forecast (6 months)</h2>
      <canvas id="forecastChart"></canvas></div>

    <div class="panel">
      <h2>Financial health score</h2>
      <div style="display:flex;gap:16px;align-items:center">
        <div class="grade" id="gradeCircle"></div>
        <div>
          <div style="font-size:30px;font-weight:700" id="scoreNum"></div>
          <div class="muted" id="scoreBreak" style="font-size:12px"></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Recommendation</h2>
      <div class="headline" id="recHead"></div>
      <ol class="rec-steps" id="recSteps"></ol>
    </div>

    <div class="panel">
      <h2>Subscription audit <span class="pill" id="subTotal"></span></h2>
      <table><thead><tr><th>Merchant</th><th class="num">Monthly</th>
        <th class="num">Annual</th></tr></thead><tbody id="subBody"></tbody></table>
    </div>

    <div class="panel">
      <h2>Unusual transactions (fraud / oops radar)</h2>
      <table><thead><tr><th>Date</th><th>Merchant</th><th class="num">Amount</th>
        <th class="num">z</th></tr></thead><tbody id="anomBody"></tbody></table>
    </div>
  </div>

  <p class="muted" style="margin-top:24px;font-size:12px">
    Educational project, not financial advice. Data:
    Kaggle bukolafatunde/personal-finance.</p>
</div>

<script>
const DATA = __DATA__;
const usd = n => '$' + Math.round(n).toLocaleString();
const gridColor = '#334155', tick = '#94a3b8';
Chart.defaults.color = tick;
Chart.defaults.font.family = "-apple-system,Segoe UI,Roboto,sans-serif";

// KPIs
const s = DATA.summary;
const kpis = [
  ['Total income', usd(s.total_income)],
  ['Total spending', usd(s.total_spending)],
  ['Net savings', usd(s.net_savings)],
  ['Savings rate', (s.savings_rate*100).toFixed(1)+'%'],
  ['Health score', DATA.score + '/100 (' + DATA.grade + ')'],
];
document.getElementById('kpis').innerHTML = kpis.map(k =>
  `<div class="kpi"><div class="label">${k[0]}</div>
   <div class="value">${k[1]}</div></div>`).join('');

// Category chart
new Chart(catChart, {type:'bar',
  data:{labels:DATA.categories.labels,
    datasets:[{data:DATA.categories.values,backgroundColor:'#38bdf8'}]},
  options:{indexAxis:'y',plugins:{legend:{display:false}},
    scales:{x:{grid:{color:gridColor}},y:{grid:{display:false}}}}});

// Trend chart
new Chart(trendChart, {type:'line',
  data:{labels:DATA.trend.months,datasets:[
    {label:'Income',data:DATA.trend.income,borderColor:'#4ade80',tension:.3},
    {label:'Spending',data:DATA.trend.spending,borderColor:'#f87171',tension:.3}]},
  options:{plugins:{legend:{labels:{boxWidth:12}}},
    scales:{x:{grid:{display:false}},y:{grid:{color:gridColor}}}}});

// Budget chart
new Chart(budgetChart, {type:'bar',
  data:{labels:DATA.budget.map(b=>b.category),
    datasets:[{data:DATA.budget.map(b=>b.over),
      backgroundColor:DATA.budget.map(b=>b.over>0?'#f87171':'#4ade80')}]},
  options:{indexAxis:'y',plugins:{legend:{display:false},
    tooltip:{callbacks:{label:c=>'Over/under: '+usd(c.raw)}}},
    scales:{x:{grid:{color:gridColor}},y:{grid:{display:false}}}}});

// Forecast chart
const hist = DATA.forecast.history, fut = DATA.forecast.forecast;
const labels = hist.map(h=>h.month).concat(fut.map(f=>f.month));
const netData = hist.map(h=>h.net).concat(fut.map(()=>null));
const savData = hist.map(()=>null).concat(fut.map(f=>f.projected_savings));
// connect the line: last hist point start
new Chart(forecastChart, {data:{labels:labels,datasets:[
    {type:'bar',label:'Actual net',data:netData,backgroundColor:'#64748b'},
    {type:'line',label:'Projected savings',data:savData,
     borderColor:'#38bdf8',tension:.2,spanGaps:true}]},
  options:{plugins:{legend:{labels:{boxWidth:12}}},
    scales:{x:{grid:{display:false},ticks:{maxRotation:90,minRotation:90,
      font:{size:8}}},y:{grid:{color:gridColor}}}}});

// Health score
const g = DATA.grade;
const gColor = DATA.score>=70?'#4ade80':DATA.score>=55?'#fbbf24':'#f87171';
const gc = document.getElementById('gradeCircle');
gc.textContent = g; gc.style.background = gColor; gc.style.color = '#0f172a';
document.getElementById('scoreNum').textContent = DATA.score + ' / 100';
const b = DATA.breakdown;
document.getElementById('scoreBreak').innerHTML =
  `savings ${b.savings_rate_pts}/50 &middot; budget ${b.budget_discipline_pts}/30 `+
  `&middot; emergency ${b.emergency_fund_pts}/20`;

// Recommendation
document.getElementById('recHead').textContent = DATA.recommendation.headline;
document.getElementById('recSteps').innerHTML =
  DATA.recommendation.steps.map(x=>`<li>${x}</li>`).join('');

// Subscriptions
document.getElementById('subTotal').textContent =
  '~' + usd(DATA.subs_annual_total) + '/yr';
document.getElementById('subBody').innerHTML = DATA.subscriptions.map(x=>
  `<tr><td>${x.merchant}</td><td class="num">${usd(x.typical_amount)}</td>
   <td class="num">${usd(x.est_annual_cost)}</td></tr>`).join('');

// Anomalies
document.getElementById('anomBody').innerHTML = DATA.anomalies.map(a=>
  `<tr><td class="muted">${a.date}</td><td>${a.merchant}</td>
   <td class="num">${usd(a.amount)}</td><td class="num">${a.zscore}</td></tr>`).join('');
</script>
</body>
</html>"""


def main():
    data = build_data()
    # inline Chart.js so the dashboard is fully self-contained (works offline,
    # no CDN needed). The library is vendored at vendor/chart.umd.js.
    with open("vendor/chart.umd.js") as f:
        chartjs = f.read()
    html = (HTML
            .replace("__CHARTJS__", chartjs)
            .replace("__DATA__", json.dumps(data))
            .replace("__NMONTHS__", str(data["summary"]["n_months"]))
            .replace("__ACC__", str(data["model_accuracy"])))
    with open("dashboard.html", "w") as f:
        f.write(html)
    print(f"Wrote dashboard.html ({len(html)//1024} KB, self-contained)")


if __name__ == "__main__":
    main()
