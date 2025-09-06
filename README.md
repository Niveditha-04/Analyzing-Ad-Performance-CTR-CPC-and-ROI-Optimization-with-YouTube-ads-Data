
A simulated data analytics project using YouTube Ads dataset: BigQuery + SQL, Python cleanup, and a Looker Studio dashboard.
I built a small, end-to-end analytics project to practice paid-media reporting: raw (simulated) data → SQL/BigQuery modeling → Python cleanup → a Looker Studio dashboard.
**Goal of this project:** To measure ad performance (like impressions, engagement, conversions, ROI, ROAS) and to explore patterns by country, device, and language.


A. **Tools used:**
**Python (Colab)** → data cleaning and basic checks
**BigQuery (SQL)** → building views for KPIs and aggregated metrics
**Looker Studio** → dashboard and visualizations
**CSV dataset** → simulated ad logs with country, device, language, impressions, views, conversions, cost, revenue

B. **Links:**
1. **Live dashboard:** https://lookerstudio.google.com/reporting/32254fee-eb7c-46f4-a702-31bfe1a6c5b8
2. **Python:** `[notebooks/01_cleaning_and_kpis.ipynb](https://colab.research.google.com/drive/1lclEigExNTd-T7JWq4Kp8cT4VvMZk8Nq?usp=sharing)` – quick data checks and KPI calculations.
   

C. **What to look for?**

Ci. **Dashboard (Looker Studio):**
  - KPI scorecards (Impressions, Cost, Revenue, ROAS).
  - Treemap: **size = Impressions**, **color = Engagement Rate** (shows scale + quality).
  - Donut: Device mix (with “Unknown” bucket for nulls).
  - Bar Chart: Return on Ad Spend (ROAS)
  - Stacked bars: Country × Device.
  - Heatmap table: **CTR, CPC, CPM** by country.
  - Dropdown: **Country & Device**

Cii. **Dataset fields (raw):**
1. ua_country – country of the ad viewer
2. ua_device – device type (mobile, desktop, TV, etc.)
3. placement_language – ad placement language
4. impressions – number of times the ad was shown
5. engaged_views – number of engaged views (watched enough to count as engagement)
6. seconds_played – how long ads were played
7. creative_duration – full length of the ad
8. conversions – number of conversions attributed
9. cost – money spent on ads
10. revenue – revenue from ads


> Note: This is **simulated** data. Benchmarks in the write-up reflect real-world ranges (CTR ~0.5–5%, CPC $0.05–$0.50, CPM $2–$20, varies by market).
