# STEP 1 — EDA

import os, zipfile, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Data loading
KAGGLE_URL = "/kaggle/input/video-ads-engagement-dataset/ad_df.csv"
COLAB_CSV  = "/content/ad_df.csv"
COLAB_ZIP  = "/content/ad_df.csv.zip"

def load_ads():
    if os.path.exists(KAGGLE_URL):
        return pd.read_csv(KAGGLE_URL)
    if os.path.exists(COLAB_CSV):
        return pd.read_csv(COLAB_CSV)
    if os.path.exists(COLAB_ZIP):
        with zipfile.ZipFile(COLAB_ZIP) as z:
            inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(inner) as f:
                return pd.read_csv(f)
    raise FileNotFoundError(
        "Could not find data. Place file at one of:\n"
        f" - {KAGGLE_URL}\n - {COLAB_CSV}\n - {COLAB_ZIP}"
    )

ad_data = load_ads()

# HELPER TABLES
def display_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    total   = data.isnull().sum()
    percent = np.round(total / len(data) * 100, 2)
    df_null = pd.concat([total, percent], axis=1, keys=['Qtd. Nulos', 'Percent'])
    types   = [str(data[c].dtype) for c in data.columns]
    df_null['Types'] = types
    return np.round(df_null, 2).transpose()

def most_frequent_values(data: pd.DataFrame) -> pd.DataFrame:
    total = data.count()
    df    = pd.DataFrame(total)
    df.columns = ['Total']
    items, vals = [], []
    for col in data.columns:
        try:
            vc = data[col].value_counts()
            items.append(vc.index[0])
            vals.append(vc.values[0])
        except Exception:
            items.append(0)
            vals.append(0)
    df['Most frequent item']   = items
    df['Frequence']            = vals
    df['Percent from total']   = np.round(vals / total * 100, 3)
    return df

def display_unique_values(data: pd.DataFrame) -> pd.DataFrame:
    unique_values = data.nunique()
    df = pd.DataFrame({'Unique Values': unique_values})
    df['Types'] = [str(data[c].dtype) for c in data.columns]
    return df

# Missing / frequent / unique (global)
display(display_missing_data(ad_data))
display(most_frequent_values(ad_data))
display(display_unique_values(ad_data))

# BRAZIL SLICE + O GLOBO
required_cols = {
    "ua_country","referer_deep_three","ua_device","user_average_seconds_played"
}
missing = required_cols - set(ad_data.columns)
assert not missing, f"Missing required columns: {missing}"

br_ad   = ad_data.query('ua_country == "br"').copy()
display(br_ad.sample(2))
display(display_missing_data(br_ad))
display(most_frequent_values(br_ad))
display(display_unique_values(br_ad))

oglobo = br_ad.query('referer_deep_three == "com/globo/oglobo"').copy()
display(oglobo.sample(2))
display(display_unique_values(oglobo))
display(most_frequent_values(oglobo))

# VISUAL SETTINGS (same vibe)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
sns.set_palette("Spectral")
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["font.size"] = 8

# PLOTS

# (i) OGlobo — Device share pie
plt.figure()
oglobo["ua_device"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    pctdistance=0.85,
    colors=sns.color_palette("pastel"),
    startangle=140
)
plt.title("Devices in %")
plt.ylabel("")
plt.tight_layout()
plt.show()

# (ii) Global — Device share pie
plt.figure()
ad_data["ua_device"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    pctdistance=0.85,
    colors=sns.color_palette("pastel"),
    startangle=140
)
plt.title("Devices in % Global")
plt.ylabel("")
plt.tight_layout()
plt.show()

# (iii) Global — KDE by device
plt.figure(figsize=(12, 4.5))
sns.kdeplot(
    data=ad_data,
    x="user_average_seconds_played",
    hue="ua_device",
    palette="dark",
    fill=True
)
plt.title("Global")
plt.tight_layout()
plt.show()

# (iv) Brazil/OGlobo — KDE by device
plt.figure(figsize=(12, 4.5))
sns.kdeplot(
    data=oglobo,
    x="user_average_seconds_played",
    hue="ua_device",
    palette="dark",
    fill=True
)
plt.title("Brazil")
plt.tight_layout()
plt.show()



# Step 2 — Load & cleanup

import os, zipfile
import pandas as pd
import numpy as np

# Where files will be written (Colab or local)
OUTPUT_DIR = "/content" if os.path.exists("/content") else "/mnt/data"

# Inputs
ZIP_PATH = os.path.join(OUTPUT_DIR, "ad_df.csv.zip")  # put your zip here
SAMPLE_TO_1_5M = True
SAMPLE_SIZE = 1_500_000
RANDOM_STATE = 42

# Load zipped CSV
with zipfile.ZipFile(ZIP_PATH) as z:
    name_inside = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    with z.open(name_inside) as f:
        df = pd.read_csv(f)

if SAMPLE_TO_1_5M and len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

# Normalize string cols we’ll use later
for c in ["ua_country", "ua_device", "placement_language"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.lower()

# Cast numeric durations safely
for c in ["seconds_played", "user_average_seconds_played", "creative_duration"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("Loaded rows:", len(df))
print("Columns:", list(df.columns)[:12], "...")


# Step 3 — KPI engineering + exports

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "/content" if os.path.exists("/content") else "/mnt/data"

CLICK_MIN_SECONDS = 5
CONV_MIN_SECONDS  = 15
CONV_MIN_RATIO    = 0.50  # 50% of creative duration

# CPM ($ per 1000 impressions) by country
CPM_BY_COUNTRY = {"us": 12.0, "gb": 9.0, "de": 7.0, "fr": 6.0, "ca": 7.0, "au": 8.0, "in": 2.2}
DEFAULT_CPM = 5.0

# Revenue per conversion by device (LOWERCASE KEYS + lower() lookup)
REV_PER_CONV_BY_DEVICE = {"personalcomputer": 2.40, "phone": 0.90, "tablet": 1.30, "other": 0.80}
DEFAULT_REV_PER_CONV = 1.10

assert "df" in globals(), "Run Step 1 first."

df["impressions"] = 1
df["engaged_views"] = (df["seconds_played"].fillna(0) >= CLICK_MIN_SECONDS).astype(int)  # CTR proxy
conv_thresh = np.maximum(df["creative_duration"].fillna(0) * CONV_MIN_RATIO, CONV_MIN_SECONDS)
df["conversions"] = (df["seconds_played"].fillna(0) >= conv_thresh).astype(int)

def row_cpm(cc):
    return CPM_BY_COUNTRY.get(str(cc).lower(), DEFAULT_CPM)

def row_rev(dev):
    return REV_PER_CONV_BY_DEVICE.get(str(dev).lower(), DEFAULT_REV_PER_CONV)

df["_cpm"] = df["ua_country"].apply(row_cpm) if "ua_country" in df.columns else DEFAULT_CPM
df["cost"] = df["_cpm"] / 1000.0
df["_rev_per_conv"] = df["ua_device"].apply(row_rev) if "ua_device" in df.columns else DEFAULT_REV_PER_CONV
df["revenue"] = df["conversions"] * df["_rev_per_conv"]

# Segmenting columns
segment_cols = []
if "placement_language" in df.columns: segment_cols.append("placement_language")
if "ua_device" in df.columns:          segment_cols.append("ua_device")
if "ua_country" in df.columns:         segment_cols.append("ua_country")
if not segment_cols:                    segment_cols = ["ua_country"]

# Aggregate to segment table
seg = (df.groupby(segment_cols, as_index=False)
         .agg(impressions=("impressions","sum"),
              engaged_views=("engaged_views","sum"),
              conversions=("conversions","sum"),
              cost=("cost","sum"),
              revenue=("revenue","sum")))

# KPIs
seg["EngagementRate"] = seg["engaged_views"] / seg["impressions"].clip(lower=1)
seg["CPC"]  = seg["cost"] / seg["engaged_views"].replace(0, np.nan)
seg["CPA"]  = seg["cost"] / seg["conversions"].replace(0, np.nan)
seg["ROI"]  = (seg["revenue"] - seg["cost"]) / seg["cost"].replace(0, np.nan)  # ratio
seg["ROAS"] = seg["revenue"] / seg["cost"].replace(0, np.nan)                  # x multiple

# Filter tiny/no-spend segments for stability
MIN_SEG_IMPRESSIONS = 200
MIN_SEG_SPEND = 1e-9
seg = seg[(seg["impressions"] >= MIN_SEG_IMPRESSIONS) & (seg["cost"] > MIN_SEG_SPEND)].reset_index(drop=True)

# Timestamp
RUN_TAG = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")

# Exports
seg_out = os.path.join(OUTPUT_DIR, f"ads_segment_metrics_clean_{RUN_TAG}.csv")
event_cols = ["impressions","engaged_views","conversions","cost","revenue",
              "ua_country","ua_device","placement_language","creative_duration","seconds_played"]
event_cols = [c for c in event_cols if c in df.columns]
event_out_df = df[event_cols].copy()
event_out = os.path.join(OUTPUT_DIR, f"ads_bigquery_table_{RUN_TAG}.csv")

seg.to_csv(seg_out, index=False)
event_out_df.to_csv(event_out, index=False)

print("Wrote:")
print(" -", seg_out)
print(" -", event_out)
print("\nSample segments:")
print(seg.head(8).to_string(index=False))


# Step 4 — Visuals (ROAS-driven)

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from textwrap import shorten

OUTPUT_DIR = "/content" if os.path.exists("/content") else "/mnt/data"
assert "seg" in globals(), "Run Step 2 first."

# Helper: compact segment label
seg_cols = [c for c in ["placement_language","ua_device","ua_country"] if c in seg.columns]
def seg_label_frame(frame):
    return frame[seg_cols].astype(str).agg(" | ".join, axis=1)

# 3.1 Top-10 segments by ROAS
top10 = seg.sort_values("ROAS", ascending=False).head(10).copy()
labels = [shorten(s, width=28, placeholder="…") for s in seg_label_frame(top10)]
vals = top10["ROAS"].values

plt.figure(figsize=(11,5))
plt.bar(labels, vals)
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROAS (revenue ÷ spend)")
plt.title("Top segments")
for i, v in enumerate(vals):
    plt.text(i, v*1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
top10_png = os.path.join(OUTPUT_DIR, f"viz_top10_roas_{RUN_TAG}.png")
plt.savefig(top10_png, dpi=160); plt.show()
print("Saved:", top10_png)

# 3.2 ROAS heatmap (Country × Device), ordered by spend
need = {"ua_country","ua_device"}
if need.issubset(seg.columns):
    seg_heat = seg.copy()
    # order by spend for readability
    rows = (seg_heat.groupby("ua_country")["cost"].sum().sort_values(ascending=False)).index
    cols = (seg_heat.groupby("ua_device")["cost"].sum().sort_values(ascending=False)).index
    pivot = (seg_heat.pivot_table(values="ROAS", index="ua_country", columns="ua_device", aggfunc="mean")
                      .reindex(index=rows, columns=cols))
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Device"); ax.set_ylabel("Country")
    ax.set_title("ROAS heatmap")
    cbar = plt.colorbar(im); cbar.set_label("ROAS")
    plt.tight_layout()
    heat_png = os.path.join(OUTPUT_DIR, f"viz_heatmap_roas_{RUN_TAG}.png")
    plt.savefig(heat_png, dpi=160); plt.show()
    print("Saved:", heat_png)
else:
    print("Heatmap skipped (need ua_country & ua_device)")

# 3.3 Engagement vs ROAS bubble (area ∝ spend)
sizes = 3000 * np.sqrt(seg["cost"] / seg["cost"].max()) + 20
plt.figure(figsize=(9,6))
plt.scatter(seg["EngagementRate"], seg["ROAS"], s=sizes, alpha=0.35)
plt.xlabel("Engagement rate")
plt.ylabel("ROAS (revenue ÷ spend)")
plt.title("Engagement vs ROAS")
plt.tight_layout()
bubble_png = os.path.join(OUTPUT_DIR, f"viz_bubble_eng_roas_{RUN_TAG}.png")
plt.savefig(bubble_png, dpi=160); plt.show()
print("Saved:", bubble_png)



# Step 5 — Budget reallocation + sensitivity

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "/content" if os.path.exists("/content") else "/mnt/data"
assert "seg" in globals(), "Run Step 2 first."

# Cohort config (spend-weighted)
LOW_SPEND_SHARE  = 0.50  # bottom 50% of portfolio spend
HIGH_SPEND_SHARE = 0.30  # top   30% of portfolio spend
MIN_SPEND = 1e-9

TARGET_LIFT = 0.18     # ~18%
TOL         = 0.001
MAX_ITERS   = 25
PCT_BOUNDS  = (0.00, 0.80)

def make_cohorts(seg_df, low_share=LOW_SPEND_SHARE, high_share=HIGH_SPEND_SHARE):
    s = seg_df.copy().sort_values("ROI", ascending=True).reset_index(drop=True)
    s["baseline_spend"] = s["cost"]
    total = s["baseline_spend"].sum()
    s["cum_low"]  = s["baseline_spend"].cumsum() / max(total, 1e-9)
    s["cum_high"] = s["baseline_spend"][::-1].cumsum()[::-1] / max(total, 1e-9)
    low_mask  = s["cum_low"]  <= low_share
    high_mask = s["cum_high"] <= high_share
    if s.loc[high_mask, "baseline_spend"].sum() <= MIN_SPEND:
        have_spend = s["baseline_spend"] > MIN_SPEND
        idx = s[have_spend].index
        cut = max(1, int(round(0.20 * len(idx))))
        high_mask = pd.Series(False, index=s.index)
        high_mask.loc[idx[-cut:]] = True
    return s, s.index[low_mask], s.index[high_mask]

def simulate(s, low_idx, high_idx, pct_to_shift):
    x = s.copy()
    x["new_spend"] = x["baseline_spend"]
    low_total = x.loc[low_idx, "baseline_spend"].sum()
    move = low_total * pct_to_shift
    # take from low
    x.loc[low_idx, "new_spend"] = x.loc[low_idx, "baseline_spend"] * (1.0 - pct_to_shift)
    # add to high (proportional to baseline)
    high_total = x.loc[high_idx, "baseline_spend"].sum()
    weights = x.loc[high_idx, "baseline_spend"] / (high_total if high_total > 0 else 1e-9)
    x.loc[high_idx, "new_spend"] = x.loc[high_idx, "baseline_spend"] + move * weights
    # efficiency & new revenue
    x["rev_per_dollar"] = x["revenue"] / x["baseline_spend"].replace(0, np.nan)
    x["new_revenue"]    = x["rev_per_dollar"] * x["new_spend"]
    base_roi = (x["revenue"].sum() - x["baseline_spend"].sum()) / x["baseline_spend"].sum()
    new_roi  = (x["new_revenue"].sum() - x["new_spend"].sum()) / x["new_spend"].sum()
    rel_lift = (new_roi - base_roi) / (base_roi if base_roi != 0 else 1e-9)
    abs_lift = (new_roi - base_roi)
    return base_roi, new_roi, rel_lift, abs_lift, x

def tune_for_target(s, low_idx, high_idx, target=TARGET_LIFT, bounds=PCT_BOUNDS, tol=TOL, iters=MAX_ITERS):
    lo, hi = bounds
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        b, n, L, A, out = simulate(s, low_idx, high_idx, mid)
        if best is None or abs(L - target) < abs(best[2] - target):
            best = (mid, b, n, L, A, out)
        if abs(L - target) <= tol:
            break
        if L < target: lo = mid
        else:          hi = mid
    return best  # (pct, base, new, rel_lift, abs_lift, seg_out)

# Cohorts
s, low_idx, high_idx = make_cohorts(seg, LOW_SPEND_SHARE, HIGH_SPEND_SHARE)
pct, base_roi, new_roi, rel_lift, abs_lift, seg_out = tune_for_target(s, low_idx, high_idx)

low_share  = s.loc[low_idx,  "baseline_spend"].sum() / s["baseline_spend"].sum()
high_share = s.loc[high_idx, "baseline_spend"].sum() / s["baseline_spend"].sum()

print("Cohorts (spend-weighted):")
print(f"  Low spend share:  {low_share*100:.1f}%  (segments={len(low_idx)})")
print(f"  High spend share: {high_share*100:.1f}%  (segments={len(high_idx)})")
print(f"Shift applied:      {pct*100:.2f}% of low-cohort spend")
print(f"Portfolio ROI:      {base_roi:.2f} → {new_roi:.2f}  (Δ={abs_lift:.2f}, {rel_lift*100:.2f}%)")


plt.figure(figsize=(6,4))
vals = [base_roi, new_roi]
plt.bar(["Before", "After"], vals)
for i, v in enumerate(vals):
    plt.text(i, v*1.01, f"{v:.2f}", ha="center", va="bottom")
plt.ylabel("ROI ( (revenue - spend) / spend )")
plt.title("Portfolio ROI")
plt.tight_layout()
before_after_png = os.path.join(OUTPUT_DIR, f"viz_roi_before_after_{RUN_TAG}.png")
plt.savefig(before_after_png, dpi=160); plt.show()
print("Saved:", before_after_png)


scenarios = []
for p in [0.10, 0.25, 0.50]:
    b, n, L, A, _ = simulate(s, low_idx, high_idx, p)
    scenarios.append({"pct_shift_low_group": p, "baseline_roi": b, "new_roi": n, "lift_pct": L, "abs_lift": A})
sens_df = pd.DataFrame(scenarios)

print("\nSensitivity:")
print(sens_df.to_string(index=False, formatters={
    "pct_shift_low_group": lambda x: f"{x*100:.0f}%",
    "baseline_roi":        lambda x: f"{x:.2f}",
    "new_roi":             lambda x: f"{x:.2f}",
    "lift_pct":            lambda x: f"{x*100:.2f}%",
    "abs_lift":            lambda x: f"{x:.2f}",
}))

plt.figure(figsize=(6,4))
x = [p*100 for p in sens_df["pct_shift_low_group"]]
y = [l*100 for l in sens_df["lift_pct"]]
plt.plot(x, y, marker="o")
for xi, yi in zip(x, y):
    plt.text(xi, yi*1.01, f"{yi:.1f}%", ha="center", va="bottom")
plt.xlabel("% of low-cohort spend shifted")
plt.ylabel("Portfolio ROI lift (%)")
plt.title("Sensitivity")
plt.grid(True, alpha=0.3)
plt.tight_layout()
sens_png = os.path.join(OUTPUT_DIR, f"viz_lift_sensitivity_{RUN_TAG}.png")
plt.savefig(sens_png, dpi=160); plt.show()
print("Saved:", sens_png)

# CSV exports with run tag
headline = pd.DataFrame({
    "metric": ["baseline_roi","new_roi","lift_pct","abs_lift","pct_shift_low_group","low_spend_share","high_spend_share"],
    "value":  [base_roi, new_roi, rel_lift,  abs_lift,  pct,                low_share,          high_share]
})
headline_out = os.path.join(OUTPUT_DIR, f"ads_budget_reallocation_headline_{RUN_TAG}.csv")
sens_out     = os.path.join(OUTPUT_DIR, f"ads_budget_reallocation_scenarios_{RUN_TAG}.csv")
headline.to_csv(headline_out, index=False)
sens_df.to_csv(sens_out, index=False)

print("\nWrote:")
print(" -", headline_out)
print(" -", sens_out)
