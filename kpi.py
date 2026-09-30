import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("subscription_campaigns.csv")

# Calculate KPIs
df["ctr_pct"] = (df.clicks / df.impressions * 100).round(2)
df["engagement_rate_pct"] = (
    df.engagements / df.impressions * 100
).round(2)
df["conversion_rate_pct"] = (
    df.conversions / df.clicks * 100
).round(2)
df["cpc"] = (df.spend_inr / df.clicks).round(2)
df["cpa"] = (
    df.spend_inr /
    df.conversions.replace(0, np.nan)
).round(2)
df["roas"] = (
    df.revenue_inr / df.spend_inr
).round(2)
df["reach_ratio"] = (
    df.reach / df.impressions
).round(3)

# First 8 rows
print("Per-record KPIs (first 8 rows):\n",
      df[["campaign", "channel", "ctr_pct",
          "conversion_rate_pct", "cpc", "cpa",
          "roas"]].head(8).to_string(index=False))

# Campaign-wise ROAS
camp = df.groupby("campaign").agg(
    spend=("spend_inr", "sum"),
    revenue=("revenue_inr", "sum"),
    conversions=("conversions", "sum"),
    ctr=("ctr_pct", "mean"),
    conv_rate=("conversion_rate_pct", "mean"),
    cpa=("cpa", "mean")
)

camp["roas"] = (
    camp.revenue / camp.spend
).round(2)

camp = camp.round(2).sort_values(
    "roas", ascending=False
)

print("\nCampaigns ranked by ROAS:\n",
      camp.to_string())

# Channel-wise ROAS
chan = df.groupby("channel").agg(
    spend=("spend_inr", "sum"),
    revenue=("revenue_inr", "sum"),
    ctr=("ctr_pct", "mean"),
    cpa=("cpa", "mean")
)

chan["roas"] = (
    chan.revenue / chan.spend
).round(2)

chan = chan.round(2).sort_values(
    "roas", ascending=False
)

print("\nChannels ranked by ROAS:\n",
      chan.to_string())

# Best and worst channel
print(
    f"\nBest channel : {chan.index[0]} "
    f"(ROAS {chan.roas.iloc[0]}, "
    f"CPA Rs {chan.cpa.iloc[0]})"
)

print(
    f"Worst channel: {chan.index[-1]} "
    f"(ROAS {chan.roas.iloc[-1]}, "
    f"CPA Rs {chan.cpa.iloc[-1]})"
)

# Recommendation
print(
    f"\nBudget recommendation: move spend to "
    f"'{camp.index[0]}', which returns "
    f"Rs {camp.roas.iloc[0]} per rupee "
    f"at a CPA of Rs {camp.cpa.iloc[0]}, "
    f"and cut '{camp.index[-1]}' "
    f"(ROAS only {camp.roas.iloc[-1]})."
)

# Two charts in 2 columns
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

# Chart 1: ROAS by campaign
camp.roas.plot.bar(
    ax=ax[0],
    color=np.where(
        camp.roas >= 1,
        "#55A868",
        "#C44E52"
    ),
    rot=35
)

ax[0].axhline(
    1, ls="--", c="k", lw=.8
)
ax[0].set_title("ROAS by campaign")
ax[0].set_ylabel("revenue / spend")

# Chart 2: TWO BARS under each channel
chan[["ctr", "roas"]].plot.bar(
    ax=ax[1],
    rot=25
)

ax[1].set_title(
    "CTR (%) and ROAS by channel"
)

plt.tight_layout()
plt.show()

print("Chart generated")
