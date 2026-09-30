import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("watchparty_checkins.csv")

city = df.groupby("city").agg(
    posts=("post_id","count"),
    engagement=("engagement","sum"),
    lat=("latitude","mean"),
    lon=("longitude","mean")
).sort_values("engagement", ascending=False)

city["avg_engagement"] = (
    city.engagement / city.posts
).round(1)

print("City-wise engagement:\n", city.round(4))
print("\nTop 3 engagement hotspots:",
      ", ".join(city.index[:3]))

print("\nEngagement by venue type:\n",
      df.groupby("venue_type").engagement
      .agg(["count","sum","mean"]).round(1))

import folium
from folium.plugins import HeatMap

m = folium.Map(
    [df.latitude.mean(), df.longitude.mean()],
    zoom_start=5
)

HeatMap(
    df[["latitude","longitude","engagement"]].values.tolist(),
    radius=13, blur=18
).add_to(m)

m.save("watchparty_heatmap.html")
print("\nInteractive heat map saved as watchparty_heatmap.html")

fig, ax = plt.subplots(1,2, figsize=(12,5))

hb = ax[0].hexbin(
    df.longitude, df.latitude,
    C=df.engagement,
    reduce_C_function=np.sum,
    gridsize=38, cmap="YlOrRd"
)

plt.colorbar(hb, ax=ax[0], label="total engagement")

for c,r in city.iterrows():
    ax[0].annotate(c, (r.lon,r.lat), fontsize=8)

ax[0].set_xlabel("longitude")
ax[0].set_ylabel("latitude")
ax[0].set_title("Engagement heat map of geo-tagged check-ins")

city.engagement.plot.barh(ax=ax[1])
ax[1].set_title("Total engagement by city")
ax[1].invert_yaxis()

plt.tight_layout()
plt.show()
