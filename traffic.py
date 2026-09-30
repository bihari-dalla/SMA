# Pract 8

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cinestream_web_traffic.csv")

p = df.groupby("page_path").agg(
    pageviews=("session_id", "count"),
    avg_time=("time_on_page_sec", "mean"),
    entries=("is_entry", "sum"),
    exits=("is_exit", "sum")
)

p["exit_rate"] = p.exits / p.pageviews * 100

print(
    "Ten most visited pages:\n",
    p.sort_values("pageviews", ascending=False).head(10).round(1)
)

print(
    "\nTop entry pages:\n",
    p.nlargest(5, "entries").entries
)

print(
    "\nTop exit pages:\n",
    p[p.pageviews >= 50]
    .nlargest(5, "exit_rate")
    .exit_rate.round(1)
)

print(
    "\nSessions by device:\n",
    df.groupby("device").session_id.nunique()
)

print(
    "\nSessions by referrer:\n",
    df.groupby("referrer_source")
    .session_id.nunique()
    .sort_values(ascending=False)
)

df = df.sort_values(["session_id", "step_in_session"])

df["next"] = df.groupby("session_id").page_path.shift(-1)

trans = df.dropna(subset=["next"]).groupby(
    ["page_path", "next"]
).size().nlargest(10)

print(
    "\nTen most common page-to-page transitions:\n",
    trans
)

paths = df.groupby("session_id").page_path.apply(
    lambda x: " > ".join(x.head(3))
)

print(
    "\nMost common three-step paths:\n",
    paths[paths.str.count(">") == 2].value_counts().head(5)
)

n = df.session_id.nunique()

signup = set(
    df[df.page_path == "/signup"].session_id
)

payment = set(
    df[df.page_path == "/payment"].session_id
)

welcome = set(
    df[df.page_path == "/welcome"].session_id
)

print(
    f"\nSign-up funnel: all sessions {n} -> "
    f"/signup {len(signup)} ({len(signup)/n:.1%})"
)

print(
    f" -> /payment {len(payment)} "
    f"({len(payment)/len(signup):.1%} of sign-ups)"
)

print(
    f" -> /welcome {len(welcome)} "
    f"({len(welcome)/len(payment):.1%} of payments)"
)

print(
    f"Drop-off: {len(signup)-len(payment)} sessions abandon "
    f"between /signup and /payment; "
    f"/signup exit rate is {p.loc['/signup','exit_rate']:.1f}%."
)

# Charts

p.pageviews.head(10).sort_values().plot.barh(
    title="Top 10 pages by pageviews"
)

plt.show()

trans.sort_values().plot.barh(
    title="Most common page transitions",
    color="orange"
)

plt.show()
