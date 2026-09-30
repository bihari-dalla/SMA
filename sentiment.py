import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

df = pd.read_csv(
    "awards_tweets.csv",
    parse_dates=["created_at"]
)

# Sentiment scores
df["polarity"] = df.tweet_text.apply(
    lambda t: round(TextBlob(t).sentiment.polarity, 3)
)

df["subjectivity"] = df.tweet_text.apply(
    lambda t: round(TextBlob(t).sentiment.subjectivity, 3)
)

# Classification
df["sentiment"] = np.where(
    df.polarity > 0, "positive",
    np.where(df.polarity < 0, "negative", "neutral")
)

# Distribution
dist = df.sentiment.value_counts()

print("Sentiment distribution:\n",
      dist.to_string())

print("\nPercentage share:\n",
      (dist / len(df) * 100).round(2).to_string())

print(
    f"\nMean polarity {df.polarity.mean():.3f}, "
    f"mean subjectivity {df.subjectivity.mean():.3f}"
)

# Engagement
df["engagement"] = df.likes + df.retweets

print("\nAverage engagement by sentiment class:\n",
      df.groupby("sentiment").agg(
          tweets=("tweet_id", "count"),
          avg_likes=("likes", "mean"),
          avg_retweets=("retweets", "mean"),
          avg_engagement=("engagement", "mean")
      ).round(1).to_string()
)

# Top positive and negative tweets
uniq = df.drop_duplicates("tweet_text")

print("\n5 most positive tweets:")
for _, r in uniq.nlargest(5, "polarity").iterrows():
    print(f"[{r.polarity:+.2f}] {r.tweet_text}")

print("\n5 most negative tweets:")
for _, r in uniq.nsmallest(5, "polarity").iterrows():
    print(f"[{r.polarity:+.2f}] {r.tweet_text}")

# Daily trend
daily = df.set_index("created_at").resample("D").polarity.mean()

# Charts
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

dist.plot.bar(
    ax=ax[0],
    color=["#55A868", "#C44E52", "#999999"],
    rot=0
)
ax[0].set_title("Tweet sentiment distribution")

daily.plot(
    ax=ax[1],
    marker="o",
    ms=4,
    color="#4C72B0"
)
ax[1].axhline(0, ls="--", c="grey")
ax[1].set_ylabel("mean polarity")
ax[1].set_title("Daily sentiment trend")

plt.tight_layout()
plt.show()

print("Chart generated")
