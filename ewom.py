# pract 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

df = pd.read_csv("streaming_device_reviews.csv")

print("Rating distribution:")
print(df.star_rating.value_counts().sort_index())

print("\nAverage rating:")
print(
    df.groupby("product").star_rating
    .agg(["count", "mean"])
    .round(2)
)

df["helpfulness"] = (
    df.helpful_votes /
    df.total_votes.replace(0, np.nan)
)

print("\n mean Helpfulness ratio:",
      round(df.helpfulness.mean(), 3))

print("\nVerified vs unverified:")
print("\nVerified vs unverified:")

print(
    df.groupby("verified_purchase").agg(
        reviews=("review_id", "count"),
        avg_rating=("star_rating", "mean"),
        avg_helpfulness=("helpfulness", "mean")
    ).round(3)
)

sia = SentimentIntensityAnalyzer()

df["compound"] = df.review_text.apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

df["sentiment"] = np.where(
    df.compound >= .05,
    "positive",
    np.where(df.compound <= -.05, "negative", "neutral")
)

df["rating_class"] = np.where(
    df.star_rating >= 4,
    "positive",
    np.where(df.star_rating <= 2, "negative", "neutral")
)

print("\nSentiment vs rating:")
print(pd.crosstab(df.rating_class, df.sentiment))

print(
    "\nAgreement between text sentiment and star rating:",
    round(
        (df.sentiment == df.rating_class).mean() * 100,
        1
    ),
    "%"
)

print(
    "Correlation between compound score and star rating:",
    round(df.compound.corr(df.star_rating), 3)
)

v = df[df.total_votes >= 20]

for s in ["positive", "negative"]:
    r = v[v.sentiment == s].nlargest(
        1, "helpful_votes"
    ).iloc[0]

    print(f"\nMost helpful {s}:")
    print(r.star_rating, r.helpful_votes, "/", r.total_votes)
    print(r.review_text)

df.star_rating.value_counts().sort_index().plot.bar()

plt.title("Star Rating Distribution")
plt.show()

plt.figure()

df.groupby("star_rating").helpfulness.mean().plot.bar()

plt.title("Mean helpfulness ratio by rating")
plt.show()

plt.figure()

pd.crosstab(
    df.rating_class,
    df.sentiment
).plot.bar(stacked=True)

plt.title("Text sentiment vs star rating")
plt.show()
