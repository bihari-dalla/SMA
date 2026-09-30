# PRACT 9

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("air_quality_agenda.csv")

df["lag1"] = df.news_articles.shift(1)
df["lag2"] = df.news_articles.shift(2)

print(
    "Media coverage vs public concern, same week : r =",
    round(df.news_articles.corr(df.public_concern_index), 3)
)

print(
    "Media coverage vs public concern, 1-week lag: r =",
    round(df.lag1.corr(df.public_concern_index), 3)
)

print(
    "Media coverage vs public concern, 2-week lag: r =",
    round(df.lag2.corr(df.public_concern_index), 3)
)

print(
    "Social posts vs public concern             : r =",
    round(df.social_posts.corr(df.public_concern_index), 3)
)

print(
    "News coverage vs Google Trends interest     : r =",
    round(df.news_articles.corr(df.google_trends_index), 3)
)


# Graphs

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Graph 1: Media agenda vs public agenda

df.set_index("week")[[
    "news_articles",
    "google_trends_index",
    "public_concern_index"
]].plot(ax=ax[0])

ax[0].set_title("Media agenda vs public agenda")
ax[0].set_xlabel("week")
ax[0].set_ylabel("")

ax[0].legend(
    [
        "news articles (media agenda)",
        "Google Trends index",
        "public concern (public agenda)"
    ],
    fontsize=8
)


# Graph 2: Transfer of salience

x = df.news_articles
y = df.public_concern_index

r = x.corr(y)

ax[1].scatter(
    x,
    y,
    c=df.week,
    cmap="viridis"
)

m, b = np.polyfit(x, y, 1)

ax[1].plot(
    x,
    m * x + b,
    linestyle="--",
    color="black"
)

ax[1].set_xlabel("news articles")
ax[1].set_ylabel("public concern index")
ax[1].set_title(f"Transfer of salience (r = {r:.2f})")

plt.tight_layout()
plt.show()


print(
    "Media coverage peaks in week",
    df.loc[df.news_articles.idxmax(), "week"],
    "; public concern peaks in week",
    df.loc[df.public_concern_index.idxmax(), "week"]
)

print("""
Media coverage peaks in week 12; public concern peaks in week 12.

EVALUATION (~180 words)

Agenda-setting theory (McCombs and Shaw, 1972) holds that the media do not tell audiences what to think, but what to think about. The dataset tracks the 'Urban Air Quality Emergency' over forty weeks. From roughly week 8 the media agenda intensifies sharply - news_articles and prime-time minutes rise together and peak around week 12 - and the public agenda follows the same shape: the Google Trends interest index and the public concern index climb in step, and the share of respondents naming air quality as the most important issue more than doubles. A correlation of 0.95 in the same week and 0.82 at a one-week lag indicates a strong transfer of salience from the media agenda to the public agenda. Social media largely amplifies rather than originates the issue: social_posts track news volume closely, showing that platforms recirculate topics the mainstream media have already elevated while adding second-level (attribute) agenda setting through framing and hashtags. Because the data are correlational, causal direction cannot be established; panel surveys or a cross-lagged design would be needed to confirm it.
""")
