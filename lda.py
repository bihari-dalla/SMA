# Pract 6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

df = pd.read_csv("show_reviews.csv")

stop = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def clean(t):
    t = re.sub(r"http\S+|@\w+|#", " ", t)
    t = re.sub(r"[^A-Za-z\s]", " ", t).lower()
    return " ".join(lem.lemmatize(w) for w in word_tokenize(t)
                    if w not in stop and len(w) > 2)

df["clean_text"] = df.review_text.apply(clean)

for _, r in df.head(3).iterrows():
    print("RAW:", r.review_text[:88])
    print("CLEAN:", r.clean_text[:88], "\n")

print("Top words:", Counter(" ".join(df.clean_text).split()).most_common(10))

cv = CountVectorizer(max_df=.85, min_df=3)
X = cv.fit_transform(df.clean_text)

print("Document-term matrix:", X.shape)

lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42,
    max_iter=50
)

W = lda.fit_transform(X)

terms = np.array(cv.get_feature_names_out())

for k, c in enumerate(lda.components_):
    print("Topic", k, terms[c.argsort()[-10:][::-1]])

df["topic"] = W.argmax(1)

names = [
    "Story & writing",
    "Subtitles & dubbing",
    "Acting & cast",
    "Streaming quality"
]

df["topic_name"] = df.topic.map(dict(enumerate(names)))
df["prob"] = W.max(1)

print("\nTopic labels:")

for i, n in enumerate(names):
    print(f"Topic {i} -> {n}")

print("\nTopic distribution:")

print(
    df.topic_name.value_counts()
    .to_frame("reviews")
    .assign(
        pct=lambda x: (x.reviews / len(df) * 100).round(1)
    )
)

print("\nAverage rating per topic:")

print(
    df.groupby("topic_name")
    .rating.mean()
    .round(2)
    .sort_values()
)

print("\nMost representative review of each topic:")

for n, g in df.groupby("topic_name"):
    r = g.loc[g.prob.idxmax()]
    print(f"[{n}, p={r.prob:.3f}]\n {r.review_text[:100]}")

plt.figure(figsize=(6,4))

df.topic_name.value_counts().plot.bar()

plt.title("Reviews per discovered topic")
plt.xlabel("topic_name")
plt.show()

plt.figure(figsize=(6,4))

pd.crosstab(
    df.topic_name,
    df.rating
).plot.bar(stacked=True)

plt.title("Rating mix within each topic")
plt.xlabel("topic_name")
plt.show()
