import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cinestream_content.csv")

df["engagement"] = df.likes + df.shares + df.comments

def classify(r):
    if r.video_length_sec > 0:
        return "Video data"
    if r.n_images > 0:
        return "Image data"
    if r.external_links > 0:
        return "Hyperlink data"
    return "Textual data"

df["data_type"] = df.apply(classify, axis=1)

s = df.groupby("data_type").agg(
    posts=("content_id", "count"),
    avg_engagement=("engagement", "mean")
).round(2)

s["share_pct"] = (s.posts / len(df) * 100).round(2)

print(s)

print("\nCross-tab:")
print(pd.crosstab(df.data_type, df.platform))


# Graph 1: Number of posts
s["posts"].plot.bar()
plt.title("Posts by social media data type")
plt.xlabel("data_type")
plt.ylabel("number of posts")
plt.tight_layout()
plt.show()


# Graph 2: Average engagement
s["avg_engagement"].plot.bar()
plt.title("Average engagement by social media data type")
plt.xlabel("data_type")
plt.ylabel("average engagement")
plt.tight_layout()
plt.show()
