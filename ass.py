# Prac 7

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("watch_sessions.csv")

baskets = df.shows_watched.str.split(",").tolist()

print(f"Sessions: {len(baskets)}  average titles per session: {np.mean([len(x) for x in baskets]):.2f}")

te = TransactionEncoder()

onehot = pd.DataFrame(
    te.fit(baskets).transform(baskets),
    columns=te.columns_
)

print(f"One-hot matrix: {onehot.shape[0]} x {onehot.shape[1]}")

print("\nTop 10 titles by support:")
print(
    onehot.mean()
    .sort_values(ascending=False)
    .head(10)
    .round(4)
)

freq = apriori(
    onehot,
    min_support=.05,
    use_colnames=True
)

freq["length"] = freq.itemsets.apply(len)

print(f"\nFrequent itemsets at min_support=0.05: {len(freq)}")

print(
    freq[freq.length > 1]
    .nlargest(8, "support")
    .to_string(index=False)
)

rules = association_rules(
    freq,
    metric="lift",
    min_threshold=1
)

rules = rules[rules.confidence >= .35]

print(f"\nRules generated: {len(rules)}")

rules["antecedents"] = rules.antecedents.apply(
    lambda x: ", ".join(sorted(x))
)

rules["consequents"] = rules.consequents.apply(
    lambda x: ", ".join(sorted(x))
)

top = rules.nlargest(10, "lift")

print("\nTop 10 rules by lift:")

print(
    top[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ]
    .round(3)
    .to_string(index=False)
)

r = top.iloc[0]

print(
    f"\nRecommender use 1: after a subscriber finishes "
    f"'{r.antecedents}', "
    f"surface '{r.consequents}' in the 'Because you watched' row - "
    f"co-watched {r.confidence:.0%} of the time, lift {r.lift:.2f}."
)

print(
    f"Recommender use 2: group '{r.consequents}' and "
    f"'{r.antecedents}' into one curated collection and promote "
    f"them together in the weekend push notification "
    f"(lift {r.lift:.2f})."
)

# Chart 1

plt.scatter(
    rules.support,
    rules.confidence,
    c=rules.lift
)

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Rules: support vs confidence")
plt.colorbar(label="Lift")
plt.show()

# Chart 2

top.set_index("antecedents").lift.sort_values().plot.barh()

plt.xlabel("Lift")
plt.title("Top 10 rules by lift")
plt.tight_layout()
plt.show()
