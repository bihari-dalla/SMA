import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("fan_interactions.csv")

G = nx.from_pandas_edgelist(
    df, "source_user", "target_user",
    edge_attr="weight", create_using=nx.DiGraph()
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", round(nx.density(G),4))
print("Weakly connected components:",
      nx.number_weakly_connected_components(G))
print("Average in-degree:",
      round(sum(dict(G.in_degree()).values())/G.number_of_nodes(),2))

U = G.to_undirected()

cent = pd.DataFrame({
    "in_degree": dict(G.in_degree()),
    "out_degree": dict(G.out_degree()),
    "betweenness": nx.betweenness_centrality(G),
    "eigenvector": nx.eigenvector_centrality(U, max_iter=1000),
    "pagerank": nx.pagerank(G),
    "clustering": nx.clustering(U)
}).round(4)

for x in ["in_degree","betweenness","eigenvector","pagerank"]:
    print("\nTop 5 by", x)
    print(cent.nlargest(5,x)[[x]])

print("\nAverage clustering:",
      round(nx.average_clustering(U),4))
print("Transitivity:",
      round(nx.transitivity(U),4))

pos = nx.spring_layout(U, seed=42, k=.35)

plt.figure(figsize=(9,7))
nx.draw_networkx_edges(U, pos, alpha=.15)
nx.draw_networkx_nodes(
    U, pos,
    node_size=1200 * cent.pagerank,
    alpha=.8
)
nx.draw_networkx_labels(
    U, pos,
    {n:n for n in cent.nlargest(8,"pagerank").index},
    font_size=9
)

plt.title("Fan interaction network (node size = PageRank)")
plt.axis("off")
plt.tight_layout()
plt.show()
