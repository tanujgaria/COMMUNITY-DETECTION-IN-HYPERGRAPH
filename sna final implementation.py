# ===== STEP 3: CODE =====

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Program started...")

# Load dataset
edges = []
with open('gene_disease.txt', 'r') as f:
    for line in f:
        edges.append(list(map(int, line.strip().split(','))))

print("Edges loaded:", len(edges))

# Create nodes
nodes = sorted(set([n for edge in edges for n in edge]))
node_index = {n:i for i,n in enumerate(nodes)}

n = len(nodes)
m = len(edges)

# Create H matrix
H = np.zeros((n, m))
for j, edge in enumerate(edges):
    for node in edge:
        H[node_index[node], j] = 1

print("H created:", H.shape)

# IRMM
import community.community_louvain as community_louvain

A = H @ H.T
np.fill_diagonal(A, 0)

G = nx.from_numpy_array(A)
partition = community_louvain.best_partition(G)

print("IRMM:", len(set(partition.values())))

# HGNN
Dv = np.diag(np.sum(H, axis=1))
De = np.diag(np.sum(H, axis=0))

Dv_inv_sqrt = np.linalg.inv(np.sqrt(Dv + 1e-5))
De_inv = np.linalg.inv(De + 1e-5)

G_hgnn = Dv_inv_sqrt @ H @ De_inv @ H.T @ Dv_inv_sqrt

X = np.eye(n)
X_new = G_hgnn @ X

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X_new)

print("HGNN:", len(set(labels)))

# FMHMO
E_sim = H.T @ H
G_e = nx.from_numpy_array(E_sim)

partition_e = community_louvain.best_partition(G_e)
num_clusters = len(set(partition_e.values()))

membership = np.zeros((n, num_clusters))

for e_idx, cluster in partition_e.items():
    for node in range(n):
        if H[node, e_idx] == 1:
            membership[node, cluster] += 1

membership = membership / (membership.sum(axis=1, keepdims=True) + 1e-5)
final_labels = np.argmax(membership, axis=1)

print("FMHMO:", len(set(final_labels)))

# Visualization
G_small = G.subgraph(range(50))
G_hgnn_graph = nx.from_numpy_array(G_hgnn)
G_small_hgnn = G_hgnn_graph.subgraph(range(50))

fig, axes = plt.subplots(1,3, figsize=(15,5))

nx.draw(G_small, node_color=[partition[i] for i in range(50)],
        node_size=30, cmap=plt.cm.rainbow, ax=axes[0])
axes[0].set_title("IRMM")

nx.draw(G_small_hgnn, node_color=labels[:50],
        node_size=30, cmap=plt.cm.viridis, ax=axes[1])
axes[1].set_title("HGNN")

nx.draw(G_small, node_color=final_labels[:50],
        node_size=30, cmap=plt.cm.plasma, ax=axes[2])
axes[2].set_title("FMHMO")

plt.show()