import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import torch

from utils import load_data

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Convert sparse adj to dense for visualization (only for small graphs like Cora)
# Note: Cora has 2708 nodes, plotting the full graph might be messy.
# We will plot a subgraph or just the structure.

adj_dense = adj.to_dense().cpu().numpy()

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_dense)

# Color mapping based on labels
colors = labels.cpu().numpy()

# Plot
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42) # Seed for reproducibility
nx.draw(G, pos, node_color=colors, cmap=plt.cm.jet, with_labels=False, node_size=10, alpha=0.6)
plt.title("Cora Dataset Graph Structure (Colored by Class)")
plt.show()