import numpy as np
# stochastic block model
import networkx as nx

# ToDo: Es sollte eine Methode geben, die als Inputfaktoren alle wichtigen Parameter hat,
#  die für die Konsistenzschranke wichtig sind.
#  Also gamma, alpha, n, n_min, n_max, lambda ...

"""
k = number of clusters
alpha = sparsity parameter
n_min = minimal cluster size
n_max = maximal cluster size
"""


def simulate_SBM_1(k, alpha, n_min, n_max):
    # generate random community sizes
    n_k = np.random.randint(n_min, n_max + 1, size=k)
    n = sum(n_k)

    # Connectivity Matrix
    r_matrix = np.random.uniform(low=0, high=0.5, size=(k, k))
    p_matrix = np.diag(np.random.uniform(low=0, high=0.5, size=(k,)))
    B = alpha * (r_matrix + p_matrix)

    graph = nx.stochastic_block_model(sizes=n_k, p=B, seed=123)

    # get adjacency matrix
    adj = np.zeros([n, n], dtype=int)
    for edge in graph.edges:
        adj[edge] += 1
    adj = adj + adj.transpose()

    # array of true labels
    y_true = np.repeat([i for i in np.arange(k)], n_k)

    return adj, y_true


"""
k = number of clusters
s = size of each cluster
r = connectivity probability between different clusters
r + p = connectivity probability in the same cluster
"""


def simulate_SBM_2(k, s, r, p):
    n = k * s
    # Community size Matrix
    com_sizes = [s for _ in range(k)]

    # Connectivity Matrix
    r_matrix = np.full((k, k), r)
    p_matrix = np.diag(np.array([p] * k))
    B = r_matrix + p_matrix

    graph = nx.stochastic_block_model(sizes=com_sizes, p=B, seed=123)

    # get adjacency matrix
    adj = np.zeros([n, n], dtype=int)
    for edge in graph.edges:
        adj[edge] += 1
    adj = adj + adj.transpose()

    # array of true labels
    y_true = np.repeat([i for i in range(k)], s)

    return adj, y_true


"""
Input:  some clustering array constructed through SBM
Do:     Permutate s entries of the clustering array
"""


def simulate_DSBM(clustering_labels):
    # ToDo
    return clustering_labels
