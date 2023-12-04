import numpy as np
import pandas as pd

from HierarchicalClustering import HierarchicalClustering
from SpectralClustering import SpectralClustering

import time

"""
Class for GALE
"""


class SubgraphSelector_Online:
    subgraph_selection_alg = 'Random'
    runtime_clustering = 0.0
    n_unused_subgraphs = 0
    n_unused_nodes = 0
    adj_matrices = []
    adj_estimates = []
    indices = []
    labels_subgraphs = []

    def __init__(self, n_nodes, n_subgraphs, size_subgraphs, n_clusters, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC', forgetting_factor=1):
        self.time_step = 0
        self.ID = ID
        self.N = n_subgraphs
        self.m = size_subgraphs
        self.K = n_clusters
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        self.forgetting_factor = forgetting_factor
        self.n_nodes = n_nodes
        time_start_selection = time.time()
        self.selectSubgraphs()
        time_end_selection = time.time()
        self.runtime_selection = time_end_selection - time_start_selection

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'subgraph_sel_alg': self.subgraph_selection_alg,
                                'base_alg': self.parent_alg,
                                'n_subgraphs': self.N,
                                'size_subgraphs': self.m,
                                'forgetting_factor': self.forgetting_factor,
                                'n_unused_nodes': self.n_unused_nodes,
                                'runtime_subgraph_clustering': self.runtime_clustering,
                                'runtime_subgraph_selection': self.runtime_selection,
                                'time_step_selector': self.time_step,
                                }])
        return var_df

    def getIndices(self):
        return self.indices

    """
    Random selection of subgraphs 
    The subgraphs will be stored in the DataFrame 'subgraphs'
    """

    def selectSubgraphs(self):
        n = self.n_nodes
        m = self.m
        N = self.N
        indices = []
        for _ in np.arange(N):
            # randomly choose m indices out of [n] (0 included, n excluded)
            index_set = np.random.choice(n, size=m, replace=False)
            index_set = np.sort(index_set)
            indices.append(index_set)

        # count oob-samples
        union = np.unique(indices)
        oob_samples = np.setdiff1d(np.arange(n), union)
        self.n_unused_nodes = len(oob_samples)

        self.indices = indices
        print(' Selected N =', N, ' subgraphs of size m =', m)

    def getAdjacencyMatrices(self, adj_matrix):
        indices = self.indices
        full_adj_matrix = adj_matrix

        adj_arr = []
        for index_set in indices:
            # get a grid to extract the submatrix
            ixgrid = np.ix_(index_set, index_set)
            adj = full_adj_matrix[ixgrid]
            adj_arr.append(adj)
        self.adj_matrices = adj_arr

    """
    Perform Clustering on each subgraph
    The Clustering results (labels) will be stored in the Dataframe 'subgraphs'
    """

    def clusterSubgraphs(self):
        clustering_results_array = []
        time_step = self.time_step
        adj_matrices = self.adj_matrices
        n_clusters = self.K
        parent_alg = self.parent_alg
        forgetting_factor = self.forgetting_factor
        N = self.N

        # static spectral clustering
        if parent_alg == 'SC':
            for adj in adj_matrices:
                SC_object = SpectralClustering(ID=self.ID, adjacency=adj, n_clusters=n_clusters)
                SC_result = SC_object.performSC()
                clustering_results_array.append(SC_result)

        # static hierarchical clustering
        if parent_alg == 'HC':
            for adj in adj_matrices:
                HC_object = HierarchicalClustering(ID=self.ID, adjacency=adj, n_clusters=n_clusters)
                HC_result = HC_object.performHC()
                clustering_results_array.append(HC_result)

        # evolutionary spectral or hierarchical clustering
        if parent_alg == 'evSC' or parent_alg == 'evHC':

            # get according estimates
            if time_step == 0:
                adj_estimates_next = adj_matrices

            else:
                adj_estimates = self.adj_estimates
                adj_estimates_next = []
                for index in np.arange(N):
                    adj_estimate = forgetting_factor * adj_matrices[index] + (1 - forgetting_factor) * adj_estimates[index]
                    adj_estimates_next.append(adj_estimate)
            self.adj_estimates = adj_estimates_next

            if parent_alg == 'evSC':
                for adj_estimate in adj_estimates_next:
                    SC_object = SpectralClustering(ID=self.ID, adjacency=adj_estimate, n_clusters=n_clusters)
                    SC_result = SC_object.performSC()
                    clustering_results_array.append(SC_result)

            if parent_alg == 'evHC':
                for adj_estimate in adj_estimates_next:
                    HC_object = HierarchicalClustering(ID=self.ID, adjacency=adj_estimate, n_clusters=n_clusters)
                    HC_result = HC_object.performHC()
                    clustering_results_array.append(HC_result)

        self.labels_subgraphs = clustering_results_array
        print(' Performed clustering algorithm', parent_alg, 'on all subgraphs for K =', n_clusters, 'clusters')

    def predict_subgraph_labels(self, adj_matrix):
        time_start_clustering = time.time()
        time_step = self.time_step
        self.getAdjacencyMatrices(adj_matrix)
        self.clusterSubgraphs()
        time_end_clustering = time.time()
        self.runtime_clustering = np.round(time_end_clustering - time_start_clustering, 4)

        time_step += 1
        self.time_step = time_step

        print('Clustered the subgraphs for time step t=', time_step)
        return self.labels_subgraphs
