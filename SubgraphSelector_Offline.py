import numpy as np
import pandas as pd

from HierarchicalClustering import HierarchicalClustering
from SpectralClustering import SpectralClustering

import time

"""
Class for GALE
"""


class SubgraphSelector_Offline:
    subgraph_selection_alg = 'Random'
    n_nodes = 0
    runtime = 0.0
    n_unused_subgraphs = 0
    n_unused_nodes = 0

    def __init__(self, SBMs, n_subgraphs, size_subgraphs, n_clusters, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC', forgetting_factor=1):
        self.ID = ID
        self.adjacencies = SBMs['adj_matrix']
        self.N = n_subgraphs
        self.m = size_subgraphs
        self.T = len(SBMs['adj_matrix'])
        self.K = n_clusters
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        self.forgetting_factor = forgetting_factor
        n_nodes = len(SBMs['adj_matrix'][0])
        print('n_nodes = ', n_nodes)
        self.n_nodes = n_nodes
        self.subgraphs_df = pd.DataFrame(index=range(n_subgraphs))

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
                                'subgraph_runtime': self.runtime,
                                }])
        return var_df

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

        self.subgraphs_df['indices'] = indices
        print(' Selected N =', N, ' subgraphs of size m =', m)

    def getAdjacencyMatrices(self):
        subgraphs_df = self.subgraphs_df
        T = self.T
        full_adjacencies = self.adjacencies

        for t in np.arange(T):
            adj_arr = []
            # get full adjacency matrix of graph of time t
            full_adj = full_adjacencies[t]
            for index_set in subgraphs_df['indices']:
                # get a grid to extract the submatrix
                ixgrid = np.ix_(index_set, index_set)
                adj = full_adj[ixgrid]
                adj_arr.append(adj)
            subgraphs_df['adj_' + str(int(t))] = adj_arr
        self.subgraphs_df = subgraphs_df

    """
    Perform Clustering on each subgraph 
    The Clustering results (labels) will be stored in the Dataframe 'subgraphs'
    """

    def clusterSubgraphs(self):
        subgraphs_for_clustering = self.subgraphs_df
        N = self.N
        n_clusters = self.K
        parent_alg = self.parent_alg
        T = self.T
        forgetting_factor = self.forgetting_factor

        # static spectral clustering
        if parent_alg == 'SC':
            for t in np.arange(T):
                clustering_results_array = []
                for adj in subgraphs_for_clustering['adj_' + str(t)]:
                    SC_object = SpectralClustering(ID=self.ID, adjacency=adj, n_clusters=n_clusters)
                    SC_result = SC_object.performSC()
                    clustering_results_array.append(SC_result)
                subgraphs_for_clustering['clus_labels_' + str(t)] = clustering_results_array

        # evolutionary spectral clustering
        if parent_alg == 'evSC':
            # save evolutionary estimates
            adj_estimates = []

            for t in np.arange(T):
                clustering_results_array = []

                # perform simple SC for the first time step
                if t == 0:
                    for index in np.arange(N):
                        adj_estimates.append(subgraphs_for_clustering['adj_' + str(t)][index])
                        SC_object = SpectralClustering(ID=self.ID, adjacency=adj_estimates[index], n_clusters=n_clusters)
                        SC_result = SC_object.performSC()
                        clustering_results_array.append(SC_result)

                # perform evolutionary SC for all other time steps
                else:
                    for index in np.arange(N):
                        adj = subgraphs_for_clustering['adj_' + str(t)][index]
                        adj_estimates[index] = forgetting_factor * adj + (1 - forgetting_factor) * adj_estimates[index]
                        SC_object = SpectralClustering(ID=self.ID, adjacency=adj_estimates[index], n_clusters=n_clusters)
                        SC_result = SC_object.performSC()
                        clustering_results_array.append(SC_result)
                subgraphs_for_clustering['clus_labels_' + str(t)] = clustering_results_array

        # static hierarchical clustering
        if parent_alg == 'HC':
            for t in np.arange(T):
                clustering_results_array = []
                for adj in subgraphs_for_clustering['adj_' + str(t)]:
                    HC_object = HierarchicalClustering(ID=self.ID, adjacency=adj, n_clusters=n_clusters)
                    HC_result = HC_object.performHC()
                    clustering_results_array.append(HC_result)
                subgraphs_for_clustering['clus_labels_' + str(t)] = clustering_results_array

        # evolutionary hierarchical clustering
        if parent_alg == 'evHC':
            # save evolutionary estimates
            adj_estimates = []

            for t in np.arange(T):
                clustering_results_array = []

                # perform simple HC for the first time step
                if t == 0:
                    for index in np.arange(N):
                        adj_estimates.append(subgraphs_for_clustering['adj_' + str(t)][index])
                        HC_object = HierarchicalClustering(ID=self.ID, adjacency=adj_estimates[index],
                                                           n_clusters=n_clusters)
                        HC_result = HC_object.performHC()
                        clustering_results_array.append(HC_result)

                # perform evolutionary HC for all other time steps
                else:
                    for index in np.arange(N):
                        adj = subgraphs_for_clustering['adj_' + str(t)][index]
                        adj_estimates[index] = forgetting_factor * adj + (1 - forgetting_factor) * adj_estimates[index]
                        HC_object = HierarchicalClustering(ID=self.ID, adjacency=adj_estimates[index],
                                                           n_clusters=n_clusters)
                        HC_result = HC_object.performHC()
                        clustering_results_array.append(HC_result)
                subgraphs_for_clustering['clus_labels_' + str(t)] = clustering_results_array

        self.subgraphs_df = subgraphs_for_clustering
        print(' Performed clustering algorithm', parent_alg, 'on all subgraphs for K =', n_clusters, 'clusters')

    def getSubgraphs(self):
        time_start = time.time()
        self.selectSubgraphs()
        self.getAdjacencyMatrices()
        self.clusterSubgraphs()
        time_end = time.time()
        self.runtime = np.round(time_end - time_start, 4)
        return self.subgraphs_df
