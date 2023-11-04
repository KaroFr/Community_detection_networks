import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from Helpers import getMembershipMatrix
from Match import match
from functools import reduce
import time
import networkx as nx

"""
Class for GALE
"""


class SubgraphSelector:
    subgraph_selection_alg = 'Random'
    parent_alg = 'SC'
    subgraphs_df = pd.DataFrame([])
    n_nodes = 0
    runtime = 0.0
    n_unused_subgraphs = 0

    def __init__(self, SBMs, n_subgraphs, size_subgraphs, n_clusters, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC'):
        self.ID = ID
        self.adjacencies = SBMs['adj_matrix']
        self.N = n_subgraphs
        self.m = size_subgraphs
        self.T = len(SBMs['adj_matrix'])
        self.K = n_clusters
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        n_nodes = len(SBMs['adj_matrix'][0])
        print('n_nodes = ', n_nodes)
        self.n_nodes = n_nodes

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'subgraph_sel_alg': self.subgraph_selection_alg,
                                'base_alg': self.parent_alg,
                                'n_subgraphs': self.N,
                                'size_subgraphs': self.m,
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
        n_clusters = self.K
        parent_alg = self.parent_alg
        T = self.T

        if parent_alg == 'SC':
            for t in np.arange(T):
                clustering_results_array = []
                for adj in subgraphs_for_clustering['adj_' + str(t)]:
                    SC_object = SpectralClustering(ID=self.ID, adjacency=adj, n_clusters=n_clusters)
                    SC_result = SC_object.performSC()
                    clustering_results_array.append(SC_result)
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
