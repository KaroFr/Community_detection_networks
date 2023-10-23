import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from Helpers import getMembershipMatrix
import time

"""
Class to perform PACE
"""


class PACE:
    algorithm = 'PACE'
    subgraph_selection_alg = 'Random'
    parent_alg = 'SC'
    subgraphs_df = pd.DataFrame([])
    counting_matrices = np.array([])
    clustering_matrices = np.array([])
    clustering_labels = np.array([])
    n_nodes = 0
    runtime = 0.0

    def __init__(self, SBMs, n_subgraphs, size_subgraphs, n_clusters, theta=0.4, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC'):
        self.ID = ID
        self.adjacencies = SBMs['adj_matrix']
        self.N = n_subgraphs
        self.m = size_subgraphs
        self.T = len(SBMs['adj_matrix'])
        print('T = ', self.T)
        self.K = n_clusters
        self.theta = theta
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        self.n_nodes = len(SBMs['adj_matrix'][0])
        print('n_nodes = ', self.n_nodes)

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'algorithm': self.algorithm,
                                'subgraph_sel_alg': self.subgraph_selection_alg,
                                'base_alg': self.parent_alg,
                                'n_subgraphs': self.N,
                                'size_subgraphs': self.m,
                                'PACE_theta': self.theta,
                                'GALE_tau': -1.0,
                                'GALE_weighted_traversal': False,
                                'GALE_n_unused_subgraphs': -1,
                                'traversal_threshold': -1.0,
                                'runtime': self.runtime,
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
        print(' PACE: Selected N =', N, ' subgraphs of size m =', m)

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
            subgraphs_df['adj_' + str(t)] = adj_arr
        self.subgraphs_df = subgraphs_df

    """
    Perform Clustering on each subgraph 
    The Clustering results will be stored in the Dataframe 'subgraphs'
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
        print(' PACE: Performed clustering algorithm', parent_alg, 'on all subgraphs for K =', n_clusters, 'clusters')

    """
    1)  Get the clustering matrices of the subgraphs
        and extend them to clustering matrices of the whole set by adding zeros
    2)  Get the counting matrices y
    """

    def getMatrices(self):
        subgraphs_to_add_matrices = self.subgraphs_df
        n_nodes = self.n_nodes
        T = self.T
        counting_matrices = []
        clustering_matrices = []

        for t in np.arange(T):
            # initiate a clustering/counting matrix containing zeros
            clustering_matrix = np.zeros([n_nodes, n_nodes], dtype=float)
            counting_matrix = np.zeros([n_nodes, n_nodes], dtype=int)

            for _, subgraph in subgraphs_to_add_matrices.iterrows():
                # get the according index set and the grid
                index_set = subgraph["indices"]
                ixgrid = np.ix_(index_set, index_set)

                # get clustering matrix of the subgraph
                clustering_labels = subgraph["clus_labels_" + str(t)]
                membership_mat = getMembershipMatrix(clustering_labels)
                clustering_matrix_subgraph = membership_mat @ membership_mat.transpose()

                # get counting matrix of the subgraph with only ones
                m = len(index_set)
                counting_matrix_subgraph = np.ones([m, m], dtype=int)

                # add up the clustering/ counting matrices
                clustering_matrix[ixgrid] += clustering_matrix_subgraph
                counting_matrix[ixgrid] += counting_matrix_subgraph

            counting_matrices.append(counting_matrix)
            clustering_matrices.append(clustering_matrix)

        self.counting_matrices = counting_matrices
        self.clustering_matrices = clustering_matrices


    """
    combine the results from the different subgraphs
    Calculate the estimate \hat{C}
    """
    def patchUp(self):
        counting_matrices = self.counting_matrices
        clustering_matrices = self.clustering_matrices
        T = self.T
        n_nodes = self.n_nodes
        theta = self.theta
        for t in np.arange(T):
            counting_matrix = counting_matrices[t]
            clustering_matrix = clustering_matrices[t]

            tau = np.quantile(counting_matrix, q=theta)

            # filter counting matrix for entries >= tau to get N
            counting_matrix_tau = np.array(
                [[x if x >= tau else 0 for x in counting_matrix[i]] for i in range(len(counting_matrix))])

            # average -> get estimate \hat{C}
            clustering_matrix_estimate = np.divide(clustering_matrix, counting_matrix_tau,
                                                   out=np.zeros([n_nodes, n_nodes]), where=counting_matrix_tau != 0)
            clustering_matrices[t] = clustering_matrix_estimate

        print(' PACE: Calculated the T=', T, ' Clustering matrices.')


    """
    Apply a final clustering algorithm to get the labels
    """
    def applyFinalClustering(self):
        T = self.T
        n_clusters = self.K
        clustering_matrices = self.clustering_matrices
        clustering_labels = []

        for t in np.arange(T):
            estimate = clustering_matrices[t]
            SC_object = SpectralClustering(adjacency=estimate, n_clusters=n_clusters)
            clustering_labels_estimate = SC_object.performSC()
            clustering_labels.append(clustering_labels_estimate)

        self.clustering_labels = clustering_labels
        print(' PACE: Applied final Spectral Clustering step')

    """
    Perform the whole algorithm
    """

    def performPACE(self):
        print('Perform PACE:')
        time_start_PACE = time.time()
        self.selectSubgraphs()
        self.getAdjacencyMatrices()
        self.clusterSubgraphs()
        self.getMatrices()
        self.patchUp()
        self.applyFinalClustering()
        time_end_PACE = time.time()
        self.runtime = np.round(time_end_PACE - time_start_PACE, 4)
        return self.clustering_labels
