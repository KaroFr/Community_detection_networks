import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from Helpers import getMembershipMatrix
import time

"""
Class to perform PACE
"""


class PACE:
    subgraphs_df = pd.DataFrame([])
    counting_matrices = np.array([])
    clustering_matrices = np.array([])
    clustering_labels = np.array([])
    n_nodes = 0
    runtime = 0.0

    def __init__(self, subgraphs_df, n_nodes, n_clusters, theta=0.4, forgetting_factor=1):
        self.subgraphs_df = subgraphs_df
        self.N = len(subgraphs_df['indices'])
        self.m = len(subgraphs_df['indices'][0])
        self.T = int((subgraphs_df.shape[1] - 1) / 2)
        self.n_nodes = n_nodes
        self.K = n_clusters
        self.theta = theta
        self.forgetting_factor = forgetting_factor

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'algorithm': 'PACE',
                                'PACE_theta': self.theta,
                                'GALE_tau': -1.0,
                                'GALE_weighted_traversal': False,
                                'GALE_n_unused_subgraphs': -1,
                                'traversal_threshold': -1.0,
                                'forgetting_factor_PACE': self.forgetting_factor,
                                'runtime': self.runtime,
                                }])
        return var_df

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
                clustering_labels = subgraph["clus_labels_" + str(int(t))]
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
        for t in np.arange(T, dtype=int):
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
        forgetting_factor = self.forgetting_factor
        T = self.T
        n_clusters = self.K
        clustering_matrices = self.clustering_matrices
        clustering_labels = []

        for t in np.arange(T, dtype=int):
            if t == 0:
                clustering_matrix_estimate = clustering_matrices[0]

            else:
                clustering_matrix_estimate = forgetting_factor * clustering_matrices[t] + (
                            1 - forgetting_factor) * clustering_matrix_estimate

            SC_object = SpectralClustering(adjacency=clustering_matrix_estimate, n_clusters=n_clusters)
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
        self.getMatrices()
        self.patchUp()
        self.applyFinalClustering()
        time_end_PACE = time.time()
        self.runtime = np.round(time_end_PACE - time_start_PACE, 4)
        return self.clustering_labels
