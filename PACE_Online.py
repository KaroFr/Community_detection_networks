import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from Helpers import getMembershipMatrix
import time

"""
Class to perform PACE
"""


class PACE_Online:
    counting_matrix = np.array([])
    clustering_matrix = np.array([])
    clustering_labels = np.array([])
    clustering_matrix_estimate = np.array([])
    n_nodes = 0
    runtime = 0.0
    tau = 0.0

    def __init__(self, indices, n_nodes, n_clusters, theta=0.4, forgetting_factor=1):
        self.time_step = 0
        self.indices = indices
        self.n_nodes = n_nodes
        self.K = n_clusters
        self.N = len(indices)
        self.m = len(indices[0])
        self.theta = theta
        self.forgetting_factor = forgetting_factor
        time_start_counting_matrix = time.time()
        self.getCountingMatrix()
        time_end_counting_matrix = time.time()
        self.runtime_counting_matrix = np.round(time_end_counting_matrix - time_start_counting_matrix, 4)


    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'algorithm': 'PACE',
                                'PACE_tau': self.tau,
                                'GALE_tau': -1.0,
                                'GALE_n_unused_subgraphs': -1,
                                'traversal_threshold': -1.0,
                                'forgetting_factor_PACE': self.forgetting_factor,
                                'runtime_PACE': self.runtime,
                                'runtime_counting_matrix_PACE': self.runtime_counting_matrix,
                                'time_step_PACE': self.time_step,
                                }])
        return var_df

    """
    Get the counting matrices y
    """

    def getCountingMatrix(self):
        n_nodes = self.n_nodes
        indices = self.indices
        theta = self.theta

        # initiate a counting matrix containing zeros
        counting_matrix = np.zeros([n_nodes, n_nodes], dtype=int)

        for index_set in indices:
            # get the according index set and the grid
            ixgrid = np.ix_(index_set, index_set)

            # get counting matrix of the subgraph with only ones
            m = len(index_set)
            counting_matrix_subgraph = np.ones([m, m], dtype=int)

            # add up the counting matrices
            counting_matrix[ixgrid] += counting_matrix_subgraph

        tau = np.quantile(counting_matrix, q=theta)

        # filter counting matrix for entries >= tau to get N
        counting_matrix_tau = np.array(
            [[x if x >= tau else 0 for x in counting_matrix[i]] for i in range(len(counting_matrix))])

        self.counting_matrix = counting_matrix_tau

    """
    1) get the clustering matrices of the subgraphs
    2) extend them to clustering matrices of the whole set by adding zeros
    3) sum up to one clustering matrix
    """

    def getClusteringMatrix(self, labels_subgraphs):
        n_nodes = self.n_nodes
        N = self.N
        indices = self.indices

        # initiate a clustering/counting matrix containing zeros
        clustering_matrix = np.zeros([n_nodes, n_nodes], dtype=float)

        for i in np.arange(N):
            # get clustering matrix of the subgraph
            clustering_labels = labels_subgraphs[i]
            membership_mat = getMembershipMatrix(clustering_labels)
            clustering_matrix_subgraph = membership_mat @ membership_mat.transpose()

            # add up the clustering/ counting matrices
            index_set = indices[i]
            ixgrid = np.ix_(index_set, index_set)
            clustering_matrix[ixgrid] += clustering_matrix_subgraph

        self.clustering_matrix = clustering_matrix

    """
    Average the summed up clustering matrix by dividing by the counting matrix
    """

    def patchUp(self):
        counting_matrix_tau = self.counting_matrix
        clustering_matrix = self.clustering_matrix
        n_nodes = self.n_nodes

        # average -> get estimate \hat{C}
        clustering_matrix_estimate = np.divide(clustering_matrix, counting_matrix_tau,
                                               out=np.zeros([n_nodes, n_nodes]), where=counting_matrix_tau != 0)
        self.clustering_matrix = clustering_matrix_estimate

        print(' PACE: Calculated the averaged clustering matrix.')

    """
    Apply a final clustering algorithm to get the labels
    """

    def applyFinalClustering(self):
        time_step = self.time_step
        forgetting_factor = self.forgetting_factor
        n_clusters = self.K
        clustering_matrix = self.clustering_matrix
        clustering_matrix_estimate = self.clustering_matrix_estimate

        if time_step == 0:
            clustering_matrix_estimate = clustering_matrix

        else:
            clustering_matrix_estimate = forgetting_factor * clustering_matrix + (
                        1 - forgetting_factor) * clustering_matrix_estimate

        SC_object = SpectralClustering(adjacency=clustering_matrix_estimate, n_clusters=n_clusters)

        self.clustering_matrix_estimate = clustering_matrix_estimate
        self.clustering_labels = SC_object.performSC()
        print(' PACE: Applied final Spectral Clustering step')

    """
    Perform the whole algorithm
    """

    def performPACE(self, labels_subgraphs):
        print('Perform PACE:')
        time_start_PACE = time.time()
        self.getClusteringMatrix(labels_subgraphs=labels_subgraphs)
        self.patchUp()
        self.applyFinalClustering()
        time_end_PACE = time.time()
        self.runtime = np.round(time_end_PACE - time_start_PACE, 4)

        self.time_step += 1
        return self.clustering_labels
