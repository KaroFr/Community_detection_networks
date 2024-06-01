import numpy as np
# for calculating eigenvalues and eigenvectors
import pandas as pd
from numpy.linalg import eigh
# KMeans
from sklearn.cluster import KMeans
from Helpers import getLaplacian

import time

"""
Class to perform Spectral Clustering
"""


class SpectralClustering:
    algorithm = 'SC'
    labels_pred = []
    runtime = 0.0

    def __init__(self, adjacency, n_clusters, P_estimate='adjacency', regularization_tau=0.0, ID=-1):
        self.ID = ID
        self.adj = adjacency
        self.P_estimate = P_estimate
        self.regularization_tau = regularization_tau
        if P_estimate not in ['adjacency', 'Laplacian', 'regularized']:
            print('The input for P_estimate needs to be either "adjacency", "Laplacian" or "regularized".')
            return
        self.K = n_clusters
        self.n_nodes = len(adjacency)

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'algorithm': self.algorithm,
                                'estimate': self.P_estimate,
                                'subgraph_sel_alg': 'None',
                                'base_alg': 'None',
                                'n_subgraphs': 1,
                                'size_subgraphs': self.n_nodes,
                                'PACE_theta': -1.0,
                                'GALE_tau': -1.0,
                                'GALE_weighted_traversal': False,
                                'GALE_n_unused_subgraphs': -1,
                                'traversal_threshold': -1.0,
                                'regularization_tau': self.regularization_tau,
                                'runtime': self.runtime,
                                }])
        return var_df

    """
    Performs Spectral Clustering
    Input:  adj: estimate for the prob. matrix P
            k: number of clusters
    Output: y_pred_sc: array of the predicted communities
    """

    def performSC(self):
        time_start_SC = time.time()
        adj = self.adj
        n_clusters = self.K

        P_estimate = self.P_estimate
        if P_estimate == 'adjacency':
            P_est = adj
        elif P_estimate == 'Laplacian':
            P_est = getLaplacian(adj)
        elif P_estimate == 'regularized':
            regularization_tau = self.regularization_tau
            P_est = getLaplacian(adj, regularization_tau)

        evalues, evectors = eigh(P_est)

        # sort by Eigenvalues
        idx_asc = np.argsort(abs(evalues))  # np.argsort gives the inidices to sort ascending! We need descending
        idx_dsc = idx_asc[::-1]

        # evalues = evalues[idx_dsc]
        evectors = evectors[:, idx_dsc]

        # get the k first Eigenvectors
        U = evectors[:, 0:n_clusters]

        # normalize Usubset
        y_pred_sc = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(U)
        self.labels_pred = y_pred_sc

        time_end_SC = time.time()
        self.runtime = np.round(time_end_SC - time_start_SC, 4)

        # print(' SC: Finished Spectral Clustering')
        return y_pred_sc
