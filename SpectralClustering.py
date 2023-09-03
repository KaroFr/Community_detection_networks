import numpy as np
# for calculating eigenvalues and eigenvectors
import pandas as pd
from numpy.linalg import eig
# KMeans
from sklearn.cluster import KMeans

import time

"""
Class to perform PACE
"""


class SpectralClustering:
    algorithm = 'SC'
    labels_pred = []
    runtime = 0.0

    def __init__(self, P_estimate, K, ID=-1):
        self.ID = ID
        self.P_estimate = P_estimate
        self.K = K
        self.n_nodes = len(P_estimate)

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'n_nodes': self.n_nodes,
                                'n_clusters': self.K,
                                'algorithm': self.algorithm,
                                'runtime': self.runtime,
                                }])
        return var_df

    """
    Performs Spectral Clustering
    Input:  P_estimate: estimate for the prob. matrix P
            k: number of clusters
    Output: y_pred_sc: array of the predicted communities
    """

    def performSC(self):
        time_start_SC = time.time()
        P_est = self.P_estimate
        n_clusters = self.K
        evalues, evectors = eig(P_est)

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
        self.runtime = time_end_SC - time_start_SC

        return y_pred_sc
