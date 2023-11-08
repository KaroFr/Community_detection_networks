import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from Helpers import getLaplacian

import time

"""
Class to perform Hierarchical Clustering
"""


class HierarchicalClustering:
    algorithm = 'HC'
    labels_pred = []
    runtime = 0.0

    def __init__(self, adjacency, n_clusters, P_estimate='adjacency', ID=-1):
        self.ID = ID
        self.adj = adjacency
        self.P_estimate = P_estimate
        if P_estimate not in ['adjacency', 'Laplacian']:
            print('The input for P_estimate needs to be either "adjacency" or "Laplacian".')
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
                                'runtime': self.runtime,
                                }])
        return var_df

    """
    Performs Hierarchical Clustering
    Output: y_pred_sc: array of the predicted communities
    """

    def performHC(self):
        time_start_HC = time.time()
        adj = self.adj
        n_clusters = self.K

        # get distance matrix
        P_estimate = self.P_estimate
        if P_estimate == 'adjacency':
            P_est = 1 - adj
        elif P_estimate == 'Laplacian':
            P_est = 1 - getLaplacian(adj)

        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average').fit(P_est)
        y_pred_HC = clustering.labels_

        time_end_HC = time.time()
        self.runtime = np.round(time_end_HC - time_start_HC, 4)

        print(' HC: Finished Agglomerative Clustering')
        return y_pred_HC
