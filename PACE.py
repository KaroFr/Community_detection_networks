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
    subgraphs = pd.DataFrame([])
    counting_matrix = np.array([])
    clustering_matrix_estimate = np.array([])
    clustering_matrix_estimate_threshold = np.array([])
    clustering_labels_estimate = np.array([])
    n_nodes = 0
    runtime = 0.0

    def __init__(self, adjacency, n_subgraphs, size_subgraphs, n_clusters, tau, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC', apply_threshold=False, threshold=0.5):
        self.ID = ID
        self.adj = adjacency
        self.T = n_subgraphs
        self.m = size_subgraphs
        self.K = n_clusters
        self.tau = tau
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        self.n_nodes = len(adjacency)
        self.apply_threshold = apply_threshold
        if apply_threshold:
            self.threshold = threshold
        else:
            self.threshold = -1

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'n_nodes': self.n_nodes,
                                'n_clusters': self.K,
                                'algorithm': self.algorithm,
                                'subgraph_sel_alg': self.subgraph_selection_alg,
                                'base_alg': self.parent_alg,
                                'n_subgraphs': self.T,
                                'size_subgraphs': self.m,
                                'PACE_tau': self.tau,
                                'apply_threshold': self.apply_threshold,
                                'clustering_mat_threshold': self.threshold,
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
        T = self.T
        adj = self.adj

        # initiate the output
        data = []

        # construct a random subgraph (T times)
        for _ in np.arange(T):
            # randomly choose m indices out of [n]
            index_set = np.random.choice(n, size=m, replace=False)
            index_set = np.sort(index_set)
            # get a grid to extract the submatrix
            ixgrid = np.ix_(index_set, index_set)
            data.append(
                {
                    'indices': index_set,
                    'subgraphs': adj[ixgrid]
                }
            )

        # load data into a DataFrame object:
        self.subgraphs = pd.DataFrame(data)
        print(' PACE: Selected T =', T, ' subgraphs of size m =', m)

    """
    Perform Clustering on each subgraph 
    The Clustering results will be stored in the Dataframe 'subgraphs'
    """

    def clusterSubgraphs(self):
        subgraphs_for_clustering = self.subgraphs
        n_clusters = self.K
        parent_alg = self.parent_alg

        clustering_results = []

        if parent_alg == 'SC':
            # perform spectral clustering on each subgraph
            for index, subgraph in subgraphs_for_clustering.iterrows():
                adj_sub = subgraph["subgraphs"]
                SC_object = SpectralClustering(ID=self.ID, P_estimate=adj_sub, K=n_clusters)
                SC_result = SC_object.performSC()
                clustering_results.append(SC_result)

        subgraphs_for_clustering["clustering_result"] = clustering_results
        self.subgraphs = subgraphs_for_clustering
        print(' PACE: Performed clustering algorithm', parent_alg, 'on all subgraphs for K =', n_clusters, 'clusters')

    """
    1)  Get the clustering matrices of the subgraphs
        and extend them to clustering matrices of the whole set by adding zeros
    2)  Get the counting matrices y
    """

    def getMatrices(self):
        subgraphs_to_add_matrices = self.subgraphs
        n_nodes = self.n_nodes

        # initiate a clustering/counting matrix containing zeros
        clustering_matrix = np.zeros([n_nodes, n_nodes])
        counting_matrix = np.zeros([n_nodes, n_nodes], dtype=int)

        for index, subgraph in subgraphs_to_add_matrices.iterrows():
            # get the according index set and the grid
            index_set = subgraph["indices"]
            ixgrid = np.ix_(index_set, index_set)

            # get clustering matrix of the subgraph
            clustering_labels = subgraph["clustering_result"]
            membership_mat = getMembershipMatrix(clustering_labels)
            clustering_matrix_subgraph = membership_mat @ membership_mat.transpose()

            # get counting matrix of the subgraph with only ones
            m = len(index_set)
            counting_matrix_subgraph = np.ones([m, m], dtype=int)

            # write the clustering matrix of the subgraph into the zero matrix
            clustering_matrix[ixgrid] += clustering_matrix_subgraph

            # write the clustering matrix of the subgraph into the zero matrix
            counting_matrix[ixgrid] += counting_matrix_subgraph

        self.clustering_matrix_estimate = clustering_matrix
        self.counting_matrix = counting_matrix

    """
    combine the results from the different subgraphs
    Calculate the estimate \hat{C}
    """

    def patchUp(self):
        tau = self.tau
        counting_matrix = self.counting_matrix
        clustering_matrix = self.clustering_matrix_estimate

        # get counting matrix N
        counting_matrix_tau = np.array(
            [[x if x > 1 else 0 for x in counting_matrix[i]] for i in range(len(counting_matrix))])

        # average -> get estimate \hat{C}
        clustering_matrix_estimate = np.divide(clustering_matrix, counting_matrix_tau,
                                               out=np.zeros_like(clustering_matrix), where=counting_matrix_tau != 0)

        self.clustering_matrix_estimate = clustering_matrix_estimate
        print(' PACE: Calculated the estimate for tau =', tau)

    """
    Perform the whole algorithm
    """

    def performPACE(self):
        print('Perform PACE:')
        time_start_PACE = time.time()
        apply_threshold = self.apply_threshold
        self.selectSubgraphs()
        self.clusterSubgraphs()
        self.getMatrices()
        self.patchUp()
        if apply_threshold:
            self.applyThresholdToEstimate()
            estimate = self.clustering_matrix_estimate_threshold
        else:
            estimate = self.clustering_matrix_estimate
        self.applyFinalClustering(estimate)
        time_end_PACE = time.time()
        self.runtime = np.round(time_end_PACE - time_start_PACE, 4)
        return self.clustering_labels_estimate


    """
    Apply a threshold to the result to get a binary clustering matrix
    """

    def applyThresholdToEstimate(self):
        threshold = self.threshold
        clust_mat = self.clustering_matrix_estimate
        clust_mat_thres = np.array([[1 if x > threshold else 0 for x in clust_mat[i]] for i in range(len(clust_mat))])
        self.clustering_matrix_estimate_threshold = clust_mat_thres
        print(' PACE: Applied threshold =', threshold, ' to get binary clustering matrix')

    """
    Apply a final clustering algorithm to get the labels
    """

    def applyFinalClustering(self, estimate):
        n_clusters = self.K
        SC_object = SpectralClustering(P_estimate=estimate, K=n_clusters)
        clustering_labels_estimate = SC_object.performSC()
        self.clustering_labels_estimate = clustering_labels_estimate
        print(' PACE: Applied final Spectral Clustering step')
