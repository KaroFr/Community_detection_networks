import numpy as np
import pandas as pd
from SpectralClustering import SpectralClustering
from Helpers import getMembershipMatrix
from Match import match
from functools import reduce
import time

# stochastic block model
import networkx as nx

"""
Class for GALE
"""


class GALE:
    algorithm = 'GALE'
    subgraph_selection_alg = 'Random'
    parent_alg = 'SC'
    subgraphs = pd.DataFrame([])
    n_nodes = 0
    sequence = []
    membership_estimate = np.array([])
    runtime = 0.0
    n_unused_subgraphs = 0

    def __init__(self, adjacency, n_subgraphs, size_subgraphs, n_clusters, tau, ID=-1, subgraph_sel_alg='Random',
                 parent_alg='SC', weightedTraversal=True):
        self.ID = ID
        self.adj = adjacency
        self.T = n_subgraphs
        self.m = size_subgraphs
        self.K = n_clusters
        self.tau = tau
        self.subgraph_selection_alg = subgraph_sel_alg
        self.parent_alg = parent_alg
        n_nodes = len(adjacency)
        self.n_nodes = n_nodes
        self.traversal_threshold = np.ceil(size_subgraphs ** 2 / (2 * n_nodes))
        self.weightedTraversal = weightedTraversal

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'ID': self.ID,
                                'algorithm': self.algorithm,
                                'subgraph_sel_alg': self.subgraph_selection_alg,
                                'base_alg': self.parent_alg,
                                'n_subgraphs': self.T,
                                'size_subgraphs': self.m,
                                'PACE_theta': -1.0,
                                'PACE_tau': -1.0,
                                'apply_threshold': False,
                                'clustering_mat_threshold': -1.0,
                                'GALE_tau': self.tau,
                                'GALE_weighted_traversal': self.weightedTraversal,
                                'GALE_n_unused_subgraphs': self.n_unused_subgraphs,
                                'traversal_threshold': self.traversal_threshold,
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
            # randomly choose m indices out of [n] (0 included, n excluded)
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
        print(' GALE: Selected T =', T, ' subgraphs of size m =', m)

    """
    Perform Clustering on each subgraph 
    The Clustering results (labels) will be stored in the Dataframe 'subgraphs'
    """

    def clusterSubgraphs(self):
        subgraphs_for_clustering = self.subgraphs
        n_clusters = self.K
        parent_alg = self.parent_alg

        clustering_results = []

        if parent_alg == 'SC':
            # perform spectral clustering on each subgraph
            for _, subgraph in subgraphs_for_clustering.iterrows():
                adj_sub = subgraph["subgraphs"]
                SC_object = SpectralClustering(ID=self.ID, P_estimate=adj_sub, K=n_clusters)
                SC_result = SC_object.performSC()
                clustering_results.append(SC_result)

        subgraphs_for_clustering["clustering_result"] = clustering_results
        self.subgraphs = subgraphs_for_clustering
        print(' GALE: Performed clustering algorithm', parent_alg, 'on all subgraphs for K =', n_clusters, 'clusters')

    """
        Get a traversal through all the subgraphs
        sequence_weighted = traversal with overlap as weight and no threshold
        """

    def getWeightedTraversal(self):
        T = self.T
        indices = self.subgraphs['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs_weighted = np.zeros((T, T))
        for t1 in np.arange(T):
            for t2 in np.arange(t1 + 1, T):
                overlap = np.intersect1d(indices[t1], indices[t2])
                n_overlap = len(overlap)
                adj_subgraphs_weighted[t1][t2] = n_overlap

        # get the spanning tree
        Graph_subgraphs_weighted = nx.Graph(adj_subgraphs_weighted)
        s_tree_weighted = nx.maximum_spanning_tree(Graph_subgraphs_weighted)

        # get the traversal (depth-first-search)
        traversal_weighted = np.array(list(nx.dfs_edges(s_tree_weighted)))

        # for weighted traversal
        sequence_weighted = [traversal_weighted[0][0]]
        sequence_weighted.extend(traversal_weighted[:, 1])
        ind = np.unique(sequence_weighted, return_index=True)[1]
        sequence_weighted_unique = [sequence_weighted[i] for i in sorted(ind)]

        self.sequence = sequence_weighted_unique

    """
    Get a traversal through all the subgraphs
    sequence = the traversal by Mukherjee et al
    """

    def getNormalTraversal(self):
        m_thres = self.traversal_threshold
        T = self.T
        indices = self.subgraphs['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs = np.zeros((T, T))
        for t1 in np.arange(T):
            for t2 in np.arange(t1 + 1, T):
                # if overlap is big enough there is a connection
                overlap = np.intersect1d(indices[t1], indices[t2])
                n_overlap = len(overlap)
                if n_overlap > m_thres:
                    adj_subgraphs[t1][t2] = 1

        # get the spanning tree
        Graph_subgraphs = nx.Graph(adj_subgraphs)

        # get the traversal (depth-first-search)
        traversal_vertices = np.array(list(nx.dfs_edges(Graph_subgraphs))).reshape(1, -1)
        vertices_list = list(traversal_vertices[0])

        # seperate the different spanning trees
        traversals = [[vertices_list[0]]]
        first_occurance = True
        predecessor = -1

        for i, num in enumerate(vertices_list[1:]):
            if first_occurance:
                traversals[-1].append(num)
                first_occurance = False
                predecessor = num
            elif not first_occurance and predecessor == num:
                traversals[-1].append(num)
                first_occurance = True
            elif not first_occurance and predecessor != num:
                traversals.append([])
                traversals[-1].append(num)
                first_occurance = True
                predecessor = num

        # delete duplicates in the different spanning trees
        traversals_unique = []
        for traversal in traversals:
            traversal = list(dict.fromkeys(traversal))
            traversals_unique.append(traversal)
        traversals_unique

        # get longest traversal
        longest_traversal = max(traversals_unique, key=len)

        self.sequence = longest_traversal

    """
    Align the subgraphs
    """

    def alignLabels(self):
        n_nodes = self.n_nodes
        K = self.K
        subgraphs_clustered = self.subgraphs['clustering_result']
        sequence = self.sequence
        indices = self.subgraphs['indices']
        T = self.T
        tau = self.tau
        traversal_threshold = self.traversal_threshold
        weightedTraversal = self.weightedTraversal

        # count how many of the subgraphs are not used because the overlap is too small
        counter_unused_subgraphs = 0

        # number of subgraphs in the traversal
        n_subgraphs = len(sequence)

        # get first subgraph: index set and clustering result
        current_index = sequence[0]
        current_subgraph_clustered = subgraphs_clustered[current_index]
        indices_current_subgraph = indices[current_index]

        # initiate array of visited subgraphs
        visited_indices = [current_index]

        # get first membership matrix and extend to n x K matrix by adding zeros
        current_membership_mat = getMembershipMatrix(current_subgraph_clustered)
        current_membership_mat_extended = np.zeros((n_nodes, K))
        current_membership_mat_extended[indices_current_subgraph] = current_membership_mat

        # initial estimate and added memberships
        membership_estimate = current_membership_mat_extended.copy()
        membership_added = current_membership_mat_extended.copy()

        counter = 1
        while len(visited_indices) < n_subgraphs:
            # get current index of the subgraph from traversal
            current_index = sequence[counter]

            # get current subgraph: index set and clustering result
            indices_current_subgraph = indices[current_index]
            current_subgraph_clustered = subgraphs_clustered[current_index]

            # calculate the overlap to all previously visited subgraphs
            visited_index_sets = [indices[i] for i in visited_indices]
            indices_previous_subgraphs = reduce(np.union1d, visited_index_sets)
            overlap = np.intersect1d(indices_current_subgraph, indices_previous_subgraphs)

            # weighted traversal:
            # if overlap is to small, go to next subgraph and visit this one later again
            if weightedTraversal and len(overlap) < traversal_threshold:
                sequence.append(current_index)
                counter += 1
                # counter > T means that the current subgraph has been visited before
                # if the overlap is still to small, we discard it as information
                if counter > T:
                    counter_unused_subgraphs += 1
                    visited_indices.append(current_index)
                continue

            # get membership matrices and extend to n x K matrix by adding zeros
            current_membership_mat = getMembershipMatrix(current_subgraph_clustered)
            current_membership_mat_extended = np.zeros((n_nodes, K))
            current_membership_mat_extended[indices_current_subgraph] = current_membership_mat

            # apply match algorithm (restrict to overlap)
            try:
                projection_mat = match(current_membership_mat_extended[overlap], membership_estimate[overlap])
            except IndexError:
                counter_unused_subgraphs += 1
                visited_indices.append(current_index)
                counter += 1
                continue

            # permutate current subgraph
            current_membership_mat_permutated = current_membership_mat_extended @ projection_mat

            # add up the membership matrices
            membership_added += current_membership_mat_permutated

            # calculate xi
            sum_vector = np.sum(membership_added, axis=1)
            xi = np.array([1 if sum_vector[j] >= tau else 0 for j in range(len(sum_vector))])
            # Todo: tau needs to be a vector not single variable

            # update membership matrix
            numerator = membership_added * xi.reshape(-1, 1)
            denominator = sum_vector.reshape(-1, 1)
            membership_estimate = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                            where=denominator != 0)

            visited_indices.append(current_index)
            counter += 1

        # save the estimate (result of GALE)
        self.membership_estimate = membership_estimate

        # count the number of unused subgraphs for the unweighted traversal
        if not weightedTraversal:
            counter_unused_subgraphs = T - n_subgraphs

        # save the number of unused subgraphs
        self.n_unused_subgraphs = counter_unused_subgraphs
        print(' GALE: ran alignLabels: number of unused subgraphs:', counter_unused_subgraphs)

    """
    Reduce the non binary membership matrix to a binary matrix 
    by setting the maximal entry to one
    - for ties take the first occurance of the maximal entry
    """

    def getBinaryMembershipmatrix(self):
        membership_estimate = self.membership_estimate
        binary_membership_estimate = np.zeros_like(membership_estimate)
        binary_membership_estimate[np.arange(len(membership_estimate)), membership_estimate.argmax(1)] = 1
        self.membership_estimate = binary_membership_estimate
        print(' GALE: Reduced result to binary membership matrix')

    """
    Perform the whole algorithm
    """

    def performGALE(self):
        print('Perform GALE:')
        time_start_GALE = time.time()
        self.selectSubgraphs()
        self.clusterSubgraphs()
        if self.weightedTraversal:
            self.getWeightedTraversal()
        else:
            self.getNormalTraversal()
        self.alignLabels()
        self.getBinaryMembershipmatrix()
        time_end_GALE = time.time()
        self.runtime = np.round(time_end_GALE - time_start_GALE, 4)
        return self.membership_estimate
