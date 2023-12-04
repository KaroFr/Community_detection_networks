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


class GALE_Offline:
    subgraphs_df = pd.DataFrame([])
    sequence = []
    membership_estimates = np.array([])
    n_nodes = 0
    runtime = 0.0
    n_unused_subgraphs = 0

    def __init__(self, subgraphs_df, n_nodes, n_clusters, theta, weightedTraversal=True):
        self.subgraphs_df = subgraphs_df
        N = len(subgraphs_df['indices'])
        self.N = N
        size_subgraphs = len(subgraphs_df['indices'][0])
        m = size_subgraphs
        self.m = m
        self.T = int((subgraphs_df.shape[1] - 1)/2)
        self.K = n_clusters
        self.tau = theta*N*m/n_nodes
        self.n_nodes = n_nodes
        self.traversal_threshold = np.ceil(size_subgraphs ** 2 / (2 * n_nodes))
        self.weightedTraversal = weightedTraversal

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'algorithm': 'GALE',
                                'PACE_theta': -1.0,
                                'GALE_tau': self.tau,
                                'GALE_weighted_traversal': self.weightedTraversal,
                                'GALE_n_unused_subgraphs': self.n_unused_subgraphs,
                                'traversal_threshold': self.traversal_threshold,
                                'runtime': self.runtime,
                                }])
        return var_df

    def getTraversal_final(self):
        N = self.N
        indices = self.subgraphs_df['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs_weighted = np.zeros((N, N))
        for t1 in np.arange(N):
            for t2 in np.arange(t1 + 1, N):
                overlap = np.intersect1d(indices[t1], indices[t2])
                n_overlap = len(overlap)
                adj_subgraphs_weighted[t1][t2] = n_overlap

        # get initial edge with maximal weight
        traversal = list(np.unravel_index(adj_subgraphs_weighted.argmax(), adj_subgraphs_weighted.shape))

        # join the nodes in already visited subgraphs
        joined_indices = np.unique((indices[traversal[0]], indices[traversal[1]]))

        for _ in np.arange(N - 2):
            # list the subgraphs not yet part of the traversal
            remaining_subgraphs = np.setdiff1d(np.arange(N), traversal)

            # for each remaining subgraph calculate the overlap
            overlaps = []
            for i in remaining_subgraphs:
                overlap = np.intersect1d(joined_indices, indices[i])
                overlaps.append(len(overlap))
            max_overlap = max(overlaps)

            # get the subgraph with the maximal overlap
            max_overlap_index = np.array(overlaps).argmax(axis=0)
            next_subgraph = remaining_subgraphs[max_overlap_index]

            # add according subgraph to traversal and join index sets
            traversal.append(next_subgraph)
            joined_indices = np.concatenate((joined_indices, indices[next_subgraph]))
            joined_indices = np.unique(joined_indices)

        self.sequence = traversal

    """
    Get a traversal through all the subgraphs
    sequence = traversal with overlap as weight and no threshold
    """

    def getWeightedTraversal(self):
        N = self.N
        indices = self.subgraphs_df['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs_weighted = np.zeros((N, N))
        for t1 in np.arange(N):
            for t2 in np.arange(t1 + 1, N):
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
    If the spanning tree is not fully connected, we get multiple paths.
    This method returns the longest of those paths.
    Differs from getAllNormalTraversals() only in the last part
    """

    def getMaximalNormalTraversal(self):
        m_thres = self.traversal_threshold
        N = self.N
        indices = self.subgraphs_df['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs = np.zeros((N, N))
        for t1 in np.arange(N):
            for t2 in np.arange(t1 + 1, N):
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

        # get longest traversal
        longest_traversal = max(traversals_unique, key=len)

        self.sequence = longest_traversal

    """
    Get a traversal through all the subgraphs
    sequence = the traversal by Mukherjee et al
    If the spanning tree is not fully connected, we get multiple paths.
    This method returns the all those paths.
    """

    def getAllNormalTraversals(self):
        m_thres = self.traversal_threshold
        N = self.N
        indices = self.subgraphs_df['indices']

        # construct graph of the subgraphs based on overlap
        # only filled on upper triangle -> that suffices to calculate the spanning tree
        adj_subgraphs = np.zeros((N, N))
        for t1 in np.arange(N):
            for t2 in np.arange(t1 + 1, N):
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

        self.sequence = traversals_unique

    """
    Align the subgraphs
    """

    def alignLabels(self):
        subgraphs_to_align = self.subgraphs_df
        T = self.T
        tau = self.tau

        membership_estimates = []

        for t in np.arange(T):
            subgraphs_clustered = subgraphs_to_align['clus_labels_' + str(int(t))]
            if t == T-1:
                membership_estimate, counter_unused_subgraphs = self.alignLabels_static(subgraphs_clustered, tau)
            else:
                membership_estimate, counter_unused_subgraphs = self.alignLabels_static(subgraphs_clustered, 0.0)
            print(' GALE: alignLabels t=', t, ', number of unused subgraphs:', counter_unused_subgraphs)
            membership_estimates.append(membership_estimate)

        self.membership_estimates = membership_estimates

    def alignLabels_static(self, subgraphs_clustered, tau):
        sequence = self.sequence
        n_nodes = self.n_nodes
        K = self.K
        indices = self.subgraphs_df['indices']
        N = self.N
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
                # counter > N means that the current subgraph has been visited before
                # if the overlap is still to small, we discard it as information
                if counter > N:
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

            # update membership matrix
            numerator = membership_added * xi.reshape(-1, 1)
            denominator = sum_vector.reshape(-1, 1)
            membership_estimate = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                            where=denominator != 0)

            visited_indices.append(current_index)
            counter += 1

        # count the number of unused subgraphs for the unweighted traversal
        if not weightedTraversal:
            counter_unused_subgraphs = N - n_subgraphs

        return membership_estimate, counter_unused_subgraphs

    """
    Reduce the non binary membership matrix to a binary matrix 
    by setting the maximal entry to one
    - for ties take the first occurance of the maximal entry
    """

    def getBinaryMembershipmatrix(self):
        membership_estimates = self.membership_estimates
        T = self.T
        for t in np.arange(T, dtype=int):
            membership_estimate = membership_estimates[t]
            binary_membership_estimate = np.zeros_like(membership_estimate)
            binary_membership_estimate[np.arange(len(membership_estimate)), membership_estimate.argmax(1)] = 1
            membership_estimates[t] = binary_membership_estimate
        self.membership_estimates = membership_estimates
        print(' GALE: Reduced result to binary membership matrices')

    """
    Perform the whole algorithm
    """

    def performGALE(self):
        print('Perform GALE:')
        time_start_GALE = time.time()
        self.getTraversal_final()
        self.alignLabels()
        self.getBinaryMembershipmatrix()
        time_end_GALE = time.time()
        self.runtime = np.round(time_end_GALE - time_start_GALE, 4)
        return self.membership_estimates
