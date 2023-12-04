import numpy as np
import pandas as pd
from Helpers import getMembershipMatrix
from Match import match
from functools import reduce
import time

"""
Class for GALE
"""


class GALE_Online:
    sequence = []
    membership_estimate = np.array([])
    n_nodes = 0
    runtime = 0.0
    n_unused_subgraphs = 0

    def __init__(self, indices, n_nodes, n_clusters, theta):
        self.indices = indices
        self.n_nodes = n_nodes
        self.K = n_clusters
        N = len(indices)
        m = len(indices[0])
        self.tau = theta * N * m / n_nodes
        self.N = N
        self.m = m


    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'algorithm': 'GALE',
                                'PACE_theta': -1.0,
                                'GALE_tau': self.tau,
                                'GALE_n_unused_subgraphs': self.n_unused_subgraphs,
                                'runtime_GALE': self.runtime,
                                }])
        return var_df

    def getTraversal(self):
        N = self.N
        indices = self.indices

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
    Align the subgraphs
    """

    def alignLabels(self, labels_subgraphs, applyTau):
        subgraphs_clustered = labels_subgraphs
        indices = self.indices
        sequence = self.sequence
        n_nodes = self.n_nodes
        K = self.K
        N = self.N
        if applyTau:
            tau = self.tau
        else:
            tau = 0.0

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
        self.n_unused_subgraphs = N - n_subgraphs
        self.membership_estimate = membership_estimate

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

    def performGALE(self, labels_subgraphs, applyTau=False):
        print('Perform GALE:')
        time_start_GALE = time.time()
        self.getTraversal()
        self.alignLabels(labels_subgraphs, applyTau)
        self.getBinaryMembershipmatrix()
        time_end_GALE = time.time()
        self.runtime = np.round(time_end_GALE - time_start_GALE, 4)
        return self.membership_estimate
