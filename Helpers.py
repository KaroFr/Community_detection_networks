import numpy as np

"""
Input: Array of clustering labels
Output: According Membership Matrix
"""


def getMembershipMatrix(clustering_labels):
    min_cluster = min(clustering_labels)
    max_cluster = max(clustering_labels)
    n_nodes = len(clustering_labels)
    n_cluster = max_cluster - min_cluster + 1
    # initiate a zero membership matrix
    membership_matrix = np.zeros((n_nodes, n_cluster))

    # loop over clustering labels and write 1 into the according entry of the membership matrix
    # for the case that the minimum node is not 0, we need to subtract the minimum
    for index, node in enumerate(clustering_labels):
        membership_matrix[index][node - min_cluster] = 1
    return membership_matrix

"""
Input: membership_matrix
Output: According clustering Matrix
"""


def getClusteringMatrix(membership_matrix):
    clustering_matrix = membership_matrix @ membership_matrix.transpose()
    return clustering_matrix

"""
Input: membership_matrix
Output: According confusion Matrix
"""


def getClusteringMatrix(membership_matrix):
    confusion_matrix = membership_matrix.transpose() @ membership_matrix
    return confusion_matrix