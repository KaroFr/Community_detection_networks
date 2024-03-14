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


def getConfusionMatrix(membership_matrix):
    confusion_matrix = membership_matrix.transpose() @ membership_matrix
    return confusion_matrix


"""
Input: membership_matrix
Output: True if membership_matrix is indeed a membership matrix
        False if not
Test if (1) maximal entry of every row is 1
        (2) every row sum equals 1
        
"""


def isMembership(membership_matrix):
    for row in np.arange(len(membership_matrix)):
        max_entry_row = np.max(membership_matrix[row])
        if max_entry_row != 1:
            return False

        sum_row = np.sum(membership_matrix[row])
        if sum_row != 1:
            return False
    return True


"""
Input: any symmetric matrix
Output: graph Laplacian matrix
"""


def getLaplacian(matrix, regularization_tau=0):
    # degree matrix
    D = matrix.sum(axis=1) + regularization_tau
    D = np.sqrt(1 / D)
    # matrix multiplication is very expensive => do elementwise multiplication
    L = np.multiply(D[np.newaxis, :], np.multiply(matrix, D[:, np.newaxis]))
    return L
