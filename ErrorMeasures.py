import numpy as np
from Helpers import getMembershipMatrix

# for the frobenius norm (for metric proposed by Sarkar and Bickel)
from numpy.linalg import norm

# for the metric of Lei Rinaldo - To iterate over permutation matrices
import itertools

# progress bar for the Lei Rinaldo Metrics
from tqdm import tqdm
from Match import match

"""
Input: Two arrays of clustering labels
Output: Difference of the confusion matrices in L0 Norm
"""


def getDifference_Confusion(clustering_labels_1, clustering_labels_2):
    membership_1 = getMembershipMatrix(clustering_labels_1)
    membership_2 = getMembershipMatrix(clustering_labels_2)
    clustering_mat_1 = membership_1 @ membership_1.transpose()
    clustering_mat_2 = membership_2 @ membership_2.transpose()
    diff_mat = clustering_mat_1 - clustering_mat_2
    # L0 Norm (count non-zero entries)
    return np.count_nonzero(diff_mat)


"""
Input: Two arrays of clustering labels
Output: Normalized difference of the confusion matrices in Frobenius Norm
This Metric is suggested by Sarkar and Bickel
"""


def SarkarMetric_fromLabels(clustering_labels_1, clustering_labels_2):
    n_clusters = len(clustering_labels_1)
    membership_1 = getMembershipMatrix(clustering_labels_1)
    membership_2 = getMembershipMatrix(clustering_labels_2)
    clustering_mat_1 = membership_1 @ membership_1.transpose()
    clustering_mat_2 = membership_2 @ membership_2.transpose()
    diff_mat = clustering_mat_1 - clustering_mat_2
    # Frobenius Norm
    frobenius_norm = norm(diff_mat, ord='fro')
    return (frobenius_norm ** 2) / (n_clusters ** 2)


"""
Input: Two arrays of clustering labels
Output: Normalized difference of the confusion matrices in Frobenius Norm
This Metric is suggested by Sarkar and Bickel
"""


def SarkarMetric_fromMatrices(clustering_matrix_1, clustering_matrix_2):
    n_clusters = len(clustering_matrix_1)
    diff_mat = clustering_matrix_1 - clustering_matrix_2
    # Frobenius Norm
    frobenius_norm = norm(diff_mat, ord='fro')
    return (frobenius_norm ** 2) / (n_clusters ** 2)


"""
1. Metric suggested by Lei, Rinaldo
Input: Two arrays of clustering labels
Output: Error measure L of Lei, Rinaldo

!!! I scaled it with 0.5 at the end
"""


def LeiRinaldoMetric_1_fromLabels(clustering_labels_estimate, clustering_labels_true):
    n_nodes = len(clustering_labels_true)
    # get the according membership matrices \Theta and \hat{\Theta}
    membership_est = getMembershipMatrix(clustering_labels_estimate)
    membership_true = getMembershipMatrix(clustering_labels_true)
    # transpose so we can use itertools.permutations on the columns
    membership_est = membership_est.T
    membership_true = membership_true.T
    # initiate an array for the differences
    L0_differences = []
    # for loop over all permutation matrices
    for membership_est_perm in tqdm(itertools.permutations(membership_est)):
        diff_mat = membership_est_perm - membership_true
        L0_differences.append(np.count_nonzero(diff_mat))
    # get minimum difference
    return 0.5 * np.min(L0_differences) / n_nodes


"""
1. Metric suggested by Lei, Rinaldo
Input: Two membership matrices
Output: Error measure L of Lei, Rinaldo
"""


def LeiRinaldoMetric_1_fromMatrices(membership_est, membership_true):
    n_nodes = len(membership_est)
    # transpose so we can use itertools.permutations on the columns
    membership_est = membership_est.T
    membership_true = membership_true.T
    # initiate an array for the differences
    L0_differences = []
    # for loop over all permutation matrices
    for membership_est_perm in tqdm(itertools.permutations(membership_est)):
        diff_mat = membership_est_perm - membership_true
        L0_differences.append(np.count_nonzero(diff_mat))
    # get minimum difference
    return 0.5 * np.min(L0_differences) / n_nodes


"""
1. Metric suggested by Lei, Rinaldo
Input: Two arrays of clustering labels
Output: Error measure L of Lei, Rinaldo
This one uses match to get the permutation matrix
"""


def LeiRinaldoMetric_1_fromLabels_match(clustering_labels_estimate, clustering_labels_true):
    n_nodes = len(clustering_labels_true)
    # get the according membership matrices \Theta and \hat{\Theta}
    membership_est = getMembershipMatrix(clustering_labels_estimate)
    membership_true = getMembershipMatrix(clustering_labels_true)

    permutation_mat = match(membership_est, membership_true)
    L0_diff = np.count_nonzero(membership_est @ permutation_mat - membership_true)

    # get minimum difference
    diff = 0.5 * L0_diff / n_nodes
    return diff


"""
1. Metric suggested by Lei, Rinaldo
Input: Two membership matrices
Output: Error measure L of Lei, Rinaldo
This one uses match to get the permutation matrix
"""


def LeiRinaldoMetric_1_fromMatrices_match(membership_est, membership_true):
    n_nodes = len(membership_est)

    permutation_mat = match(membership_est, membership_true)
    L0_diff = np.count_nonzero(membership_est @ permutation_mat - membership_true)

    # get minimum difference
    diff = 0.5 * L0_diff / n_nodes
    return diff


"""
2. Metric suggested by Lei, Rinaldo
Input: Two arrays of clustering labels
Output: Error measure \Tilde{L} of Lei, Rinaldo

!!! I scaled it with 0.5 at the end
"""


def LeiRinaldoMetric_2(clustering_labels_estimate, clustering_labels_true):
    # get number of clusters
    k = np.max(clustering_labels_true) + 1
    # get sizes of the clusters
    _, n_k = np.unique(clustering_labels_true, return_counts=True)

    # get the according membership matrices \Theta and \hat{\Theta}
    membership_est = getMembershipMatrix(clustering_labels_estimate)
    membership_true = getMembershipMatrix(clustering_labels_true)
    # transpose so we can use itertools.permutations on the columns
    membership_est = membership_est.T
    membership_true = membership_true.T

    # initiate array for differences over all permutations
    differences_over_perm = []
    # for loop over all permutation matrices
    for membership_est_perm in itertools.permutations(membership_est):
        diff_mat = membership_est_perm - membership_true

        # initiate array for differences over all clusters
        differences_over_clusters = []
        # for loop over all clusters
        for cluster in np.arange(k):
            indices = np.where(clustering_labels_true == cluster)[0]
            difference = np.count_nonzero(diff_mat[:, indices])
            differences_over_clusters.append(difference / n_k[cluster])
        # get maximum difference over all clusters
        differences_over_perm.append(np.max(differences_over_clusters))

    # get minimum difference over all permutations
    return 0.5 * np.min(differences_over_perm)
