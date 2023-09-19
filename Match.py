import numpy as np

"""
Input:  Two membership matrices that need to be matched
        Both need to be of the same set of nodes!
        The order of the two doesn't matter
Output: Best permutation matrix to align the clusterings
        The permutation matrix needs to be applied to membership_1!!
"""


def match(membership_1, membership_2):
    k = len(membership_1[0])
    confusion_mat = membership_1.transpose() @ membership_2
    projection_mat = np.zeros((k, k))

    max_value = np.max(confusion_mat)
    while max_value >= 0:
        # find the first occurance of max_value in the confusion matrix
        row, col = np.argwhere(confusion_mat == max_value)[0]

        # set according row and column to zeros
        confusion_mat[row] = -1
        confusion_mat[:, col] = -1

        # set according entry in the projection matrix to 1
        projection_mat[row, col] = 1

        # set max_value to new maximal value
        max_value = np.max(confusion_mat)

    return projection_mat
