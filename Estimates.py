import numpy as np
# for calculating eigenvalues and eigenvectors
from numpy.linalg import eig
# KMeans
from sklearn.cluster import KMeans

"""
get the normalized Graph Laplacian of a matrix
Input: Matrix A to form the Laplacian of
Output: Laplacian L
"""


def getLaplacian(A):
    # get degree matrix
    D = A.sum(axis=1)  # sum in each row of the adjacency matrix
    # D holds the diagonal entries of the degree matrix, all the others are zero
    # => for D^-1/2 we can just take the square root of  1/D
    D_sqrt = np.sqrt(1 / D)
    # calculate the Laplacian L
    # matrix multiplication is very expensive => do elementwise multiplication
    L = np.multiply(D_sqrt, np.multiply(A, D_sqrt[:, np.newaxis]))
    return L
