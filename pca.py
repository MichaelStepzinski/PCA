# Author:   Michael Stepzinski
# Date:     28 November 2021
# Purpose:  CS422 Machine Learning Project 4 pca file

import numpy as np
from matplotlib import pyplot as plt

# Function: compute_z
# Purpose:  centers and scales data
# Input:    centering and scaling flags
# Returns:  centered and scaled data
def compute_Z(X, centering=True, scaling=False):
    if centering:
        mean = np.mean(X, axis=0)
        X = X - mean
    if scaling:
        std_dev = np.std(X, axis=0)
        X = X / std_dev
    return np.around(X, decimals=2)

# Function: compute_covariance_matrix
# Purpose:  computes covariance matrix
# Input:    data matrix
# Returns:  covariance matrix
def compute_covariance_matrix(Z):
    Y = Z.transpose()
    Z = np.dot(Y, Z)
    return Z

# Function: find_pcs
# Purpose:  compute principal components and corresponding eigenvalues
# Input:    covariance matrix
# Returns:  eigenvalues, eigenvectors
def find_pcs(COV):
    eigenvalues, eigenvectors = np.linalg.eig(COV)
    return eigenvalues, eigenvectors

# Function: project_data
# Purpose:  project data onto principal components
# Input:    data(Z), Principal Components(PCS), Eigenvalues(L), num of PCS to project onto(k), variance to account for(var)
# Returns:  projected data
def project_data(Z, PCS, L, k, var):
    if k == 0:
        Lsum = np.sum(L)
        percentVar = L / Lsum
        accountedfor = 0.0
        for val in percentVar:
            accountedfor += val
            k += 1
            # if the variance is greater than the percent accounted for
            if var <= accountedfor:
                break

    # construct u
    vector_len = len(PCS[0])
    u = np.zeros((vector_len, k))

    for n in range(k):
        u[:,n] = PCS[:,n]

    v = np.dot(Z, u)

    return v