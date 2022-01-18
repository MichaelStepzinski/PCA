# Author:   Michael Stepzinski
# Date:     28 November 2021
# Purpose:  CS422 Machine Learning Project 4 compress file

import pca
import numpy as np
from matplotlib import pyplot as plt
import os

# Function: compress_images
# Purpose:  find eigenfaces, write compressed images to directory
# Input:    image data, number of principal components to project onto
# Returns:  reconstructed image data
def compress_images(DATA,k):

    # data is centered and (optionally) scaled
    data = pca.compute_Z(DATA)
    # COV is covariance of data
    COV = pca.compute_covariance_matrix(data)
    # L, PCS is eigenvalues and corresponding eigenvectors (aka pricipal components) of COV
    L, PCS = pca.find_pcs(COV)

    # length of a PC vector is needed for u
    # u is k largest PCS used for projection
    # calculate u
    vector_len = len(PCS[0])
    u = np.zeros((vector_len, k))
    for n in range(k):
        u[:,n] = PCS[:,n]

    # project data onto u
    Z_star = pca.project_data(data, PCS, L, 100, 0)

    # calculate X_compressed from projected data and transpose of principal components
    X_compressed = np.dot(Z_star, u.transpose())

    # rescale images to be between 0 and 255
    X_compressed = X_compressed - X_compressed.min()
    X_compressed = X_compressed * 255 / X_compressed.max()

    # check if /Output/ exists, if not create
    path = 'Output/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    # save num_images to path as greyscale pngs
    num_images = range(len(X_compressed[0]))
    #num_images = range(100) # uncomment to get the 100 faces used for reconstruction, aka eigenfaces
    for n in num_images:
        temp = np.array(X_compressed[:,n])
        #temp = np.array(Z_star[:,n]) # uncomment for eigenfaces
        temp = temp.reshape(60, 48)
        plt.imsave(path + str(n), temp, cmap='gray', format='png')

    return X_compressed

# Function: load_data
# Purpose:  load data
# Input:    input directory
# Returns:  data as np array of floats
def load_data(input_dir):
    filelist = os.listdir(input_dir)
    temp = plt.imread(input_dir + '/' + filelist[0])
    img_height = len(temp)
    img_width = len(temp[0])
    shapetuple = ((img_height*img_width),len(filelist))
    data = np.empty(shapetuple)

    for index in range(len(filelist)):
        temp = plt.imread(input_dir + '/' + filelist[index])
        temp = temp.flatten()
        data[:,index] = temp

    data.astype(float)
    return data