Author:  Michael Stepzinski
Date:	 November 28, 2021
Purpose: CS422 Project 4 PCA Project Writeup

I have two files, compress.py and pca.py

compress.py

compress_images
	This function compresses images by using a large dataset of images and finding 
	their principal components. Using functions from pca.py, the data is centered 
	and scaled, covariance is calculated, and then eigenvalues & vectors are found. 
	The number of images is found and the principal components are used to project 
	the data. Compressed image data is then found, and rescaled to proper greyscale 
	image format, a directory to store the images is created, then the reconstructed 
	images are saved.
	
load_data
	This function loads image data into the program as a numpy float array. The os 
	library is imported to open and read the file, image height and width is saved, 
	then used to load all image data.


pca.py

compute_z
	This function centers and scales data by using the numpy library.

compute_covariance_matrix
	This function calculates the covariance matrix of input data by using the numpy 
	library.

find_pcs
	This function calculates the eigenvalues and corresponding eigenvectors by using 
	the numpy library.

project_data
	This function projects data onto principal components. The function first checks 
	for k and var, decides which one is desired, and then computes k anyway. This 
	way, the function can calculate how many principal components are needed to 
	account for the data variance as desired. Then matrix/vector of principal 
	components is then constructed and used to project the input data onto them.

