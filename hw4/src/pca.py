import math_util
import numpy as np


class PCA:
    def compute_pca(self, X, num_components=2):
        '''
        Compute the principal components by first pre-processing the input data in order to get our 
        zero centered features, our eigen values, and eigen vectors. From there, we sort the eigen
        values in descending order and then sort our eigen vectors based on the eigen values. With 
        our now sorted eigen vectors, we select the top N-components based on the input value to
        return the eigen vectors with the largest eigen value. Finally, we project the data 
        onto the largest vectors.

        :param X: The features data
        :param num_components: The number of components to reduce our features too

        :return the largest eigen vectors and the project data onto the
        largest eigen vectors
        '''
        # Pre-process data to remove duplicate code
        X_centered, eigen_values, eigen_vectors = self.preprocess_data(X)

        # Get the indices to sort in decreasing order
        sorted_indices = np.argsort(-eigen_values)

        # Sort the Vectors
        sorted_vectors = eigen_vectors[:, sorted_indices]

        # With our sorted vectors, select the top N-components
        largest_vectors = sorted_vectors[:, :num_components]

        # Project the data onto the top N-components
        projection = np.dot(X_centered, largest_vectors)

        return largest_vectors, projection

    def whiten_data(self, X):
        '''
        Whitens our principal component(s) by first pre-processing the input data in order to get our 
        zero centered features, our eigen values, and eigen vectors. From there, we can 
        "whiten" our vectors by dividing them by the square-root of their eigen values. 
        Finally, we project the data onto the whitened eigen vectors.

        :param X: The features (principal components at this point) data

        :return the projected data onto the whitened eigen vectors
        '''
        # Pre-Process data to remove duplicate code
        X_centered, eigen_values, eigen_vectors = self.preprocess_data(X)

        # Whiten our Eigen Vectors
        whitened_eigen_vectors = eigen_vectors / np.sqrt(eigen_values)

        # Project the data onto the whitened vectors
        projection = np.dot(X_centered, np.dot(
            whitened_eigen_vectors, eigen_vectors))

        return projection

    def preprocess_data(self, X):
        '''
        Helper function that pre-processes the feature data by performing the following:
        1) Zero center our data by subtracting the mean
        2) Calculate the covariance matrix of our data
        3) Compute the Eigen Values and Eigen Vectors of our covariance matrix

        :param X: The current feature data

        :return The zero centered features, eigen values, and eigen vectors
        '''
        # Center our features around zero
        X_centered = X - math_util.calculate_mean(X=X, axis=0)

        # Calculate the Covariance Matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute Eigen values and vectors
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        return X_centered, eigen_values, eigen_vectors