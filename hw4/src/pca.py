import math_util
import numpy as np


class PCA:

    def __init__(self, num_components=2):
        '''
        Constructor that takes in the number of components to be selected when we
        are training our model. Default number of components is set to two.

        :param num_components: Number of components to select from our computed
        eigen vectors

        :return None
        '''
        self.num_components = num_components

    def train_model(self, X):
        '''
        Compute the principal components by first pre-processing the input data 
        in order to get our zero centered features, our eigen values, and eigen
        vectors. From there, we sort the eigen values in descending order and 
        then sort our eigen vectors based on the eigen values. With our now 
        sorted eigen vectors, we select the top N-components based on the input 
        value and set these as our training eigen vectors for future projection.

        :param X: The features data
        :param num_components: The number of components to reduce our features

        :return None
        '''
        # Pre-process data to remove duplicate code
        X_centered, eigen_values, eigen_vectors = self.preprocess_data(X)

        # Get the indices to sort in decreasing order
        sorted_indices = np.argsort(-eigen_values)

        # Sort the Vectors
        sorted_vectors = eigen_vectors[:, sorted_indices]

        # With our sorted vectors, select the top N-components
        eigen_vectors = sorted_vectors[:, :self.num_components]
        return eigen_vectors

    def whiten_data(self, X):
        '''
        Whitens our principal component(s) by first pre-processing the input 
        data in order to get our zero centered features, our eigen values, and 
        eigen vectors. From there, we can "whiten" our vectors by dividing them 
        by the square-root of their eigen values. Finally, we project the data 
        onto the whitened eigen vectors.

        :param X: The features (principal components at this point) data

        :return the projection matrix onto the whitened eigen vectors
        '''
        # Pre-Process data to remove duplicate code
        X_centered, eigen_values, eigen_vectors = self.preprocess_data(X)

        # Whiten our Eigen Vectors
        whitened_eigen_vectors = eigen_vectors / np.sqrt(eigen_values)
        # return whitened_eigen_vectors

        projection = np.dot(
            np.dot(eigen_vectors, whitened_eigen_vectors), eigen_vectors.T)
        return projection

    def evaluate_model(self, X, projection):
        '''
        Evaluates our model by projecting our features data into the eigen vectors
        space passed in as a parameter.

        :param X: The features data
        :param projection: The projection matrix (or eigen vectors) to project our 
        features data into

        :return the features data projected into the eigen vectors spsace
        '''
        return np.dot(X, projection)

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