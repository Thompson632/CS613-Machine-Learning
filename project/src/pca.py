import math_util
import numpy as np


class PCA:
    def __init__(self, num_components=2):
        '''
        Constructor that takes in the number of components to be selected when 
        we are training our model. Default number of components is set to two.

        :param num_components: Number of components to select from our computed
        eigenvectors

        :return None
        '''
        self.num_components = num_components

    def fit(self, X):
        '''
        Compute the principal components by first pre-processing the input data 
        in order to get our zero centered features, our eigenvalues, and eigen
        vectors. From there, we sort the eigenvalues in descending order and 
        then sort our eigenvectors based on the eigenvalues. With our now 
        sorted eigenvectors, we select the top N-components based on the input 
        value and set these as our training eigenvectors for future projection.

        :param X: The features data
        :param num_components: The number of components to reduce our features

        :return the eigenvectors of the largest principal components
        '''
        # Pre-process data to remove duplicate code
        eigenvalues, eigenvectors = self.preprocess(X)

        # Sort the values and vectors
        _, sorted_vectors = self.sort_eigen(eigenvalues, eigenvectors)

        # With our sorted vectors, select the top N-components
        largest_eigenvectors = sorted_vectors[:, :self.num_components]
        
        return largest_eigenvectors

    def whiten(self, X):
        '''
        Whitens our principal component(s) by first pre-processing the input 
        data in order to get our zero centered features, our eigenvalues, and 
        eigenvectors. From there, we can "whiten" our vectors by dividing them 
        by the square-root of their eigenvalues. Finally, we project the data 
        onto the whitened eigenvectors.

        :param X: The features (principal components at this point) data

        :return the whitened eigenvectors of the principal components
        '''
        # Pre-process data to remove duplicate code
        eigenvalues, eigenvectors = self.preprocess(X)

        # Whiten our eigenvectors
        whitened_eigenvectors = eigenvectors / np.sqrt(eigenvalues)
        
        return whitened_eigenvectors

    def predict(self, X, projection):
        '''
        Evaluates our model by projecting our features data into the 
        eigenvectors space passed in as a parameter.

        :param X: The features data
        :param projection: The projection matrix (or eigenvectors) to project 
        our features data into

        :return the features data projected into the eigenvectors space
        '''
        return np.dot(X, projection)

    def preprocess(self, X):
        '''
        Helper function that pre-processes the feature data by performing the 
        following:
        1) Zero center our data by subtracting the mean
        2) Calculate the covariance matrix of our data
        3) Compute the eigenvalues and eigenvectors of our covariance matrix

        :param X: The current feature data

        :return the eigenvalues and eigenvectors
        '''
        # Center our features around zero
        X_centered = X - math_util.calculate_mean(X, axis=0)

        # Calculate the Covariance Matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        return eigenvalues, eigenvectors

    def sort_eigen(self, eigenvalues, eigenvectors):
        '''
        Sorts the eigenvalues and eigenvectors in decreasing order
        to have the largest principal components at the first indices
        of our values and vectors.

        :param eigenvalues: The eigenvalues we want to sort
        :param eigenvectors: The eigenvectors we want to sort

        :return the sorted eigenvalues and eigenvectors
        '''
        # Get the indices to sort in decreasing order
        sorted_indices = np.argsort(-eigenvalues)

        # Sort the values
        sorted_values = eigenvalues[sorted_indices]

        # Sort the vectors
        sorted_vectors = eigenvectors[:, sorted_indices]

        return sorted_values, sorted_vectors

    def determine_min_components(self, X, threshold=0.95):
        '''
        Determines the minimum number of components required in 
        order to reconstruct an image based on a default threshold
        of 95%. We achieve this by doing the following:
        (1) Eigenvalues and vectors of our features data
        (2) Sort our eigenvalues and eigenvectors in descending order
        (3) Calculate the absolute value of all eigenvalues and then
        we take the sum of them
        (4) Iterate over each sorted eigenvalue and do the following:
          (a) Get the eigenvalue at the given index
          (b) Calculate the absolute value of the eigvenvalue
          and then we take the sum of it
          (c) Divide our sum of the given eigenvalue by the the sum of
          all the eigenvalues
          (d) If our quotient is greater than or equal to 0.95, this
          is the minimum number of components we need
        (5) Get the minimum number of components from our eigenvectors
        and return them
        
        :param X: The features data
        :param threshold: The threshold for determing the minimum
        number of components
        
        :return the eigenvectors for the minimum number of components
        '''
        # Minimum components required for the default threshold
        # set to 95%
        min_components = 0

        # Pre-process data to remove duplicate code
        eigenvalues, eigenvectors = self.preprocess(X)

        # Sort the values and vectors
        sorted_values, sorted_vectors = self.sort_eigen(
            eigenvalues, eigenvectors)

        # Calculates the absolute value of our eigen values to be used when
        # determing the threshold
        sorted_abs_values = np.abs(sorted_values)
        sorted_abs_values_sum = np.sum(sorted_abs_values)

        for index, _, in enumerate(sorted_values):
            # Get our eigenvalue at this index
            eigenvalue = sorted_values[:index]
            # Take the absolute value of our eigenvalue
            eigenvalue_abs = np.abs(eigenvalue)
            # Calculate the sum of our eigenvalue
            eigenvalue_abs_sum = np.sum(eigenvalue_abs)

            # Divide the sum of the absolute value of this eigenvalue by the 
            # sum of the absolute value of all eigenvalues. If the quotient of 
            # this is greater than or equal to the threshold, then this index 
            # represents the minimum number of components required to achieve 
            # 95% reconstruction of the image
            if eigenvalue_abs_sum / sorted_abs_values_sum >= threshold:
                # Adding a one here because our arrays begin at index 0.
                # Therefore, the minimum number of components will be the index
                # that meets the threshold + one
                min_components = index + 1
                break

        # With our sorted vectors, select the minimum number of components
        # to achieve 95% image reconstruction
        largest_eigenvectors = sorted_vectors[:, :min_components]
        
        return min_components, largest_eigenvectors