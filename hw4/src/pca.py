import numpy as np


class PCA:

    def compute_pca(self, X, num_components=2):
        '''
        Computes the Principal Component Analysis of our feature data to find the 
        most relevant features. First we calculate the covariance matrix to remove
        un-needed features. Then with our covariance matrix, we use compute eigen-
        decomposition to get the eigen values and vectors that will be used to
        determine our most relevant feature. With the eigen vectors, we can return
        the number of features we want based on the num_components parameter
        (default to 2 components). Finally, we want to project our data based on
        the found principal components and return that output.

        :param X: The features data
        :param num_components: The number of components (or features)

        :return the data projected to the principal components
        '''
        # Set our number of components
        self.num_components = num_components

        # Calculate the Covariance Matrix of our Features
        cov_matrix = np.cov(X, rowvar=False)

        # Compute Eigendecomposition
        eigen_values, eigen_vectors = self.compute_eigendecomposition(
            cov_matrix)

        # Largest eigen values
        largest_values = eigen_values[:num_components]

        # Largest eigen vectors
        largest_vectors = eigen_vectors[:, :num_components]

        # Project the data based on the number of components
        nonwhitened_projection = self.project_data(X, largest_vectors)

        # Whiten our Eigen Vectors
        whitened_vectors = self.whitened_vectors(eigen_values, eigen_vectors)

        # Largest eigen values
        largest_whitened_values = eigen_values[:num_components]

        # Largest eigen vectors
        largest_whitened_vectors = eigen_vectors[:, :num_components]

        # Whiten the data
        whitened_projection = self.project_data(X, largest_whitened_vectors)
        return nonwhitened_projection, whitened_projection

    def compute_eigendecomposition(self, covariance_matrix):
        '''
        Computes the eigen values and vectors based on the calculated
        covariance matrix of our original data. It then sorts our data
        in descending order and returns the sorted eigen values and 
        vectors.

        :param covariance_matrix: The calculated covariance matrix from our 
        original data

        :return the sorted eigen values and vectors
        '''
        # Compute Eigendecomposition
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

        # Get the indices to sort in decreasing order
        sorted_indices = np.argsort(-eigen_values)

        # Sort the values
        sorted_values = eigen_values[sorted_indices]

        # Sort the Vectors
        sorted_vectors = eigen_vectors[:, sorted_indices]

        return sorted_values, sorted_vectors

    def project_data(self, X, eigen_vectors):
        projection = np.dot(eigen_vectors.T, X.T).T
        return projection

    def whitened_vectors(self, eigen_values, eigen_vectors):
        num_features = np.shape(eigen_vectors)[1]

        for i in range(num_features):
            current_value = eigen_values[i]
            current_vector = eigen_vectors[:, i]

            whitened_vector = current_vector / current_value

            eigen_vectors[:, i] = whitened_vector

        return eigen_vectors