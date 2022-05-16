import numpy as np
import math_util


class LDA:
    def __init__(self, num_components=2):
        '''
        Constructor that takes in the number of components to be selected when 
        we are training our model. Default number of components is set to two.

        :param num_components: Number of components to select from our computed
        eigenvectors

        :return None
        '''
        self.num_components = num_components

    def train_model(self, X, y):
        '''
        Trains our model for Linear Disciminant Analysis
        
        :param X: The features data
        :param y: The target data
        
        :return the largest eigenvector
        '''
        # Get the number of features
        num_features = np.shape(X)[1]
        
        # Get the unique classes
        unique_classes = np.unique(y)
        
        # Calculate the overall mean of the features
        overall_feature_mean = math_util.calculate_mean(X, axis=0)
        
        # Create our SW and SB Matrices of num_features by num_features
        SW = np.zeros((num_features, num_features))
        SB = np.zeros((num_features, num_features))
        
        # For each class...
        for c in unique_classes:
            print("\n\nClass C:", c)
            
            # Get the classes data
            class_data = X[y == c]
            # Get the number of class observations
            num_class_observations = np.shape(class_data)[0]
            
            # Calculate class mean
            class_mean = math_util.calculate_mean(class_data,axis=0)
            
            # Center our features around zero
            class_mean_centered = class_data - class_mean
            print("\nClass Data:\n", class_data)
            print("Class Mean:\n", class_mean)
            print("Class Data - Class Mean:\n", class_mean_centered)
            
            # Update the within class scatter matrix
            SW += np.dot(class_mean_centered.T, class_mean_centered)
            print("\nClass Mean Centered Transposed:\n", class_mean_centered.T)
            print("Class Mean Centered:\n", class_mean_centered)
            print("SW:\n", SW)
            
            # Calculate the difference between the class mean and the overall mean
            mean_diff = (class_mean - overall_feature_mean)
            print("\nClass Mean:\n", class_mean)
            print("Overall Feature Mean:\n", overall_feature_mean)
            print("Class Mean - Overall Mean\n:", mean_diff)
            # Reshape our difference
            mean_diff = np.reshape(mean_diff, (num_features, 1))
            print("Mean Diff Reshaped:\n", mean_diff)
            
            # Update the class scatter difference
            SB += num_class_observations * np.dot(mean_diff, mean_diff.T)
            print("\nNum Class Observations:\n", num_class_observations)
            print("Mean Diff:\n", mean_diff)
            print("Mean Diff Transposed:\n", mean_diff.T)
            print("SB:\n", SB)

        # Compute SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        print("\n\nSW:\n", SW)
        print("SW Inverse\n:", np.linalg.inv(SW))
        print("SB\n", SB)
        print("A:\n", A)
        
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        print("\nEigenvalues:\n", eigenvalues)
        print("Eigenvectors:\n", eigenvectors)
        
        # Sort the values and vectors
        _, sorted_vectors = self.sort_eigen(eigenvalues, eigenvectors)

        # With our sorted vectors, select the top N-components
        largest_eigenvectors = sorted_vectors[:, :self.num_components]
        print("\nLargest Eigenvectors:\n", largest_eigenvectors)
        
        return largest_eigenvectors

    def evaluate_model(self, X, projection):
        '''
        Evaluates our model by projecting our features data into the 
        eigenvectors space passed in as a parameter.

        :param X: The features data
        :param projection: The projection matrix (or eigenvectors) to project 
        our features data into

        :return the features data projected into the eigenvectors spsace
        '''
        return np.dot(X, projection)
    
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
        
        print("\nSorted Eigenvalues:\n", sorted_values)
        print("\nSorted Eigenvectors:\n", sorted_vectors)

        return sorted_values, sorted_vectors