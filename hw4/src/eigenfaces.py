from pca import PCA
import numpy as np
import plot
import math_util


class Eigenfaces:
    def __init__(self, X, means, stds, person_index):
        '''
        Constructor that takes in the stabilized and zscored features data
        along with the features mean and standard deviation values. The
        constructor also takes in an index for the person we want to 
        reconstuct.

        :param X: The features data
        :param means: The features mean values
        :param stds: The features standard deviation values
        :param person_index: The index of the person we want to reconstruct

        :return None
        '''
        self.X = X
        self.means = means
        self.stds = stds

        self.person_index = person_index

    def build_eigenfaces(self, num_components=1):
        '''
        Build all of the required images based on the proposed requirements. 
        We build the following:
        (1) The image based on the primary principal component of the data
        (2) The reconstructed image based on the primary principal component 
        of the data
        (3) The original image we plan to used for compression / reconstruction
        (4) The 95% reconstructed original image

        :param num_components: The default number of components we are to train 
        our PCA model with

        :return None
        '''
        # Primary Component as Image
        model = PCA(num_components)
        eigenvectors = model.train_model(self.X)
        reshaped_eigenvectors = np.reshape(eigenvectors, (87, 65))
        plot.plot_eigenfaces(title="Primary Component as Image",
                             eigenvectors=reshaped_eigenvectors)

        # Reconstructed Image One Component
        person_row = self.X[self.person_index]
        one_component = self.reconstruct(model, eigenvectors, person_row)
        plot.plot_eigenfaces(title="Reconstructed Image Using One Component",
                             eigenvectors=one_component)

        # Image Used for Compression/Reconstruction
        num_features = np.shape(self.X)[1]
        model = PCA(num_features)
        eigenvectors = model.train_model(self.X)
        person_row = self.X[self.person_index]
        all_components = self.reconstruct(model, eigenvectors, person_row)
        plot.plot_eigenfaces(title="Original Image",
                             eigenvectors=all_components)

        # Determine Minimum Components for 95% Reconstruction
        min_num_components, min_eigenvectors = model.determine_min_components(
            self.X)
        print("\nMinimum Number of Components Necessary to Perform 95% Reconstruction:",
              min_num_components)

        # 95% Reconstructed Image with Min-Components
        model = PCA(min_num_components)
        eigenvectors = model.train_model(self.X)
        person_row = self.X[self.person_index]
        min_components = self.reconstruct(model, min_eigenvectors, person_row)
        plot.plot_eigenfaces(
            title="95% Reconstruction Image", eigenvectors=min_components)

        # Determine Minimum Components for 100% Reconstruction
        min_num_components, min_eigenvectors = model.determine_min_components(
            self.X, threshold=1)
        print("\nMinimum Number of Components Necessary to Perform 100% Reconstruction:",
              min_num_components)

        # 100% Reconstructed Image with Min-Components
        model = PCA(min_num_components)
        eigenvectors = model.train_model(self.X)
        person_row = self.X[self.person_index]
        min_components = self.reconstruct(model, min_eigenvectors, person_row)
        plot.plot_eigenfaces(
            title="100% Reconstruction Image", eigenvectors=min_components)

    def reconstruct(self, model, eigenvectors, person_row):
        '''
        Method that helps us reconstruct our image based on our trained model, 
        eigenvectors, and the desired row to be projected onto the eigenvectors. 
        In order to do this, we need to perform the following:
        (1) Projected the person onto the eigenvectors
        (2) Calculate x_hat by taking the dot product of the eigenvectors 
        and our transposed Z
        (3) Un-zscore the data
        (4) Unstabilize the data
        (5) Reshape the data into an 87 x 65 image

        :param model: The PCA model we have trained
        :param eigenvectors: The learned eigenvectors based on the features data
        :param person_row: The person's image row that we want to reconstruct

        :return the reconstructed and reshape x_hat
        '''
        # Project the person onto the eigen vectors
        z = model.evaluate_model(person_row, eigenvectors)

        # With our project data, we want to calculate x_hat
        x_hat = np.dot(eigenvectors, z.T)

        # Un-zscore our data
        x_hat = math_util.un_zscore_data(x_hat, self.means, self.stds)

        # Unstabilize our data
        x_hat = math_util.unstablize_data(x_hat)

        # Reshape our x_hat to 87 x 65
        x_hat_reshape = np.reshape(x_hat, (87, 65))

        # Return our x_hat_reshape to be displayed as an image
        return x_hat_reshape
