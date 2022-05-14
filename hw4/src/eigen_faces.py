from pca import PCA
import numpy as np
import plot
import math_util

class Eigenfaces:
    def build_eigenfaces(self, X, means, stds, num_components=1):
        pca = PCA()
        
        # Z-Scored Plot
        eigen_vectors, _ = pca.compute_pca(X, num_components)
        reshaped_largest_eigen_vector = np.reshape(eigen_vectors, (87, 65))
        plot.plot_eigen_faces(title="Z-Scored", eigen_vectors=reshaped_largest_eigen_vector)
        
        # Un-zscored Plot
        x_unzscored = math_util.un_zscore_data(X, means, stds)
        eigen_vectors, _ = pca.compute_pca(x_unzscored, num_components)
        reshaped_eigen_vectors = np.reshape(eigen_vectors, (87, 65))
        plot.plot_eigen_faces(title="Unzscored", eigen_vectors=reshaped_eigen_vectors)
        
        # Unstabilized
        x_unstabilized = math_util.unstablize_data(x_unzscored)
        eigen_vectors, _ = pca.compute_pca(x_unstabilized, num_components)
        reshaped_eigen_vectors = np.reshape(eigen_vectors, (87, 65))
        plot.plot_eigen_faces(title="Unstabilized", eigen_vectors=reshaped_eigen_vectors)