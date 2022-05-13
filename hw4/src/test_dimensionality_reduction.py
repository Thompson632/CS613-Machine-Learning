import data_util
import math_util
from pca import PCA
import plot


def load_wildfaces(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    X, y = data_util.get_features_actuals(data)
    x_stabilized = math_util.stabilize_data(X)

    x_train_zscored, _, _ = math_util.zscore_data(x_stabilized)

    return x_train_zscored, y


def pca(filename, num_components):
    X, _ = load_wildfaces(filename)

    model = PCA()

    # Compute Principal Components
    pca = model.compute_pca(X, num_components)
    wpca = model.whiten_data(pca)

    # Plot Non-Whitened Data
    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=pca)

    # Plot Whitened Data
    plot.plot_pca_scatterplot(title="Whitened PCA", data=wpca)
    
    # Plot Non-Whited and Whitened Data
    plot.plot_pca_scatterplot_overlay(title="Non-Whitened and Whitened PCA", pca=pca, wpca=wpca)


pca(filename="lfw20.csv", num_components=2)