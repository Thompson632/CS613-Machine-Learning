import data_util
import math_util
from pca import PCA
import plot


def load_wildfaces(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    X, y = data_util.get_features_actuals(data)
    x_stabilized = math_util.stabilize_data(X)

    means, stds = math_util.calculate_mean_std_of_features(x_stabilized)
    x_train_zscored = math_util.zscore_data(x_stabilized, means, stds)

    return x_train_zscored, y


def pca(filename, num_components):
    print("DIMENSIONALITY REDUCTION FOR VISUALIZATION")

    X, y = load_wildfaces(filename)

    pca = PCA()
    projections = pca.compute_pca(X, num_components)

    plot.plot_pca_scatterplot(y=y, projections=projections)


pca(filename="lfw20.csv", num_components=2)