import data_util
import math_util
import plot
from pca import PCA
from knn import KNN
from evaluator import Evaluator
import numpy as np


def load_wildfaces_pca(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    X, y = data_util.get_features_actuals(data)
    x_stabilized = math_util.stabilize_data(X)

    x_train_zscored, _, _ = math_util.zscore_data(x_stabilized)

    return x_train_zscored, y


def load_wildfaces_knn(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    x_train_stabilized = math_util.stabilize_data(x_train)
    x_valid_stabilized = math_util.stabilize_data(x_valid)

    x_train_zscored, means, stds = math_util.zscore_data(x_train_stabilized)
    x_valid_zscored, _, _ = math_util.zscore_data(
        x_valid_stabilized, means_train=means, stds_train=stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def pca(filename, num_components):
    X, _ = load_wildfaces_pca(filename)

    pca = PCA()
    non_whitened = pca.compute_pca(X, num_components)
    whitened = pca.whiten_data(pca)

    # Plot Non-Whitened Data
    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=non_whitened)

    # Plot Whitened Data
    plot.plot_pca_scatterplot(title="Whitened PCA", data=whitened)

    # Plot Non-Whited and Whitened Data
    plot.plot_pca_scatterplot_overlay(
        title="Non-Whitened and Whitened PCA", pca=non_whitened, wpca=whitened)


def knn(filename, k):
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    # Number of dimensions
    D = np.shape(X_train)[1]

    knn = KNN(k=k)
    knn.train_model(X_train, y_train)
    valid_preds = knn.evaluate_model(X_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", D, "(Original) \nAccuracy:", valid_accuracy)


def knn_pca(filename, k, num_components):
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    # Number of dimensions
    D = num_components

    pca = PCA()
    pca_train = pca.compute_pca(X_train, num_components)
    pca_valid = pca.compute_pca(X_valid, num_components)

    knn = KNN(k=k)
    knn.train_model(pca_train, y_train)
    valid_preds = knn.evaluate_model(pca_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", D, "Accuracy:", valid_accuracy)


def knn_pca_whiten(filename, k, num_components=None):
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    # Number of dimensions
    D = num_components

    pca = PCA()
    pca_train = pca.compute_pca(X_train, num_components)
    pca_train_whiten = pca.whiten_data(pca_train)
    pca_valid = pca.compute_pca(X_valid, num_components)
    pca_whiten_valid = pca.whiten_data(pca_valid)

    knn = KNN(k=k)
    knn.train_model(pca_train_whiten, y_train)
    valid_preds = knn.evaluate_model(pca_whiten_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", D, "Accuracy:", valid_accuracy)


pca(filename="lfw20.csv", num_components=2)
knn(filename="lfw20.csv", k=5)
knn_pca(filename="lfw20.csv", k=5, num_components=100)
knn_pca_whiten(filename="lfw20.csv", k=5, num_components=100)