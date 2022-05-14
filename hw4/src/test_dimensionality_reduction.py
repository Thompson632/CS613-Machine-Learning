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

    means, stds = math_util.calculate_feature_mean_std(x_stabilized)
    x_zscored = math_util.z_score_data(x_stabilized, means, stds)

    return x_zscored, y


def load_wildfaces_knn(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    x_train_stabilized = math_util.stabilize_data(x_train)
    x_valid_stabilized = math_util.stabilize_data(x_valid)

    means, stds = math_util.calculate_feature_mean_std(x_train_stabilized)
    x_train_zscored = math_util.z_score_data(x_train_stabilized, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid_stabilized, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def pca(filename, num_components):
    X, _ = load_wildfaces_pca(filename)

    pca = PCA()
    _, non_whitened = pca.compute_pca(X, num_components)
    whitened = pca.whiten_data(non_whitened)

    # Plot Non-Whitened Data
    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=non_whitened)

    # Plot Whitened Data
    plot.plot_pca_scatterplot(title="Whitened PCA", data=whitened)

    # Plot Non-Whited and Whitened Data
    plot.plot_pca_scatterplot_overlay(
        title="Non-Whitened and Whitened PCA", pca=non_whitened, wpca=whitened)


def knn(filename, k):
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    num_components = np.shape(X_train)[1]

    knn = KNN(k)
    knn.train_model(X_train, y_train)
    valid_preds = knn.evaluate_model(X_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components,
          "(Original) \nAccuracy:", valid_accuracy)


def knn_pca(filename, k, num_components):
    print("\nK-NEAREST NEIGHBORS (KNN) WITH PCA")
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    pca = PCA()
    _, pca_train = pca.compute_pca(X_train, num_components)
    _, pca_valid = pca.compute_pca(X_valid, num_components)

    knn = KNN(k)
    knn.train_model(pca_train, y_train)
    valid_preds = knn.evaluate_model(pca_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)

    print("\nK-NEAREST NEIGHBORS (KNN) WITH PCA WHITENED")
    pca_train_whiten = pca.whiten_data(pca_train)
    pca_whiten_valid = pca.whiten_data(pca_valid)

    knn = KNN(k)
    knn.train_model(pca_train_whiten, y_train)
    valid_preds = knn.evaluate_model(pca_whiten_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)


pca(filename="lfw20.csv", num_components=2)
knn(filename="lfw20.csv", k=1)
knn_pca(filename="lfw20.csv", k=1, num_components=100)