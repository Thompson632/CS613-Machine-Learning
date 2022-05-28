import data_util
import math_util
import plot
from pca import PCA
from knn import KNN
from eigenfaces import Eigenfaces
from evaluator import Evaluator
import numpy as np


def load_wildfaces_pca(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    X, y = data_util.get_features_actuals(data)
    x_stabilized = math_util.stabilize_data(X)

    means, stds = math_util.calculate_feature_mean_std(x_stabilized)
    x_zscored = math_util.z_score_data(x_stabilized, means, stds)

    return x_zscored, y, means, stds


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
    X, _, _, _ = load_wildfaces_pca(filename)

    model = PCA(num_components)

    eigenvectors = model.fit(X)
    z = model.predict(X, eigenvectors)

    whitened_eigenvectors = model.whiten(z)
    z_whitened = model.predict(z, whitened_eigenvectors)

    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=z)
    plot.plot_pca_scatterplot(title="Whitened PCA", data=z_whitened)
    plot.plot_pca_scatterplot_overlay(
        title="Non-Whitened and Whitened PCA", pca=z, wpca=z_whitened)


def knn(filename, k):
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    num_components = np.shape(X_train)[1]

    knn = KNN(k)
    knn.fit(X_train, y_train)
    valid_preds = knn.predict(X_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components,
          "(Original) \nAccuracy:", valid_accuracy)


def knn_pca(filename, k, num_components):
    print("\nK-NEAREST NEIGHBORS (KNN) WITH PCA")
    X_train, y_train, X_valid, y_valid = load_wildfaces_knn(filename)

    pca_model = PCA(num_components)

    eigenvectors = pca_model.fit(X_train)
    z_train = pca_model.predict(X_train, eigenvectors)
    z_valid = pca_model.predict(X_valid, eigenvectors)

    knn_model = KNN(k)
    knn_model.fit(z_train, y_train)
    valid_preds = knn_model.predict(z_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)

    print("\nK-NEAREST NEIGHBORS (KNN) WITH PCA WHITENED")
    whitened_eigenvectors = pca_model.whiten(z_train)
    z_train_whiten = pca_model.predict(
        z_train, whitened_eigenvectors)
    z_valid_whiten = pca_model.predict(
        z_valid, whitened_eigenvectors)

    knn_model = KNN(k)
    knn_model.fit(z_train_whiten, y_train)
    valid_preds = knn_model.predict(z_valid_whiten)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)


def eigenfaces_compression(filename, num_components, person_index):
    X, _, means, stds = load_wildfaces_pca(filename)

    ef = Eigenfaces(X, means, stds, person_index)
    ef.build_eigenfaces(num_components)


pca(filename="lfw20.csv", num_components=2)
knn(filename="lfw20.csv", k=1)
knn_pca(filename="lfw20.csv", k=1, num_components=100)
eigenfaces_compression(filename="lfw20.csv", num_components=1, person_index=224)