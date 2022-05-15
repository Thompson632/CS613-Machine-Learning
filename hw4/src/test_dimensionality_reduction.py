import data_util
import math_util
import plot
from pca import PCA
from knn import KNN
from eigen_faces import Eigenfaces
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

    eigen_vectors = model.train_model(X)
    non_whitened = model.evaluate_model(X, eigen_vectors)

    whitned_eigen_vectors = model.whiten_data(non_whitened)
    whitened = model.evaluate_model(non_whitened, whitned_eigen_vectors)

    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=non_whitened)
    plot.plot_pca_scatterplot(title="Whitened PCA", data=whitened)
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

    pca_model = PCA(num_components)

    eigen_vectors = pca_model.train_model(X_train)
    pca_train = pca_model.evaluate_model(X_train, eigen_vectors)
    pca_valid = pca_model.evaluate_model(X_valid, eigen_vectors)

    knn_model = KNN(k)
    knn_model.train_model(pca_train, y_train)
    valid_preds = knn_model.evaluate_model(pca_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)

    print("\nK-NEAREST NEIGHBORS (KNN) WITH PCA WHITENED")
    whitened_eigen_vectors = pca_model.whiten_data(pca_train)
    pca_train_whiten = pca_model.evaluate_model(
        pca_train, whitened_eigen_vectors)
    pca_whiten_valid = pca_model.evaluate_model(
        pca_valid, whitened_eigen_vectors)

    knn_model = KNN(k)
    knn_model.train_model(pca_train_whiten, y_train)
    valid_preds = knn_model.evaluate_model(pca_whiten_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("K =", k, "D =", num_components, "\nAccuracy:", valid_accuracy)


def eigen_faces_compression(filename, num_components):
    X, _, means, stds = load_wildfaces_pca(filename)

    ef = Eigenfaces()
    ef.build_eigenfaces(X, means, stds, num_components=num_components)


pca(filename="lfw20.csv", num_components=2)
knn(filename="lfw20.csv", k=1)
knn_pca(filename="lfw20.csv", k=1, num_components=100)
eigen_faces_compression(filename="lfw20.csv", num_components=1)