import math_util
import data_util
import plot
from evaluator import Evaluator
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree
from pca import PCA


def load_data(filename):
    data = data_util.load_data(filename, rows_to_skip=1)
    data = data_util.shuffle_data(data, 0)
    print("Data Shape:", data.shape)
    print("Total Empty Values:", len(data[data == '']))

    # Filling empty values with 0
    data[data == ''] = 0

    training, validation = data_util.get_train_valid_data(data)
    print("Training Shape:", training.shape)
    print("Validation Shape:", validation.shape)

    x_train, y_train = data_util.get_features_actuals(training)
    print("Training x_train Shape:", x_train.shape)
    print("Training y_train Shape:", y_train.shape)

    x_valid, y_valid = data_util.get_features_actuals(validation)
    print("Validation x_train Shape:", x_valid.shape)
    print("Validation y_train Shape:", y_valid.shape)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    x_train_bias = data_util.add_bias_feature(x_train_zscored)
    print("Training x_train_bias Shape:", x_train_bias.shape)
    x_valid_bias = data_util.add_bias_feature(x_valid_zscored)
    print("Validation x_valid_bias Shape:", x_valid_bias.shape)

    return x_train_bias, y_train, x_valid_bias, y_valid

def load_data_pca(filename):
    data = data_util.load_data(filename, rows_to_skip=1)
    data = data_util.shuffle_data(data, 0)
    
    # Filling empty values with 0
    data[data == ''] = 0

    X, y = data_util.get_features_actuals(data)

    means, stds = math_util.calculate_feature_mean_std(X)
    x_zscored = math_util.z_score_data(X, means, stds)

    return x_zscored, y, means, stds


def logistic_regression(filename, learning_rate, epochs, stability):
    print("\nLOGISTIC REGRESSION CLASSIFIER:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    eval = Evaluator()
    model = LogisticRegression(learning_rate, epochs, stability)
    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    _, _ = model.train_model(X_train, y_train, X_valid, y_valid)

    train_preds = model.evaluate_model(X_train)
    valid_preds = model.evaluate_model(X_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


def logistic_regression_pca(filename, learning_rate, epochs, stability, num_components):
    print("\nLOGISTIC REGRESSION CLASSIFIER WITH PCA:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    pca_model = PCA(num_components)
    print("\nNum Components:", num_components)

    eigenvectors = pca_model.train_model(X_train)
    z_train = pca_model.evaluate_model(X_train, eigenvectors)
    z_valid = pca_model.evaluate_model(X_valid, eigenvectors)

    eval = Evaluator()
    model = LogisticRegression(learning_rate, epochs, stability)
    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    _, _ = model.train_model(z_train, y_train, z_valid, y_valid)

    train_preds = model.evaluate_model(z_train)
    valid_preds = model.evaluate_model(z_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


def decision_tree(filename, min_observation_split, min_information_gain):
    print("\nDECISION TREE CLASSIFIER:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    eval = Evaluator()
    model = DecisionTree(min_observation_split=min_observation_split,
                         min_information_gain=min_information_gain)

    print("\nMin Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    model.train_model(X_train, y_train)

    train_preds = model.evaluate_model(X_train)
    valid_preds = model.evaluate_model(X_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


def decision_tree_pca(filename, min_observation_split, min_information_gain, num_components):
    print("\nDECISION TREE CLASSIFIER WITH PCA:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    pca_model = PCA(num_components)
    print("\nNum Components:", num_components)

    eigenvectors = pca_model.train_model(X_train)
    z_train = pca_model.evaluate_model(X_train, eigenvectors)
    z_valid = pca_model.evaluate_model(X_valid, eigenvectors)

    eval = Evaluator()
    model = DecisionTree(min_observation_split=min_observation_split,
                         min_information_gain=min_information_gain)

    print("\nMin Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    model.train_model(z_train, y_train)

    train_preds = model.evaluate_model(z_train)
    valid_preds = model.evaluate_model(z_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)
    
def pca(filename, num_components):
    X, _, _, _ = load_data_pca(filename)

    model = PCA(num_components)

    eigenvectors = model.train_model(X)
    z = model.evaluate_model(X, eigenvectors)

    whitened_eigenvectors = model.whiten_data(z)
    z_whitened = model.evaluate_model(z, whitened_eigenvectors)

    plot.plot_pca_scatterplot(title="Non-Whitened PCA", data=z)
    plot.plot_pca_scatterplot(title="Whitened PCA", data=z_whitened)
    plot.plot_pca_scatterplot_overlay(
        title="Non-Whitened and Whitened PCA", pca=z, wpca=z_whitened)


# logistic_regression(filename="ONLY-NUMBERS_NCAAB_Games_2014_2020.csv",
#                     learning_rate=0.1, epochs=10000, stability=10e-7)
# logistic_regression_pca(filename="ONLY-NUMBERS_NCAAB_Games_2014_2020.csv",
#                         learning_rate=0.1, epochs=10000, stability=10e-7,
#                         num_components=100)
# decision_tree(filename="ONLY-NUMBERS_NCAAB_Games_2014_2020.csv",
#               min_observation_split=2, min_information_gain=0)
# decision_tree_pca(filename="ONLY-NUMBERS_NCAAB_Games_2014_2020.csv",
#                   min_observation_split=2, min_information_gain=0,
#                   num_components=100)
pca(filename="ONLY-NUMBERS_NCAAB_Games_2014_2020.csv", num_components=2)