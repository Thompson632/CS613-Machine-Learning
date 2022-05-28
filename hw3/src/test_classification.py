import data_util
import math_util
from evaluator import Evaluator
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def load_spambase(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_mean_std_of_features(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def load_ctg(filename):
    data = data_util.load_data(filename, 2)
    data = np.delete(data, -2, axis=1)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_mean_std_of_features(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def naive_bayes(stability_constant, filename):
    print("\n======================================================")
    print("NAIVES BAYES CLASSIFIER:")

    x_train, y_train, x_valid, y_valid = load_spambase(filename)

    nb = NaiveBayes(stability_constant=stability_constant)
    nb.fit(x_train, y_train)
    valid_preds = nb.predict(x_valid)

    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    print("Validation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


def multi_class_naive_bayes(stability_constant, filename):
    print("\n======================================================")
    print("MULTI-CLASS NAIVE BAYES CLASSIFIER:")

    x_train, y_train, x_valid, y_valid = load_ctg(filename)

    nb = NaiveBayes(stability_constant=stability_constant)
    nb.fit(x_train, y_train)
    valid_preds = nb.predict(x_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)


def decision_tree(filename, min_observation_split, min_information_gain):
    print("\n======================================================")
    print("DECISION TREE CLASSIFIER:")

    x_train, y_train, x_valid, y_valid = load_spambase(
        filename)

    dt = DecisionTree(min_observation_split=min_observation_split,
                      min_information_gain=min_information_gain)
    dt.fit(x_train, y_train)
    valid_preds = dt.predict(x_valid)

    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    print("Validation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


def multi_class_decision_tree(filename, min_observation_split, min_information_gain):
    print("\n======================================================")
    print("MULTI-CLASS DECISION TREE CLASSIFIER:")

    x_train, y_train, x_valid, y_valid = load_ctg(filename)

    dt = DecisionTree(min_observation_split=min_observation_split,
                      min_information_gain=min_information_gain)
    dt.fit(x_train, y_train)
    valid_preds = dt.predict(x_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)


naive_bayes(stability_constant=1e-4, filename="spambase.data")
decision_tree(filename="spambase.data",
              min_observation_split=2, min_information_gain=0)
multi_class_naive_bayes(stability_constant=1e-4, filename="CTG.csv")
multi_class_decision_tree(
    filename="CTG.csv", min_observation_split=2, min_information_gain=0)
