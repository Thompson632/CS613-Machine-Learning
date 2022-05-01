import util
import math_util
from evaluator import Evaluator
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def naive_bayes(stability_constant):
    print("NAIVES BAYES CLASSIFIER:")

    data = util.load_data("spambase.data")
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)
    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    means, stds = math_util.compute_training_mean_std_by_feature(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    nb = NaiveBayes(stability_constant=stability_constant)
    nb.train_model(x_train_zscored, y_train)
    valid_preds = nb.evaluate_model(x_valid_zscored)

    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    print("Validation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)

    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train,
                            x_valid_zscored, y_valid, use_nb=True, is_multi=False)


def multi_class_naive_bayes(stability_constant):
    print("MULTI-CLASS NAIVE BAYES CLASSIFIER:")

    data = util.load_data("CTG.csv", rows_to_skip=2)
    data = np.delete(data, -2, axis=1)
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)
    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    means, stds = math_util.compute_training_mean_std_by_feature(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    nb = NaiveBayes(stability_constant=stability_constant)
    nb.train_model(x_train_zscored, y_train)
    valid_preds = nb.evaluate_model(x_valid_zscored)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)

    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train,
                            x_valid_zscored, y_valid, use_nb=True, is_multi=True)


def decision_tree():
    print("DECISION TREE:")

    data = util.load_data("spambase.data")
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)
    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    means, stds = math_util.compute_training_mean_std_by_feature(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    dt = DecisionTree()
    dt.train_model(x_train_zscored, y_train)
    valid_preds = dt.evaluate_model(x_valid_zscored)

    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    print("Validation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)

    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train,
                            x_valid_zscored, y_valid, use_nb=False, is_multi=False)


def multi_class_decision_tree():
    print("MULTI-CLASS DECISION TREE:")

    data = util.load_data("CTG.csv", rows_to_skip=2)
    data = np.delete(data, -2, axis=1)
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)
    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    means, stds = math_util.compute_training_mean_std_by_feature(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    dt = DecisionTree()
    dt.train_model(x_train_zscored, y_train)
    valid_preds = dt.evaluate_model(x_valid_zscored)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)

    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train,
                            x_valid_zscored, y_valid, use_nb=False, is_multi=True)


def compare_against_sklearn(x_train, y_train, x_valid, y_valid, use_nb=True, is_multi=False):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier

    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    model = None

    if use_nb:
        model = GaussianNB()
        model.fit(x_train, y_train)
    else:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
        model.fit(x_train, y_train)

    y_predict = model.predict(x_valid)

    if not is_multi:
        print("\nsklearn Precision:", precision_score(y_valid, y_predict))
        print("sklearn Recall:", recall_score(y_valid, y_predict))
        print("sklearn F-Measure:", f1_score(y_valid, y_predict))

    print("sklearn Accuracy:", accuracy_score(y_valid, y_predict))


naive_bayes(stability_constant=1e-4)
print("")
multi_class_naive_bayes(stability_constant=1e-4)
print("\n")
decision_tree()
print("")
multi_class_decision_tree()
