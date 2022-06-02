import data_util
import math_util
from evaluator import Evaluator
from svm import SVM
import numpy as np


def load_data(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    x_train_bias = data_util.add_bias_feature(x_train_zscored)
    x_valid_bias = data_util.add_bias_feature(x_valid_zscored)

    return x_train_bias, y_train, x_valid_bias, y_valid


def svm(filename, learning_rate, epochs):
    print("\n======================================================")
    print("SUPPORT VECTOR MACHINE CLASSIFIER:")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    model = SVM(lr=learning_rate, epochs=epochs, log_verbose=True)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

    eval = Evaluator()

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


svm(filename="spambase.data", learning_rate=0.001, epochs=1000)