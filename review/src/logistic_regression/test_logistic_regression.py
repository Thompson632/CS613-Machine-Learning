import numpy as np
import data_util
import math_util
from logistic_regression import LogisticRegression
from multi_class_logistic_regression import MultiClassLogisticRegression
from evaluator import Evaluator


def load_data(filename, should_read_as_strings=False, should_convert_to_float=False):
    data = data_util.load_data(filename, should_read_as_strings)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)

    x_train, y_train = data_util.get_features_actuals(
        training, should_convert_to_float)
    x_valid, y_valid = data_util.get_features_actuals(
        validation, should_convert_to_float)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    x_train_bias = data_util.add_bias_feature(x_train_zscored)
    x_valid_bias = data_util.add_bias_feature(x_valid_zscored)

    return x_train_bias, y_train, x_valid_bias, y_valid


def binary_logistic_regression(filename, learning_rate, epochs, stability):
    print("\n======================================================")
    print("BINARY LOGISTIC REGRESSION")

    X_train, y_train, X_valid, y_valid = load_data(filename)

    model = LogisticRegression(
        lr=learning_rate, epochs=epochs, stability_constant=stability, log_verbose=False)
    train_losses, valid_losses = model.fit(
        X_train, y_train, X_valid, y_valid)

    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

    eval = Evaluator()
    eval.plot_mean_log_loss(train_losses, valid_losses, epochs)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("Training Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)

    # 8. Plot a Precision-Recall Graph
    training_precisions, training_recalls = eval.evaluate_precision_recall_with_threshold(
        y_train, train_preds, 0, 1, 11)
    valid_precisions, valid_recalls = eval.evaluate_precision_recall_with_threshold(
        y_valid, valid_preds, 0, 1, 11)
    eval.plot_precision_recall(
        training_precisions, training_recalls, valid_precisions, valid_recalls)


def multi_class_logistic_regression(filename, learning_rate, epochs, stability):
    print("\n======================================================")
    print("MULTI-CLASS LOGISTIC REGRESSION:")

    X_train, y_train, X_valid, y_valid = load_data(
        filename, should_read_as_strings=True, should_convert_to_float=True)

    unique_classes = np.unique(y_train)

    model = MultiClassLogisticRegression(
        lr=learning_rate, epochs=epochs, stability_constant=stability, log_verbose=False, unique_classes=unique_classes)
    model.fit(X_train, y_train, X_valid, y_valid)

    valid_preds = model.predict(X_valid)

    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)

    confusion_matrix = eval.compute_confusion_matrix(
        y_valid, valid_preds, unique_classes)
    print("Confusion Matrix:\n", confusion_matrix)


binary_logistic_regression(filename="spambase.data",
                           learning_rate=0.1, epochs=10000, stability=10e-7)
multi_class_logistic_regression(filename="iris.data",
                                learning_rate=0.1, epochs=10000, stability=10e-7)