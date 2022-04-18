from multi_class_logistic_regression import MultiClassLogisticRegression
import util
from logistic_regression import LogisticRegression
from evaluator import Evaluator
import numpy as np


def binary_logistic_regression(learning_rate, epochs, stability):
    print("Binary Logistic Regression:")

    # Load data
    data = util.load_data("spambase.data", False)
    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training, False)
    x_valid, y_valid = util.get_features_actuals(validation, False)

    # Get mean and std of training data
    means, stds = util.compute_training_mean_std(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = util.z_score_data(x_valid, means, stds)
    
    # Add bias feature to the training data
    x_train_bias = util.add_bias_feature(x_train_zscored)
    
    # Add bias feature to the validation data
    x_valid_bias = util.add_bias_feature(x_valid_zscored)

    eval = Evaluator()
    lr = LogisticRegression(learning_rate, epochs, stability)

    # 5. Train Logistic Regression Model
    train_losses, valid_losses = lr.train_model(
        x_train_bias, y_train, x_valid_bias, y_valid)

    # Evaluate Model
    train_preds = lr.evaluate_model(x_train_bias)
    valid_preds = lr.evaluate_model(x_valid_bias)

    # 6. Plot Training and Validation Loss
    eval.plot_mean_log_loss(train_losses, valid_losses, epochs)

    # 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the training data
    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    # 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the validation data
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


def multi_class_logistic_regression(learning_rate, epochs, stability):
    print("Multi-Class Logistic Regression:")

    # Load Data
    data = util.load_data("iris.data", True)

    # Shuffle data
    data = util.shuffle_data(data, 0)

    # Unique Classes
    unique_classes = np.unique(data[:, -1])

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training, True)
    x_valid, y_valid = util.get_features_actuals(validation, True)

    means, stds = util.compute_training_mean_std(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = util.z_score_data(x_valid, means, stds)
    
    # Add bias feature to the training data
    x_train_bias = util.add_bias_feature(x_train_zscored)
    
    # Add bias feature to the validation data
    x_valid_bias = util.add_bias_feature(x_valid_zscored)

    eval = Evaluator()
    mc = MultiClassLogisticRegression(
        learning_rate, epochs, stability, unique_classes)

    # 5(b). Train Logistic Regression Model
    mc.train_model(x_train_bias, y_train, x_valid_bias, y_valid)

    # 6. Applies the models to each validation sample to determine the most likely class.
    valid_preds = mc.evaluate_model(x_valid_bias)

    # 7. Computes the validation accuracy
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("\nValidation Accuracy:", valid_accuracy)

    # 8. Creates a confusion matrix for the validation data
    confusion_matrix = eval.compute_confusion_matrix(
        y_valid, valid_preds, unique_classes)
    print("Confusion Matrix:\n", confusion_matrix)


binary_logistic_regression(learning_rate=0.1, epochs=10000, stability=10e-7)
print("\n\n")
multi_class_logistic_regression(
    learning_rate=0.1, epochs=10000, stability=10e-7)