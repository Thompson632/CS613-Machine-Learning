import util
import math_util
from evaluator import Evaluator
from naive_bayes import NaiveBayes
import numpy as np


def naive_bayes():
    print("Naive Bayes Classifier:")

    # Load data
    data = util.load_data("spambase.data", False)
    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training, False)
    x_valid, y_valid = util.get_features_actuals(validation, False)

    # Get mean and std of training data
    means, stds = math_util.compute_training_mean_std_by_feature(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = math_util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    # Add bias feature to the training data
    x_train_bias = util.add_bias_feature(x_train_zscored)

    # Add bias feature to the validation data
    x_valid_bias = util.add_bias_feature(x_valid_zscored)

    # 5. Divides the training data into two groups: Spam samples, Non-Spam samples
    split_data_dict = util.split_data_by_class(x_train_bias, y_train)

    # Instantiate NaiveBayes
    nb = NaiveBayes(split_data_dict)

    # 6. Create Normal models for each feature for each class
    nb.create_normal_models_for_each_class(False)

    # 7. Classifies each validation sample using these models and chooses the class label
    # based on which class probability is higher
    valid_preds = nb.evaluate_models(x_valid_bias, y_valid)

    # 8. Computes the following statistics using the validation data results:
    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    # (a) Precision
    print("\nValidation Precision:", valid_precision)
    # (b) Recall
    print("Validation Recall:", valid_recall)
    # (c) F-Measure
    print("Validation F-Measure:", valid_f_measure)
    # (d) Accuracy
    print("Validation Accuracy:", valid_accuracy)


naive_bayes()
print("\n\n")