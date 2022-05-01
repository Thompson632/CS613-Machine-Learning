import data_util
import math_util
from naive_bayes import NaiveBayes
import numpy as np

data = data_util.load_data("theory.csv")
#data = data_util.shuffle_data(data, 0)
print("Data:\n", data)

#training, validation = data_util.get_train_valid_data(data)
x_train, y_train = data_util.get_features_actuals(data)
#x_valid, y_valid = data_util.get_features_actuals(validation)

print("\nX:\n", x_train)
print("\ny:\n", y_train)

means, stds = math_util.calculate_mean_std_of_features(x_train)
x_train_zscored = math_util.z_score_data(x_train, means, stds)
#x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

means, stds = math_util.calculate_mean_std_of_features(x_train)
print("\nMeans:", means)
print("\nStd:", stds)
x_train_zscored = math_util.z_score_data(x_train, means, stds)
print("\nTraining Z:\n", x_train_zscored)
#x_valid_zscored = math_util.z_score_data(x_valid, means, stds)
#print("Validation Z:", x_valid_zscored)


nb = NaiveBayes(stability_constant=1e-4)
nb.train_model(x_train, y_train)
# valid_preds = nb.evaluate_model(x_valid)

# eval = Evaluator()
# valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
#     y_valid, valid_preds)

# print("Validation Precision:", valid_precision)
# print("Validation Recall:", valid_recall)
# print("Validation F-Measure:", valid_f_measure)
# print("Validation Accuracy:", valid_accuracy)