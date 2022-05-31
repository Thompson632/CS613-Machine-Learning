import data_util
import math_util
from naive_bayes import NaiveBayes

import numpy as np

data = data_util.load_data("theory.csv")
print("Data:\n", data)

x_train, y_train = data_util.get_features_actuals(data)

print("\nX:\n", x_train)
print("\ny:\n", y_train)

means, stds = math_util.calculate_mean_std_of_features(x_train)
x_train_zscored = math_util.z_score_data(x_train, means, stds)

means, stds = math_util.calculate_mean_std_of_features(x_train)
print("\nMeans:", means)
print("\nStd:", stds)
x_train_zscored = math_util.z_score_data(x_train, means, stds)
print("\nTraining Z:\n", x_train_zscored)

x_valid = np.array([[2.42], [4.56]])
x_valid_zscored = math_util.z_score_data(x_valid, means, stds)
print("\nValidation Z:\n", x_valid_zscored)

x_valid = np.array([[0.23], [0.4]])

nb = NaiveBayes(stability_constant=1e-4, log_verbose=True)
nb.fit(x_train, y_train)

valid_preds = nb.predict(x_valid_zscored)
print("\nPredictions: ", valid_preds)