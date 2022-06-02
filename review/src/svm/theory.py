from svm import SVM
import data_util
import numpy as np

data = data_util.load_data("data.csv")
print("Data:\n", data)

X, y = data_util.get_features_actuals(data)
X_bias = data_util.add_bias_feature(X)
print("\nX:\n", X)
print("y:\n", y)
print("X_bias:\n", X_bias)

print("\n======================================================")
print("SVM MODEL")
model = SVM()
model.fit(X_bias, y)
print("Weights:\n", model.weights)
print("Bias:\n", model.bias)
print("Prediction:\n",model.predict(X_bias))

print("\n======================================================")
print("SVM SLIDES IN-CODE FORMULAS")
num_observations, num_features = np.shape(X)
print("\nnum_observations:", num_observations)
print("num_features:", num_features)

y_diagonal_array = np.diag(y)
print("\ny_diagonal_array:\n", y_diagonal_array)

first_term = np.dot(y_diagonal_array, X_bias)
print("\nFirst Term:\n", first_term)
second_term = np.dot(first_term, X_bias.T)
print("Second Term:\n", second_term)
third_term = np.dot(second_term, y_diagonal_array)
print("Third Term:\n", third_term)
full_term = np.linalg.pinv(third_term)
print("Full Term:\n", full_term)

full_term_size = np.shape(full_term)[0]
print("\nFull Term Size:\n", full_term_size)
ones = np.ones((num_observations, 1))
print("Ones:\n", ones)

a = np.dot(full_term, full_term_size)
print("\nalpha ALL:\n", a)

tmp = []
for row in range(len(a)):
    for column in range(len(a[row])):
        if row + column == num_observations - 1:
            tmp.append(round(a[row][column], 2))

a = np.array(tmp)
print("\nalpha OPPOSITE DIAG:\n", a)
w_first_term = np.dot(X_bias.T, y_diagonal_array)
w_second_term = np.dot(w_first_term, a)
print("weights:\n", w_second_term)

gx = np.dot(X_bias, w_second_term)
print("\nPrediction:\n", gx)
