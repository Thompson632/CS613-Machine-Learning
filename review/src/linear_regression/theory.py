import numpy as np
import data_util

from closed_form import LinearRegressionClosedForm
from evaluator import Evaluator

# xg = [-2, -5, -3, 0, -8, -2, 1, 5, -1, 6]

# x = [[1, -2], [1, -5], [1, -3], [1, 0], [1, -8],
#      [1, -2], [1, 1], [1, 5], [1, -1], [1, 6]]


# y = [1, -4, 1, 3, 11, 5, 0, -1, -3, 1]

# x = np.array(x)
# y = np.array([y]).T

data = data_util.load_data("theory.csv")
print("Data:\n", data)

X = data[:, 0]

num_observations = np.shape(X)[0]
X = X.reshape(num_observations, 1)
X = data_util.add_bias_feature(X)
y = data[:, -1]

print("\nX:\n", X)
print("\ny:\n", y)

model = LinearRegressionClosedForm(print_weights=True)
model.fit(X, y)

y_hat = model.predict(X)
print("\ny_hat:\n", y_hat)

print("y:\n", y)

eval = Evaluator()
print("")
print("RMSE:", eval.compute_rmse(y_hat, y))
print("MAPE:", eval.compute_mape(y_hat, y))
print("SMAPE:", eval.compute_smape(y_hat, y))
