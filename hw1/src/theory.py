import numpy as np

from closed_form import LinearRegressionClosedForm
from evaluator import Evaluator

xg = [-2, -5, -3, 0, -8, -2, 1, 5, -1, 6]

x = [[1, -2], [1, -5], [1, -3], [1, 0], [1, -8],
     [1, -2], [1, 1], [1, 5], [1, -1], [1, 6]]


y = [1, -4, 1, 3, 11, 5, 0, -1, -3, 1]

x = np.array(x)
y = np.array([y]).T


model = LinearRegressionClosedForm()
model.fit(x, y)

y_hat = model.predict(x)
print("y_hat:\n", y_hat)

print("y:\n", y)

eval = Evaluator()
print("")
print("RMSE:", eval.compute_rmse(y_hat, y))
print("MAPE:", eval.compute_mape(y_hat, y))
print("SMAPE:", eval.compute_smape(y_hat, y))
