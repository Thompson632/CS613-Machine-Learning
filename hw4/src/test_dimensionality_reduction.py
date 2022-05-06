from hashlib import new
import data_util
import math_util
import numpy as np

def load_wildfaces(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    X, y = data_util.get_features_actuals(data)
    x_stabilized = math_util.stabilize_data(X)

    means, stds = math_util.calculate_mean_std_of_features(x_stabilized)
    x_train_zscored = math_util.z_score_data(x_stabilized, means, stds)

    # TODO: Might need to merge these two back together
    new_data = data_util.merge_arrays(x_train_zscored, y)
    #return new_data
    return x_train_zscored, y


X, y = load_wildfaces("lfw20.csv")
# data= load_wildfaces("lfw20.csv")
# print("Data:", data.shape)

# Covariance of our features
cov_matrix = np.cov(X.T)
print("Cov:", cov_matrix.shape)

# Perform Eigendecomposition
values, vectors = np.linalg.eig(cov_matrix)

# Get highest vectors

# Plot data