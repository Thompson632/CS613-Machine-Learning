import data_util
import math_util
from evaluator import Evaluator
import numpy as np

def load_spambase(filename):
    data = data_util.load_data(filename)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def load_ctg(filename):
    data = data_util.load_data(filename, 2)
    data = np.delete(data, -2, axis=1)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)
    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid
