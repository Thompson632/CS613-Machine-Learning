import numpy as np


def compute_mean(feature):
    '''
    Computes the mean based on the feature data.

    :param feature: The current feature (or column) to be averaged

    :return the average of the feature data
    '''
    return np.mean(feature)


def compute_std(feature):
    '''
    Computes the standard deviation based on the feature data.

    :param feature: the current feature (or column) to be standardized

    :return the standard deviation of the feature data    
    '''
    return np.std(feature, ddof=1)


def compute_training_mean_std_by_feature(X):
    '''
    Computes the mean and standard deviation for each feature.

    :param X: The feature data

    :return the vector of means and stds
    '''
    num_features = np.shape(X)[1]

    means = []
    stds = []

    for i in range(num_features):
        current_feature = X[:, i]

        mean = compute_mean(current_feature)
        std = compute_std(current_feature)

        means.append(mean)
        stds.append(std)

    return means, stds


def z_score_data(X, means, stds):
    '''
    Z-scores the data by subtracting the mean of the current column
    from the current column and dividing by the standard deviation
    of the current column

    :param X: The features data
    :param means: The means vector for the training data
    :param: stds: The standard deviation vector for the training data

    :return the z-scored data
    '''
    num_features = np.shape(X)[1]

    for i in range(num_features):
        current_feature = X[:, i]

        numerator = current_feature - means[i]
        denominator = stds[i]

        zscore = numerator / denominator

        X[:, i] = zscore

    return X


def compute_mean_std_of_features_list(list_of_feature_tuples):
    '''
    Computes the mean and standard deviation given 
    a list of feature tuples and their observation data.

    :param list_of_features: The list of features to compute
    the mean and standard deviation 

    :return the list of mean and standard deviation tuples for each
    feature
    '''
    list_of_features_mean_std_tuples = [(compute_mean(feature_tuple), compute_std(
        feature_tuple), len(feature_tuple)) for feature_tuple in zip(*list_of_feature_tuples)]
    return list_of_features_mean_std_tuples
