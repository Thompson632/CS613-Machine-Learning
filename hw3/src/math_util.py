import numpy as np


def compute_mean(X, axis=None):
    '''
    Computes the mean based on the feature data.

    :param X: The current feature (or column) to be averaged
    :param axis: Optional parameter that if set will
    compute the mean of the specified axis

    :return the average of the feature data
    '''
    if axis == None:
        return np.mean(X)
    else:
        return np.mean(X, axis=axis)


def compute_std(X, axis=None):
    '''
    Computes the standard deviation based on the feature data.

    :param X: The current feature (or column) to be standardized
    :param axis: Optional parameter that if set will
    compute the standard deviation of the specified axis

    :return the standard deviation of the feature data    
    '''
    if axis == None:
        return np.std(X, ddof=1)
    else:
        return np.std(X, axis=axis, ddof=1)
    
def compute_variance(X, axis=None):
    '''
    Computes the variance based on the feature data.
    
    :param X: The current feature (or column) to be standardized
    :param axis: Optional parameter that if set will
    compute the variance of the specified axis

    :return the variance of the feature data  
    '''
    if axis == None:
        return np.var(X, ddof=1)
    else:
        return np.var(X, ddof=1, axis=axis)


def compute_initial_classifier_probability(classifier_observations, total_observations):
    '''
    Computes the initial classifier probably by dividing the observations for
    this particular classifier by the total observations in the data set:

    :param classifier_observations: The number of observations in
    this classifier
    :param total_observations: The number of observations in 
    this dataset

    :return the initial classifier_probability
    '''
    return classifier_observations / total_observations


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

def get_indices_of_max_values(arr, axis=None):
    '''
    Helper function to get the indices of maximum values of the array.
    
    :param arr: The array to be searched
    :param axis: Optional parameter that if set will
    get the indices of the maximum values along the axis

    :return the array with the indices
    '''
    if axis == None:
        return np.argmax(arr)
    else:
        return np.argmax(arr, axis=axis)