import numpy as np


def calculate_mean(X, axis=None):
    '''
    Calculates the mean based on the feature data.

    :param X: The current feature (or column) to be averaged
    :param axis: Optional parameter that if set will
    compute the mean of the specified axis

    :return the average of the feature data
    '''
    if axis is None:
        return np.mean(X)

    return np.mean(X, axis=axis)


def calculate_std(X, axis=None):
    '''
    Calculates the standard deviation based on the feature data.

    :param X: The current feature (or column) to be standardized
    :param axis: Optional parameter that if set will
    compute the standard deviation of the specified axis

    :return the standard deviation of the feature data    
    '''
    if axis is None:
        return np.std(X, ddof=1)

    return np.std(X, axis=axis, ddof=1)

def calculate_feature_mean_std(X):
    '''
    Calculates the mean and standard deviation for each feature.
    
    :param X: The feature data
    
    :return the vector of means and stds
    '''
    num_features = np.shape(X)[1]

    means = []
    stds = []

    for i in range(num_features):
        current_feature = X[:, i]

        mean = calculate_mean(current_feature)
        std = calculate_std(current_feature)

        means.append(mean)
        stds.append(std)

    return means, stds


def z_score_data(X, means, stds):
    '''
    Z-scores the data by subtracting the mean of the current column
    from the current column and dividing by the standard deviation
    of the current column.
    
    :param X: The features data
    :param means: The means vector for the training data
    :param stds: The standard deviation vector for the training data
    
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


def un_zscore_data(X, means, stds):
    '''
    Un-zscores the data by multiplying the standard devation of the 
    features and adding the mean of the features.

    :param X: The features data
    :param means: The means vector for the training data
    :param stds: The standard deviation vector for the training data

    :return the un-z-scored data
    '''
    unzscored = X * stds + means
    return unzscored


def stabilize_data(X):
    '''
    Helper function that stabilizes the pixel data by dividing each columns
    values by 255 for numeric stability purposes.

    :param X: The features data

    :return the stabilized data
    '''
    stabilized = np.divide(X, 255)
    return stabilized


def unstablize_data(X):
    '''
    Helper function that unstabilizes the pixel data by multiplying each columns
    values by 255.

    :param X: The features data

    :return the stabilized data
    '''
    unstabilized = np.multiply(X, 255)
    return unstabilized
