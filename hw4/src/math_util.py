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


def zscore_data(X, means_train=None, stds_train=None):
    '''
    Z-scores the data by subtracting the mean of the current column
    from the current column and dividing by the standard deviation
    of the current column

    :param X: The features data
    :param means_train: The mean of the training data. Optional because
    we want to calculate the mean of the training data to be used when
    we zscore the validation data
    :param stds_train: The standard deviation of the training data.
    Optional becauase we want to calculate the standard deviation 
    of the training data to be used when we zscore the validation
    data

    :return the z-scored data, the mean of each column, and the
    standard deviation of each column
    '''
    means = None
    stds = None

    if means_train is None and stds_train is None:
        means = calculate_mean(X=X, axis=0)
        stds = calculate_std(X=X, axis=0)
    else:
        means = means_train
        stds = stds_train

    zscored = X - means / stds
    return zscored, means, stds


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
