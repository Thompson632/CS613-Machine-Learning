import numpy as np


def load_data(filename, should_read_as_strings=False):
    '''
    Reads in a file as a numpy ndarray and returns the data. 
    Added a flag to read in the data as strings in the case
    of multi-class classification where our target class is a label
    instead of binary.

    :param filename: The name of the filename to be loaded
    :param should_read_as_strings: Flag to read in the data as 
    strictly strings

    :return the numpy ndarray of data
    '''
    if should_read_as_strings:
        return np.loadtxt(filename, delimiter=',', dtype=str)

    return np.loadtxt(filename, delimiter=',')


def shuffle_data(data, reseed_val):
    '''
    Shuffles and returns the data.

    :param: data: The data that is to be shuffled
    :param reseed_val: The value to be used when reseeding the random
    number generator

    :return data shuffled
    '''
    np.random.seed(reseed_val)
    np.random.shuffle(data)
    return data


def get_train_valid_data(data):
    '''
    Gets the training and validation data by first calculating the index of the first 2/3 of data.
    Once that is found, we set the training data from 0 to the index of the first 2/3 of data
    and the validation data from the index to the end of the data.

    :param data: Data to be searched

    :return training, validation data
    '''
    # Get the index value of the 2/3 data
    training_index = round(len(data) * (2/3))

    # Set our training and validation data sets based on the 2/3 index
    training = data[:training_index]  # begin-training_index
    validation = data[training_index:]  # training_index-end

    return training, validation


def get_features_actuals(data, should_convert_to_float=False):
    '''
    Helper function that retrieves the features and
    actual values from the data. 

    :param data: The data we are using
    :param should_convert_to_float: If set to true, will convert
    the string data to float

    :return the features data as X
    :return the actual data as y
    '''
    X = data[:, : -1]

    if should_convert_to_float:
        X = X.astype(float)

    y = data[:, -1]
    return X, y


def add_bias_feature(X):
    '''
    Adds a bias feature of ones to the first column.
    Doing this after we z-score due to divide by 0
    when calculating the mean/std for each feature.

    :param X: The features data

    :return the features data with a bias feature of
    ones prepended to the original data
    '''
    num_observations = np.shape(X)[0]
    ones = np.ones((num_observations, 1))
    X = np.concatenate((ones, X), axis=1)
    return X


def convert_data_to_binary(X, y, current_class):
    '''
    Helper method to manipulate the data by arranging the data of the current class
    we are modeling as the first subset of data in our array and the second set of data
    that is not equal to the class we are modeling on the bottom. Once that is completed, 
    we do the smae thing for the actual values but instead set the class we are modeling
    from its label to 1 and 0 for the data we are not modeling. 

    :param X: The features data
    :param y: The actuals data
    :param current_class: The current class we are modeling

    :return the sorted feature data by the class we are modeling
    :return the binary actual data sorted by the class we are modeling
    '''
    x_one = X[y == current_class]
    x_zero = X[y != current_class]
    x_sorted = np.vstack((x_one, x_zero))

    y_one = np.ones(np.shape(x_one)[0])
    y_zero = np.zeros(np.shape(x_zero)[0])
    y_sorted_binary = np.hstack((y_one, y_zero))

    return x_sorted, y_sorted_binary


def compute_training_mean_std(X):
    '''
    Computes the mean and standard deviation for each column.

    :param X: The feature data

    :return the vector of means and stds
    '''
    num_features = np.shape(X)[1]

    means = []
    stds = []

    for i in range(num_features):
        current_feature = X[:, i]

        mean = np.mean(current_feature)
        # Need ddof for one less degree of freedom
        std = np.std(current_feature, ddof=1)

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