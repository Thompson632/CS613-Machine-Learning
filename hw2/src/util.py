import numpy as np


def load_data(filename):
    '''
    Reads in a csv file ignoring the first rows (headers)
    and only returning the columns as specified in the parameter.

    :param filename: The file name specified at runtime

    :return numpy ndarray of data
    '''
    return np.loadtxt(filename, delimiter=',')


def load_multi_class_data(filename):
    '''
    Loads the multi-class data by taking the columns as data and classes.

    :param filename: The file name specified at runtime
    :param data_columns: The data columns to load
    :param class_columns: The class columns to load

    :return the numpy ndarray of data
    '''
    return np.loadtxt(filename, delimiter=',', dtype=str)


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


def get_features_actuals(data):
    '''
    Helper function that retrieves the features and
    actual values from the data. Currently it retrieves
    the features as the last two columns (Temp of Water
    and Length of Fish) in the ndarray and the first column
    (Age) as the actuals.
    and the

    :param data: The data we are using

    :return the features data as X
    :return the actual data as y
    '''
    X = data[:, : -1]
    y = data[:, -1]
    return X, y


def get_multi_class_features_actuals(data, class_one, class_two):
    '''
    Helper function that retrieves the features and actual values from the data
    based on the classes we are used for multi-class classification.

    :param data: The data we are using
    :param class_one: The current class
    :param class_two: The class we are using to compare

    :return the features data as X
    :return the actual data as y
    '''
    x_list = []
    y_list = []

    for observation in data:
        X, y = convert_multi_class_observation(
            observation, class_one, class_two)
        x_list.append(X)
        y_list.append(y)

    x_list = np.array(x_list)
    y_list = np.array(y_list)
    return x_list, y_list


def convert_multi_class_observation(observation, class_one, class_two):
    '''
    Helper method converts our features observation from a string array to an
    array of floats. Also, converts our y-values from their string representation
    to a 1 or zero. If the current y-observation is equal to class one, that means
    we are evaluating for class one and this should be set to 1 (or true). If the
    current y-observation is not equal to class one but it is equal to class two,
    that means we are using class two to compare against class one.

    :param observation: The current observation in the data
    :param class_one: The current class
    :param class_two: The class we are using to compare

    :return the features data as Xi
    :return the binary value as yi
    '''
    # Get the current observation features
    X = observation[: -1]
    # Get the current observation actual
    y = observation[-1]

    # Convert our X data from a string array to a float array
    X = [float(i) for i in X]

    y_int = 0
    if y == class_one:
        y_int = 1
    elif y == class_two:
        y_int = 0

    return X, y_int


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
