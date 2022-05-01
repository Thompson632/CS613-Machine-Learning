import numpy as np


def load_data(filename, rows_to_skip=None):
    '''
    Reads in a file as a numpy ndarray and returns the data. 
    Added a flag to read in the data as strings in the case
    of multi-class classification where our target class is a label
    instead of binary.

    :param filename: The name of the filename to be loaded
    :param rows_to_skip: Optional parameter that if set will
    skip the specified rows in the file            

    :return the numpy ndarray of data
    '''
    if rows_to_skip is not None:
        return np.loadtxt(filename, delimiter=',', skiprows=rows_to_skip)

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


def get_features_actuals(data):
    '''
    Helper function that retrieves the features and
    actual values from the data. 

    :param data: The data we are using

    :return the features data as X
    :return the actual data as y
    '''
    X = data[:, : -1]
    y = data[:, -1]
    return X, y


def create_mean_var_prior_arrays(num_classes, num_features):
    '''
    Helper function that creates mean and variance numpy arrays
    based on the number of classes and features in a particular 
    dataset. This function also creates a classes prior probability
    array.

    :param num_classes: The number of classes in a particular data set
    :param num_features: The number of features in a particular data set

    :return the initial array of means with num_classes as rows
    and num_features as columns
    :return the initial array of variances with num_classes as rows
    and num_features as columns
    :return the initial array of class prior probabilities with a 
    size of num_classes
    '''
    means = np.zeros((num_classes, num_features))
    variances = np.zeros((num_classes, num_features))
    priors = np.zeros(num_classes)

    return means, variances, priors

def split_on_feature(data, feature_index, threshold):
    left = np.array([sample for sample in data if sample[feature_index] <= threshold])
    right = np.array([sample for sample in data if sample[feature_index] > threshold])
    return left, right
    