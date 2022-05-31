import numpy as np
import pandas as pd


def load_data(filename, columns=None):
    '''
    Reads in a csv file ignoring the first rows (headers)
    and only returning the columns as specified in the parameter.

    :param filename: The file name specified at runtime
    :param columns: The columns of data to retrieve

    :return numpy ndarray
    '''
    if columns is not None:
        return np.loadtxt(filename, delimiter=',', skiprows=1, dtype='int', usecols=columns)

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
    Gets the training and validation data by first calculating the index of the 
    first 2/3 of data. Once that is found, we set the training data from 0 to the 
    index of the first 2/3 of data and the validation data from the index to the 
    end of the data.

    :param data: The data to be split

    :return training, validation data
    '''
    training_index = round(len(data) * (2/3))

    training = data[:training_index]
    validation = data[training_index:]

    return training, validation


def get_features_actuals(data):
    '''
    Helper function that retrieves the features and
    actual values from the data. 

    :param data: The data we are using

    :return the features data as X
    :return the actual data as y
    '''
    X = data[:, -2:]  # Every Column Except First
    y = data[:, 0]  # First Column is our Target
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


def convert_list_to_numpy_array(list):
    '''
    Helper function that converts the provided list of data
    into a numpy array. We need to do this because we are 
    provided a list of lists of arrays. We want the array
    data aka our training or validation data.

    :param list: The list to be iterated over

    :return the numpy array of just data for the current fold
    '''
    tmp_list = []

    for list_of_arr in list:
        for arr in list_of_arr:
            tmp_list.append(arr)

    return np.asarray(tmp_list)