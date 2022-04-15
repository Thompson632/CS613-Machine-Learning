import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    '''
    Reads in a csv file ignoring the first rows (headers)
    and only returning the columns as specified in the parameter.

    :param filename: The file name specified at runtime
    :param columns: The columns of data to retrieve

    :return numpy ndarray
    '''
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


def get_features_actuals(data):
    '''
    Helper function that retrieves the features and
    actual values from the data. Currently it retrieves
    the features as the last two columns (Temp of Water
    and Length of Fish) in the ndarray and the first column
    (Age) as the actuals.
    and the

    :param data: Data to be searched

    :return the features (X) and actuals (y)
    '''
    X = data[:, : -1]
    y = data[:, -1]
    return X, y


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


def add_bias_feature(X):
    '''
    Helper function remove duplicate code by adding a bias feature.

    :param X: Features

    :return X: Features with a bias
    '''
    N = np.shape(X)[0]

    ones = np.ones((N, 1))
    X = np.concatenate((ones, X), axis=1)
    return X


def compute_training_mean_std(X):
    '''
    Computes the mean and standard deviation for each column.

    :param X: The feature data

    :return the vector of means and stds
    '''
    N = np.shape(X)[1]

    means = []
    stds = []

    for i in range(N):
        current_feature = X[:, i]

        mean = np.mean(current_feature)
        # Need ddof for one less degree of freedom
        std = np.std(current_feature, ddof=1)

        means.append(mean)
        stds.append(std)

    return means, stds


def z_score_data(X, means, stds):
    '''
    Z-scores the data by taking the subtracting each column by its
    mean value and dividing by its standard deviation.

    :param X: The features data
    :param means: The means vector for the training data
    :param: stds: The standard deviation vector for the training data

    :return the z-scored data
    '''
    N = np.shape(X)[1]

    for i in range(N):
        current_feature = X[:, i]

        numerator = current_feature - means[i]
        denominator = stds[i]
        
        zscore = numerator / denominator

        X[:, i] = zscore

    return X

def plot_mean_log_loss(type, mean_log_loss, epochs):
    '''
    Plots the mean log loss as a function of the epoch
    
    :param type: Type of data
    :param mean_log_loss: Mean of the log loss
    :param epochs: Number of iterations
    
    :return none
    '''
    fig = plt.figure(figsize = (8, 6))
    plt.plot([i for i in range(epochs)], mean_log_loss, 'r-')
    plt.title(type)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Log Loss')
    plt.show()
