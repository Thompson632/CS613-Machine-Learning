import numpy as np


def load_data(filename, rows_to_skip=None):
    '''
    Reads in a file as a numpy ndarray and returns the data. 
    Added a flag for skipping rows if it value is provided.

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
    Gets the training and validation data by first calculating the index of the 
    first 2/3 of data. Once that is found, we set the training data from 0 to the 
    index of the first 2/3 of data and the validation data from the index to the 
    end of the data.

    :param data: The data to be split

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
    X = data[:, 1:] # Every column starting from 1 is our features
    y = data[:, 0] # First column is our target
    return X, y


def create_mean_std_prior_arrays(num_classes, num_features):
    '''
    Helper function that creates mean and standard deviation numpy arrays
    based on the number of classes and features in a particular 
    dataset. This function also creates a classes prior probability
    array.

    :param num_classes: The number of classes in a particular data set
    :param num_features: The number of features in a particular data set

    :return the initial array of means with num_classes as rows
    and num_features as columns
    :return the initial array of standard deviation with num_classes as rows
    and num_features as columns
    :return the initial array of class prior probabilities with a 
    size of num_classes
    '''
    means = np.zeros((num_classes, num_features))
    stds = np.zeros((num_classes, num_features))
    priors = np.zeros(num_classes)

    return means, stds, priors


def merge_arrays(X, y):
    '''
    Helper function that merges two arrays into one by first reshape our 
    one-dimensional y-array into a two dimensional array. Once the two 
    arrays are compatible, we add the reshaped y-array to the end of the 
    two-dimensional X-array.

    :param X: The features data
    :param y: The targets data

    :return the merged features and targets data
    '''
    num_observations = np.shape(X)[0]
    y = y.reshape(num_observations, 1)
    data = np.concatenate((y, X), axis=1)
    return data


def split_on_feature(data, feature_index, threshold):
    '''
    Helper function used in determing the best feature to split our data
    on when building our decision tree. For the left sub-tree, we iterate
    through all the rows of data for the particular feature and we check
    to see if the current row is less than or to the threshold. If it is, 
    we add it to our left sub-tree array. For the right sub-tree, we iterate
    through all the rows of data for the particular feature provided index
    and we check to see if the current row is greater than the threshold. 
    If it is , we add it to our right sub-tree array. 

    :param data: The current data
    :param feature_index: The index of the current feature we are trying
    to split on
    :param threshold: The threshold value to be used for splitting. 
    In our case, this will be the mean of the entire column

    :return the left and right sub-trees for splitting
    '''
    left = np.array(
        [observation for observation in data if observation[feature_index] <= threshold])
    right = np.array(
        [observation for observation in data if observation[feature_index] > threshold])

    return left, right


def get_observation_counts(X=None, y=None, c=None):
    '''
    Helper function that gets the observations present in an array
    where a value in the array meets the criteria. We then get the number
    of observations once we have the observations. Finally, we return the
    observations and the number of observations for the given array and
    criteria.

    :param X: Optional value that if set will search the X array
    for the given criteria
    :param y: The array to be used in determining if the criteria is met
    :param c: The criteria we are checking if our arrays meet

    :return the observations that met the criteria
    :return the number of observation that met the criteria
    '''
    observations = 0
    count = 0

    if X is not None:
        observations = X[y == c]
    else:
        observations = y[y == c]

    count = np.shape(observations)[0]
    return observations, count
