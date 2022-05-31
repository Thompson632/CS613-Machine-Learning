import numpy as np
from closed_form import LinearRegressionClosedForm
import math


class LinearRegressionLocallyWeighted:
    def __init__(self, k=1, print_weights=False):
        '''
        Constructor that initializes our k value for computing the distances,
        X and y training data, and our LinearRegressionClosedForm model.

        :param k: Optional parameter k
        :param print_weights: Flag to print the calculated weights

        :return none
        '''
        self.k = k
        self.print_weights = print_weights

        self.X_train = None
        self.y_train = None

        self.model = LinearRegressionClosedForm()

    def fit(self, X, y):
        '''
        Trains our locally weighted linear regression model by initializing
        our training features and targets data to be used when predicting.

        :param X: The features data
        :param y: The target data

        :return none
        '''
        self.X_train = X
        self.num_train_observations = np.shape(self.X_train)[0]

        y = y.reshape(self.num_train_observations, 1)
        y_mat = np.mat(y)

        self.y_train = y_mat

    def predict(self, X):
        '''
        Evaluates our model by iterating through all of the validation
        observations, computing the local weights, and computes
        y_hat based off the locally computed weights.

        :param X: The features data

        :return the predictions
        '''
        num_valid_observations = np.shape(X)[0]

        y_hat_array = np.zeros(num_valid_observations)

        for i in range(num_valid_observations):
            validation_sample = X[i]

            weights = self.compute_local_weights(
                self.X_train, validation_sample, self.y_train, self.k, self.num_train_observations)
            
            if self.print_weights:
                print("Validation Sample", i, "Weights:\n", weights)

            y_hat = self.model.compute_y_hat(validation_sample, weights)
            y_hat_array[i] = y_hat

        return y_hat_array

    def compute_local_weights(self, X, x, Y, k, N):
        '''
        Computes the local weights for the current observation. 
        We first need to compute the local diaganal weights matrix. 
        Once we have that computed, we calculate the weights matrix
        for the current observation and return the result.

        :param X: Training Features
        :param x: Validation sample
        :param Y: Training Actuals
        :param k: Value used in gaussian similarity metric
        :param N: Number of observations in the training dataset

        return weights matrix for the current observation
        '''
        diagonal_weights_mat = self.compute_local_diagonal_weights(
            X, x, k, N)

        xt_d = np.dot(X.T, diagonal_weights_mat)
        xt_dx = np.dot(xt_d, X)
        xt_dx_inv = np.linalg.pinv(xt_dx)
        xt_dy = np.dot(xt_d, Y)
        return np.dot(xt_dx_inv, xt_dy)

    def compute_local_diagonal_weights(self, X, x, k, N):
        '''
        Computes the local diagonal weights matrix by iterating
        over the number of observations, computing the gaussian similarity metric, 
        and adding the result to our diagonal matrix.

        :param X: The training data 
        :param x: The validation sample
        :param k: Value used in gaussian similarity metric
        :param N: Number of observations in the training dataset

        :return local diagonal weights matrix
        '''
        diagonal_array = np.eye(N)
        diagonal_weights_mat = np.mat(diagonal_array)

        k_squared = (k**2)

        for i in range(N):
            diagonal_weights_mat[i, i] = self.compute_gaussian_similarity_metric(
                X[i], x, k_squared)

        return diagonal_weights_mat

    def compute_gaussian_similarity_metric(self, xi, x, k_squared):
        '''
        Computes the gaussian similarity metric by calculating the distance
        using the manhattan distance formula for the observations.

        :param xi: Current observation in our diagonal weights loop
        :param x: Current validation sample we originally started with
        :param k_squared: Use in formula

        :return gaussian similarity metric computation
        '''
        distance = self.compute_manhattan_distance(xi, x)
        return math.e ** -(distance**2 / k_squared)

    def compute_manhattan_distance(self, xi, x):
        '''
        Computes the distance between two observations

        :param xi: Current observation in our diagonal weights loop
        :param x: Current observation we originally started with

        :return distance between observations
        '''
        return sum(abs(val1 - val2) for val1, val2 in zip(xi, x))