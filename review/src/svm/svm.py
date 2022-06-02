import numpy as np


class SVM:
    def __init__(self, lr=0.001, epochs=1000, lambda_value=0.01, log_verbose=False):
        '''
        Constructor that initializes our linear support vector machine.

        :param lr: The learning rate. Default is 0.001
        :param epochs: The number of iterations. Default is 1000
        :param lambda_value: Value used to define a trade off between the weights
        and the calculated cost function. Default is 0.01
        :param log_verbose: Flag to have extra logging for output. Default is false

        :return none
        '''
        self.lr = lr
        self.epochs = epochs
        self.lambda_value = lambda_value
        self.log_verbose = log_verbose

        self.weights = 0
        self.bias = 0

    def fit(self, X, y):
        '''
        Trains a support vector machine using gradient descent.

        :param X: The features data
        :param y: The targets data

        :return none
        '''
        # Converts our target values from 0, 1 to -1, 1
        y = np.where(y <= 0, -1, 1)

        num_features = np.shape(X)[1]

        self.weights = np.zeros(num_features)

        for _ in range(self.epochs):
            for index, xi in enumerate(X):
                dw, db = self.compute_gradients(index, xi, y)

                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db

        if self.log_verbose:
            print("\nLearning Rate:", self.lr)
            print("Epochs:", self.epochs)
            print("Lambda:", self.lambda_value)
            print("Num Weights:\n", len(self.weights))
            print("Weights:\n", self.weights)
            print("Bias:\n", self.bias)

    def compute_gradients(self, index, xi, y):
        '''
        Computes the gradients by first calculating the y_hat for the features data.
        Next, it calculates the partial derivatives of the weights and bias' based
        on whether or not the calculated y_hat is greater than or equal to 1.

        :param index: The index of the given feature sample data
        :param xi: The feature sample data
        :param y: The target data

        :return derivative of weights with respect to loss
        :return derivative of bias with respect to loss
        '''
        y_hat = self.compute_y_hat(xi)
        condition = y[index] * y_hat >= 1

        dw = 0
        db = 0

        if condition:
            dw = 2 * self.lambda_value * self.weights
            db = 0
        else:
            dw = 2 * self.lambda_value * self.weights - np.dot(xi, y[index])
            db = y[index]

        return dw, db

    def predict(self, X):
        '''
        Evaluate the model for the provided data by computing the linear predictions
        based on the calculated weights and bias. Once predicted, we take the sign 
        of the predictions to have our predictions be returned as 1 or -1.

        :param X: The features data

        :return the classifier predictions
        '''
        y_hat = self.compute_y_hat(X)
        return np.sign(y_hat)

    def compute_y_hat(self, X):
        '''
        Computes our y_hat based on the features, weights, and bias

        :param X: The features data

        :return the predicted value
        '''
        return np.dot(X, self.weights) - self.bias