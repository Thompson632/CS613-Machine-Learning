import numpy as np


class LogisticRegression():

    def __init__(self, lr, epochs, stability_constant):
        '''
        Constructor that takes in a learning rate value
        and the number of epochs to be ran in our 
        training model

        :param lr: The learning rate
        :param epochs: The number of iterations
        :param stability_constant: The constant to stabilize
        our logs to ensure we do not get log(0) = infinity

        :return none
        '''
        self.lr = lr
        self.epochs = epochs
        self.stability_constant = stability_constant

    def compute_sigmold(self, y_hat):
        '''
        Computes the logistic function (or sigmold) using the
        y_hat computed using gradient descent

        :param y_hat: Gradient descent computed prediction

        :return the logistic function (or sigmold)
        '''
        return 1 / (1 + np.exp(-y_hat))

    def compute_y_hat(self, X, weights, bias):
        '''
        Computes our y_hat based on the features, weights, and bias

        :param X: The features
        :param weights: The learned weights
        :bias: The learned bias

        :return the predicted value
        '''
        return np.dot(X, weights) + bias

    def compute_mean_log_loss(self, y, y_hat):
        '''
        Computes the mean log loss of the data

        :param y: Actual value
        :param y_hat: Predicted value

        :return the mean log loss of the data
        '''
        first_term = y * np.log(y_hat + self.stability_constant)
        one_minus_y_hat = 1 - y_hat + self.stability_constant

        second_term = (1 - y) * np.log(one_minus_y_hat)
        third_term = -(first_term + second_term)
        return third_term.mean()

    def compute_derivatives_of_weights_and_bias(self, X, y, y_hat):
        '''
        Computes the derivatives of weights and bias
        with respect to the loss.

        :param X: Features
        :param y: Actual values
        :param y_hat: Predicted values

        :return dw: Derivative of weights
        :return db: Derivative of bias
        '''
        num_observations = X.shape[0]

        # Derivative of weights with respect to loss
        dw = (1 / num_observations) * np.dot(X.T, (y_hat - y))

        # Derivative of bias with respect to loss
        db = (1 / num_observations) * np.sum((y_hat - y))

        return dw, db

    def train_model(self, x_train, y_train, x_valid, y_valid):
        '''
        Trains a logistic regression model using gradient descent
        and sigmold (or the logistic function). The eventual output will
        be the learned weights and bias as well as the probability of the training 
        and validation data.

        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features
        :param y_valid: Validation actuals

        :return training mean log loss
        :return validation mean log loss
        :return learned weights
        :return learned bias
        '''
        num_features = np.shape(x_train)[1]

        weights = np.zeros(num_features)
        bias = 0

        training_losses = []
        validation_losses = []

        for i in range(self.epochs):
            # Compute gradients and train / valid probability
            dw, db, train_probability, valid_probability = self.compute_gradients(
                x_train, y_train, x_valid, y_valid, weights, bias)

            # Update our weights and bias
            weights = weights - self.lr * dw
            bias = bias - self.lr * db

            # Calculate log loss of training datax
            training_loss = self.compute_mean_log_loss(
                y_train, train_probability)
            training_losses.append(training_loss)

            # Calculate log loss of validation data
            validation_loss = self.compute_mean_log_loss(
                y_valid, valid_probability)
            validation_losses.append(validation_loss)

        return training_losses, validation_losses, weights, bias

    def compute_gradients(self, x_train, y_train, x_valid, y_valid, weights, bias):
        # Compute y_hat
        train_y_hat = self.compute_y_hat(x_train, weights, bias)
        valid_y_hat = self.compute_y_hat(x_valid, weights, bias)

        # Compute sigmold (or logistic function)
        train_probability = self.compute_sigmold(train_y_hat)
        valid_probability = self.compute_sigmold(valid_y_hat)

        # Get gradients of loss
        dw, db = self.compute_derivatives_of_weights_and_bias(
            x_train, y_train, train_probability)

        return dw, db, train_probability, valid_probability

    def evaluate_model(self, X, weights, bias):
        '''
        Evaluates the model for the provided data by computing
        the gradient descent and sigmold. Once evaluated, we
        want to clean our data by considering a threshold. If 
        the current value in y_hat is greater than 0.5, we set that
        value to 1. Otherwise, we will set it to 0.

        :param X: The training or validation data
        :param weights: The learned weights
        :param bias: The learned bias

        :return the predicted values evaluated with a threshold
        for the training or validation data
        '''
        # Compute y_hat
        y_hat = self.compute_y_hat(X, weights, bias)

        # Compute sigmold (or logistic function)
        probability = self.compute_sigmold(y_hat)
        return probability.flatten()
