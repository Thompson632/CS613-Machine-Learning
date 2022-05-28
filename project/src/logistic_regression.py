import numpy as np


class LogisticRegression():
    def __init__(self, lr=0.1, epochs=1000, stability_constant=10e-7):
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

        self.weights = 0
        self.bias = 0

    def compute_sigmoid(self, y_hat):
        '''
        Computes the logistic function (or sigmoid) using the
        y_hat computed using gradient descent

        :param y_hat: Gradient descent computed prediction

        :return the logistic function (or sigmoid)
        '''
        return 1 / (1 + np.exp(-y_hat))

    def compute_y_hat(self, X):
        '''
        Computes our y_hat based on the features, weights, and bias

        :param X: The features

        :return the predicted value
        '''
        return np.dot(X, self.weights) + self.bias

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

    def compute_partial_derivatives_of_weights_and_bias(self, X, y, y_hat):
        '''
        Computes the partial derivatives of weights and bias
        with respect to the loss.

        :param X: Features
        :param y: Actual values
        :param y_hat: Predicted values

        :return the derivative of weights
        :return the derivative of bias
        '''
        num_observations = X.shape[0]

        # Derivative of weights with respect to loss
        dw = (1 / num_observations) * np.dot(X.T, (y_hat - y))

        # Derivative of bias with respect to loss
        db = (1 / num_observations) * np.sum((y_hat - y))

        return dw, db

    def fit(self, x_train, y_train, x_valid, y_valid):
        '''
        Trains a logistic regression model using gradient descent
        and sigmoid (or the logistic function). The eventual output will
        be the training and validation data mean log loss.

        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features
        :param y_valid: Validation actuals

        :return training mean log loss
        :return validation mean log loss
        '''
        num_features = np.shape(x_train)[1]

        self.weights = np.zeros(num_features)

        training_losses = []
        validation_losses = []

        for i in range(self.epochs):
            # Compute gradients and train / valid probability
            dw, db, train_probability, valid_probability = self.compute_gradients(
                x_train, y_train, x_valid)

            # Update our weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # Calculate log loss of training data
            training_loss = self.compute_mean_log_loss(
                y_train, train_probability)
            training_losses.append(training_loss)

            # Calculate log loss of validation data
            validation_loss = self.compute_mean_log_loss(
                y_valid, valid_probability)
            validation_losses.append(validation_loss)

        return training_losses, validation_losses

    def compute_gradients(self, x_train, y_train, x_valid):
        '''
        Computes the gradients by first calculating the y_hat for the 
        training and validation data. Next, it compute the sigmoid (or logistic function)
        in order to get the probably values for each training set. Once that is done,
        we are able to calculate the derivates of the weights and bias'.

        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features

        :return derivative of weights with respect to loss
        :return derivative of bias with respect to bias
        :return the training data probability
        :return the validation data probability
        '''
        # Compute y_hat
        train_y_hat = self.compute_y_hat(x_train)
        valid_y_hat = self.compute_y_hat(x_valid)

        # Compute sigmoid (or logistic function)
        train_probability = self.compute_sigmoid(train_y_hat)
        valid_probability = self.compute_sigmoid(valid_y_hat)

        # Get gradients of loss
        dw, db = self.compute_partial_derivatives_of_weights_and_bias(
            x_train, y_train, train_probability)

        return dw, db, train_probability, valid_probability

    def predict(self, X):
        '''
        Evaluates the model for the provided data by computing
        the gradient descent and sigmoid. Once evaluated, we
        want to clean our data by considering a threshold. If 
        the current value in y_hat is greater than 0.5, we set that
        value to 1. Otherwise, we will set it to 0.

        :param X: The training or validation data

        :return the predicted values
        '''
        # Compute y_hat
        y_hat = self.compute_y_hat(X)

        # Compute sigmoid (or logistic function)
        probabilities = self.compute_sigmoid(y_hat).flatten()
        prob_threshold = self.evaluate_with_threshold(probabilities)

        return prob_threshold

    def evaluate_with_threshold(self, y_hat, threshold=0.5):
        '''
        Iterates through the y_hat values sets the values to 1 if there
        value is greater than the threshold (0.5) provided or to 0
        if the value is less than the threshold provided.

        :param y_hat: The predicted values
        :param threshold: The static threshold value

        :return the predictions evaluated with a threshold value
        '''
        y_hat_with_threshold = []

        # Add threshold for computing the classifiers
        # If the current y_hat_val is greater than or equal to the threshold, set to 1. Otherwise, set to 0.
        y_hat_t = [1 if y_hat_val >= threshold else 0 for y_hat_val in y_hat]
        y_hat_with_threshold.append(y_hat_t)

        # Flatten to one dimension
        y_hat_with_threshold = np.array(y_hat_with_threshold).flatten()
        return y_hat_with_threshold
