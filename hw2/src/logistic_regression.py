import numpy as np


class LogisticRegression():
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        
    def compute_sigmold(self, y_hat):
        '''
        Computes the logistic function (or sigmold) using the
        y_hat computed using gradient descent

        :param y_hat: Gradient descent computed prediction

        :return the logistic function (or sigmold)
        '''
        return 1 / (1 + np.exp(-y_hat))

    def compute_gradient_descent(self, X, weights, bias):
        '''
        Computes y_hat using gradient descent
        '''
        return np.dot(X, weights) + bias

    def compute_mean_log_loss(self, y, y_hat):
        '''
        Computes the mean log loss of the data

        :param y: Actual value
        :param y_hat: Predicted value

        :return the mean log loss of the data
        '''
        first_term = y * np.log(y_hat)
        second_term = (1 - y) * np.log(1 - y_hat)
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

    def train_logistic_regression_model(self, x_train, y_train, x_valid, y_valid):
        '''
        Traings a logistic regression model using gradient descent
        and sigmold (or the logistic function).
        
        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features
        :param y_valid: Validation actuals
        
        :return training predictions
        :return training mean log loss
        :return validation predictions
        :return validation mean log loss
        '''
        num_features = np.shape(x_train)[1]
        
        weights = np.zeros(num_features)
        bias = 0
        
        training_preds = []
        training_losses = []
        
        validation_preds = []
        validation_losses = []
        
        for i in range(self.epochs):
            # Compute Gradient Descent
            train_gd = self.compute_gradient_descent(x_train, weights, bias)
            valid_gd = self.compute_gradient_descent(x_valid, weights, bias)

            # Compute sigmold (or logistic function) using gradient descent
            train_y_hat = self.compute_sigmold(train_gd)
            training_preds.append(train_y_hat)
            
            # Compute sigmold (or logistic function) using gradient descent
            valid_y_hat = self.compute_sigmold(valid_gd)
            validation_preds.append(valid_y_hat)
            
            # Get gradients of loss
            dw, db = self.compute_derivatives_of_weights_and_bias(x_train, y_train, train_y_hat)

            # Update our weights and bias
            weights = weights - self.lr * dw
            bias = bias - self.lr * db

            # Calculate log loss of training data
            training_loss = self.compute_mean_log_loss(y_train, train_y_hat)
            training_losses.append(training_loss)
            
            # Calculate log loss of validation data
            validation_loss = self.compute_mean_log_loss(y_valid, valid_y_hat)
            validation_losses.append(validation_loss)
            
        return training_preds, training_losses, validation_preds, validation_losses