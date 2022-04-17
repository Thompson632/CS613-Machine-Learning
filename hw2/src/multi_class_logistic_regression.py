import numpy as np
from logistic_regression import LogisticRegression


class MultiClassLogisticRegression:
    def __init__(self, lr, epochs, stability_constant, unique_classes):
        '''
        Constructor that takes in a learning rate value,
        the number of epochs to be ran in our training model,
        stability constant for our log loss function, and
        the unique classes for this model.

        :param lr: The learning rate
        :param epochs: The number of iterations
        :param stability_constant: The constant to stabilize
        our logs to ensure we do not get log(0) = infinity
        :param unique_classes: The unique number of classes to
        be used for this multi-class logistic regression model

        :return none
        '''
        self.lr = lr
        self.epochs = epochs
        self.stability_constant = stability_constant

        self.unique_classes = unique_classes
        self.class_models = []

        self.class_training_losses = []
        self.class_validation_losses = []

    def train_model(self, x_train, y_train, x_valid, y_valid):
        '''
        Trains a logistic regression model using gradient descent
        and sigmold (or the logistic function) for each of the classes.
        This function will generate the following number of binary
        classifiers: K(K-1)/2

        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features
        :param y_valid: Validation actuals

        :return none
        '''
        for c in self.unique_classes:
            x_train_sorted, y_train_sorted_binary = self.convert_data_to_binary(
                x_train, y_train, c)
            x_valid_sorted, y_valid_sorted_binary = self.convert_data_to_binary(
                x_valid, y_valid, c)

            model = LogisticRegression(
                self.lr, self.epochs, self.stability_constant)

            train_losses, valid_losses = model.train_model(
                x_train_sorted, y_train_sorted_binary, x_valid_sorted, y_valid_sorted_binary)

            self.class_training_losses.append([c, train_losses])
            self.class_validation_losses.append([c, valid_losses])

            self.class_models.append([c, model])

    def convert_data_to_binary(self, X, y, current_class):
        '''
        Helper method to manipulate the data by arranging the data of the current class
        we are modeling as the first subset of data in our array and the second set of data
        that is not equal to the class we are modeling on the bottom. Once that is completed, 
        we do the smae thing for the actual values but instead set the class we are modeling
        from its label to 1 and 0 for the data we are not modeling. 

        :param X: The features data
        :param y: The actuals data
        :param current_class: The current class we are modeling
        
        :return the sorted feature data by the class we are modeling
        :return the binary actual data sorted by the class we are modeling
        '''
        x_one = X[y == current_class]
        x_zero = X[y != current_class]
        x_sorted = np.vstack((x_one, x_zero))

        y_one = np.ones(np.shape(x_one)[0])
        y_zero = np.zeros(np.shape(x_zero)[0])
        y_sorted_binary = np.hstack((y_one, y_zero))

        return x_sorted, y_sorted_binary

    def evaluate_model(self, X):
        '''
        Evaluates the data on for each of the learned models. Once computed,
        we compute the largest mean likelihood for each class and assign the 
        observation to the class. Finally, we return the predicted values for 
        comparison.

        :param X: The training or validation data

        :return the predicted values 
        '''
        probabilities = []

        # Evaluate the validation data for each of our learned models (weights and bias)
        # that are stored in our LogisticRegression class.
        for class_model in self.class_models:
            label, lr = class_model
            probability = lr.evaluate_model(X)
            probabilities.append([label, probability])

        mean_of_losses = []

        # Compute the mean of all our training losses
        for class_model in self.class_training_losses:
            label, mle = class_model
            mean_of_losses.append(np.mean(mle))

        # Max Mean Likelihood
        max_mle = max(mean_of_losses)

        num_observations = np.shape(X)[0]
        num_preds = len(probabilities)

        preds = []

        # For each observation...
        for i in range(num_observations):
            class_to_assign = None

            # For each prediction...
            for j in range(num_preds):
                current_probability = probabilities[j][1][i]

                # If our current predicted probability is greater than
                # the largest mean likelihood, we will assign this class to this observation
                if current_probability > max_mle:
                    class_to_assign = probabilities[j][0]

            preds.append(class_to_assign)

        return np.array(preds)