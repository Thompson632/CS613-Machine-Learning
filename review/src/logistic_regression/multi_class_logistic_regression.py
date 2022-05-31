import numpy as np
from logistic_regression import LogisticRegression
import data_util


class MultiClassLogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, stability_constant=10e-7, log_verbose=False, unique_classes=None):
        '''
        Constructor that takes in a learning rate value,
        the number of epochs to be ran in our training model,
        stability constant for our log loss function, and
        the unique classes for this model.

        param lr: The learning rate. Default is 0.1
        :param epochs: The number of iterations. Default is 1000
        :param stability_constant: The constant to stabilize
        our logs to ensure we do not get log(0) = infinity. Default is 10e-7
        :param log_verbose: Flag to have extra logging for output. Default is false
        :param unique_classes: The unique number of classes to
        be used for this multi-class logistic regression model

        :return none
        '''
        self.lr = lr
        self.epochs = epochs
        self.stability_constant = stability_constant
        self.log_verbose = log_verbose

        self.unique_classes = unique_classes
        self.class_models = []

    def fit(self, x_train, y_train, x_valid, y_valid):
        '''
        Trains a logistic regression model using gradient descent
        and sigmoid (or the logistic function) for each of the classes.
        This function will generate the following number of binary
        classifiers: K(K-1)/2

        :param x_train: Training features
        :param y_train: Training actuals
        :param x_valid: Validation features
        :param y_valid: Validation actuals

        :return none
        '''
        for c in self.unique_classes:
            x_train_sorted, y_train_sorted_binary = data_util.convert_data_to_binary(
                x_train, y_train, c)
            x_valid_sorted, y_valid_sorted_binary = data_util.convert_data_to_binary(
                x_valid, y_valid, c)

            model = LogisticRegression(
                lr=self.lr, epochs=self.epochs, stability_constant=self.stability_constant, log_verbose=self.log_verbose)

            # Not storing returned training and validation loss as not needed for assigning a class
            model.fit(x_train_sorted, y_train_sorted_binary,
                      x_valid_sorted, y_valid_sorted_binary)

            self.class_models.append([c, model])

    def predict(self, X):
        '''
        Evaluates the data on for each of the learned models. Once computed,
        we compute the largest mean likelihood for each class and assign the 
        observation to the class. Finally, we return the predicted values for 
        comparison.

        :param X: The training or validation data

        :return the predicted values 
        '''
        probabilities = []

        # Evaluate the validation data for each of our learned models
        for class_model in self.class_models:
            label, lr = class_model
            probability = lr.predict(X)
            probabilities.append([label, probability])

        num_observations = np.shape(X)[0]

        # Assign a class to each observation based on the max likelihood
        class_preds = self.assign_class_to_observation(
            num_observations, probabilities)

        return class_preds

    def assign_class_to_observation(self, num_observations, probabilities):
        '''
        Method used to loop through all observations to determine the correct
        class to assign to the observation. For each observation, we have an
        inner loop that loops through the predictions for that specific
        observation and updates a local variable with the max likelihood
        that the observation belongs to the class. After we have looped through
        all predictions of the observations, we will assign the class that was last
        set to a local variable to later be returned.

        :param num_observations: The number of observations in our validation
        data set
        :param probabilities: The list of probabilities for each observation for 
        each class label

        :return the predicted classes for each observation in our validation 
        data set
        '''
        num_preds = len(probabilities)

        class_preds = []

        # For each observation...
        for i in range(num_observations):
            # Class to assign to this specific observation
            class_to_assign = None

            # Default this to 0 to start
            max_likelihood = 0.0

            # For each model...
            for j in range(num_preds):
                # Get this models predicted probability for this observation
                model_predicted_prob = probabilities[j][1][i]

                # If the current predicted probability for this model is larger
                # than the current largest likelihood, we will assign this models
                # class to this observation and will update the largest likelihood
                # value for the next iteration
                if model_predicted_prob > max_likelihood:
                    class_to_assign = probabilities[j][0]
                    max_likelihood = model_predicted_prob

            class_preds.append(class_to_assign)

        return np.array(class_preds)
