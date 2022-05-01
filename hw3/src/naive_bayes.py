import numpy as np
import data_util
import math_util


class NaiveBayes:
    def __init__(self, stability_constant):
        '''
        Constructor that takes in a stability constant to be used 
        to avoid division by zero. The constructor also
        initializes attributes that will be used throughout
        the life of this class.

        :param stability_constant: The stability constant to be used
        to avoid divide by zero in the probability calculation

        :return none
        '''
        self.classes = None
        self.num_classes = 0

        self.class_means = None
        self.class_vars = None
        self.class_priors = None

        self.stability_constant = stability_constant

    def train_model(self, X, y):
        '''
        Trains our model by computing the mean and variance for each
        column associated with the specific class we are training on
        and the prior probability of each class. The calculated
        values for each class will be used later when we evaluate
        the models.

        :param X: The training features
        :param y: The training targets

        :return none
        '''
        # Get number of observations and features
        num_observations, num_features = np.shape(X)

        # Get the unique classes
        self.classes = np.unique(y)

        # Get the number of classes
        self.num_classes = len(self.classes)

        # Create the class mean, variance, and prior probability arrays
        self.class_means, self.class_vars, self.class_priors = data_util.create_mean_var_prior_arrays(
            self.num_classes, num_features)

        # For each class...
        for i, c in enumerate(self.classes):
            # Get the observations associated with this class
            class_observations = X[y == c]
            # Get the number of observations with this class
            class_count = np.shape(class_observations)[0]

            # Compute this class' prior probability
            self.class_priors[i] = math_util.calculate_prior_probability(
                class_count, num_observations)

            # Compute the mean and variance of all observations
            self.class_means[i, :] = math_util.calculate_mean(
                class_observations, 0)
            self.class_vars[i, :] = math_util.calculate_variance(
                class_observations, 0)

    def evaluate_model(self, X):
        '''
        Evaluates our models by iterating through each observation
        in the validation data, assigning the class with the highest
        probability, and returning the predicted classes.

        :param X: The validation features data

        :return the predicted classes 
        '''
        class_preds = []

        # For each observation...
        for x in X:
            # Calculate the probabilities for each class for
            # this observation
            class_probabilities = self.compute_class_probabilities(x)

            # Get the index of the max class probability
            max_probability_index = np.argmax(class_probabilities)

            # Get the class associated with this index
            class_to_assign = self.classes[max_probability_index]

            # Assign the class
            class_preds.append(class_to_assign)

        return np.array(class_preds)

    def compute_class_probabilities(self, x):
        '''
        Computes the probabilities of each classes being assigned to
        the given observation. We compute the probability for each class
        by using the Gaussian Probability Density Function (or Normal Distribution).
        We take the logs of the class' prior probability and the output of
        the Gaussian PDF to stabilize our calculations.

        :param x: The current observation we are trying to assign

        :return the list of probabilities for each calss
        '''
        posteriors = []

        # For each class...
        for c in range(self.num_classes):
            # Get the prior probability of this class
            class_prior = self.class_priors[c]
            # Take the log of it for stability
            prior = np.log(class_prior)

            # Get the mean values for this class
            means = self.class_means[c]

            # Get the variance values for this class
            vars = self.class_vars[c]

            # Compute the probability using the Gaussian
            # Probability Density Function
            gpdf = self.compute_gaussian_pdf(x, means, vars)

            # Get the summation of our probabilities
            sum_gpdf = np.sum(gpdf)

            # Compute the class' posterior probability
            posterior = prior + sum_gpdf
            posteriors.append(posterior)

        return posteriors

    def compute_gaussian_pdf(self, x, means, vars):
        '''
        Computes the probability for this observation using the 
        Gaussian Probability Density Function or otherwise known
        as the normal distribution. For stability purposes, we take
        the log of the output of the Gaussian PDF.

        Function Reference: https://en.wikipedia.org/wiki/Gaussian_function

        :param x: The observation we are evaluating
        :param means: The calculated mean values for a particular class
        :param vars: The calculated variance values for a particular class

        :return the log of the calculated probability
        '''
        exp_numerator = -np.power((x - means), 2)
        exp_denominator = (2 * vars + self.stability_constant)

        numerator = np.exp(exp_numerator / exp_denominator)
        denominator = np.sqrt(2 * np.pi * vars + self.stability_constant)

        pdf = numerator / denominator
        return np.log(pdf)
