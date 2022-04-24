from re import I
import numpy as np
import math
import math_util


class NaiveBayes():
    def __init__(self, split_data_dict):
        '''
        Constructor that takes in the split data as a dictionary
        and is later used to create the normal models for
        each feature in each class

        :param split_data_dict: The data split by class

        :return none
        '''
        self.split_data_dict = split_data_dict
        self.normal_models = None

    def create_normal_models_for_each_class(self, should_print_models):
        '''
        Creates the normal models for each feature in each class
        by calculating the mean and standard deviation of each feature.

        :param should_print_models: Flag to print the models for
        visualization purposes

        :return the created normal models
        '''
        self.normal_models = {}

        # For each class and list of feature tuples...
        for class_value, list_of_feature_tuples in self.split_data_dict.items():
            self.normal_models[class_value] = math_util.compute_mean_std_of_features_list(
                list_of_feature_tuples)

        if should_print_models:
            self.print_normal_models()

    def print_normal_models(self):
        '''
        Helper method used to print the normal (mean and standard deviation)
        models for each feature.

        :param none

        :return none
        '''
        for class_value in self.normal_models:
            print("\nClass:", class_value)
            print("Mean and Standard Deviation Values:")

            for i in range(len(self.normal_models[class_value])):
                print("Feature", i, ":", self.normal_models[class_value][i])

    def evaluate_models(self, X, y):
        '''
        Evaluates our validation data against each feature model for each class.

        :param X: The validation feature data
        :param y: The target data

        :returns the probabilities for each y
        '''
        num_observations = np.shape(X)[0]

        # Predictions
        y_hat = np.zeros(num_observations)

        # For each observation...
        for i in range(num_observations):
            probability_dict = self.compute_probability(X[i])
            print(probability_dict)

            target_class = y[i]

            # TODO: Assign the correct class. Defaulting currently to set up for metrics
            y_hat[i] = target_class

        return y_hat

    def compute_probability(self, current_feature):
        '''
        Computes the probability for the current validation feature
        we are iterating through. We do this by iterating through each
        class and their normal models in order to compute the probability
        using the gaussian probability density function.
        
        :param current_feature: The current validation feature
        
        :return the probability dictionary for this feature
        '''
        probability_dict = {}

        num_observation_in_current_feature = len(current_feature)

        # For each class in our normal models...
        for class_value in self.normal_models:
            # Computed model for each feature in this class
            class_feature_models = self.normal_models[class_value]

            # Initial probability will be the number of observations in
            # the current class divided by the number of observations
            # in the current validation feature
            num_observations_in_current_class = class_feature_models[0][2]
            probability = num_observations_in_current_class / \
                num_observation_in_current_feature
            # print("\nInitial Probability for this Class:", probability)

            for i in range(len(class_feature_models)):
                # Get the observation
                current_observation = current_feature[i]
                # print("Current Observation:", current_observation)

                # Get the mean of the current feature
                mean_value = class_feature_models[i][0]
                # print("Mean:", mean_value)

                # Get the std of the current feature
                std_value = class_feature_models[i][1]
                # print("Std:", std_value)

                # Compute Gaussian PDF
                predicted_probability = self.compute_gaussian_pdf(
                    current_observation, mean_value, std_value)
                #print("Predicted Probability for Feature",
                #      i, ":", predicted_probability)
                    
                probability *=predicted_probability
                #print("Probability for Feature:", i, ":", probability)

            probability_dict[class_value] = probability

        return probability_dict

    def compute_gaussian_pdf(self, x, mean, std):
        '''
        Computes the gaussian probability densition function
        for a single observation.

        Function Reference: https://en.wikipedia.org/wiki/Gaussian_function

        :param x: The observation
        :param mean: The mean of the observation
        :param std: The standard deviation of the observation

        :return the gaussian probability density function calculation
        '''
        denominator = std * math.sqrt(2 * math.pi)
        if denominator == 0:
            #print("Denominator is 0. Defaulting denominator to 1!")
            denominator = 1

        exp_numerator = (x - mean)**2
        exp_denominator = 2 * (std**2)
        exp = math.exp(-(exp_numerator / exp_denominator))

        pdf = (1 / denominator) * exp
        return pdf
