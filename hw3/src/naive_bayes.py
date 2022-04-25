import numpy as np
import math_util
import math


class NaiveBayes():
    def __init__(self, epsilon):
        '''
        Constructor that takes in an epsilon to be used in the 
        calculation of the Gaussian Probability Density Function
        to increase the variance when performing calculations.
        
        :param epsilon: The constant epsilion value
        
        :return none
        '''
        self.num_total_observations = 0
        self.num_total_features = 0

        self.classifiers = None
        self.num_classifiers = 0

        self.classifier_mean = {}
        self.classifier_stds = {}
        self.classifier_initial_probs = {}

        self.epsilon = epsilon

    def train_model(self, X, y):
        '''
        Computes the normal models (mean and standard deviation) of each feature 
        for each classifier in the dataset.
        
        :param X: The training features data
        :param y: The training targets data
        
        :return none
        '''
        # Set our total observtation and feature of the training data
        self.num_total_observations, self.num_total_features = np.shape(X)

        # Get our unique classifiers of this dataset
        self.classifiers = np.unique(y)
        # Get the number of classifiers in this dataset
        self.num_classifiers = len(self.classifiers)

        # For each classifier...
        for classifier in range(self.num_classifiers):
            # Get the observations where y is equal to the classifier
            classifier_observations = X[y == classifier]

            # Get the number of observations for this classifier
            num_classifier_observations = np.shape(classifier_observations)[0]

            # 6. Create Normal models for each feature for each class
            self.classifier_mean[str(classifier)] = math_util.compute_mean(
                classifier_observations, 0)
            self.classifier_stds[str(classifier)] = math_util.compute_variance(
                classifier_observations, 0)
            self.classifier_initial_probs[str(classifier)] = math_util.compute_initial_classifier_probability(
                num_classifier_observations, self.num_total_observations)
            
            # print("\nClass:", classifier)
            # print("Mean:\n", self.classifier_mean[str(classifier)] )
            # print("Stds:\n", self.classifier_stds[str(classifier)])
            # print("Prob:", self.classifier_initial_probs[str(classifier)])

    def evaluate_model(self, X):
        '''
        Evaluates our models using the learned models and the 
        Gaussian Probability Density Function to compute the 
        probability that each observation belongs to a specific
        classifier. We set the class with the highest probability 
        to each observation.
        
        :param X: The validation features data
        
        return the classifier predictions
        '''
        num_observations = np.shape(X)[0]
        
        # Create a 2-D numpy array with num_observations as row count
        # and num_classifier as column count
        probabilities = np.zeros((num_observations, self.num_classifiers))
        
        # For each classifier...
        for classifier in range(self.num_classifiers):
            # Get the initial probability for this classifier
            classifier_initial_prob = self.classifier_initial_probs[str(classifier)]
            
            # Get the mean and std for this classifier
            mean = self.classifier_mean[str(classifier)]
            std = self.classifier_stds[str(classifier)]
            
            # Calculate the probability given the normal models
            probabilities_of_classifier = self.compute_gaussian_pdf(X, mean, std)
            
            # Set the probabilities for every observation for this classifier
            probabilities[:, classifier] = probabilities_of_classifier + np.log(classifier_initial_prob)
        
        return math_util.get_indices_of_max_values(probabilities, axis=1)
        

    def compute_gaussian_pdf(self, X, mean, std):
        '''
        Computes the gaussian probability density function
        for all observation.

        Function Reference: https://en.wikipedia.org/wiki/Gaussian_function

        :param X: The observations
        :param mean: The mean of the observations
        :param std: The standard deviation of the observations

        :return the gaussian probability density function calculation
        '''
        const_numerator = -self.num_total_features
        const_denominator = 2 * np.log(2 * math.pi)
        const_val = 0.5 * np.sum(np.log(std + self.epsilon))
        const = const_numerator / const_denominator - const_val
        
        prob_numerator = np.power(X - mean, 2)
        prob_denominator = std + self.epsilon
        prob = 0.5 * np.sum(prob_numerator / prob_denominator, 1)
        
        # Since we are taking the log, the original product makes this into
        # a sum, so we can subtract our probability from our constant 
        return const - prob
