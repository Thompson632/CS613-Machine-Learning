import numpy as np
from collections import Counter
from decision_tree import DecisionTree
import data_util


class RandomForest:
    def __init__(self, forest_size, min_observation_split, min_information_gain):
        '''
        Constructor that creates our random forest classifier. The constructor 
        takes in a value for the number of trees that will be in our forest. 
        The constructor also initializes a list to store our decision tree models.

        :param forest_size: The number of trees in our forest
        :param min_observation_split: The minimum number of observations required 
        to continue splitting the feature data for the decision tree
        :param min_information_gain: The minimum information gain value required 
        to continue splitting the feature data for the decision tree

        :return none
        '''
        self.forest_size = forest_size

        self.min_observation_split = min_observation_split
        self.min_information_gain = min_information_gain

        self.trees = []

    def fit(self, X, y):
        '''
        Method that is used to train our model or in this case, we are building 
        n-decision trees based on the provided forest_size variable at initialization. 
        The function will randomly split our features and target data, build a 
        decision tree, and add it to our forest for future classification.

        :param X: The features data
        :param y: The target data

        :return none
        '''
        for i in range(self.forest_size):
            random_X, random_y = self.random_sample_data(X, y)

            model = DecisionTree(min_observation_split=self.min_observation_split,
                                 min_information_gain=self.min_information_gain)
            model.fit(random_X, random_y)

            self.trees.append(model)

    def random_sample_data(self, X, y):
        '''
        Method that is used to randomly split our features and target data. Based 
        on the number of observations, we randomly return n-number of indices,
        and then only return the features and target data corresponding to the 
        randomly generated indices.

        :param X: The features data
        :param y: The target data

        :return random X and y data
        '''
        num_observations = np.shape(X)[0]

        random_sample_indices = np.random.choice(
            a=num_observations, size=num_observations, replace=True)

        random_X = X[random_sample_indices]
        random_y = y[random_sample_indices]

        return random_X, random_y

    def predict(self, X):
        '''
        Evaluates our model (or forest of decision trees) by doing the following:
        (1) Iterates through each of our decision trees and makes the classifer
        predicition for the validation data, and adds it to a list for further
        processing
        (2) Converts our list of decision tree arrays to a 2D numpy array
        for easier classifier predictions
        (2) Converts our list of decision tree classifier arrays to a 2D array
        with rows as predictions and columns as decision tree number (1 - through
        self.forest_size)
        (3) Loops through each of our decision tree classifier predicitions, 
        selects the most common, and adds it to a list of classifier 
        predictions to be returned

        :param X: The features validation data

        :return the classifier predictions
        '''
        # For each tree, predict the class
        class_tree_preds_list = ([tree.predict(X) for tree in self.trees])

        # Convert tree classifier predictions to a 2D numpy array
        class_tree_preds_arr = data_util.convert_list_to_2d_array(
            class_tree_preds_list)

        # Get most common class amongst all trees
        class_preds = self._predict(class_tree_preds_arr)

        return np.array(class_preds)

    def _predict(self, class_tree_preds):
        '''
        Method that utilizes majority voting to determine the most common class 
        predicted amongst all of the decision trees in our forest.
        We determine the most common class to assign by doing the following:
        (1) Loop through all the decision tree predictions
          (a) Create a Counter dictionary of class predictions to number of 
          class prediction occurrences
          (b) Get the most common class predicition
          (c) Extract the most common class prediction
          (d) Add it to our list of class predictions
        (2) Return the list of classifier predictions

        :param class_tree_preds: The 2D numpy array of all predictions amongst 
        the decision trees

        :return the list of classifier predictions
        '''
        class_preds = []

        # For each decision tree's predictions, get most common class prediction
        for tree_pred in class_tree_preds:
            # Create a counter for our predictions
            tree_pred_counter = Counter(tree_pred)

            # Determine the most common class
            most_common = tree_pred_counter.most_common(1)

            # Extract the most common class to be assigned
            most_common_class = most_common[0][0]
            class_preds.append(most_common_class)

        return class_preds