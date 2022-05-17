import math_util
import data_util

import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left_subtree=None, right_subtree=None, leaf_value=None):
        '''
        Constructor that creates either a decision node or a leaf node based on whether or not a given
        feature value meets the required threshold for the given feature. If this class is a decision
        node, all attributes will be set except for the leaf_value attribute. If this class is a leaf
        node, the only attribute that will be set is the leaf_value attribute.

        :param feature_index: The index of the feature in the dataset
        :param threshold: The minimum threshold value for this given feature. In our case,
        we are taking the threshold to be the mean of the entire feature
        :param left_subtree: The left sub-tree where feature values were less than
        or equal to the threshold value
        :param right_subtree: The right sub-tree where feature values were greater
        than the threshold value
        :param leaf_value: The value (or classification) of this node if it is a leaf

        return this node
        '''
        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

        # For leaf node
        self.leaf_value = leaf_value

class DecisionTree:
    def __init__(self, min_observation_split, min_information_gain):
        '''
        Constructor that creates our decision tree classifier. The construction takes in
        a value for the minimum observations required to continue splitting the feature data, 
        minimum information gain required to continuing splitting the feature data, and
        the maximum depth to grow our tree. In our testing class, we default to require at 
        least two observations to continue splitting and that our information gain is greater than 0.

        :param min_observation_split: The minimum number of observations required to
        continue splitting the feature data
        :param min_information_gain: The minimum information gain value required to
        continue splitting the feature data

        :return none
        '''
        self.root = None

        self.min_observation_split = min_observation_split
        self.min_information_gain = min_information_gain

    def train_model(self, X, y):
        '''
        Method that is used to train our model or in this case, we are building
        a decision tree to be used in order to classify an observation.

        :param X: The features data
        :param y: The actual data

        :return none
        '''
        # To make it easier when manipulating our data, we are going to
        # merge our X and y back into one data set
        data = data_util.merge_arrays(X, y)

        # Build our tree from our training data
        self.root = self.build_tree(data=data)

    def build_tree(self, data):
        '''
        We build our decision tree by first finding the feature that has the most 
        information gain value and then we recursively build out each left and 
        right subtree until we have reached our pre-set determination criteria. 
        The determination criteria by default requires the minimum number of 
        observations to continue building the tree to be 2 and that our feature's
        calculated information gain is greater than 0.

        :param data: The data that we will use to find the most informative feature
        and to compute the leaf node's value (or classification)

        :return the decision node or leaf node to be added to our decision tree
        '''
        # Split our data back into X and y
        X, y = data_util.get_features_actuals(data)

        # Get number of observations and features
        num_observations, num_features = np.shape(X)

        # Stop building our tree if we don't have at least 2 observations
        if num_observations >= self.min_observation_split:
            # Find the most informative feature
            most_informative_feature = self.find_most_informative_feature(
                data, num_features)

            # Check that we have an informative feature
            if most_informative_feature is not None:
                # If our most informative feature's information gain is greater than the
                # minimum value we set, keep building the tree
                if most_informative_feature["information_gain"] > self.min_information_gain:
                    # Build the left sub-tree using the split data of our most informative feature
                    left_subtree = self.build_tree(
                        data=most_informative_feature["left_subtree"])

                    # Build the right sub-tree using the split data of our most informative feature
                    right_subtree = self.build_tree(
                        data=most_informative_feature["right_subtree"])

                    # Build and return decision node
                    return Node(feature_index=most_informative_feature["feature_index"],
                                threshold=most_informative_feature["threshold"], left_subtree=left_subtree,
                                right_subtree=right_subtree)

        # We have reached the bottom of our branch and need to classify this leaf node
        leaf_value = self.classify_leaf_node(y)

        # Build and return our leaf node
        return Node(leaf_value=leaf_value)

    def find_most_informative_feature(self, data, num_features):
        '''
        Finds the feature with the highest information value by iterating through each feature, 
        calculating the mean of the feature that will be used as our threshold value per Professor
        Burlick's instructions, and splits our data into two arrays based on the threshold value.
        Once we have the split arrays, we compute the information gain of the feature, compare it 
        to the current max information gain variable, and update the max information gain and
        max information gain feature value if it is greater than. Finally, we return a dictionary
        containing the attributes of the feature with the max information that is required to 
        create a node in our decision tree.

        :param data: The data we are iterating searching
        :param num_features: The number of features in the current data

        :return a dictionary of attributes pertaining to the feature with the most information gain
        '''
        # Starting maximum information gain value
        max_info_gain = -1

        # This will be our most informative feature
        max_info_gain_feature = None

        # For each feature...
        for feature_index in range(num_features):
            # Get all observations for this feature
            feature_values = data[:, feature_index]
            # Calculate the mean of the values for our threshold value
            feature_mean = math_util.calculate_mean(feature_values)

            # Split our data into two sub-arrays based on the threshold value (or feature mean)
            left, right = data_util.split_on_feature(
                data=data, feature_index=feature_index, threshold=feature_mean)

            # If we have at least one observation in either of the sub-arrays
            if len(left) >= 1 and len(right) >= 1:
                # Get the y-values of all our data and the split
                # left and right datasets
                data_y = data[:, -1]
                left_y = left[:, -1]
                right_y = right[:, -1]

                # Calculate this features information gain
                feature_information_gain = self.calculate_information_gain(
                    data_y, left_y, right_y)

                # If this feature's information gain is greater than the
                # current max information gain, update the max information
                # gain value and set this feature as the most informative
                if feature_information_gain > max_info_gain:
                    max_info_gain = feature_information_gain

                    max_info_gain_feature = {
                        "feature_index": feature_index,
                        "threshold": feature_mean,
                        "left_subtree": left,
                        "right_subtree": right,
                        "information_gain": feature_information_gain
                    }

        return max_info_gain_feature

    def calculate_information_gain(self, data_y, left_y, right_y):
        '''
        Calculates the feature's total information gain by first computing the entropy
        of the entire data set and then calculating the entropy of the split arrays
        based on the feature's threshold value. Finally, we return this feature's
        total information gain by subtracting the feature's information gain from
        the total data set's entropy.

        Function Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees

        :param data_y: The data set's y-values
        :param left_y: The left branch data set's y-values
        :param right_y: The right branch data set's y-values

        :return the feature's total information gain
        '''
        # Get the number of observations
        num_observations = len(data_y)

        # Calculate the total entropy for all the data
        total_entropy = self.calculate_entropy(data_y)

        # Calculate left information gain
        left_probability = len(left_y) / num_observations
        left_entropy = self.calculate_entropy(left_y)
        left_info_gain = left_probability * left_entropy

        # Calculate right information gain
        right_probability = len(right_y) / num_observations
        right_entropy = self.calculate_entropy(right_y)
        right_info_gain = right_probability * right_entropy

        # Calculate feature information gain
        feature_info_gain = left_info_gain + right_info_gain

        # Calculate and return the total information gain of this feature
        # by subtracting the features information gain from the total
        # datasets' entropy
        return total_entropy - feature_info_gain

    def calculate_entropy(self, y):
        '''
        Calculates the entropy of the provided target values by first
        calculating the entropy for each unique class in the provided data.

        Function Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)

        :param y: The target values

        :return the sum of each class' entropy calculation 
        '''
        # Get our unique classes
        unique_classes = np.unique(y)

        # Get number of total observations
        num_observations = len(y)

        # Default entropy for this feature value
        entropy = 0

        # For each class..
        for c in unique_classes:
            # Get the number of observations for this class
            _, class_count = data_util.get_observation_counts(y=y, c=c)

            # Calculate this class' probability
            class_probability = class_count / num_observations

            # Calculate this class' entropy
            class_entropy = -class_probability * np.log2(class_probability)

            # Add it to our feature's total entropy
            entropy = entropy + class_entropy

        return entropy

    def classify_leaf_node(self, y):
        '''
        Classifies the leaf node value by iterating through the each of the unique
        classes in the data set and assigning the class that has the most occurences
        in the dataset.

        :param y: The target values

        :return the class value to assign to this leaf node
        '''
        # Get our unique classes
        unique_classes = np.unique(y)

        # Represents the max number of class occurences
        # in the data
        max_class_count = 0

        # Represents the class to assign to this leaf node
        class_to_assign = None

        # For each class...
        for c in unique_classes:
            # Get the number of observations for this class
            _, class_count = data_util.get_observation_counts(y=y, c=c)

            # If the current class count is greater than the max class count,
            # update the max class count, and temporarily assign this class
            if class_count > max_class_count:
                max_class_count = class_count
                class_to_assign = c

        return class_to_assign

    def evaluate_model(self, X):
        '''
        Evaluates our model (or tree) by iterating through each observation in the validation
        data, recursively search our model (or tree) until we reach a leaf node, return the 
        class to assign to this observation, and finally return an array of our class
        predicitions for the data.

        :param X: The validation features data

        :return the predicted classes
        '''
        class_preds = []

        # For each observation...
        for x in X:
            # Get the class to assign to this observation
            class_to_assign = self.search_tree(x, self.root)

            # Assign the class
            class_preds.append(class_to_assign)

        return np.array(class_preds)

    def search_tree(self, x, tree):
        '''
        Recursive function that searches our decision tree by using our 
        current decision node's threhold value to determine what branch (left or right)
        we need to traverse in order to get to the correct leaf node classification
        for this particular observation. When we eventually reach a node that has 
        a leaf value, we return this as our prediction.

        :param x: The current observation we are attempting to classify
        :param tree: The current root node of our decision tree

        :return the classification for this observation
        '''

        # First, check to see if we are at a leaf node
        # and if we are, then return the prediction value
        # for this observation
        if tree.leaf_value is not None:
            return tree.leaf_value

        # Get the feature value
        feature_value = x[tree.feature_index]

        # Now we need to determine which branch we are going to traverse
        # We can do this by comparing our feature value against the current
        # node's threshold value similar to what we did when we built the tree
        if feature_value <= tree.threshold:
            return self.search_tree(x, tree.left_subtree)
        else:
            return self.search_tree(x, tree.right_subtree)