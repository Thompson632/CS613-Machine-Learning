from node import Node
import math_util
import data_util

import numpy as np


class DecisionTree:
    def __init__(self, min_observation_split=2, min_information_gain=1e-7, max_tree_depth=float("inf")):
        '''
        Constructor that creates our decision tree classifier. The construction takes in
        a value for the minimum observations required to continue splitting the feature data, 
        minimum information gain required to continuing splitting the feature data, and
        the maximum depth to grow our tree.

        :param min_observation_split: The minimum number of observations required to
        continue splitting the feature data
        :param min_information_gain: The minimum information gain value required to
        continue splitting the feature data
        :param max_tree_depth: The maximum depth to grow our tree

        :return none
        '''
        self.root = None

        self.min_observation_split = min_observation_split
        self.min_information_gain = min_information_gain
        self.max_tree_depth = max_tree_depth

    def train_model(self, X, y):
        # To make it easier when manipulating our data, we are going to
        # merge our X and y back into one data set
        data = data_util.merge_arrays(X, y)

        # Build our tree from our training data
        self.root = self.build_tree(data=data)

    def build_tree(self, data, current_depth=0):
        # Split our data back into X and y
        X, y = data_util.get_features_actuals(data)

        # Get number of observations and features
        num_observations, num_features = np.shape(X)

        # Stop building our tree if we don't have at least 2 observation
        # and our current_depth is less than or equal to infinity. The former
        # will generally stop the building of this tree
        if num_observations >= self.min_observation_split and current_depth <= self.max_tree_depth:
            # Find the most informative
            most_informative_feature = self.find_most_informative_feature(
                data, num_features)

            # Check that we have an informative feature
            if most_informative_feature is not None:
                # If our most informative feature's information gain is greater than the
                # minimum value we set, keep building the tree
                if most_informative_feature["information_gain"] > self.min_information_gain:
                    # Increment the current depth of our tree
                    current_depth = current_depth + 1
                    # Build left sub-tree
                    left_subtree = self.build_tree(
                        data=most_informative_feature["left_subtree"], current_depth=current_depth)
                    # Build right sub-tree
                    right_subtree = self.build_tree(
                        data=most_informative_feature["right_subtree"], current_depth=current_depth)

                    # Build and return decision node
                    return Node(feature_index=most_informative_feature["feature_index"],
                                threshold=most_informative_feature["threshold"], left_subtree=left_subtree,
                                right_subtree=right_subtree)

        # If we are here, we have completed building our tree and need to calculate the value for our leaf
        leaf_value = self.calculate_leaf_value(y)

        # Build and return our leaf node
        return Node(leaf_value=leaf_value)

    def find_most_informative_feature(self, data, num_features):
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
            if len(left) > 0 and len(right) > 0:
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
        # Get our unique classes
        unique_classes = np.unique(y)

        # Get number of total observations
        num_observations = len(y)

        # Default entropy for this feature value
        entropy = 0

        # For each class..
        for c in unique_classes:
            # Get the observations for this class
            class_observations = y[y == c]
            # Get the number of observation for this class
            class_count = len(class_observations)

            # Calculate this class' probability
            class_probability = class_count / num_observations

            # Calculate this class' entropy
            class_entropy = -class_probability * np.log2(class_probability)

            # Add it to our feature's total entropy
            entropy = entropy + class_entropy

        return entropy

    def calculate_leaf_value(self, y):
        # Get our unique classes
        unique_classes = np.unique(y)

        # Represents the max number of class occurences
        # in the data
        max_class_count = 0

        # Represents the class to assign to this leaf node
        class_to_assign = None

        # For each class...
        for c in unique_classes:
            # Get the observations for this class
            class_observations = y[y == c]
            # Get the number of observation for this class
            class_count = len(class_observations)

            # If the current class count is greater than the max class count,
            # update the max class count, and temporarily assign this class
            if class_count > max_class_count:
                max_class_count = class_count
                class_to_assign = c

        return class_to_assign

    def evaluate_model(self, X):
        class_preds = []

        # For each observation...
        for x in X:
            # Get the class to assign to this observation
            class_to_assign = self.search_tree(x, self.root)

            # Assign the class
            class_preds.append(class_to_assign)

        return np.array(class_preds)

    def search_tree(self, x, tree):
        # First, check to see if we are at a leaf node
        # and if we are, then return the prediction value
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