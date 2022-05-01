import numpy as np
from node import Node
import math_util
import util


class DecisionTree:
    def __init__(self, min_samples_split, min_information_gain=1e-7, max_depth=float("inf")):
        self.root = None

        self.min_samples_split = min_samples_split
        self.min_information_gain = min_information_gain
        self.max_depth = max_depth

    def train_model(self, X, y):
        self.root_node = self.build_tree(X, y)

    def build_tree(self, X, y, current_depth):
        # Merge our X and y into one for easier splitting
        data = np.concatenate((X, y), axis=1)

        # Get our num observations and features
        num_observations, num_features = np.shape(X)

        if num_observations >= self.min_samples_split and current_depth <= self.max_depth:
            # Find the best feature split
            best_feature_split = self.find_best_feature(data, num_features)

            if best_feature_split["information_gain"] > self.min_information_gain:
                # Build left sub-tree
                left_sub_tree = self.build_tree(
                    best_feature_split["left_sub_tree"], current_depth + 1)
                # Build right sub-tre
                right_sub_tree = self.build_tree(
                    best_feature_split["right_sub_tree"], current_depth + 1)

                # Return the decision node
                return Node(best_feature_split["feature_index"], best_feature_split["threshold"], left_sub_tree, right_sub_tree, best_feature_split["information_gain"])

        # Compute the leaf value
        leaf_value = self.compute_leaf_value(y)
        return Node(value=leaf_value)

    def find_best_feature(self, data, num_features):
        # Starting maximum information gain value
        max_info_gain = 0

        # This will be the best node for this
        best_node = None

        # For each feature...
        for feature_index in range(num_features):
            # Get all rows for this column
            feature_values = data[:, feature_index]
            # Compute the mean of the data for our initial threshold for this feature
            feature_mean = math_util.compute_mean(feature_values)
            # Split the data based on this feature's threshold value
            left, right = util.split_on_feature(
                data, feature_index, feature_mean)

            # If both our split data sets have at minimum 1 sample in them...
            if len(left) > 0 and len(right) > 0:
                # Get the y-values of all our data and the split
                # left and right datasets
                data_y = data[:, -1]
                left_y = left[:, -1]
                right_y = right[:, -1]

                # Compute information gain
                feature_information_gain = self.compute_information_gain(
                    data_y, left_y, right_y)

                # If this feature's information gain is greater
                # than the current max info, update the max info gain,
                # and save this feature as our best node
                if feature_information_gain > max_info_gain:
                    max_info_gain = feature_information_gain

                    best_node = {
                        "feature_index": feature_index,
                        "threshold": feature_mean,
                        "left_sub_tree": left,
                        "right_sub_tree": right,
                        "information_gain": feature_information_gain
                    }

        return best_node

    def compute_information_gain(self, parent, left_child, right_child):
        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)

        total_gain = self.compute_entropy(parent)
        left_gain = left_weight * self.compute_entropy(left_child)
        right_gain = right_weight * self.compute_entropy(right_child)

        return total_gain - (left_gain + right_gain)

    def compute_entropy(self, y):
        unique_classes = np.unique(y)
        entropy = 0
        for c in unique_classes:
            class_probability = len(y[y == c]) / len(y)
            entropy = entropy + (-class_probability *
                                 np.log2(class_probability))

        return entropy

    def compute_leaf_value(self, y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def evaluate_model(self, X):
        preds = []

        for x in X:
            prediction = self.evaluate(x, self.root)
            preds.append(prediction)

        return np.array(preds)

    def evaluate(self, x, tree):
        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_index]

        if feature_value <= tree.threshold:
            return self.evaluate(x, tree.left)
        else:
            return self.evaluate(x, tree.right)
