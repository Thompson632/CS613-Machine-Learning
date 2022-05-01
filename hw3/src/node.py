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