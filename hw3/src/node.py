class Node:
    def __init__(self, feature_index=None, threshold_value=None, left_sub_tree=None, right_sub_tree=None, information_gain=None, value=None):
        # For our decision nodes
        self.feature_index = feature_index
        self.threshold = threshold_value

        # Left sub-tree
        self.left_sub_tree = left_sub_tree
        # Right sub-tree
        self.right_sub_tree = right_sub_tree

        # Information Gain
        self.information_gain = information_gain

        # For our leaf nodes
        self.value = value