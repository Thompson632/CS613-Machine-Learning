import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=1):
        '''
        Constructor that takes in a value for the k-nearest neighbors 
        when sorting the indices values once the distances have been
        calculated. We also set the values of our X and y training
        data to None upon initialization.

        :param k: The k-nearest neighbors value

        :return None
        '''
        self.k = k

        self.X_train = None
        self.y_train = None

    def train_model(self, X, y):
        '''
        Training method that sets our X and y training data to be
        used when we evaluate our model.

        :param X: The training features data
        :param y: The training target data

        :return None
        '''
        self.X_train = X
        self.y_train = y

    def evaluate_model(self, X):
        '''
        Evaluates our model by iterating through each validation point to 
        assigned a class to the observation.

        :param X: The validation features data

        :return the array of classification predictions
        '''
        class_preds = []

        # For each validation observation...
        for x in X:
            # Determine the class to assign
            class_to_assign = self.evaluate(x)
            class_preds.append(class_to_assign)

        return np.array(class_preds)

    def evaluate(self, x):
        '''
        Evaluates our model by doing the following:
        (1) Computing the distance between the validation point and all points 
        in our training data
        (2) Sorting the distances 
        (3) Get the indices based on the provided k-nearest neighbors value at 
        instantiation
        (4) Extracting the correct class labels from our training data using the 
        indices found above
        (5) Create a Counter to keep track of the number of occurrences of the 
        class label
        (6) Get the most common label from our Counter
        (7) Return the class label of the most common class

        :param x: The current validation point

        :return the class label associated with the most common class
        '''
        # Compute the distances for the given validation point for each
        # observation in the training data
        distances = self.compute_distances(x)

        # Sort our distances
        sorted_distances = np.argsort(distances)

        # Get the indices for the first K-neigbors
        k_indices = sorted_distances[:self.k]

        # Get class labels for the indices
        k_labels = self.extract_class(k_indices)

        # Create a counter for our k-nearest neighbors labels
        k_labels_counter = Counter(k_labels)

        # Determine the most common label
        most_common = k_labels_counter.most_common(1)

        # Extract the most common label for classification
        most_common_label = most_common[0][0]

        return most_common_label

    def compute_distances(self, x):
        '''
        Computes the distances between the current point and every
        point in the training data.

        :param x: The given validation point

        :return the list of distances for the given validation point
        '''
        distances = []

        # For each observation in the training data..
        for xt in self.X_train:
            # Compute the distance between the validation point
            # and the current training point
            distance = self.compute_euclidean_distance(x, xt)
            distances.append(distance)

        return distances

    def extract_class(self, k_indices):
        '''
        Extracts the class associated with the given k-nearest neighbors
        indices by return the value of the index in our training data.

        :param k_indices: The indices for k-nearest neighbors for
        the validation point

        :return the list of classes for the given validation point
        '''
        k_class = []

        for index in k_indices:
            k_class.append(self.y_train[index])

        return k_class

    def compute_euclidean_distance(self, x1, x2):
        '''
        Computes the distance between two points using the squared euclidean 
        distance formula.

        :param x1: Point 1
        :param x2: Point 2

        :return the distance between the two points
        '''
        square_val = np.square(x1 - x2)
        sum_val = np.sum(square_val)
        distance = np.sqrt(sum_val)
        return distance