import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=1):
        self.k = k
        
        self.X_train = None
        self.y_train = None
        
    def train_model(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def evaluate_model(self, X):
        y_pred = [self.evaluate(x) for x in X]
        return np.array(y_pred)
    
    def evaluate(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.calculate_euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def calculate_euclidean_distance(self, x1, x2):    
        # square_val = np.square(x1 - x2)
        # sum_val = np.sum(square_val)
        # distance = np.sqrt(sum_val)
        # return distance
        return np.sqrt(np.sum((x1 - x2) ** 2))