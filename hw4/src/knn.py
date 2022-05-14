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
        class_preds = []
        
        for x in X:
            class_to_assign = self.evaluate(x)
            class_preds.append(class_to_assign)
            
        return np.array(class_preds)
    
    def evaluate(self, x):
        distances = [self.calculate_euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def calculate_euclidean_distance(self, x1, x2):    
        square_val = np.square(x1 - x2)
        sum_val = np.sum(square_val)
        distance = np.sqrt(sum_val)
        return distance