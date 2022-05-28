import numpy as np
from scipy import stats
from decision_tree import DecisionTree


class RandomForest():
    def __init__(self, forest_size, tree_depth=5):
        self.forest_size = forest_size
        self.tree_depth = tree_depth
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        np.random.seed(0)
        for i in range(self.forest_size):
            # if(i % (self.forest_size // 100) == -1 % (self.forest_size // 100)):
            #    print("training tree",i)
            #print("training tree",i)
            feature_indices = np.random.choice(X.shape[1],
                                               size=self.tree_depth,
                                               replace=False).tolist()
            print(type(feature_indices))
            print(feature_indices)
            # print(np.shape(feature_indices))
            #X_features = X[:, feature_indices]
            #decisionTree = DTL(X,Y,feature_indices,default=0)
            feature_indices = sorted(feature_indices)
            print(feature_indices)
            dt = DecisionTree()
            dt.fit(X, y)
            self.forest.append(dt)
        return self.forest

    def predict(self, X):
        predictions = np.empty((X.shape[0], self.forest_size))
        for row, obs in enumerate(X):
            for tree in range(self.forest_size):
                #pred = self.forest[tree].predict(obs)[0]
                predictions[row, tree] = self.forest[tree].predict(obs)[0]

        # print(predictions)
        predictions = stats.mode(predictions, axis=1)[0]
        predictions = np.reshape(predictions, (predictions.size,))
        return predictions