import numpy as np

class LinearRegressionClosedForm:
    def __init__(self):
        '''
        Constructor that initializes a weights vector based on 
        the training data
        
        :param none
        
        :return none
        '''
        self.weights = None

    def fit(self, X, y):
        '''
        Trains a closed form linear regression model.
        
        :param X: The features data
        :param y: The target data
        
        :return none
        '''
        num_observations = np.shape(X)[0]
        
        y = y.reshape(num_observations, 1)
        y_mat = np.mat(y)
        
        self.weights = self.compute_weights(X, y_mat)

    def predict(self, X):
        '''
        Evaluates our trained closed form linear regression model by 
        taking the dot product of our validation data and the learned
        weights model.
        
        :param X: The validation data
        
        :return the predictions
        '''
        y_hat = self.compute_y_hat(X)
        return y_hat
    
    def compute_weights(self, X, y):
        '''
        Computes the weights based on the features and actuals

        :param X: Features
        :param y_mat: Actuals

        :return the weights (or coefficients)
        '''
        xt_x = np.dot(X.T, X)
        xt_x_inv = np.linalg.pinv(xt_x)
        xt_x_inv_xt = np.dot(xt_x_inv, X.T)
        xt_x_inv_xt_y = np.dot(xt_x_inv_xt, y)
        return xt_x_inv_xt_y
    
    def compute_y_hat(self, X, weights=None):
        '''
        Computes y_hat by taking the dot product of our
        features and weights

        :param X: The features data
        :param weights: Optional weights parameter. If provided, will compute
        y_hat using the weights provided. Otherwise, will use the model
        computed weights

        :return the predictions
        '''
        if weights is not None:
            return np.dot(X, weights).flatten()
        
        return np.dot(X, self.weights).flatten()