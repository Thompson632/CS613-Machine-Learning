import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def read_data(self, filename):
        # Read the file - skip header (first row) and index (first column)
        data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype='int', usecols=[1, 2, 3])
        
        # Seed random number generator prior to shuffling of the data
        np.random.seed(0)
        
        # Shuffle the data
        np.random.shuffle(data)
        
        # Get the index value of the 2/3 data
        # For this dataset it would be 44 * (2/3) = 29
        training_index = round(len(data) * (2/3))
                
        # Set our training and validation data sets based on the 2/3 index
        training = data[0:training_index]
        validation = data[training_index:]
        
        # Return training, validation data sets
        return training, validation
    
    def calculate_closed_form_direct_solution(self, X, y):
        # Add bias feature
        # TODO: Make this configurable in constructor
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        
        # Calculate the weights based on our features and actuals
        weights= self.calculate_weights(X, y)

        # calculate y_hat (or prediction) based off the features and weights
        y_hat = self.calculate_y_hat(X, weights)

        return y_hat
    
    def calculate_weights(self, X, y):
        # Number of observations
        N = X.shape[0]

        # Convert y from horizontal array to vertical array of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)

        # Error with no inverse for the given validation matrix         
        return np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y)
        
    def calculate_y_hat(self, X, weights):
        return np.dot(X, weights).flatten().astype(int)
     
    def calculate_rmse(self, actual, predicted):
        differences = np.subtract(actual, predicted)
        differences_squared= np.square(differences)
        mse = differences_squared.mean()
        rmse = np.sqrt(mse)
        return rmse
    
    def calculate_mape(self, actual, predicted):
        abs = np.abs((actual - predicted) / actual)
        mape = np.mean(abs) * 100
        return mape
    
    def calculate_local_weight_regression(self, X, y, k):
        # Number of observations
        N = np.shape(X)[0]
        
        # Add bias feature
        # TODO: Make this configurable in constructor
        X = np.append(X, np.ones((N, 1)), axis=1)
        
        # Create y_hat array based on the size of observations
        y_hat_array = np.zeros(N)
        
        # Iterate over all observations, calculate the diagonal weights matrix, calculate all weights matrix, and finally calculate y_hat
        for i in range(N):
            weights = self.calculate_weights_from_diagonal_weights_matrix(X, X[i], y, k, N)
            y_hat = self.calculate_y_hat(X[i], weights);
            y_hat_array[i] = y_hat
            
        return y_hat_array
    
    def calculate_weights_from_diagonal_weights_matrix(self, X, x, y, k, N):        
        # Convert y from horizontal array to vertical array of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)
        
        diagonal_weights_matrix = self.calculate_diagonal_weights_matrix_for_single_observation(X, x, k, N)
        return np.linalg.pinv(X.T * (diagonal_weights_matrix * X)) * (X.T * diagonal_weights_matrix * Y) 
       
    def calculate_diagonal_weights_matrix_for_single_observation(self, X, x, k, N):
        # Create 2D-array with ones on the diagonal and zeros everywhere for the single observation
        diagonal_array = np.eye(N)
        # Convert array to a matrix
        diagonal_weights_matrix = np.mat(diagonal_array)
        
        # Calculate k_squared for the Gaussian Similarity Metric calculation
        k_squared = -(k**2)
        
        # Iterate over the the number of observation, 
        for i in range(N):
            # Using i,i at x,y in matrix to set the returned value to the diagonal
            diagonal_weights_matrix[i, i] = self.calculate_weight_using_gaussian_similarity(X[i], x, k_squared)
        
        return diagonal_weights_matrix
    
    def calculate_weight_using_gaussian_similarity(self, xi, x, k_squared):
        distance = self.calculate_manhattan_distance(xi, x)
        return np.exp(np.dot(distance, distance.T) / k_squared)
    
    def calculate_manhattan_distance(self, xi, x):
        return sum(abs(val1 - val2) for val1, val2 in zip(xi, x))
        

def closed_form_direct_solution(lr, training, validation):
    print("Closed Form Direct Solution\n")
    # TODO: Clean up this duplicate code
    # X should be features Temp of Water and Length of Fish because we want to predict age
    training_x = training[:,-2:]
    validation_x = validation[:,-2:]
    # y should be the actual age values because that is what we are looking to predict
    training_y = training[:,1]
    validation_y = validation[:,1]
    
    # TODO: Clean up duplicate code
    # calculate training direct solution
    training_preds = lr.calculate_closed_form_direct_solution(training_x, training_y)
    rmse = lr.calculate_rmse(training_y, training_preds)
    print("RMSE Training: ", rmse)
    mape = lr.calculate_mape(training_y, training_preds)
    print("MAPE Training: ", mape)
    
    print("")
    
    # calculate direct solution with validation dataset
    actual = lr.calculate_closed_form_direct_solution(validation_x, validation_y)
    rmse = lr.calculate_rmse(validation_y, actual)
    print("RMSE Validation: ", rmse)
    mape = lr.calculate_mape(validation_y, actual)
    print("MAPE Validation: ", mape)
    
    print("")
    
def s_folds_cross_validation(lr):
    print("***** NOT IMPLEMENTED *****\n")
    
def locally_weighted(lr, training, validation):   
    print("Locally Weight Regression\n") 
    # TODO: Clean up this duplicate code
    # X should be features Temp of Water and Length of Fish because we want to predict age
    training_x = training[:,-2:]
    validation_x = validation[:,-2:]
    # y should be the actual age values because that is what we are looking to predict
    training_y = training[:,1]
    validation_y = validation[:,1]
    
    training_preds = lr.calculate_local_weight_regression(training_x, training_y, 1)
    training_rmse = lr.calculate_rmse(training_y, training_preds)
    print("RMSE Training: ", training_rmse)
    training_mape = lr.calculate_mape(training_y, training_preds)
    print("MAPE Training: ", training_mape)
    
    print("")
    
    validation_preds = lr.calculate_local_weight_regression(validation_x, validation_y, 1)
    validation_rmse = lr.calculate_rmse(validation_y, validation_preds)
    print("RMSE Validation: ", validation_rmse)
    validation_mape = lr.calculate_mape(validation_y, validation_preds)
    print("MAPE Validation: ", validation_mape)
    
    
def main():
    lr = LinearRegression()
    
    # Parse file and returning training and validation data
    training,validation = lr.read_data("x06Simple.csv")
    
    closed_form_direct_solution(lr, training, validation)
    #s_folds_cross_validation(lr)
    locally_weighted(lr, training, validation)

if __name__ == '__main__':
    main()