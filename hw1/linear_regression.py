import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def read_data(self, filename):
        # Read the file - skip header (first row) and index (first column)
        data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype='int', usecols=[1, 2, 3])
        
        # Seed random number generator prior to shuffling of the data
        np.random.seed(1)
        
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
    
    def compute_closed_form_direct_solution(self, X, y):
        weights= self.compute_weights(X, y)

        # Add the bias feature (or column) of ones to match what we did
        # when we calculated the weights
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        # Compute y_hat (or prediction) based off the features and weights
        y_hat = self.compute_y_hat(X, weights)
        y_hat = y_hat.flatten().astype(int)

        return y_hat
    
    def compute_weights(self, X, y):
        # Number of observations
        N = X.shape[0]
        
        # Add bias feature (or column) of ones
        X = np.append(X, np.ones((N, 1)), axis=1)

        # Convert y from horizontal array to vertical array of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)

        # Error with no inverse for the given validation matrix         
        weights = np.linalg.pinv(np.dot(X.T,X)) * np.dot(X.T,Y)
        return weights
        
    def compute_y_hat(self, X, weights):
         return np.dot(X, weights)
     
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
    # Compute training direct solution
    training_preds = lr.compute_closed_form_direct_solution(training_x, training_y)
    rmse = lr.calculate_rmse(training_y, training_preds)
    print("RMSE Training: ", rmse)
    mape = lr.calculate_mape(training_y, training_preds)
    print("MAPE Training: ", mape)
    
    print("")
    
    # Compute direct solution with validation dataset
    actual = lr.compute_closed_form_direct_solution(validation_x, validation_y)
    rmse = lr.calculate_rmse(validation_y, actual)
    print("RMSE Validation: ", rmse)
    mape = lr.calculate_mape(validation_y, actual)
    print("MAPE Validation: ", mape)
    
    print("")
    
def s_folds_cross_validation(lr):
    print("***** NOT IMPLEMENTED *****\n")
    
def locally_weighted(lr, training, validation):
    print("***** NOT IMPLEMENTED *****\n")
    
    # TODO: Clean up this duplicate code
    # X should be features Temp of Water and Length of Fish because we want to predict age
    training_x = training[:,-2:]
    validation_x = validation[:,-2:]
    # y should be the actual age values because that is what we are looking to predict
    training_y = training[:,1]
    validation_y = validation[:,1]
    
    
def main():
    lr = LinearRegression()
    
    # Parse file and returning training and validation data
    training,validation = lr.read_data("x06Simple.csv")
    
    closed_form_direct_solution(lr, training, validation)
    s_folds_cross_validation(lr)
    locally_weighted(lr, training, validation)
    

if __name__ == '__main__':
    main()