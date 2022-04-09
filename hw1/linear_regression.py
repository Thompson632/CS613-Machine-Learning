import numpy as np
from random import randrange


class LinearRegression:
    def load_data(self, filename, columns):
        return np.loadtxt(filename, delimiter=',', skiprows=1, dtype='int', usecols=columns)

    def shuffle_data(self, data, reseed_val):
        # Seed random number
        np.random.seed(reseed_val)

        # Shuffle the data
        np.random.shuffle(data)
        return data

    def get_features_actuals(self, data):
        X = data[:, -2:]
        y = data[:, 1]
        return X, y

    def get_train_valid_data(self, data):
        # Get the index value of the 2/3 data
        # For this dataset it would be 44 * (2/3) = 29
        training_index = round(len(data) * (2/3))

        # Set our training and validation data sets based on the 2/3 index
        training = data[0:training_index]  # begin-29
        validation = data[training_index:]  # 29-end

        # Return training, validation
        return training, validation

    def compute_closed_form(self, X, y):
        # Bias feature
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        # Calculate the weights for our features and actuals
        weights = self.compute_weights(X, y)

        # calculate y_hat (or prediction) based off the features and weights
        y_hat = self.compute_y_hat(X, weights)

        return y_hat

    def compute_weights(self, X, y):
        # Number of observations
        N = X.shape[0]

        # Convert y from horizontal array to vertical array of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)

        # Error with no inverse for the given validation matrix
        return np.linalg.pinv(np.dot(X.T, X)) * np.dot(X.T, Y)

    def compute_y_hat(self, X, weights):
        return np.dot(X, weights).flatten().astype(int)

    def compute_rmse(self, actual, predicted):
        differences = np.subtract(actual, predicted)
        differences_squared = np.square(differences)
        mse = differences_squared.mean()
        rmse = np.sqrt(mse)
        return rmse

    def compute_mape(self, actual, predicted):
        abs = np.abs((actual - predicted) / actual)
        mape = np.mean(abs) * 100
        return mape

    def compute_s_folds_cross_validation(self, data, S):
        # Create a list of RMSE values to track
        rmse_list = []

        # 2. 20 times does the following:
        # 2(c). Create S folds
        for i in range(1, 21):
            shuffled_data = self.shuffle_data(data, i)

            # 2(d) For i = to 1
            for j in range(1, S + 1):
                # 2(d)(i) Split the data by selecting fold i as validation data and
                # remaining (S - 1) folds as training data
                training, validation = self.split_data_by_s_folds(
                    shuffled_data, S, j - 1)

                # Convert the returned list back to a numpy array to get our training and
                # validation X, y
                training_arr = self.convert_list_to_numpy_array(training)
                validation_arr = self.convert_list_to_numpy_array(validation)

                # 2(d)(ii) Train a linear regression model using the direct solution
                xTrain, yTrain = self.get_features_actuals(training_arr)
                xValid, yValid = self.get_features_actuals(validation_arr)
                
                train_preds = self.compute_closed_form(xTrain, yTrain)                
                valid_preds = self.compute_closed_form(xValid, yValid)
    
                # 2(d)(iii) Compute the squared error for each sample in the current
                # current validation fold
                # 2(e) You should now have N squared errors. Compute the RMSE for these.
                valid_rmse = self.compute_rmse(yValid, valid_preds)
                rmse_list.append(valid_rmse)

        # 3. You should now have RMSE values. Compute the mean and standard
        # deviation of these. The former should give us a better "overall" mean,
        # whereas the latter should give us feel for the variance of the models that
        # were created.
        self.compute_mean_std_of_rmse(rmse_list)

    def split_data_by_s_folds(self, data, num_folds, i):
      # Create a list of the data as a copy
        data_copy = list(data)
        # Create a list for the validation_data
        validation_data = list()
        # Create a list for the split data
        split_data = list()
        # Determine the fold size (or number of observation per fold)
        # In our case, we have 44 observations so the fold size
        # would be the following: 44 (length of data set) / 4 (num_folds) = 11
        num_observations_per_fold = int(len(data) / num_folds)

        # Iterate over the number of folds we need to split the data on
        for _ in range(num_folds):
            # Create a list for the current fold
            current_fold = list()

            # Now we want to build our list for the current fold by iterating over the
            # data until we reach the number of observations per fold we desire
            while len(current_fold) < num_observations_per_fold:
                # Create a random number within the range of our data
                random_index = randrange(len(data_copy))
                # Get that data stored at the random number index and at it to our
                # current fold
                current_fold.append(data_copy.pop(random_index))

            # Add the fold we just build to a list that will be returned as
            # the data split data
            split_data.append(current_fold)

        # Get validation data by taking the i fold from the split_data
        validation_fold = list()
        validation_fold = split_data.pop(i)
        validation_data.append(validation_fold)

        return split_data, validation_data

    def convert_list_to_numpy_array(self, list):
      # Temp list to remove array complexity
        tmp_list = []
        # List of lists of arrays is size S-Folds - 1
        for list_of_arr in list:
            # List of Array is the size of N-observations
            for arr in list_of_arr:
                # Array is of size three because there are three columns in our dataset
                tmp_list.append(arr)

        # Convert the tmp list we created as a numpy array
        return np.asarray(tmp_list)

    def compute_mean_std_of_rmse(self, rmse_list):
        rmse_arr = np.asarray(rmse_list)
        mean_rmse = rmse_arr.mean()
        std_rmse = rmse_arr.std()

        print("Mean RMSE:", mean_rmse)
        print("Standard Deviation RMSE:", std_rmse)

    def compute_locally_weighted(self, X, y, k):
        # Number of observations
        N = np.shape(X)[0]

        # Bias feature
        X = np.append(X, np.ones((N, 1)), axis=1)

        # Create y_hat array based on the size of observations
        y_hat_array = np.zeros(N)

        # For all observations, calculate the diagonal weights matrix, calculate all the weights matrix, and finally, calculate y_hat
        for i in range(N):
            weights = self.compute_weights_from_diagonal_weights(
                X, X[i], y, k, N)
            y_hat = self.compute_y_hat(X[i], weights)
            y_hat_array[i] = y_hat

        return y_hat_array

    def compute_weights_from_diagonal_weights(self, X, x, y, k, N):
        # Convert y from horizontal array to vertical array of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)

        diagonal_weights_matrix = self.calculate_diagonal_weights_matrix_for_single_observation(
            X, x, k, N)
        return np.linalg.pinv(X.T * (diagonal_weights_matrix * X)) * (X.T * diagonal_weights_matrix * Y)

    def calculate_diagonal_weights_matrix_for_single_observation(self, X, x, k, N):
        # Create 2D-array with ones on the diagonal and zeros everywhere for the single observation
        diagonal_array = np.eye(N)
        # Convert array to a matrix
        diagonal_weights_matrix = np.mat(diagonal_array)

        # Calculate k_squared for the Gaussian Similarity Metric calculation
        k_squared = -(k**2)

        # Iterate over the the number of observation
        for i in range(N):
            # Using i,i at x,y in matrix to set the returned value to the diagonal
            diagonal_weights_matrix[i, i] = self.compute_weight_using_gaussian_similarity(
                X[i], x, k_squared)

        return diagonal_weights_matrix

    def compute_weight_using_gaussian_similarity(self, xi, x, k_squared):
        distance = self.compute_manhattan_distance(xi, x)
        return np.exp(np.dot(distance, distance.T) / k_squared)

    def compute_manhattan_distance(self, xi, x):
        return sum(abs(val1 - val2) for val1, val2 in zip(xi, x))
