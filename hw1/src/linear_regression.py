import numpy as np
from random import randrange
import math

class LinearRegression:
    def load_data(self, filename, columns):
        '''
        Reads in a csv file ignoring the first rows (headers)
        and only returning the columns as specified in the parameter.

        :param filename: The file name specified at runtime
        :param columns: The columns of data to retrieve

        :return numpy ndarray
        '''
        return np.loadtxt(filename, delimiter=',', skiprows=1, dtype='int', usecols=columns)

    def shuffle_data(self, data, reseed_val):
        '''
        Shuffles and returns the data.

        :param: data: The data that is to be shuffled
        :param reseed_val: The value to be used when reseeding the random 
        number generator

        :return data shuffled
        '''
        np.random.seed(reseed_val)
        np.random.shuffle(data)
        return data

    def get_features_actuals(self, data):
        '''
        Helper function that retrieves the features and
        actual values from the data. Currently it retrieves 
        the features as the last two columns (Temp of Water 
        and Length of Fish) in the ndarray and the first column
        (Age) as the actuals.
        and the

        :param data: Data to be searched

        :return the features (X) and actuals (y)
        '''
        X = data[:, -2:]
        y = data[:, 0]
        return X, y

    def get_train_valid_data(self, data):
        '''
        Gets the training and validation data by first calculating the index of the first 2/3 of data. 
        Once that is found, we set the training data from 0 to the index of the first 2/3 of data 
        and the validation data from the index to the end of the data.

        :param data: Data to be searched

        :return training, validation data
        '''
        # Get the index value of the 2/3 data
        training_index = round(len(data) * (2/3))

        # Set our training and validation data sets based on the 2/3 index
        training = data[:training_index]  # begin-training_index
        validation = data[training_index:]  # training_index-end

        return training, validation

    def get_bias_observations_reshape_y(self, X, y):
        '''
        Helper function to remove duplicate code by returning
        the number of observation, adding a bias feature, reshaping
        our actuals array to match the features, and converting it 
        to a matrix.

        :param X: Features
        :param y: Actuals

        :return N: Number of observations
        :return X: Features with a bias
        :return Y: Actuals as a matrix
        '''
        # Number of observations
        N = np.shape(X)[0]

        # Bias feature
        ones = np.ones((N, 1))
        X = np.concatenate((ones, X), axis=1)

        # Convert y from horizontal array to vertical matrix of height N (or observations)
        y = y.reshape(N, 1)
        Y = np.mat(y)

        return N, X, Y

    def compute_model_using_direct_solution_and_apply(self, training, validation, should_print_weights):
        '''
        Computes the linear regression model using the clsoed form direct solution and returns
        the actual / predicted training and validation data.

        :param training: Training datasaet
        :param validation: Validation dataset
        :param should_prints_weights: Flag to print the weights

        :return yTrain: Actual training data
        :return train_preds: Predicted training data
        :return yValid: Actual validation data
        :return valid_preds: Predicted validation data
        '''
        # Get Training X (features) and y (actuals)
        xTrain, yTrain = self.get_features_actuals(training)

        # Add bias to xTrain
        N, XTrainBias, y_mat = self.get_bias_observations_reshape_y(
            xTrain, yTrain)

        # Compute the linear regression model using the direct solution.
        learned_model = self.compute_weights(XTrainBias, y_mat)
        if should_print_weights:
            print("Learned Model:\n", learned_model, "\n")

        # Apply learned model to the training samples.
        train_preds = self.compute_y_hat(XTrainBias, learned_model)

        # Get Validation X (features) and y (actuals)
        xValid, yValid = self.get_features_actuals(validation)
        # Add bias to xValid to match shape
        N, xValidBias, y_mat = self.get_bias_observations_reshape_y(
            xValid, yValid)

        # Apply the learned model to the validation samples.
        valid_preds = self.compute_y_hat(xValidBias, learned_model)

        # Return the actual / predicted training and actual / prediction validation data
        return yTrain, train_preds, yValid, valid_preds

    def compute_weights(self, X, y_mat):
        '''
        Computes the weights based on the features and actuals

        :param X: Features
        :param y_mat: Actuals

        :return the weights (or coefficients)
        '''
        xt_x = np.dot(X.T, X)
        xt_x_inv = np.linalg.pinv(xt_x)
        xt_x_inv_xt = np.dot(xt_x_inv, X.T)
        xt_x_inv_xt_y = np.dot(xt_x_inv_xt, y_mat)
        return xt_x_inv_xt_y

    def compute_y_hat(self, X, weights):
        '''
        Computes y_hat by taking the dot product of our
        features and weights

        :param X: Features
        :param weights: Weights

        :return the predictions
        '''
        return np.dot(X, weights).flatten()

    def compute_se(self, actual, predictions):
        '''
        Computes the squared error based on the actual data
        and our trained model predictions

        :param actual: Actual real data
        :param predictions: Training model preditions

        :return the squared error
        '''
        differences = np.subtract(actual, predictions)
        differences_squared = np.square(differences)
        return differences_squared

    def compute_mse(self, actual, predictions):
        '''
        Computes the mean squared error based on the actual data
        and our trained models predictions.

        :param actual: Actual real data
        :param predictions: Training model predictions

        :return the mean squared error
        '''
        differences_squared = self.compute_se(actual, predictions)
        mse = differences_squared.mean()
        return mse

    def compute_rmse(self, actual, predicted):
        '''
        Computes the root mean squared error based on the actual
        data and our trained models predictions.

        :param actual: Actual real data
        :param predictions: Training model predictions

        :return the root mean squared error        
        '''
        mse = self.compute_mse(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse

    def compute_rmse_from_squared_errors(self, squared_errors_list):
        '''
        Computes the root mean squared error based on a list of squared
        errors calculated while iterating through our S-Folds.

        :param squared_errors_list: Squared errors

        :return the root mean squared error
        '''
        mse = np.mean(squared_errors_list)
        rmse = np.sqrt(mse)
        return rmse

    def compute_mape(self, actual, predicted):
        '''
        Computes the mean absolute percent error based on the actual
        data and our trained models predictions.

        :param actual: Actual real data
        :param predictions: Training model predictions

        :return the mean absolute percent error
        '''
        abs = np.abs((np.subtract(actual, predicted)) / actual)
        mape = np.mean(abs) * 100
        return mape

    def compute_s_folds_cross_validation(self, data, S):
        '''
        Computes our root mean squared error over the course of 20
        different runs with dividing the data into different S-Fold.

        :param data: Dataset
        :param S: Number of folds or different parts to divide
        our data into

        :return None
        '''
        # Create a list of RMSE values to track
        rmse_list = []

        # 2. 20 times does the following:
        for i in range(1, 21):
            # 2(a) Seeds the random generator to the current run (out of 20)
            # 2(b) Shuffles the rows of data
            shuffled_data = self.shuffle_data(data, i)

            # List of squared errors
            squared_errors_list = []

            # 2(c) Create S folds
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
                yTrain, train_preds, yValid, valid_preds = self.compute_model_using_direct_solution_and_apply(
                    training_arr, validation_arr, False)

                # 2(d)(iii) Compute the squared error for each sample in the current
                # current validation fold
                squared_errors = self.compute_se(yValid, valid_preds)
                squared_errors_list.append(squared_errors)

            # 2(e) You should now have N squared errors. Compute the RMSE for these.
            rmse = self.compute_rmse_from_squared_errors(squared_errors_list)
            rmse_list.append(rmse)

        # 3. You should now have RMSE values. Compute the mean and standard
        # deviation of these. The former should give us a better "overall" mean,
        # whereas the latter should give us feel for the variance of the models that
        # were created.
        self.compute_mean_std_of_rmse(rmse_list)

    def split_data_by_s_folds(self, data, num_folds, i):
        '''
        Computes our training and validation data by splitting
        the data into different folds (or parts). Once the data
        is evenly split into different folds, we take the first
        data fold of data as our validation data set and leave the
        other three to be used as our training data.

        :param data: The dataset
        :param num_fold: The number of folds (or parts) to divide our data into
        :param i: Current fold we are creating

        :return training and validation data
        '''
        # Create a list of the data as a copy
        data_copy = list(data)
        # Create a list for the training data
        training_data = list()
        # Determine the fold size (or number of observation per fold)
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
                # Pop the data stored at the random number index and add it to our current fold
                current_fold.append(data_copy.pop(random_index))

            # Add the fold we just build to a list that will be returned as
            # the data split data
            training_data.append(current_fold)

        # Get validation data by taking the i fold from the split_data
        # Create a list for the validation_data
        validation_data = list()
        validation_fold = training_data.pop(i)
        validation_data.append(validation_fold)

        return training_data, validation_data

    def convert_list_to_numpy_array(self, list):
        '''
        Helper function that converts the provided list of data
        into a numpy array. We need to do this because we are 
        provided a list of lists of arrays. We want the array
        data aka our training or validation data.

        :param list: The list to be iterated over

        :return the numpy array of just data for the current fold
        '''
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
        '''
        Computes the mean and standard deviation of our
        calculated root mean squared errors once we
        have completed the S-Folds Cross Validation.

        :param rmse_list: List of root mean squared errors

        :return None
        '''
        rmse_arr = np.asarray(rmse_list)
        mean_rmse = rmse_arr.mean()
        std_rmse = rmse_arr.std()

        print("Mean RMSE:", mean_rmse)
        print("Standard Deviation RMSE:", std_rmse)

    def compute_locally_weighted(self, training, validation, k):
        '''
        Computes the locally weighted regression by iterating over each
        validation sample, computing the distances between the current
        validation sample and all the training data to generate our weights matrix.
        Once that is generated, we evaluate our model against the current
        validation sample.

        :param training: Training dataset
        :param validation: Validation dataset
        :param k: Value used in gaussian similarity function

        :return actual and predicted validation values
        '''
        xTrain, yTrain = self.get_features_actuals(training)
        xValid, yValid = self.get_features_actuals(validation)

        # Training Observations, Features with Bias, and Actuals as Matrix
        num_train_observations, xTrainBias, yTrainMat = self.get_bias_observations_reshape_y(
            xTrain, yTrain)

        # Validation Observations, Features with Bias, and Actuals as Matrix
        num_valid_observations, xValidBias, yValidMat = self.get_bias_observations_reshape_y(
            xValid, yValid)

        # Predictions Array Based on the Number of Validation Observations
        y_hat_array = np.zeros(num_valid_observations)

        # 4. Then for each validation sample
        for i in range(num_valid_observations):
            # Get the current validation sample
            validation_sample = xValidBias[i]

            # 4(b). Use the weight matrix to compute a local model via the direct method
            weights = self.compute_local_weights(
                xTrainBias, validation_sample, yTrainMat, k, num_train_observations)

            # Evaluate the validation sample using the local model
            y_hat = self.compute_y_hat(validation_sample, weights)
            y_hat_array[i] = y_hat

        return yValid, y_hat_array

    def compute_local_weights(self, X, x, Y, k, N):
        '''
        Computes the local weights for the current observation. 
        We first need to compute the local diaganal weights matrix. 
        Once we have that computed, we calculate the weights matrix
        for the current observation and return the result.

        :param X: Training Features
        :param x: Validation sample
        :param Y: Training Actuals
        :param k: Value used in gaussian similarity metric
        :param N: Number of observations in the training dataset

        return weights matrix for the current observation
        '''
        # Compute the necessary distances of the validation sample to the training
        # data in order to establish your weight matrix
        diagonal_weights_mat = self.compute_local_diagonal_weights(
            X, x, k, N)

        xt_d = np.dot(X.T, diagonal_weights_mat)
        xt_dx = np.dot(xt_d, X)
        xt_dx_inv = np.linalg.pinv(xt_dx)
        xt_dy = np.dot(xt_d, Y)
        return np.dot(xt_dx_inv, xt_dy)

    def compute_local_diagonal_weights(self, X, x, k, N):
        '''
        Computes the local diagonal weights matrix by iterating
        over the number of observations, computing the gaussian similarity metric, 
        and adding the result to our diagonal matrix.

        :param X: Training Data
        :param x: Validation sample
        :param k: Value used in gaussian similarity metric
        :param N: Number of observations in the training dataset

        :return local diagonal weights matrix
        '''
        # Create 2D-array with ones on the diagonal and zeros everywhere for the single observation
        diagonal_array = np.eye(N)
        # Convert array to a matrix
        diagonal_weights_mat = np.mat(diagonal_array)

        # Calculate k_squared for the Gaussian Similarity Metric calculation
        k_squared = (k**2)

        for i in range(N):
            # Computing distances of the training data to the validation sample
            diagonal_weights_mat[i, i] = self.compute_gaussian_similarity_metric(
                X[i], x, k_squared)

        return diagonal_weights_mat

    def compute_gaussian_similarity_metric(self, xi, x, k_squared):
        '''
        Computes the gaussian similarity metric by calculating the distance
        using the manhattan distance formula for the observations.

        :param xi: Current observation in our diagonal weights loop
        :param x: Current validation sample we originally started with
        :param k_squared: Use in formula

        :return gaussian similarity metric computation
        '''
        distance = self.compute_manhattan_distance(xi, x)
        return math.e ** -(distance**2 / k_squared)

    def compute_manhattan_distance(self, xi, x):
        '''
        Computes the distance between two observations

        :param xi: Current observation in our diagonal weights loop
        :param x: Current observation we originally started with

        :return distance between observations
        '''
        return sum(abs(val1 - val2) for val1, val2 in zip(xi, x))