import numpy as np


class Evaluator:
    def compute_se(self, y, y_hat):
        '''
        Computes the squared error based on the actual data
        and our trained model predictions

        :param y: Actual real data
        :param y_hat: Training model preditions

        :return the squared error
        '''
        differences = np.subtract(y, y_hat)
        differences_squared = np.square(differences)
        return differences_squared

    def compute_mse(self, y, y_hat):
        '''
        Computes the mean squared error based on the actual data
        and our trained models predictions.

        :param y: Actual real data
        :param y_hat: Training model predictions

        :return the mean squared error
        '''
        differences_squared = self.compute_se(y, y_hat)
        mse = differences_squared.mean()
        return mse

    def compute_rmse(self, y, y_hat):
        '''
        Computes the root mean squared error based on the actual
        data and our trained models predictions.

        :param y: Actual real data
        :param y_hat: Training model predictions

        :return the root mean squared error        
        '''
        mse = self.compute_mse(y, y_hat)
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

    def compute_mape(self, actual, y_hat):
        '''
        Computes the mean absolute percent error based on the actual
        data and our trained models predictions.

        :param actual: Actual real data
        :param predictions: Training model predictions

        :return the mean absolute percent error
        '''
        abs = np.abs((np.subtract(actual, y_hat)) / actual)
        mape = np.mean(abs) * 100
        return mape

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

    def compute_smape(self, y, y_hat):
        '''
        Computes the symmetric mean absolute percentage error based on the 
        actual data and our trained models predictions

        :param y: Actual real data
        :param y_hat: Training model predictions

        :return the symmetric mean absolute percent error
        '''
        return 1/len(y) * np.sum(2 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat))*100)