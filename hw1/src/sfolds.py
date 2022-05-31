from random import randrange
import data_util
from evaluator import Evaluator
from closed_form import LinearRegressionClosedForm


class LinearRegressionSFolds:
    def __init__(self, folds, print_weights=False):
        '''
        Constructor that initializes our folds to be used when performing
        cross validation and our Evaluator class to be used when computing
        metrics.
        
        :param folds: The different folds for cross validation
        :param print_weights: Flag to print the calculated weights
        
        :return none
        '''
        self.folds = folds
        self.print_weights = print_weights
        self.eval = Evaluator()
    
    def fit(self, data):
        '''
        Trains our s-folds cross valid linear regression model.
        
        :param data: The data set
        
        :return none
        '''
        N = len(data)
        
        for fold in self.folds:
            print("\nS-Fold:", fold)
            self.cross_validation(data, fold)
        
        print("\nS-Fold (N):", N)
        self.cross_validation(data, N)
    
    def cross_validation(self, data, S):
        '''
        Computes our root mean squared error over the course of 20
        different runs with dividing the data into different S-Fold.

        :param data: Dataset
        :param S: Number of folds or different parts to divide
        our data into

        :return None
        '''
        rmse_list = []

        for i in range(1, 21):
            shuffled_data = data_util.shuffle_data(data, i)

            squared_errors_list = []

            for j in range(1, S + 1):
                training, validation = self.split_data_by_fold(
                    shuffled_data, S, j - 1)
                
                training_arr = data_util.convert_list_to_numpy_array(training)
                validation_arr = data_util.convert_list_to_numpy_array(validation)
                
                x_train, y_train = data_util.get_features_actuals(training_arr)
                x_valid, y_valid = data_util.get_features_actuals(validation_arr)
                
                x_train_bias = data_util.add_bias_feature(x_train)
                x_valid_bias = data_util.add_bias_feature(x_valid)
                
                model = LinearRegressionClosedForm(print_weights=self.print_weights)
                model.fit(x_train_bias, y_train)
                
                valid_preds = model.predict(x_valid_bias)

                squared_errors = self.eval.compute_se(y_valid, valid_preds)
                squared_errors_list.append(squared_errors)

            rmse = self.eval.compute_rmse_from_squared_errors(squared_errors_list)
            rmse_list.append(rmse)

        self.eval.compute_mean_std_of_rmse(rmse_list)

    def split_data_by_fold(self, data, num_folds, i):
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
        data_copy = list(data)
        training_data = list()
        num_observations_per_fold = int(len(data) / num_folds)

        for _ in range(num_folds):
            current_fold = list()

            while len(current_fold) < num_observations_per_fold:
                random_index = randrange(len(data_copy))
                current_fold.append(data_copy.pop(random_index))
                
            training_data.append(current_fold)

        validation_data = list()
        validation_fold = training_data.pop(i)
        validation_data.append(validation_fold)

        return training_data, validation_data