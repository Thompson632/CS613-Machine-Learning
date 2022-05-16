import numpy as np


class Evaluator:
    def evaluate_accuracy(self, y, y_hat):
        '''
        Evaluates the accuracy of our actual and predicted data

        :param y: The actual data
        :param y_hat: The predicted data

        :return accuracy
        '''
        num_observations = np.shape(y)[0]
        return (1 / num_observations) * np.sum(y == y_hat)