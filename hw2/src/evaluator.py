import numpy as np
import matplotlib.pyplot as plt

class Evaluator():
    
    def compute_precision(self, tp, fp):
        '''
        Computes the percentage of things that were classified as position
        and actually were positive.

        :param tp: True Positive (hit)
        :param fp: False Positive (false alarm)

        :return precision
        '''
        return (tp / (tp + fp))

    def compute_recall(self, tp, fn):
        '''
        Computes the percentage of true positives (sensitivity) correctly identified

        :param tp: True Positive (hit)
        :param fn: False Negative (miss)

        :return recall
        '''
        return (tp / (tp + fn))

    def compute_f_measure(self, precision, recall):
        '''
        Computes the weighted harmonic mean and recall

        :param precision: Percentage of things actually true
        :param recall: Percentage of true positives

        :return f-measure 
        '''
        return (2 * precision * recall) / (precision + recall)

    def evaulate_accuracy(self, actual, predicted):
        '''
        Evaluates the accuracy of our actual and predicted data

        :param actual: The actual data
        :param predicted: The predicted data

        :return accuracy
        '''
        num_observations = np.shape(actual)[0]
        return (1 / num_observations) * np.sum(actual == predicted)

    def evaluate_classifier(self, actual, predicted):
        '''
        Evaluates our classifier by calculating the true positive,
        true negative, false positive, and false negative. Once those
        are calculated, the precision, recall, f-measure, and accuracy
        are calculated and returned

        :param actual: The actual data
        :param predicted: The predicted data

        :return precision
        :return recall
        :return f-measure
        :return accuracy
        '''
        predicted = np.array(predicted)
        # True Positive: Hit
        TP = np.sum(np.logical_and(predicted == 1, actual == 1))
        # True Negative: Correct rejection
        TN = np.sum(np.logical_and(predicted == 0, actual == 0))
        # False Positive: False Alarm (Type 1 error)
        FP = np.sum(np.logical_and(predicted == 1, actual == 0))
        # False Negative: Miss (Type 2 error)
        FN = np.sum(np.logical_and(predicted == 0, actual == 1))

        precision = self.compute_precision(TP, FP)
        recall = self.compute_recall(TP, FN)
        f_measure = self.compute_f_measure(precision, recall)
        accuracy = self.evaulate_accuracy(actual, predicted)

        return precision, recall, f_measure, accuracy
    
    def plot_mean_log_loss(self, type, mean_log_loss, epochs):
        '''
        Plots the mean log loss as a function of the epoch

        :param type: Type of data
        :param mean_log_loss: Mean of the log loss
        :param epochs: Number of iterations

        :return none
        '''
        fig = plt.figure(figsize=(8, 6))
        plt.plot([i for i in range(epochs)], mean_log_loss, 'r-')
        plt.title(type)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Log Loss')
        plt.show()


    def plot_precision_recall(self, type, precision, recall):
        '''
        Plots the precision recall graph.

        :param type: Type of data
        :param precision: Data precision
        :param recall: Data recall

        :return none
        '''
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')

        ax.set_title(type)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        ax.set_xlim([0, 1, ])
        ax.xaxis.set_ticks(np.arange(0, 1, 0.1))
        ax.set_ylim([0, 1, ])
        ax.yaxis.set_ticks(np.arange(0, 1, 0.1))

        plt.show()