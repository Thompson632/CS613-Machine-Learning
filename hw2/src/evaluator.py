from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Evaluator():
    
    def compute_precision(self, tp, fp):
        '''
        Computes the percentage of things that were classified as position
        and actually were positive.

        :param tp: True Positive (hit)
        :param fp: False Positive (false alarm)

        :return precision
        '''
        try:
            precision = (tp / (tp + fp))
        except:
            precision = 1
        
        return precision

    def compute_recall(self, tp, fn):
        '''
        Computes the percentage of true positives (sensitivity) correctly identified

        :param tp: True Positive (hit)
        :param fn: False Negative (miss)

        :return recall
        '''
        try:
            recall = (tp / (tp + fn))
        except:
            recall = 1
        
        return recall

    def compute_f_measure(self, precision, recall):
        '''
        Computes the weighted harmonic mean and recall

        :param precision: Percentage of things actually true
        :param recall: Percentage of true positives

        :return f-measure 
        '''
        return (2 * precision * recall) / (precision + recall)

    def evaluate_accuracy(self, actual, predicted):
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
        # True Positive: Hit
        TP = np.sum(np.logical_and(predicted == 1, actual == 1))
        # True Negative: Correct rejection
        TN = np.sum(np.logical_and(predicted == 0, actual == 0))
        # False Positive: False Alarm (Type 1 error)
        FP = np.sum(np.logical_and(predicted == 1, actual == 0))
        # False Negative: Miss (Type 2 error)
        FN = np.sum(np.logical_and(predicted == 0, actual == 1))
        
        TP = 0
        FP = 0
        FN = 0
        
        for i in range(actual.shape[0]):
            if actual[i] == predicted[i] == 1:
                TP +=1
            if predicted[i] == 1 and actual[i] != predicted[i]:
                FP +=1
            if predicted[i] == 0 and actual[i] != predicted[i]:
                FN +=1

        precision = self.compute_precision(TP, FP)
        recall = self.compute_recall(TP, FN)
        f_measure = self.compute_f_measure(precision, recall)
        accuracy = self.evaluate_accuracy(actual, predicted)

        return precision, recall, f_measure, accuracy
    
    def evaluate_prob_threshold(self, actual, predicted):
        precision_scores = []
        recall_scores = []
        
        probability_thresholds = np.linspace(0, 1, num=11)
                
        for p in probability_thresholds:
            y_test_preds = []
            
            for prob in predicted:
                if prob > p:
                    y_test_preds.append(1)
                else:
                    y_test_preds.append(0)
                    
            precision, recall, f_measure, accuracy = self.evaluate_classifier(actual, y_test_preds)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        return precision_scores, recall_scores
    
    def plot_mean_log_loss(self, train_log_loss, valid_log_loss, epochs):
        '''
        Plots the mean log loss as a function of the epoch

        :param train_log_loss: Training data mean log loss
        :param valid_log_loss: Validation data mean log loss
        :param epochs: Number of iterations

        :return none
        '''
        fig = plt.figure(figsize=(8, 6))        
        plt.title("Training vs. Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Mean Log Loss')
        plt.plot([i for i in range(epochs)], train_log_loss, label="Training")
        plt.plot([i for i in range(epochs)], valid_log_loss, label="Validation")
        plt.legend()
        plt.show()

    def plot_precision_recall(self, train_precisions, train_recalls, valid_precisions, valid_recalls):
        '''
        Plots the training vs. validation precision-recall graph.
        
        :param train_precisions: Training precision data
        :param train_recalls: Training recall data
        :param valid_precisions: Validation precision data
        :param valid_recalls: Validation recall data

        :return none
        '''
        fig, ax = plt.subplots()
        ax.plot(train_recalls, train_precisions, label="Training")
        ax.plot(valid_recalls, valid_precisions, label="Validation")

        ax.set_title("Training vs. Validation Precision-Recall")
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')

        ax.set_xlim([0, 1, ])
        ax.xaxis.set_ticks(np.arange(0, 1, 0.1))
        ax.set_ylim([0, 1, ])
        ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
        
        plt.legend()
        plt.show()