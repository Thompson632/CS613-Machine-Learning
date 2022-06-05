import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    def compute_classification_error_types(self, y, y_hat):
        '''
        Computes the classification error types between the actual and predicted data.

        :param y: The actual data
        :param y_hat: The predicted data

        :return the true positive count
        :return the true negative count (future)
        :return the false positive count
        :return the false negative count
        '''
        # True Positive: Hit
        TP = 0
        # True Negative: Correct rejection
        TN = 0
        # False Positive: False Alarm (Type 1 error)
        FP = 0
        # False Negative: Miss (Type 2 error)
        FN = 0

        num_observations = np.shape(y)[0]

        for i in range(num_observations):
            y_val = y[i]
            y_hat_val = y_hat[i]

            # Evaluate True Positive
            if y_val == y_hat_val == 1:
                TP = TP + 1

            # Evaluate True Negative
            if y_val == y_hat_val == 0:
                TN = TN + 1

            # Evaluate False Positive
            if y_val == 0 and y_hat_val == 1:
                FP = FP + 1

            # Evaluate False Negative
            if y_val == 1 and y_hat_val == 0:
                FN = FN + 1

        return TP, TN, FP, FN

    def evaluate_y_hat_with_threshold(self, y_hat, threshold):
        '''
        Iterates through the y_hat values sets the values to 1 if there
        value is greater than the threshold (0.5) provided or to 0
        if the value is less than the threshold provided.

        :param y_hat: The predicted values
        :param threshold: The static threshold value

        :return the predictions evaluated with a threshold value
        '''
        y_hat_with_threshold = []

        # Add threshold for computing the classifiers
        # If the current y_hat_val is greater than or equal to the threshold, set to 1. Otherwise, set to 0.
        y_hat_t = [1 if y_hat_val >= threshold else 0 for y_hat_val in y_hat]
        y_hat_with_threshold.append(y_hat_t)

        # Flatten to one dimension
        y_hat_with_threshold = np.array(y_hat_with_threshold).flatten()
        return y_hat_with_threshold

    def compute_precision(self, tp, fp):
        '''
        Computes the percentage of things that were classified as position
        and actually were positive.

        :param tp: True Positive (hit)
        :param fp: False Positive (false alarm)

        :return precision
        '''
        # Wrap in try-catch incase of division by zero
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
        # Wrap in try-catch incase of division by zero
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

    def evaluate_accuracy(self, y, y_hat):
        '''
        Evaluates the accuracy of our actual and predicted data

        :param y: The actual data
        :param y_hat: The predicted data

        :return accuracy
        '''
        num_observations = np.shape(y)[0]
        print(np.sum(y == y_hat))
        return (1 / num_observations) * np.sum(y == y_hat)

    def compute_precision_and_recall(self, y, y_hat):
        '''
        Computes the precision and recall by first calculating the 
        actual and non-thresholded predicted classification errors. 
        Then computes the precision and recall using the
        true positive, false positive, and false negative counts.

        :param y: The actual data
        :param y_hat: The predicted data

        :return precision
        :return recall
        '''
        # Compute the classification errors
        TP, TN, FP, FN = self.compute_classification_error_types(y, y_hat)
        
        print("TP:", TP)
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)

        precision = self.compute_precision(TP, FP)
        recall = self.compute_recall(TP, FN)

        return precision, recall

    def evaluate_classifier(self, y, y_hat):
        '''
        Evaluates our classifier by first evaluating y_hat with a threshold
        of 0.5, then we calculate the classification error, and finally,
        we calculate the precision, recall, f-measure, and accuracy.

        :param y: The actual data
        :param y_hat: The predicted data

        :return precision
        :return recall
        :return f-measure
        :return accuracy
        '''
        # Evaluate y_hat with a threshold value of 0.5
        # y_hat_with_threshold = self.evaluate_y_hat_with_threshold(y_hat, 0.5)

        # Calculate the precision and recall for the current increment
        precision, recall = self.compute_precision_and_recall(
            y, y_hat)

        f_measure = self.compute_f_measure(precision, recall)
        accuracy = self.evaluate_accuracy(y, y_hat)

        return precision, recall, f_measure, accuracy

    def evaluate_precision_recall_with_threshold(self, y, y_hat, threshold_start, threshold_end, increments):
        '''
        Evaluates the precision and recall values for each increment as provided.

        :param y: The actual data
        :param y_hat: The predicted data
        :param threshold_start: The starting point of the threshold
        :param threshold_end: The ending point of the threshold
        :param increments: Number of increments between threshold start and end

        :return the list of precisions
        :return the list of recalls
        '''
        precision_list = []
        recall_list = []

        threshold_increments = np.linspace(
            threshold_start, threshold_end, num=increments)

        # For each of the increments (ex. 0.1, 0.2,..., 1.0)
        for increment in threshold_increments:
            y_hat_thresholds = []

            # If the current y_hat_val is greater than or equal to the increment, append 1. Otherwise, add 0.
            [y_hat_thresholds.append(1) if y_hat_val >= increment else y_hat_thresholds.append(
                0) for y_hat_val in y_hat]

            # Recalculate the precision and recall for the current increment
            precision, recall = self.compute_precision_and_recall(
                y, y_hat_thresholds)

            # Add recalculated precision and recall to our lists
            precision_list.append(precision)
            recall_list.append(recall)

        return precision_list, recall_list

    def compute_confusion_matrix(self, y, y_hat, classes):
        '''
        Computes the confusion matrix for our multi-class classifier
        by creating a confusion matrix of size class x class. 

        :param y: Actual values
        :param y_hat: Predicted values
        :param classes: Unique classes

        :return the confusion matrix of the number of classes
        '''
        # Number of classes
        num_classes = len(classes)

        # Create our confusion matrix classes x classes
        confusion_matrix = np.zeros((num_classes, num_classes))

        # For each row...
        for i in range(num_classes):
            # For each column...
            for j in range(num_classes):
                y_true = (y == classes[i])
                y_hat_true = (y_hat == classes[j])
                confusion_matrix[i, j] = np.sum(y_true & y_hat_true)

        return confusion_matrix

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
        plt.plot([i for i in range(epochs)],
                 valid_log_loss, label="Validation")
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