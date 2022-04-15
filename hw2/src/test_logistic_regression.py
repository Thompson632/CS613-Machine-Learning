import util
from logistic_regression import LogisticRegression
import numpy as np
from evaluator import Evaluator

# Define learning rate and epochs
learning_rate = 0.33
epochs = 100

# Instantiate our LogisticRegression class
lr = LogisticRegression(learning_rate, epochs)
# Instantiate our Evaluator class
eval = Evaluator()

# Load data
data = util.load_data("spambase.data")
# Shuffle data
data = util.shuffle_data(data, 0)

# Split and get training and validation
training, validation = util.get_train_valid_data(data)

# Get training features and actuals
x_train, y_train = util.get_features_actuals(training)
# Get validation features and actuals
x_valid, y_valid = util.get_features_actuals(validation)

# Get mean and std of training data
means, stds = util.compute_training_mean_std(x_train)

# 4. Z-Score our training data with the means and std
x_train_zscored = util.z_score_data(x_train, means, stds)

# 4. Z-Score our validation data with the means and std
x_valid_zscored = util.z_score_data(x_valid, means, stds)

# 5. Train Logistic Regression Model
train_preds, train_losses, valid_preds, valid_losses = lr.train_logistic_regression_model(
    x_train_zscored, y_train, x_valid_zscored, y_valid)
#print("\nTraining Preds:\n", train_preds)
#print("\nTraining Losses:\n", train_losses)
#print("\nValidation Preds:\n", valid_preds)
#print("\nValidation Losses:\n", valid_losses)

# 6. Plot Training Mean Log Loss
#eval.plot_mean_log_loss("Training", train_losses, epochs)

# 6. Plot Validation Mean Log Loss
#eval.plot_mean_log_loss("Validation", valid_losses, epochs)

# 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the training data
train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(y_train, train_preds)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F-Measure:", train_f_measure)
print("Training Accuracy: ", train_accuracy)

print("")

# 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the validation data
valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(y_valid, valid_preds)
print("Validation Precision:", valid_precision)
print("Validation Recall:", valid_recall)
print("Validation F-Measure:", valid_f_measure)
print("Validation Accuracy: ", valid_accuracy)

# 8. Plot a Precision-Recall Graph
eval.plot_precision_recall("Training", train_precision, train_recall)