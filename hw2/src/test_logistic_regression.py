from logistic_regression import LogisticRegression
import util
from logistic_regression import LogisticRegression
import numpy as np

# Define learning rate and epochs
learning_rate = 0.33
epochs = 100

# Instantiate our LogisticRegression class
lr = LogisticRegression(learning_rate, epochs)

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
train_preds, train_losses, valid_preds, valid_losses = lr.train_logistic_regression_model(x_train_zscored, y_train, x_valid_zscored, y_valid)
print("\nTraining Preds:\n", train_preds)
print("\nTraining Losses:\n", train_losses)
print("\nValidation Preds:\n", valid_preds)
print("\nValidation Losses:\n", valid_losses)

# 6. Plot Training Mean Log Loss
#util.plot_mean_log_loss("Training", train_losses, epochs)

# 6. Plot Validation Mean Log Loss
#util.plot_mean_log_loss("Validation", valid_losses, epochs)
