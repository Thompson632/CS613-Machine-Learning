import util
from logistic_regression import LogisticRegression
from evaluator import Evaluator

def binary_logistic_regression(learning_rate, epochs):
    # Define stability
    stability = 10e-7

    # Instantiate our LogisticRegression class
    lr = LogisticRegression(learning_rate, epochs, stability)
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
    train_losses, valid_losses, weights, bias = lr.train_logistic_regression_model(
        x_train_zscored, y_train, x_valid_zscored, y_valid)
    train_preds = lr.evaluate_model(x_train_zscored, weights, bias)
    valid_preds = lr.evaluate_model(x_valid_zscored, weights, bias)

    #print("Training Actuals:", y_train)
    #print("Training Preds:", train_preds)
    #print("\nValidation Actuals:", y_valid)
    #print("Validation Preds:", valid_preds)

    # 6. Plot Training and Validation Loss
    eval.plot_mean_log_loss(train_losses, valid_losses, epochs)

    # 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the training data
    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    # 7. Compute the precision, recall, and f-measure and accuracy of the learned model on the validation data
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)

    # 8. Plot a Precision-Recall Graph
    training_precisions, training_recalls = eval.evaluate_precision_recall_with_threshold(
        y_train, train_preds, 0, 1, 11)
    valid_precisions, valid_recalls = eval.evaluate_precision_recall_with_threshold(
        y_valid, valid_preds, 0, 1, 11)
    eval.plot_precision_recall(
        training_precisions, training_recalls, valid_precisions, valid_recalls)
    
def multi_class_logistic_regression(learning_rate, epochs):
    # Define stability
    stability = 10e-7

    # Instantiate our LogisticRegression class
    lr = LogisticRegression(learning_rate, epochs, stability)
    # Instantiate our Evaluator class
    eval = Evaluator()

    # Load data
    data = util.load_data("iris.data")
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
    
#binary_logistic_regression(learning_rate=0.1, epochs=1000)

