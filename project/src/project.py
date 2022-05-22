import math_util
import data_util
from evaluator import Evaluator
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree

def generate_game_fields(fields, target):
    away_columns = []
    home_columns = []

    for field in fields:
        away_columns.append("away_"+field)
        home_columns.append("home_" + field)
    result = away_columns + home_columns
    result.append(target)
    return result


def load_data(filename, columns):
    data = data_util.load_data(filename, columns=columns)
    data = data_util.shuffle_data(data, 0)
    print("Data Shape:", data.shape)

    training, validation = data_util.get_train_valid_data(data)
    print("Training Shape:", training.shape)
    print("Validation Shape:", validation.shape)

    x_train, y_train = data_util.get_features_actuals(training)
    print("Training x_train Shape:", x_train.shape)
    print("Training y_train Shape:", y_train.shape)

    x_valid, y_valid = data_util.get_features_actuals(validation)
    print("Validation x_train Shape:", x_valid.shape)
    print("Validation y_train Shape:", y_valid.shape)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    x_train_bias = data_util.add_bias_feature(x_train_zscored)
    print("Training x_train_bias Shape:", x_train_bias.shape)
    x_valid_bias = data_util.add_bias_feature(x_valid_zscored)
    print("Validation x_valid_bias Shape:", x_valid_bias.shape)

    return x_train_bias, y_train, x_valid_bias, y_valid

def logistic_regression(filename, learning_rate, epochs, stability, columns):
    print("\nLOGISTIC REGRESSION CLASSIFIER:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename, columns)

    eval = Evaluator()
    model = LogisticRegression(learning_rate, epochs, stability)
    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    _, _ = model.train_model(X_train, y_train, X_valid, y_valid)

    train_preds = model.evaluate_model(X_train)
    valid_preds = model.evaluate_model(X_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)

def decision_tree(filename, min_observation_split, min_information_gain, columns):
    print("\nDECISION TREE CLASSIFIER:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename, columns)

    eval = Evaluator()
    model = DecisionTree(min_observation_split=min_observation_split,
                         min_information_gain=min_information_gain)

    print("\nMin Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    model.train_model(X_train, y_train)

    train_preds = model.evaluate_model(X_train)
    valid_preds = model.evaluate_model(X_valid)

    train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
        y_train, train_preds)
    print("\nTraining Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F-Measure:", train_f_measure)
    print("Training Accuracy:", train_accuracy)

    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)
    print("\nValidation Precision:", valid_precision)
    print("Validation Recall:", valid_recall)
    print("Validation F-Measure:", valid_f_measure)
    print("Validation Accuracy:", valid_accuracy)


# Define our most informative features based on our knowledge
fields = ['offensive_rating', 'effective_field_goal_percentage', 'total_rebound_percentage', 'free_throw_attempt_rate',
          'free_throw_percentage', 'three_point_attempt_rate', 'three_point_field_goal_percentage', 'turnover_percentage', 'true_shooting_percentage']

# Prepend away and home to each field
game_fields = generate_game_fields(fields,"home_win")

logistic_regression(filename="games.csv", learning_rate=0.1,
                    epochs=1000, stability=10e-7, columns=game_fields)
decision_tree(filename="games.csv", min_observation_split=2,
              min_information_gain=0, columns=game_fields)
