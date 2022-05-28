import math_util
import data_util
from evaluator import Evaluator
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree
from bracket import Bracket
from random_forest import RandomForest


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

    training, validation = data_util.get_train_valid_data(data)

    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)

    means, stds = math_util.calculate_feature_mean_std(x_train)
    x_train_zscored = math_util.z_score_data(x_train, means, stds)
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    return x_train_zscored, y_train, x_valid_zscored, y_valid


def logistic_regression(filename, learning_rate, epochs, stability, game_fields):
    print("\nLOGISTIC REGRESSION CLASSIFIER:\n")

    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    eval = Evaluator()

    model = LogisticRegression(learning_rate, epochs, stability)
    train_losses, valid_losses = model.fit(
        X_train, y_train, X_valid, y_valid)

    eval.plot_mean_log_loss(train_losses, valid_losses, epochs)

    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

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


def decision_tree(filename, min_observation_split, min_information_gain, game_fields):
    print("\nDECISION TREE CLASSIFIER:\n")

    print("\nMin Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    eval = Evaluator()

    model = DecisionTree()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

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


def random_forest(filename, forest_size, game_fields):
    print("\nRANDOM FOREST CLASSIFIER:\n")

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    eval = Evaluator()

    model = RandomForest(forest_size)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)

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


def bracket_logistic_regression(filename, learning_rate, epochs, stability, fields, game_fields, year):
    print(year, "LOGISTIC REGRESSION BRACKET PREDICTION:\n")

    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    model = LogisticRegression(learning_rate, epochs, stability)
    _, _ = model.fit(
        X_train, y_train, X_valid, y_valid)

    bracket = Bracket(year, model, fields, game_fields)
    bracket.run_bracket()


def bracket_decision_tree(filename, min_observation_split, min_information_gain, fields, game_fields, year):
    print(year, "DECISION TREE BRACKET PREDICTION:\n")

    print("\nMin Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    X_train, y_train, _, _ = load_data(filename, game_fields)

    model = DecisionTree(min_observation_split=min_observation_split,
                         min_information_gain=min_information_gain)
    model.fit(X_train, y_train)

    bracket = Bracket(year, model, fields, game_fields)
    bracket.run_bracket()


# Define our most informative features based on our knowledge
fields = ['offensive_rating', 'effective_field_goal_percentage', 'total_rebound_percentage', 'free_throw_attempt_rate',
          'free_throw_percentage', 'three_point_attempt_rate', 'three_point_field_goal_percentage', 'turnover_percentage', 'true_shooting_percentage']

# Prepend away and home to each field
game_fields = generate_game_fields(fields, "home_win")

# Manual Feature Selection Classifiers
logistic_regression(filename="games.csv", learning_rate=0.1,
                    epochs=1000, stability=10e-7, game_fields=game_fields)
decision_tree(filename="games.csv", min_observation_split=2,
              min_information_gain=0, game_fields=game_fields)
random_forest(filename="games.csv", forest_size=2, game_fields=game_fields)

# PCA Feature Selection Classifiers

# Manual Feature Selection Bracket Logic
bracket_logistic_regression(filename="games.csv", learning_rate=0.1,
                            epochs=1000, stability=10e-7, fields=fields, game_fields=game_fields, year=2018)
bracket_decision_tree(filename="games.csv", min_observation_split=2,
                      min_information_gain=0, fields=fields, game_fields=game_fields, year=2018)