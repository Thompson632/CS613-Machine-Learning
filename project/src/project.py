import math_util
import data_util
from evaluator import Evaluator
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree
from bracket import Bracket
from random_forest import RandomForest
from pca import PCA


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
    print("\n======================================================")
    print("LOGISTIC REGRESSION CLASSIFIER:")

    print("Learning Rate:", learning_rate)
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
    print("\n======================================================")
    print("DECISION TREE CLASSIFIER:")

    print("Min Observation Split:", min_observation_split)
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
    print("\n======================================================")
    print("RANDOM FOREST CLASSIFIER:")

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


def pca_logistic_regression(filename, learning_rate, epochs, stability, game_fields):
    print("\n======================================================")
    print("PCA LOGISTIC REGRESSION CLASSIFIER:")

    print("Learning Rate:", learning_rate)
    print("Epochs:", epochs)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    pca_model = PCA()
    min_components, _ = pca_model.determine_min_components(X_train)

    print("\n100% Number of Components:", X_train.shape[1])
    print("95% Minimum Number of Components:", min_components)

    pca_model = PCA(min_components)
    eigenvectors = pca_model.fit(X_train)

    z_train = pca_model.predict(X_train, eigenvectors)
    z_valid = pca_model.predict(X_valid, eigenvectors)

    lr_model = LogisticRegression(learning_rate, epochs, stability)
    train_losses, valid_losses = lr_model.fit(
        z_train, y_train, z_valid, y_valid)

    eval = Evaluator()

    eval.plot_mean_log_loss(train_losses, valid_losses, epochs)

    train_preds = lr_model.predict(z_train)
    valid_preds = lr_model.predict(z_valid)

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


def pca_decision_tree(filename, min_observation_split, min_information_gain, game_fields):
    print("\n======================================================")
    print("PCA DECISION TREE CLASSIFIER:")

    print("Min Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    pca_model = PCA()
    min_components, _ = pca_model.determine_min_components(X_train)

    print("\n100% Number of Components:", X_train.shape[1])
    print("95% Minimum Number of Components:", min_components)

    pca_model = PCA(min_components)
    eigenvectors = pca_model.fit(X_train)

    z_train = pca_model.predict(X_train, eigenvectors)
    z_valid = pca_model.predict(X_valid, eigenvectors)

    dt_model = DecisionTree()
    dt_model.fit(z_train, y_train)

    train_preds = dt_model.predict(z_train)
    valid_preds = dt_model.predict(z_valid)

    eval = Evaluator()

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


def pca_random_forest(filename, forest_size, game_fields):
    print("\n======================================================")
    pass


def bracket_logistic_regression(filename, learning_rate, epochs, stability, fields, game_fields, year):
    print("\n======================================================")
    print(year, "LOGISTIC REGRESSION BRACKET PREDICTION:")

    print("\nLearning Rate:", learning_rate)
    print("Epochs:", epochs)

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    model = LogisticRegression(learning_rate, epochs, stability)
    _, _ = model.fit(
        X_train, y_train, X_valid, y_valid)

    bracket = Bracket(year, model, fields, game_fields)
    bracket.run_bracket()


def bracket_decision_tree(filename, min_observation_split, min_information_gain, fields, game_fields, year):
    print("\n======================================================")
    print(year, "DECISION TREE BRACKET PREDICTION:")

    print("Min Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    X_train, y_train, _, _ = load_data(filename, game_fields)

    model = DecisionTree(min_observation_split=min_observation_split,
                         min_information_gain=min_information_gain)
    model.fit(X_train, y_train)

    bracket = Bracket(year, model, fields, game_fields)
    bracket.run_bracket()


def manual_feature_selection_classifiers(game_fields):
    logistic_regression(filename="games.csv", learning_rate=0.1,
                        epochs=1000, stability=10e-7, game_fields=game_fields)
    decision_tree(filename="games.csv", min_observation_split=2,
                  min_information_gain=0, game_fields=game_fields)
    # TODO: This is broken
    # random_forest(filename="games.csv", forest_size=2, game_fields=game_fields)


def pca_feature_selection_classifiers(game_fields):
    pca_logistic_regression(filename="games.csv", learning_rate=0.1,
                            epochs=1000, stability=10e-7, game_fields=game_fields)
    pca_decision_tree(filename="games.csv", min_observation_split=2,
                      min_information_gain=0, game_fields=game_fields)


def manual_feature_selection_bracket(fields, game_fields, year):
    bracket_logistic_regression(filename="games.csv", learning_rate=0.1,
                                epochs=1000, stability=10e-7, fields=fields, game_fields=game_fields, year=year)
    bracket_decision_tree(filename="games.csv", min_observation_split=2,
                          min_information_gain=0, fields=fields, game_fields=game_fields, year=year)


# Define our most informative features based on our knowledge
fields = ['offensive_rating', 'effective_field_goal_percentage', 'total_rebound_percentage', 'free_throw_attempt_rate',
          'free_throw_percentage', 'three_point_attempt_rate', 'three_point_field_goal_percentage', 'turnover_percentage', 'true_shooting_percentage']

# Prepend away and home to each field
game_fields = generate_game_fields(fields, "home_win")

# Manual Feature Selection Classifiers
manual_feature_selection_classifiers(game_fields=game_fields)

# PCA Feature Selection Classifiers
pca_feature_selection_classifiers(game_fields=game_fields)

# Manual Feature Selection Bracket Logic
manual_feature_selection_bracket(
    fields=fields, game_fields=game_fields, year=2018)