import math_util
import data_util
from evaluator import Evaluator
from logistic_regression import LogisticRegression
from decision_tree import DecisionTree
from bracket import Bracket
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

    # TODO: Fit and Predict with RF Model
    model = None

    eval = Evaluator()

    # train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
    #     y_train, train_preds)
    # print("\nTraining Precision:", train_precision)
    # print("Training Recall:", train_recall)
    # print("Training F-Measure:", train_f_measure)
    # print("Training Accuracy:", train_accuracy)

    # valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
    #     y_valid, valid_preds)
    # print("\nValidation Precision:", valid_precision)
    # print("Validation Recall:", valid_recall)
    # print("Validation F-Measure:", valid_f_measure)
    # print("Validation Accuracy:", valid_accuracy)


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
    print("PCA RANDOM FOREST CLASSIFIER:")

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    pca_model = PCA()
    min_components, _ = pca_model.determine_min_components(X_train)

    print("\n100% Number of Components:", X_train.shape[1])
    print("95% Minimum Number of Components:", min_components)

    pca_model = PCA(min_components)
    eigenvectors = pca_model.fit(X_train)

    z_train = pca_model.predict(X_train, eigenvectors)
    z_valid = pca_model.predict(X_valid, eigenvectors)

    # TODO: Fit and Predict with RF Model
    rf_model = None

    eval = Evaluator()

    # train_precision, train_recall, train_f_measure, train_accuracy = eval.evaluate_classifier(
    #     y_train, train_preds)
    # print("\nTraining Precision:", train_precision)
    # print("Training Recall:", train_recall)
    # print("Training F-Measure:", train_f_measure)
    # print("Training Accuracy:", train_accuracy)

    # valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
    #     y_valid, valid_preds)
    # print("\nValidation Precision:", valid_precision)
    # print("Validation Recall:", valid_recall)
    # print("Validation F-Measure:", valid_f_measure)
    # print("Validation Accuracy:", valid_accuracy)


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


def bracket_random_forest(filename, fields, game_fields, year):
    print("\n======================================================")
    print(year, "RANDOM FOREST BRACKET PREDICTION:")

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    # TODO: Fit and Predict with RF Model
    model = None

    bracket = Bracket(year, model, fields, game_fields)
    bracket.run_bracket()


def bracket_pca_logistic_regression(filename, learning_rate, epochs, stability, fields, game_fields, year):
    print("\n======================================================")
    print(year, "PCA LOGISTIC REGRESSION BRACKET PREDICTION:")

    print("\nLearning Rate:", learning_rate)
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
    _, _ = lr_model.fit(
        z_train, y_train, z_valid, y_valid)

    bracket = Bracket(year, lr_model, fields, game_fields)
    bracket.run_bracket()


def bracket_pca_decision_tree(filename, min_observation_split, min_information_gain, fields, game_fields, year):
    print("\n======================================================")
    print(year, "PCA DECISION TREE BRACKET PREDICTION:")

    print("Min Observation Split:", min_observation_split)
    print("Min Information Gain:", min_information_gain)

    X_train, y_train, _, _ = load_data(filename, game_fields)

    pca_model = PCA()
    min_components, _ = pca_model.determine_min_components(X_train)

    print("\n100% Number of Components:", X_train.shape[1])
    print("95% Minimum Number of Components:", min_components)

    pca_model = PCA(min_components)
    eigenvectors = pca_model.fit(X_train)

    z_train = pca_model.predict(X_train, eigenvectors)

    dt_model = DecisionTree(min_observation_split=min_observation_split,
                            min_information_gain=min_information_gain)
    dt_model.fit(z_train, y_train)

    bracket = Bracket(year, dt_model, fields, game_fields)
    bracket.run_bracket()


def bracket_pca_random_forest(filename, fields, game_fields, year):
    print("\n======================================================")
    print(year, "PCA RANDOM FOREST BRACKET PREDICTION:")

    X_train, y_train, X_valid, y_valid = load_data(filename, game_fields)

    pca_model = PCA()
    min_components, _ = pca_model.determine_min_components(X_train)

    print("100% Number of Components:", X_train.shape[1])
    print("95% Minimum Number of Components:", min_components)

    pca_model = PCA(min_components)
    eigenvectors = pca_model.fit(X_train)

    z_train = pca_model.predict(X_train, eigenvectors)
    z_valid = pca_model.predict(X_valid, eigenvectors)

    # TODO: Fit and Predict with RF Model
    rf_model = None

    bracket = Bracket(year, rf_model, fields, game_fields)
    bracket.run_bracket()


def run_manual_feature_selection_classifiers(file_path, game_fields):
    logistic_regression(filename=file_path, learning_rate=0.1,
                        epochs=1000, stability=10e-7, game_fields=game_fields)
    decision_tree(filename=file_path, min_observation_split=2,
                  min_information_gain=0, game_fields=game_fields)
    # TODO: Implement Random Forest
    random_forest(filename=file_path, forest_size=2, game_fields=game_fields)


def run_pca_feature_selection_classifiers(file_path, game_fields):
    pca_logistic_regression(filename=file_path, learning_rate=0.1,
                            epochs=1000, stability=10e-7, game_fields=game_fields)
    pca_decision_tree(filename=file_path, min_observation_split=2,
                      min_information_gain=0, game_fields=game_fields)
    # TODO: Implement Random Forest
    pca_random_forest(filename=file_path, forest_size=2,
                      game_fields=game_fields)


def run_manual_feature_selection_bracket(file_path, fields, game_fields, year):
    bracket_logistic_regression(filename=file_path, learning_rate=0.1,
                                epochs=1000, stability=10e-7, fields=fields, game_fields=game_fields, year=year)
    bracket_decision_tree(filename=file_path, min_observation_split=2,
                          min_information_gain=0, fields=fields, game_fields=game_fields, year=year)
    # TODO: Implement Random Forest
    bracket_random_forest(filename=file_path, fields=fields,
                          game_fields=game_fields, year=year)


def run_pca_feature_selection_bracket(file_path, fields, game_fields, year):
    bracket_pca_logistic_regression(filename=file_path, learning_rate=0.1,
                                    epochs=1000, stability=10e-7, fields=fields, game_fields=game_fields, year=year)
    bracket_pca_decision_tree(filename=file_path, min_observation_split=2,
                              min_information_gain=0, fields=fields, game_fields=game_fields, year=year)
    # TODO: Implement Random Forest
    bracket_pca_random_forest(
        filename=file_path, fields=fields, game_fields=game_fields, year=year)


# File Path
file_path = "games.csv"

# Define our most informative features based on our knowledge
fields = ['offensive_rating', 'effective_field_goal_percentage', 'total_rebound_percentage', 'free_throw_attempt_rate',
          'free_throw_percentage', 'three_point_attempt_rate', 'three_point_field_goal_percentage', 'turnover_percentage', 'true_shooting_percentage']

# Prepend away and home to each field
game_fields = generate_game_fields(fields, "home_win")

run_manual_feature_selection_classifiers(
    file_path=file_path, game_fields=game_fields)
run_pca_feature_selection_classifiers(
    file_path=file_path, game_fields=game_fields)
run_manual_feature_selection_bracket(
    file_path=file_path, fields=fields, game_fields=game_fields, year=2018)
run_pca_feature_selection_bracket(
    file_path=file_path, fields=fields, game_fields=game_fields, year=2018)