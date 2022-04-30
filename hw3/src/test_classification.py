import util
import math_util
from evaluator import Evaluator
from naive_bayes import NaiveBayes
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def naive_bayes():
    print("NAIVES BAYES CLASSIFIER:")

    # Load data
    data = util.load_data("spambase.data")
    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    # Get mean and std of training data
    means, stds = math_util.compute_training_mean_std_by_feature(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = math_util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    # Instantiate NaiveBayes
    nb = NaiveBayes(stability_constant=1e-4)

    # 6. Create Normal models for each feature for each class
    nb.train_model(x_train_zscored, y_train)

    # 7. Classifies each validation sample using these models and chooses the class label
    # based on which class probability is higher
    valid_preds = nb.evaluate_model(x_valid_zscored)

    # 8. Computes the following statistics using the validation data results:
    eval = Evaluator()
    valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
        y_valid, valid_preds)

    # (a) Precision
    print("Validation Precision:", valid_precision)
    # # (b) Recall
    print("Validation Recall:", valid_recall)
    # # (c) F-Measure
    print("Validation F-Measure:", valid_f_measure)
    # (d) Accuracy
    print("Validation Accuracy:", valid_accuracy)
    
    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train, x_valid_zscored, y_valid, use_nb=True, is_multi=False)


def multi_class_naive_bayes():
    print("MULTI-CLASS NAIVE BAYES CLASSIFIER:")

    # Load data
    data = util.load_data("CTG.csv", rows_to_skip=2)
    # "The second to last column of the dataset can also be used for classification but for our purposes DISCARD it."
    data = np.delete(data, -2, axis=1)

    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    # Get mean and std of training data
    means, stds = math_util.compute_training_mean_std_by_feature(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = math_util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    # Instantiate NaiveBayes
    nb = NaiveBayes(stability_constant=1e4)

    # 6. Create Normal models for each feature for each class
    nb.train_model(x_train_zscored, y_train)

    # 7. Classifies each validation sample using these models and chooses the class label
    # based on which class probability is higher
    valid_preds = nb.evaluate_model(x_valid_zscored)

    # 8. Computes the following statistics using the validation data results:
    eval = Evaluator()
    valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    print("Validation Accuracy:", valid_accuracy)
    
    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train, x_valid_zscored, y_valid, use_nb=True, is_multi=True)
    
def decision_tree():
    print("DECISION TREE:")

    # Load data
    data = util.load_data("spambase.data")
    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    # Get mean and std of training data
    means, stds = math_util.compute_training_mean_std_by_feature(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = math_util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    # # Instantiate NaiveBayes
    # nb = NaiveBayes(stability_constant=1e-4)

    # # 6. Create Normal models for each feature for each class
    # nb.train_model(x_train_zscored, y_train)

    # # 7. Classifies each validation sample using these models and chooses the class label
    # # based on which class probability is higher
    # valid_preds = nb.evaluate_model(x_valid_zscored)

    # # 8. Computes the following statistics using the validation data results:
    # eval = Evaluator()
    # valid_precision, valid_recall, valid_f_measure, valid_accuracy = eval.evaluate_classifier(
    #     y_valid, valid_preds)

    # # (a) Precision
    # print("\nValidation Precision:", valid_precision)
    # # # (b) Recall
    # print("Validation Recall:", valid_recall)
    # # # (c) F-Measure
    # print("Validation F-Measure:", valid_f_measure)
    # # (d) Accuracy
    # print("Validation Accuracy:", valid_accuracy)
    
    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train, x_valid_zscored, y_valid, use_nb=False, is_multi=False)
    
def multi_class_decision_tree():
    print("MULTI-CLASS DECISION TREE:")

    # Load data
    data = util.load_data("CTG.csv", rows_to_skip=2)
    # "The second to last column of the dataset can also be used for classification but for our purposes DISCARD it."
    data = np.delete(data, -2, axis=1)

    # Shuffle data
    data = util.shuffle_data(data, 0)

    training, validation = util.get_train_valid_data(data)

    x_train, y_train = util.get_features_actuals(training)
    x_valid, y_valid = util.get_features_actuals(validation)

    # Get mean and std of training data
    means, stds = math_util.compute_training_mean_std_by_feature(x_train)

    # 4. Z-Score our training data with the means and std
    x_train_zscored = math_util.z_score_data(x_train, means, stds)

    # 4. Z-Score our validation data with the means and std
    x_valid_zscored = math_util.z_score_data(x_valid, means, stds)

    # # Instantiate NaiveBayes
    # nb = NaiveBayes(stability_constant=1e4)

    # # 6. Create Normal models for each feature for each class
    # nb.train_model(x_train_zscored, y_train)

    # # 7. Classifies each validation sample using these models and chooses the class label
    # # based on which class probability is higher
    # valid_preds = nb.evaluate_model(x_valid_zscored)

    # # 8. Computes the following statistics using the validation data results:
    # eval = Evaluator()
    # valid_accuracy = eval.evaluate_accuracy(y_valid, valid_preds)
    # print("Validation Accuracy:", valid_accuracy)
    
    # For verification purposes only, we compare our results with the sklearn
    # GaussianNB implementation with our z-scored features.
    compare_against_sklearn(x_train_zscored, y_train, x_valid_zscored, y_valid, use_nb=False, is_multi=True)
    
    
def compare_against_sklearn(x_train, y_train, x_valid, y_valid, use_nb=True, is_multi=False):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    
    model = None
    
    if use_nb:
        model = GaussianNB()
        model.fit(x_train, y_train)
    else:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
        model.fit(x_train, y_train)
        
    y_predict = model.predict(x_valid)
    
    if not is_multi:
        print("\nsklearn Precision:", precision_score(y_valid, y_predict))
        print("sklearn Recall:", recall_score(y_valid, y_predict))
        print("sklearn F-Measure:", f1_score(y_valid, y_predict))
    
    print("sklearn Accuracy:", accuracy_score(y_valid, y_predict))



naive_bayes()
print("\n")
multi_class_naive_bayes()
print("\n\n")
decision_tree()
print("\n")
multi_class_decision_tree()