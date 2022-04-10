from linear_regression import LinearRegression
import numpy as np

def print_results(lr, y, preds, type):
    #print(type, "actual:", y)
    #print(type, "preds:", preds)
    #print("")
    
    rmse = lr.compute_rmse(y, preds)
    print(type, "RMSE:", rmse)
    mape = lr.compute_mape(y, preds)
    print(type, "MAPE:", mape)
    
def test_closed_form(lr, data):
    print("Closed Form (Direct) Linear Regression\n")
    
    data = lr.shuffle_data(data, 0)
    training, validation = lr.get_train_valid_data(data)
    N = 29
    xTrain = np.ones((N, 1))
    xTrain = np.append(xTrain, np.ones((N, 1)), axis=1)
    xTrain, yTrain = lr.get_features_actuals(training)
    xValid, yValid = lr.get_features_actuals(validation)
    
    train_preds = lr.compute_closed_form(xTrain, yTrain)
    valid_preds = lr.compute_closed_form(xValid, yValid)
    
    print_results(lr, yTrain, train_preds, "Training")
    print("")
    print_results(lr, yValid, valid_preds, "Validation")
    
def test_s_folds_cross_validation(lr, data):
    print("S-Folds Cross-Validation")
    s_fold_list = [4, 11, 22]
    
    for s_fold in s_fold_list:
        print("\nS-Fold:", s_fold)
        lr.compute_s_folds_cross_validation(data, s_fold)
    
def test_locally_weighted(lr, data):
    print("Locally-Weight Linear Regression\n")
    
    data = lr.shuffle_data(data, 0)
    training, validation = lr.get_train_valid_data(data)
    xTrain, yTrain = lr.get_features_actuals(training)
    xValid, yValid = lr.get_features_actuals(validation)
    
    train_preds = lr.compute_locally_weighted(xTrain, yTrain, 1)
    valid_preds = lr.compute_locally_weighted(xValid, yValid, 1)
    
    print_results(lr, yTrain, train_preds, "Training")
    print("")
    print_results(lr, yValid, valid_preds, "Validation")

def main():
    lr = LinearRegression()
    
    columns = [1, 2, 3]
    data = lr.load_data("x06Simple.csv", columns)
    
    test_closed_form(lr, data)
    print("\n-------------------------------------")
    test_s_folds_cross_validation(lr, data)
    print("\n-------------------------------------")
    test_locally_weighted(lr, data)

if __name__ == '__main__':
    main()