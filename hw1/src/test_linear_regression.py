from linear_regression import LinearRegression
import numpy as np

def print_results(lr, y, preds, type):
    print(type, "Actuals:", y)
    print(type, "Preds:", preds)
    print("")

    rmse = lr.compute_rmse(y, preds)
    print(type, "RMSE:", rmse)

    mape = lr.compute_mape(y, preds)
    print(type, "MAPE:", mape)


def test_closed_form(lr):
    print("Closed Form (Direct) Linear Regression\n")

    # 1. Reads in the data, ignorning the first row (header) and first column (index)
    columns = [1, 2, 3]
    data = lr.load_data("x06Simple.csv", columns)

    # 2. Shuffle the rows of data
    data = lr.shuffle_data(data, 0)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining
    # for validation
    training, validation = lr.get_train_valid_data(data)

    # 4. Computes the linear regression model using the direct solution
    # 5. Applies the learned model to the validation samples
    yTrain, train_preds, yValid, valid_preds = lr.compute_model_using_direct_solution_and_apply(
        training, validation, True)

    # 6. Computes the root mean squared error (RMSE) and mean absolute
    # percent error (MAPE) for the training and validation sets
    print_results(lr, yTrain, train_preds, "Training")
    print("")
    print_results(lr, yValid, valid_preds, "Validation")


def test_s_folds_cross_validation(lr):
    print("S-Folds Cross-Validation")

    # 1. Reads in the data, ignorning the first row (header) and first column (index)
    columns = [1, 2, 3]
    data = lr.load_data("x06Simple.csv", columns)

    s_fold_list = [4, 11, 22]

    N = len(data)

    for s_fold in s_fold_list:
        print("\nS-Fold:", s_fold)
        lr.compute_s_folds_cross_validation(data, s_fold)

    print("\nS-Fold (N):", N)
    lr.compute_s_folds_cross_validation(data, N)


def test_locally_weighted(lr):
    print("Locally-Weighted Linear Regression\n")

    # 1. Reads in the data, ignorning the first row (header) and first column (index)
    columns = [1, 2, 3]
    data = lr.load_data("x06Simple.csv", columns)

    # 2. Shuffle the rows of data
    data = lr.shuffle_data(data, 0)

    # 3. Selects the first 2/3 (round up) of the data for training and the remaining
    # for validation
    training, validation = lr.get_train_valid_data(data)

    yValid, valid_preds = lr.compute_locally_weighted(training, validation, 1)

    # 6. Computes the RMSE and MAPE over the validation data
    print_results(lr, yValid, valid_preds, "Validation")


def main():
    lr = LinearRegression()

    test_closed_form(lr)
    print("\n-------------------------------------")
    test_s_folds_cross_validation(lr)
    print("\n-------------------------------------")
    test_locally_weighted(lr)


if __name__ == '__main__':
    main()
