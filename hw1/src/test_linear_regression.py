import data_util
from closed_form import LinearRegressionClosedForm
from sfolds import LinearRegressionSFolds
from locally_weighted import LinearRegressionLocallyWeighted
from evaluator import Evaluator

def load_data(filename, columns):
    data = data_util.load_data(filename, columns)
    data = data_util.shuffle_data(data, 0)

    training, validation = data_util.get_train_valid_data(data)

    x_train, y_train = data_util.get_features_actuals(training)
    x_valid, y_valid = data_util.get_features_actuals(validation)
    
    x_train_bias = data_util.add_bias_feature(x_train)
    x_valid_bias = data_util.add_bias_feature(x_valid)
    
    return x_train_bias, y_train, x_valid_bias, y_valid

def closed_form(filename, columns):
    print("\n======================================================")
    print("CLOSED FORM LINEAR REGRESSION:")
    
    X_train, y_train, X_valid, y_valid = load_data(filename, columns)
    
    model = LinearRegressionClosedForm()
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    
    eval = Evaluator()
    
    print("Training RMSE:", eval.compute_rmse(y_train, train_preds))
    print("Training MAPE:", eval.compute_mape(y_train, train_preds))
    print("\nValidation RMSE:", eval.compute_rmse(y_valid, valid_preds))
    print("Validation MAPE:", eval.compute_mape(y_valid, valid_preds))

def s_folds(filename, columns):
    print("\n======================================================")
    print("S-FOLDS LINEAR REGRESSION:")
    
    data = data_util.load_data(filename, columns)
    
    folds = [4, 11, 22]

    model = LinearRegressionSFolds(folds)
    model.fit(data)
    
def locally_weighted(filename, columns):
    print("\n======================================================")
    print("LOCALLY WEIGHTED LINEAR REGRESSION:")
    
    X_train, y_train, X_valid, y_valid = load_data(filename, columns)
    
    model = LinearRegressionLocallyWeighted()
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    valid_preds = model.predict(X_valid)
    
    eval = Evaluator()
    
    print("Training RMSE:", eval.compute_rmse(y_train, train_preds))
    print("Training MAPE:", eval.compute_mape(y_train, train_preds))
    print("\nValidation RMSE:", eval.compute_rmse(y_valid, valid_preds))
    print("Validation MAPE:", eval.compute_mape(y_valid, valid_preds))
    

columns = [1, 2, 3]

closed_form(filename="x06Simple.csv", columns=columns)
s_folds(filename="x06Simple.csv", columns=columns)
locally_weighted(filename="x06Simple.csv", columns=columns)