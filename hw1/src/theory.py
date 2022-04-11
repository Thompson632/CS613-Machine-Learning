import numpy as np

def compute_weights(X, y_mat):
    '''
    Computes the weights based on the features and actuals

    :param X: Features
    :param y_mat: Actuals

    :return the weights (or coefficients)
    '''
    xt_x = np.dot(X.T, X)
    xt_x_inv = np.linalg.pinv(xt_x)
    xt_x_inv_xt = np.dot(xt_x_inv, X.T)
    xt_x_inv_xt_y = np.dot(xt_x_inv_xt, y_mat)
    return xt_x_inv_xt_y

def compute_y_hat(X, weights):
    '''
    Computes y_hat by taking the dot product of our
    features and weights

    :param X: Features
    :param weights: Weights

    :return the predictions
    '''
    return np.dot(X, weights)

def compute_se(actual, predictions):
    '''
    Computes the squared error based on the actual data
    and our trained model predictions

    :param actual: Actual real data
    :param predictions: Training model preditions

    :return the squared error
    '''
    differences = np.subtract(actual, predictions)
    differences_squared = np.square(differences)
    return differences_squared

def compute_mse(actual, predictions):
    '''
    Computes the mean squared error based on the actual data
    and our trained models predictions.

    :param actual: Actual real data
    :param predictions: Training model predictions

    :return the mean squared error
    '''
    differences_squared = compute_se(actual, predictions)
    mse = differences_squared.mean()
    return mse

def compute_rmse(actual, predicted):
    '''
    Computes the root mean squared error based on the actual
    data and our trained models predictions.

    :param actual: Actual real data
    :param predictions: Training model predictions
    :return the mean squared error        
    '''
    mse = compute_mse(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse

def compute_mape(actual, predicted):
    '''
    Computes the mean absolute percent error based on the actual
    data and our trained models predictions.

    :param actual: Actual real data
    :param predictions: Training model predictions

    :return the mean absolute percent error
    '''
    abs = np.abs((np.subtract(actual, predicted)) / actual)
    mape = np.mean(abs) * 100
    return mape

def compute_smape(actual, predicted):
    '''
    Computes the symmetric mean absolute percentage error based on the 
    actual data and our trained models predictions
    
    :param actual: Actual real data
    :param predictions: Training model predictions
    
    :return the symmetric mean absolute percent error
    '''
    return 1/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))*100)


xg = [-2, -5, -3, 0, -8, -2, 1, 5, -1, 6]

x = [[1, -2], [1, -5], [1, -3], [1, 0], [1, -8],
     [1, -2], [1, 1], [1, 5], [1, -1], [1, 6]]


y = [1, -4, 1, 3, 11, 5, 0, -1, -3, 1]

x = np.array(x)
y = np.array([y]).T


model = compute_weights(x,y)
print("Model:\n", model)

y_hat = compute_y_hat(x,model)
print("y_hat:\n", y_hat)

print("y:\n", y)

print("")
print("RMSE:",compute_rmse(y_hat, y))
print("MAPE:",compute_mape(y_hat, y))
print("SMAPE:",compute_smape(y_hat, y))