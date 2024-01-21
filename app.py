import argparse
import pandas as pd
import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn import preprocessing

from csvload import FULL_COLUMNS, df, train, test, all_column_train, all_column_y, all_column_test, all_columns_test_result, no_na_train, no_na_train_y, no_na_test, no_na_test_y

def MSE(predictions, results):
    sum = 0.0
    for i in range(len(predictions)):
        sum += (predictions[i] - results['Outcome SE/refraction\n(after surgery vision)'][i])**2
    return sum/len(predictions)

def MAE(predictions, results):
    sum = 0.0
    for i in range(len(predictions)):
        sum += abs(predictions[i] - results['Outcome SE/refraction\n(after surgery vision)'][i])
    return sum/len(predictions)

def fit_to_model(args, model):
    if args.model_type == "Full":
        data = all_column_train[FULL_COLUMNS]
        y_data = all_column_y
        test_data = all_column_test[FULL_COLUMNS]
        test_results = all_columns_test_result
    else:
        data = no_na_train
        y_data = no_na_train_y
        test_data = no_na_test
        test_results = no_na_test_y
    
    model.fit(data, y_data)
    predictions = np.array(model.predict(test_data))
    predictions = np.array([prediction[0] for prediction in predictions])
    test_results.reset_index(inplace=True)
    prediction_errors = []
    if len(predictions) != test_results.shape[0]:
        print("Wrong Prediction Sizing")
        return -1
    for i in range(len(predictions)):
        prediction_errors.append(abs(predictions[i] - test_results['Outcome SE/refraction\n(after surgery vision)'][i]))
    return np.mean(prediction_errors)

def MLR(args):
    regr = LinearRegression()
    return fit_to_model(args, regr)

def logistic(args):
    regr = LogisticRegression()
    return fit_to_model(args, regr)

def SGD(args):
    if args.model_type == "Full":
        data = all_column_train[FULL_COLUMNS]
        y_data = all_column_y
        test_data = all_column_test[FULL_COLUMNS]
        test_results = all_columns_test_result
    else:
        data = no_na_train
        y_data = no_na_train_y
        test_data = no_na_test
        test_results = no_na_test_y
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    test_data = scaler.transform(test_data)
    n_iter = 100
    regr = SGDRegressor(max_iter=n_iter)
    regr.fit(data, y_data.values.ravel())
    predictions = np.array(regr.predict(test_data))
    test_results.reset_index(inplace=True)
    return MAE(predictions, test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Eyelabs AI")
    parser.add_argument("-tech", "--technique", required=True, help="enter technique to run predictions (MLR, Log, SGD)")
    parser.add_argument("-mt", "--model_type", required=True, help="Full or All")
    args = parser.parse_args()
    if args.technique == "MLR":
        result = MLR(args)
    elif args.technique == "Log":
        result = logistic(args)
    elif args.technique == "SGD":
        result = SGD(args)
    print(result)
    
    # data = all_column_train[FULL_COLUMNS]
    # y_data = all_column_y
    # test_data = all_column_test[FULL_COLUMNS]
    # test_results = all_columns_test_result
    # regr = LinearRegression()
    # regr.fit(data, y_data)
    # predictions = np.array(regr.predict(test_data))
    # test_results.reset_index(inplace=True)
    # for i in range(len(predictions)):
    #     print(predictions[i][0])
    #     print(test_results['IOL Power'])
        # print(predictions[i][0] - test_results['IOL Power'][i])
    