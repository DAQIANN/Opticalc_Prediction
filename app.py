import argparse
import pandas as pd
import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

from csvload import FULL_COLUMNS, df, train, test, all_column_train, all_column_y, all_column_test, all_columns_test_result, no_na_train, no_na_train_y, no_na_test, no_na_test_y

def MAE(predictions, results):
    sum = 0.0
    for i in range(len(predictions)):
        sum += abs(predictions[i][0] - results['IOL Power'][i])
    return sum/len(predictions)

def MLR(args):
    regr = LinearRegression()
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
    
    regr.fit(data, y_data)
    predictions = np.array(regr.predict(test_data))
    test_results.reset_index(inplace=True)
    prediction_errors = []
    if len(predictions) != test_results.shape[0]:
        print("Wrong Prediction Sizing")
        return -1
    for i in range(len(predictions)):
        prediction_errors.append((predictions[i][0] - test_results['IOL Power'][i]))
    return MAE(predictions, test_results)

def logistic(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Eyelabs AI")
    parser.add_argument("-tech", "--technique", required=True, help="enter technique to run predictions (MLR, Log")
    parser.add_argument("-mt", "--model_type", required=True, help="Full or All")
    args = parser.parse_args()
    if args.technique == "MLR":
        result = MLR(args)
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
    