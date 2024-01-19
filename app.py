import argparse
import pandas as pd
import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from csvload import FULL_COLUMNS, df, train, test, all_column_train, all_column_y, all_column_test, all_columns_test_result, no_na_train, no_na_train_y, no_na_test, no_na_test_y

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
    
    pass

def logistic(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for Eyelabs AI")
    parser.add_argument("-tech", "--technique", required=True, help="enter technique to run predictions (MLR, Log")
    parser.add_argument("-mt", "--model_type", required=True, help="Full or All")
    args = parser.parse_args()
