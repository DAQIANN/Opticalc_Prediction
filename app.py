import argparse
import pandas as pd
import import_ipynb

from csvload import df

FULL_COLUMNS = ['Gender', 'Eye', 'Q', 'SA', 'RMS', 'Astig EKR65', 'K1', 'K1 Axis', 'K2', 'K2 Axis', 'Axial Length', 'ACD', 'IOL Power']
PREDICTION_COLUMN = ['Outcome SE/refraction\n(after surgery vision)']

full_column_df = df[FULL_COLUMNS]
no_na_row_df = df.dropna()

def MLR(args):
    pass

def logistic(args):
    pass

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Script for Eyelabs AI")
    # parser.add_argument("-csvf", "--csv_file", required=True, help="enter name of csv file to load")
    # args = parser.parse_args()
    # main(args)
    print(full_column_df.columns)