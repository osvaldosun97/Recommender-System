import numpy as np

import pandas as pd
from sklearn import preprocessing

from preprocessing.utils import optimize_dtypes


def train():
    """
    Function for running various steps for train phase

    :return:
    """
    pass




if __name__ == '__main__':
    print("Welcome to Haneul's Recommendation engine")
    # df = pd.DataFrame({"age":[22,21,23,33,35,33,33,55],
    #                    "country":["Korea", "Canada", "Canada", "Korea",
    #                               "Korea", "Korea", "Canada", "Singapore"],
    #                     "age2":[22.0,21.2,23.2,33.3,35.5,33.76,33.1,55.2],
    #                     "age3":[23123123, 34333434, -999999669999, -9999999978678678, -33333333, 3.44,3.33,4.44]
    #                   })

    df = pd.read_parquet("datasets/h&m/customers.parquet")
    print(df.head())

    df = optimize_dtypes(df, ['float', 'integer', 'object'], verbose=True)

    # TODO: Need to make this process dynamic, given dataset we need to figure out its dtypes then
    # apply appropriate preprocessing steps.
    # le = preprocessing.LabelEncoder()
    # df['country'] =