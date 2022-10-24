import numpy as np

import pandas as pd
from sklearn import preprocessing

from metrics import calc_rmse
from models.utils import df_to_rating_matrix
from models.matrix_factorization import run_matrix_factorization, train_test_split_rating_mat
from preprocessing.utils import optimize_dtypes


def train():
    """
    Function for running various steps for train phase

    :return:
    """
    pass




if __name__ == '__main__':
    print("Welcome to Haneul's Recommendation engine")
    # TODO: Need to make this process dynamic, given dataset we need to figure out its dtypes then
    # apply appropriate preprocessing steps.
    # le = preprocessing.LabelEncoder()
    # df['country'] =

    # rating_df = pd.read_parquet("datasets/ml-25m/ratings_sampled.parquet").sample(10_000)
    rating_df = pd.DataFrame({"user": np.array([[u] * 5 for u in [0, 1, 2, 3]]).reshape(-1),
                       'item': [0, 1, 2, 3, 4] * 4,
                       'ratings': [2, 1, 5, 4, 5,
                                   5, 4, 1, np.nan, 2,
                                   1, 1, 5, 2, 2,
                                   1, np.nan, np.nan, 4, 3]
                       })
    # FIXME : TypeError: No matching signature found
    # rating_df = optimize_dtypes(rating_df, True)

    rating_matrix = df_to_rating_matrix(rating_df, "movieId", "userId", "rating")
    print(f"rating_matrix shape : {rating_matrix.shape}")

    # FIXME : how to train/test split rating matrix?
    # train_mat, test_mat = train_test_split_rating_mat()
    emb_dim, n_steps, lr, _lambda = 3, 100, 0.01, 0.01
    mf_method = "sgd"
    P, Q = run_matrix_factorization(rating_matrix, emb_dim, n_steps, lr, _lambda, mf_method)
    print(f"RMSE = {np.round(calc_rmse(P, Q, rating_matrix), 4)}")
