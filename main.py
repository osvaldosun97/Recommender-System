import numpy as np
import pandas as pd

from sklearn import preprocessing
from timeit import default_timer as timer

from metrics import calc_rmse
from models.utils import df_to_rating_matrix
from models.matrix_factorization import MatrixFactorization, train_test_split_rating_mat,\
    MatrixFactorization_rating_matrix_version
from preprocessing.utils import optimize_dtypes, train_test_split_dataframe


def train():
    """
    Function for running various steps for train phase

    :return:
    """
    pass




if __name__ == '__main__':
    print("Welcome to Haneul's Recommendation engine")
    # TODO: Need to make this process dynamic, given dataset we need to figure out its dtypes then
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
    item_col_nm, user_col_nm, rating_col_nm = "user", "item", "ratings"
    print(f"n_users = {rating_df[user_col_nm].nunique()}")
    print(f"n_items = {rating_df[item_col_nm].nunique()}")
    # train_rating_df, test_rating_df = train_test_split_dataframe(rating_df, test_ratio=0.3, method="random")

    emb_dim = 3
    epoch = 100
    lr = 0.01
    _lambda = 0.01
    # Rating matrix version
    rating_matrix = df_to_rating_matrix(rating_df, item_col_nm, user_col_nm, rating_col_nm)
    MF_rmv = MatrixFactorization_rating_matrix_version(item_col_nm, user_col_nm, rating_col_nm,
                                                       emb_dim, lr, _lambda)
    start = timer()
    MF_rmv.fit(rating_matrix, epoch=100, verbose=True)
    end = timer()
    print(f"took {round(end - start, 3)} seconds")
    print(f"====== predicted ======")
    print(np.round(MF_rmv.predict(), 2))
    print("====== original rating matrix")
    print(rating_matrix)

    print(f"final RMSE = {round(calc_rmse(MF_rmv.P, MF_rmv.Q, rating_matrix), 3)}")



    # # FIXME : how to train/test split rating matrix?
    # # train_mat, test_mat = train_test_split_rating_mat()
    # emb_dim, n_steps, lr, _lambda = 3, 100, 0.01, 0.01
    # mf_method = "sgd"
    # tr_P, tr_Q = run_matrix_factorization(train_rating_df, emb_dim, n_steps, lr, _lambda, mf_method)
    # print(f"RMSE = {np.round(calc_rmse(tr_P, tr_Q, train_rating_df), 4)}")
    #
    # verbose = True
    # if verbose:
    #     print()
    #     print(f"======= original rating matrix ========")
    #     print(rating_matrix)
    #     print()
    #     print(f"======= predicted rating matrix by dot(P, Q^T) ========== ")
    #     print(np.dot(P, Q.T))
    #     print()
    #     print(f"========== P =============")
    #     print(P)
    #     print()
    #     print(f"========== Q^T =============")
    #     print(Q.T)
