import numpy as np
import pandas as pd

from sklearn import preprocessing
from timeit import default_timer as timer

from metrics import calc_rmse
from models.utils import df_to_rating_matrix
from models.matrix_factorization import MatrixFactorization, train_test_split_rating_mat,\
    MatrixFactorization_rating_matrix_version
from preprocessing.utils import optimize_dtypes, train_test_split_dataframe


if __name__ == '__main__':
    print("Welcome to Haneul's Recommendation engine")
    rating_df = pd.read_parquet("datasets/ml-25m/ratings_sampled.parquet").sample(10_000)
    # FIXME : TypeError: No matching signature found
    rating_df = optimize_dtypes(rating_df, verbose=False)
    train_df, test_df = train_test_split_dataframe(rating_df, test_ratio=0.3, method='random',
                                                   time_col=None, verbose=True)
    user_col_nm, item_col_nm, rating_col_nm = "userId", "movieId", "rating"

    emb_dim = 3
    epoch = 100
    lr = 0.01
    _lambda = 0.01

    n_users, n_items = train_df[[user_col_nm, item_col_nm]].nunique()
    MF_model = MatrixFactorization(emb_dim, lr, _lambda, n_users, n_items)
    MF_model.fit(train_df, user_col_nm, item_col_nm, rating_col_nm, epoch=epoch, verbose=True)

    print("====== prediction on trained dataset ======")
    train_df['pred_rating'] = MF_model.predict(train_df)
    print(train_df.head())
    print(f"RMSE = {round(calc_rmse(train_df['rating'], train_df['pred_rating']), 3)}")


    recommend_df = MF_model.batch_reccomendation(test_df)
    print(f"===== recommend_df ========")
    print(recommend_df.head())

    for userid in test_df[user_col_nm].unique():
        recc_items = MF_model.recommend_item(userid, topk=3)
        print(f"recommend user{userid} -> movies:{recc_items}")

