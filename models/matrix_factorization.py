"""
Matrix Factorization based methods
"""
import numpy as np
import pandas as pd

from metrics import calc_rmse, calc_rmse_rating_matrix
from timeit import default_timer as timer


def train_test_split_rating_mat(rating_matrix, train_ratio):
    """
    split matrix into train and test rating matrix

    :param rating_matrix:
    :param train_ratio: float, ratio of train_matrix
    :return: train_mat, test_mat : np.array, np.array
    """

    bool_matrix = (np.random.rand(n_user, n_item) < train_ratio)
    train_R_matrix = replace_zero_to_nan(R * bool_matrix)
    test_R_matrix = replace_zero_to_nan(R * ~bool_matrix)

    return train_R_matrix, test_R_matrix

def initialize_emb_vector(n, K, dist="normal"):
    """
    intializes embedding vector that will be learned.

    :param n: int, length of vector
    :param K: int, embedding dimension
    :param dist: str, distribution to intialize emb_vector from
    :return:
    """
    if dist == "normal":
        emb_vector = np.random.normal(loc=0, scale=1.0/K, size=(n, K))
    return emb_vector


class MatrixFactorization:
    def __init__(self, emb_dim, lr, _lambda, n_users, n_items):
        self.emb_dim = emb_dim
        self.lr = lr
        self._lambda = _lambda
        self.n_users = n_users
        self.n_items = n_items

        self.P = initialize_emb_vector(self.n_users, self.emb_dim, dist="normal")
        self.Q = initialize_emb_vector(self.n_items, self.emb_dim, dist="normal")

    def fit(self, train_df, user_col, item_col, rating_col, epoch=10, verbose=True):
        """
        train user/item embedding vector

        :param epoch: int, number of training iteration
        :param verbose: Bool
        """
        train_df = train_df.copy(deep=True)
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        # re-index for efficient lookup of embedding vectors.
        self.userid2idx = {u_id: i for i, u_id in enumerate(train_df[user_col].unique())}
        self.itemid2idx = {u_id: i for i, u_id in enumerate(train_df[item_col].unique())}
        train_df[user_col] = train_df[user_col].apply(lambda x: self.userid2idx[x])
        train_df[item_col] = train_df[item_col].apply(lambda x: self.itemid2idx[x])

        for i_epoch in range(epoch):
            for row in train_df.itertuples():
                user_id = getattr(row, self.user_col)
                item_id = getattr(row, self.item_col)
                r_ui = getattr(row, self.rating_col)
                pred_r_ui = np.dot(self.P[user_id],
                                   self.Q[item_id])
                e_ui = r_ui - pred_r_ui
                self.P[user_id] = self.P[user_id] + self.lr * \
                                  (e_ui*self.Q[item_id] - self._lambda*self.P[user_id])
                self.Q[item_id] = self.Q[item_id] + self.lr * \
                                  (e_ui*self.P[user_id] - self._lambda*self.Q[item_id])
            if verbose:
                if i_epoch % 10 == 0:
                    user_emb = self.P[train_df[user_col]]
                    item_emb = self.Q[train_df[item_col]]
                    y_pred = np.sum(user_emb * item_emb, axis=1)
                    print(f"{i_epoch} : RMSE =  ",
                          np.round(calc_rmse(train_df["rating"], y_pred), 4))

    def predict(self, test_df) -> pd.Series:
        test_df = test_df.copy(deep=True)
        # TODO: Handle cold-start problem
        test_df[self.user_col] = test_df[self.user_col].apply(lambda x: self.userid2idx[x])
        test_df[self.item_col] = test_df[self.item_col].apply(lambda x: self.itemid2idx[x])

        user_emb = self.P[test_df[self.user_col]]
        item_emb = self.Q[test_df[self.item_col]]
        predictions = np.sum(user_emb * item_emb, axis=1)

        return predictions


class MatrixFactorization_rating_matrix_version:
    def __init__(self, item_col, user_col, rating_col, emb_dim, lr, _lambda):
        self.item_col = item_col
        self.user_col = user_col
        self.rating_col = rating_col

        self.emb_dim = emb_dim
        self.lr = lr
        self._lambda = _lambda

    def _initialize_emb_vector(self, R):
        self.n_users, self.n_items  = R.shape
        self.P = initialize_emb_vector(self.n_users, self.emb_dim, dist="normal")
        self.Q = initialize_emb_vector(self.n_items, self.emb_dim, dist="normal")

    def fit(self, R, epoch=10, verbose=False):
        """
        rating matrix as input version.
        """
        self._initialize_emb_vector(R)

        regularization = True
        for i_epoch in range(epoch):
            for i in range(self.n_users):
                for j in range(self.n_items):
                    r_ui = R[i, j]
                    if np.isnan(r_ui):  # skip empty elements
                        continue
                    pred_r_ui = np.dot(self.P[i, :], self.Q[j, :])
                    e_ui = r_ui - pred_r_ui
                    if not regularization:
                        self.P[i, :] = self.P[i, :] + self.lr * e_ui * self.Q[j, :]
                        self.Q[j, :] = self.Q[j, :] + self.lr * e_ui * self.P[i, :]
                        continue
                    # Done for user_i with all item embedding vectors that had interactions
                    # then move on(all j is done, next i) to next user, repeat the process.
                    # updates user emb vector(P_i) using item emb vector Q_j
                    self.P[i, :] = self.P[i, :] + self.lr * (e_ui * self.Q[j, :] - self._lambda * self.P[i,:])
                    # use updated user emb vector to update Q_j
                    self.Q[j, :] = self.Q[j, :] + self.lr * (e_ui * self.P[i, :] - self._lambda * self.Q[j, :])
            if verbose:
                if i_epoch % 10 == 0:
                    print(f"epoch = {i} : RMSE =  ", np.round(calc_rmse_rating_matrix(self.P, self.Q, R), 4))
        print("Done training")

    def predict(self):
        return np.dot(self.P, self.Q.T)
