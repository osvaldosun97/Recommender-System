"""
Matrix Factorization based methods
"""
import numpy as np

from metrics import calc_rmse
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


class MatrixFactorization:
    def __init__(self, rating_df, item_col, user_col, rating_col, emb_dim, lr, _lambda):
        self.rating_df = rating_df
        self.item_col = item_col
        self.user_col = user_col
        self.rating_col = rating_col

        self.emb_dim = emb_dim
        self.lr = lr
        self._lambda = _lambda

        self.n_users, self.n_items  = self.rating_df[[self.user_col, self.item_col]].nunique()

        self.P = initialize_emb_vector(self.n_users, self.emb_dim, dist="normal")
        self.Q = initialize_emb_vector(self.n_items, self.emb_dim, dist="normal")

    def fit(self, epoch=10, verbose=True):
        """
        train user/item embedding vector

        :param epoch: int, number of training iteration
        :param verbose: Bool
        """

        regularization = True
        for i in range(epoch):
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
                if i % 10 == 0:
                    print(f"epoch = {i} : RMSE =  ", np.round(calc_rmse(self.P, self.Q, self.rating_df), 4))




def run_matrix_factorization_rating_matrix(R, K, n_steps, lr, _lambda, method="sgd", verbose=True, timeit=True):
    """
    rating matrix as input version.
    """
    if timeit:
        start_time = timer()
    # FIXME : should be added as function parameter
    regularization = True

    n_users, n_items = R.shape
    P = initialize_emb_vector(n_users, K)
    Q = initialize_emb_vector(n_items, K)

    for n_step in range(1, n_steps):
        for i in range(n_users):
            for j in range(n_items):
                r_ui = R[i, j]
                if np.isnan(r_ui): # skip empty elements
                    continue
                pred_r_ui = np.dot(P[i, :], Q[j, :])
                e_ui = r_ui - pred_r_ui
                if not regularization:
                    P[i, :] = P[i, :] + lr * e_ui * Q[j, :]
                    Q[j, :] = Q[j, :] + lr * e_ui * P[i, :]
                    continue
                # Done for user_i with all item embedding vectors that had interactions
                # then move on(all j is done, next i) to next user, repeat the process.
                P[i, :] = P[i, :] + lr * (e_ui * Q[j, :] - _lambda * P[i,:])  # updates user emb vector(P_i) using item emb vector Q_j
                Q[j, :] = Q[j, :] + lr * (e_ui * P[i, :] - _lambda * Q[j, :])  # use updated user emb vector to update Q_j
        if verbose:
            if n_step % 10 == 0:
                print(f"{n_step} : RMSE =  ", np.round(calc_rmse(P, Q, R), 4))

    if timeit:
        end_time = timer()
        print(f"seconds took : {round(end_time - start_time, 3)}s")
    return P, Q

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
