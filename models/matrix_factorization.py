"""
Matrix Factorization based methods
"""
import numpy as np

from metrics import calc_rmse


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


def run_matrix_factorization(R, K, n_steps, lr, _lambda, method="sgd", verbose=True):
    """
    run matrix factorization and return factorized matrix.

    Assumes that rows are users and columns are items

    :param R: np.array, original rating matrix
    :param K: int, embeddinng dimension
    :param method: str
    :return: np.array, predicted rating matrix
    """
    # FIXME : should be added as function parameter
    regularization = True

    n_users, n_items = R.shape()
    P = initialize_emb_vector(n_users, K)
    Q = initialize_emb_vector(n_items, K)

    for n_step in range(1, n_steps):
        for i in range(n_users):
            for j in range(n_items):
                r_ui = R[i, j]
                if np.isnan(r_ui): # skip empty elements
                    continue
                e_ui = r_ui - np.dot(P[i, :], Q[j, :])
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
