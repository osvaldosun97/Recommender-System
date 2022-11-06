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

    def batch_reccomendation(self, test_df) -> pd.DataFrame:
        col_names = [self.user_col, self.item_col, "relevant_score", "rank"]
        user_df = test_df[[self.user_col]].copy()
        user_df['user_idx'] = user_df[self.user_col].apply(self.map_id2idx, args=(self.userid2idx, ))

        warm_user_df = user_df.loc[~user_df['user_idx'].isna()].copy()
        warm_user_df['user_idx'] = warm_user_df['user_idx'].astype(int)
        warm_user_recc_df = self.batch_recommend_warm_users(warm_user_df)
        warm_user_recc_df = warm_user_recc_df.reindex(columns=col_names)

        # TODO: recc item to cold-user based on given tactic
        #   ex -> just recc MP to all
        #   ex2 -> perform MAB on cold-users -> where arms are binned by popularity.
        cold_user_df = user_df.loc[user_df['user_idx'].isna()].copy()
        cold_user_recc_df = self.batch_recommend_cold_users(cold_user_df)
        return pd.concat([warm_user_recc_df, cold_user_recc_df])

    def fit(self, train_df, user_col, item_col, rating_col, epoch=10, verbose=True):
        """
        train user/item embedding vector

        :param epoch: int, number of training iteration
        :param verbose: Bool
        """
        train_df = train_df.copy(deep=True)
        self.train_df = train_df
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.unique_user_ids = train_df[self.user_col].unique()
        self.unique_item_ids = train_df[self.item_col].unique()

        # For cold-users
        self.mp_df_sorted = (self.train_df
                              .groupby("movieId")
                              .agg({"rating": ["mean", "count"]})['rating']
                              .sort_values(["mean", "count"], ascending=False))

        # re-index for efficient lookup of embedding vectors.
        self.userid2idx = {u_id: i for i, u_id in enumerate(self.unique_user_ids)}
        self.itemid2idx = {u_id: i for i, u_id in enumerate(self.unique_item_ids)}
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

    def get_topK(self, user_emb, topk=5) -> pd.Series:
        """
        Given user_emb, retrieve relevance score for all items from
        previous training sample and return top K most relevant items
        ordered from most relevant to least relevant

        :param int, number of items to return
        :return pd.DataFrame : predicted score for topK items.
        """
        user_emb_rep = np.tile(user_emb, (self.n_items, 1))
        relevant_scores = np.sum(user_emb_rep * self.Q, axis=1)
        recc_df = pd.DataFrame(relevant_scores, columns=['relevant_score'])
        recc_df[self.item_col] = self.unique_item_ids
        topk_recc_df = recc_df.sort_values("relevant_score", ascending=False)[:topk]
        return topk_recc_df

    def recommend_item(self, userid, topk=5) -> pd.Series:
        try:
            user_emb = self.P[self.userid2idx[userid]]
            recc_items = self.get_topK(user_emb, topk=topk)[self.item_col].tolist()
        except KeyError:
            recc_items = self.mp_df_sorted.index[:topk].tolist()
        return recc_items

    def batch_recommend_cold_users(self, cold_user_df:pd.DataFrame, topk=5, method='most_popular') -> pd.DataFrame:
        """
        batch recommendation for cold-start users
        """
        n_rows = len(cold_user_df)
        if method == "most_popular":
            mp_df_sorted = (self.train_df
                             .groupby("movieId")
                             .agg({"rating": ["mean", "count"]})['rating']
                             .sort_values(["mean", "count"], ascending=False))
            recc_result_df = pd.DataFrame(np.repeat(cold_user_df[self.user_col], topk))
            topk_items = mp_df_sorted.index[:topk].tolist()
            recc_result_df[self.item_col] = topk_items*n_rows
            recc_result_df['relevant_score'] = np.nan
            recc_result_df['rank'] = list(range(1, topk+1))*n_rows
        else:
            raise NameError(f"method named {method} does not exists.")

        return recc_result_df

    def batch_recommend_warm_users(self, warm_user_df:pd.DataFrame, topk=5) -> pd.DataFrame:
        """
        batch recommendation for warm-start users
        """
        user_embs = self.P[warm_user_df['user_idx']]
        user_ids = []
        topk_recc_dfs = []
        for user_emb, user_id in zip(user_embs, warm_user_df[self.user_col]):
            topk_recc_df = self.get_topK(user_emb, topk=topk)
            user_ids += [user_id]*topk
            topk_recc_dfs.append(topk_recc_df)
        recc_result_df = pd.concat(topk_recc_dfs)
        recc_result_df[self.user_col] = user_ids
        recc_result_df["rank"] = list(range(1, topk+1))*len(warm_user_df)
        return recc_result_df

    def map_id2idx(self, value, _map):
        """
        map userid to id that has been trained. For new users, return np.nan

        :param value: int, id to map to idx.
        :param _map: dictionary, for mapping
        :return: float, index
        """
        try:
            return _map[value]
        except KeyError:
            return np.nan

    def predict(self, test_df) -> pd.Series:
        """
        *Assumes there are no-cold-start users/items.

        :param test_df:
        :return:
        """
        test_df = test_df.copy(deep=True)
        # TODO: Handle cold-start problem
        try:
            test_df[self.user_col] = test_df[self.user_col].apply(lambda x: self.userid2idx[x])
        except KeyError:
            raise Exception("There are cold-start users, please use batch_recommendation function instead")
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
