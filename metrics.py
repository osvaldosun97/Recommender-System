import pandas as pd
import numpy as np



def calc_mse_rating_matrix(P, Q, R):
    R_pred = np.dot(P, Q.T)
    mse = np.nanmean((R - R_pred) ** 2)
    return mse

def calc_rmse_rating_matrix(P,Q,R):
    return np.sqrt(calc_mse(P, Q, R))

def calc_mse(y: pd.Series, pred_y: pd.Series) -> float:
    return np.mean((y - pred_y)**2)
        def calc_rmse(y: pd.Series, pred_y: pd.Series) -> float:
        return np.sqrt(calc_mse(y, pred_y))