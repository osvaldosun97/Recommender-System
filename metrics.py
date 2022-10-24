import pandas as pd
import numpy as np



def calc_mse(P, Q, R):
    R_pred = np.dot(P, Q.T)
    mse = np.nanmean((R - R_pred) ** 2)
    return mse

def calc_rmse(P,Q,R):
    return np.sqrt(calc_mse(P,Q,R))
