

def df_to_rating_matrix(df, item_col, user_col, rating_col):
    """
    Converts given dataframe into rating matrix
    df must contain user, item, rating columns

    :param df: pd.DataFrame

    :return: rating_matrix : np.array
    """
    return df.pivot(index=item_col, columns=user_col, values=rating_col).values