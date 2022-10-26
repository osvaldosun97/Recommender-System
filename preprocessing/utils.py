import numpy as np
import pandas as pd



def analyze_data():
    pass

def get_new_dtype(desc_series, dtype, c):
    """
    :param desc_series:
    :param dtype:
    :param c: int, confidence you want. higher confidence means that even if new value comes in for
           column it will still be within range of new dtype.
    :return:
    """
    _min, _max = desc_series[['min', "max"]]
    diff = _max - _min
    if dtype == "integer":
        if diff*c < (np.iinfo(np.int8).max - np.iinfo(np.int8).min):
            # check if int8 is plausible
            return "int8"
        elif diff * c < (np.iinfo(np.int16).max - np.iinfo(np.int16).min):
            return "int16"
        elif diff * c < (np.iinfo(np.int32).max - np.iinfo(np.int32).min):
            return "int32"
        else:
            return "int64"

    elif dtype == "float":
        float16_max = np.finfo(np.float16).max
        float16_min = np.finfo(np.float16).min
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min

        if diff*c < (float16_max - float16_min):
            # check if float16 is plausible
            return "float16"
        elif diff * c < (float32_max - float32_min):
            return "float32"
        else:
            return "float64"
    else:
        raise KeyError("Unsupported dtype")

def optimize_dtypes(df, verbose, types=['float', 'integer', 'object']):
    """
    For given dataframe convert dtypes for maximum efficiency and
    memory saving.

    This function assumes proper dtypes are already assigned.
     ex: object dtype must be converted to appropriate dtype.

    :param df : pd.DataFrame, dataframe to optimize
    :param types : list, datatypes to optimize

    :return: df : pd.DataFrame, optimized dataframe.
    """
    n_rows = len(df)
    new_dtype_dict = {}
    if verbose:
        print(f"n_rows = {n_rows}")
        print("=== BEFORE optimization ===")
        print("memory usage(Bytes)")
        print(df.memory_usage(deep=True, index=False))
        print("dtypes")
        print(df.dtypes)

    for type in types:
        if "int" in type:
            int_df = df.select_dtypes(include=['integer'])
            int_cols = int_df.columns
            if len(int_cols) == 0:
                continue
            int_df_desc = int_df.describe()

            # optimize integers.
            # TODO: check "reduced numeric ranges might lead to overflow"
            for col in int_cols:
                new_dtype = get_new_dtype(int_df_desc[col], 'integer', 2)
                new_dtype_dict[col] = new_dtype

        elif "float" in type:
            float_df = df.select_dtypes(include=['float'])
            float_cols = float_df.columns
            float_df_desc = float_df.describe()
            for col in float_cols:
                new_dtype = get_new_dtype(float_df_desc[col], 'float', 2)
                new_dtype_dict[col] = new_dtype

        elif type == 'object' or type == 'string':
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            to_cat_cols = []
            for col in str_cols:
                if df[col].nunique()*10 < n_rows:
                    new_dtype_dict[col] = "category"

    df = df.astype(new_dtype_dict)
    if verbose:
        print(f"=== AFTER optimization ===")
        print("memory usage(Bytes) ")
        print(df.memory_usage(deep=True, index=False))
        print("dtypes")
        print(df.dtypes)

    return df


def train_test_split_dataframe(df, test_ratio=0.3, method='random', time_col=None):
    """
    Splits pandas dataframe into train and test dataframe using given method and test_ratio.
    If method is 'temporal' use time_col to order them if exists.

    :param df:
    :param test_ratio:
    :param method:
    :param time_col:
    :return:
    """
    if method == 'random':
        test_df = df.sample(frac=test_ratio)
        train_df = df.loc[~df.index.isin(test_df.index)].copy()
    elif method == 'temporal':
        if time_col:
            df.sort_values(time_col, inplace=True)
        tr_upper_index = round(len(df)*(1-test_ratio))
        train_df = df.iloc[:tr_upper_index, :].copy()
        test_df = df.iloc[tr_upper_index:, :].copy()

    return train_df, test_df