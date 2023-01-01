# utility set of funcs for manipulating dataframes & tensord

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy



# get a scaler for scaling/normalising the data (in a func because I change it routinely)
def get_scaler():
    # uncomment the one yu want
    # return StandardScaler()
    # return RobustScaler()
    return MinMaxScaler()


def check_inf(dataframe):
    col_name = dataframe.columns.to_series()[np.isinf(dataframe).any()]
    if len(col_name) > 0:
        print("***")
        print("*** Infinity in cols: ", col_name)
        print("***")


def remove_debug_columns(dataframe: DataFrame) -> DataFrame:
    drop_list = dataframe.filter(regex='^%').columns
    if len(drop_list) > 0:
        for col in drop_list:
            dataframe = dataframe.drop(col, axis=1)
        dataframe.reindex()
    return dataframe


scaler = None


# Normalise a dataframe
def norm_dataframe(dataframe: DataFrame) -> DataFrame:

    global scaler

    check_inf(dataframe)

    df = dataframe.copy()
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'], utc=True)
        # dates = pd.to_datetime(df['date'])
        start_date = datetime(2020, 1, 1).astimezone(timezone.utc)
        df['date'] = dates.astype('int64')
        df['days_from_start'] = (dates - start_date).dt.days
        df['day_of_week'] = dates.dt.dayofweek
        df['day_of_month'] = dates.dt.day
        df['week_of_year'] = dates.dt.isocalendar().week
        df['month'] = dates.dt.month
        # df['year'] = dates.dt.year

    df = remove_debug_columns(df)

    df.set_index('date')
    df.reindex()

    cols = df.columns
    scaler = get_scaler()

    df = pd.DataFrame(scaler.fit_transform(df), columns=cols)

    return df


# De-Normalise a dataframe - note this relies on the scaler still being valid
def denorm_dataframe(dataframe: DataFrame) -> DataFrame:

    global scaler

    df = dataframe.copy()

    cols = df.columns

    df = pd.DataFrame(scaler.inverse_transform(dataframe), columns=cols)

    return df


# slit a dataframe into two, based on the supplied ratio
def split_dataframe(dataframe: DataFrame, ratio: float) -> (DataFrame, DataFrame):
    split_row = int(ratio * dataframe.shape[0])
    df1 = dataframe.iloc[0:split_row].copy()
    df2 = dataframe.iloc[split_row + 1:].copy()
    return df1, df2


# slit an array into two, based on the supplied ratio
def split_array(array, ratio: float) -> (DataFrame, DataFrame):
    split_row = int(ratio * np.shape(array)[0])
    a1 = array[0:split_row].copy()
    a2 = array[split_row + 1:].copy()
    return a1, a2


# remove outliers from normalised dataframe
def remove_outliers(df_norm: DataFrame, buys, sells):
    # for col in df_norm.columns.values:
    #     if col != 'date':
    #         df_norm = df_norm[(df_norm[col] <= 3.0)]
    # return df_norm
    df = df_norm.copy()
    df['%temp_buy'] = buys.copy()
    df['%temp_sell'] = sells.copy()
    #
    df2 = df[((df >= -3.0) & (df <= 3.0)).all(axis=1)]
    # df_out = df[~((df >= -3.0) & (df <= 3.0)).all(axis=1)] # for debug
    ndrop = df_norm.shape[0] - df2.shape[0]
    if ndrop > 0:
        b = df2['%temp_buy'].copy()
        s = df2['%temp_sell'].copy()
        df2.drop('%temp_buy', axis=1, inplace=True)
        df2.drop('%temp_sell', axis=1, inplace=True)
        df2.reindex()
        # if dbg_verbose:
        print("    Removed ", ndrop, " outliers")
        # print(" df2:", df2)
        # print(" df_out:", df_out)
        # print ("df_norm:", df_norm.shape, "df2:", df2.shape, "df_out:", df_out.shape)
    else:
        # no outliers, just return originals
        df2 = df_norm
        b = buys
        s = sells
    return df2, b, s


# build a 'viable' dataframe sample set. Needed because the positive labels are sparse
def build_viable_dataset(size: int, df_norm: DataFrame, buys, sells):
    # if dbg_verbose:
    #     print("     df_norm:{} size:{} buys:{} sells:{}".format(df_norm.shape, size, buys.shape[0], sells.shape[0]))

    # copy and combine the data into one dataframe
    df = df_norm.copy()
    df['%temp_buy'] = buys.copy()
    df['%temp_sell'] = sells.copy()

    # df_buy = df[( (df['%temp_buy'] > 0) ).all(axis=1)]
    # df_sell = df[((df['%temp_sell'] > 0)).all(axis=1)]
    # df_nosig = df[((df['%temp_buy'] == 0) & (df['%temp_sell'] == 0)).all(axis=1)]

    df_buy = df.loc[df['%temp_buy'] == 1]
    df_sell = df.loc[df['%temp_sell'] == 1]
    df_nosig = df.loc[(df['%temp_buy'] == 0) & (df['%temp_sell'] == 0)]

    # make sure there aren't too many buys & sells
    # We are aiming for a roughly even split between buys, sells, and 'no signal' (no buy or sell)
    max_signals = int(2 * size / 3)
    buy_train_size = df_buy.shape[0]
    sell_train_size = df_sell.shape[0]

    if max_signals > df_nosig.shape[0]:
        max_signals = int((size - df_nosig.shape[0])) - 1

    if ((df_buy.shape[0] + df_sell.shape[0]) > max_signals):
        # both exceed max?
        sig_size = int(max_signals / 2)
        # if dbg_verbose:
        #     print("     sig_size:{} max_signals:{} buys:{} sells:{}".format(sig_size, max_signals, df_buy.shape[0],
        #                                                                     df_sell.shape[0]))

        if (df_buy.shape[0] > sig_size) & (df_sell.shape[0] > sig_size):
            # resize both buy & sell to 1/3 of requested size
            buy_train_size = sig_size
            sell_train_size = sig_size
        else:
            # only one them is too big, so figure out which
            if (df_buy.shape[0] > df_sell.shape[0]):
                buy_train_size = max_signals - df_sell.shape[0]
            else:
                sell_train_size = max_signals - df_buy.shape[0]

        # if dbg_verbose:
        #     print("     buy_train_size:{} sell_train_size:{}".format(buy_train_size, sell_train_size))

    if buy_train_size < df_buy.shape[0]:
        df_buy, _ = train_test_split(df_buy, train_size=buy_train_size, shuffle=False)
    if sell_train_size < df_sell.shape[0]:
        df_sell, _ = train_test_split(df_sell, train_size=sell_train_size, shuffle=False)

    # extract enough rows to fill the requested size
    fill_size = size - buy_train_size - sell_train_size - 1
    # if dbg_verbose:
    #     print("     df_nosig:{} fill_size:{}".format(df_nosig.shape, fill_size))

    if fill_size < df_nosig.shape[0]:
        df_nosig, _ = train_test_split(df_nosig, train_size=fill_size, shuffle=False)

    # print("viable df - buys:{} sells:{} fill:{}".format(df_buy.shape[0], df_sell.shape[0], df_nosig.shape[0]))

    # concatenate the dataframes
    frames = [df_buy, df_sell, df_nosig]
    df2 = pd.concat(frames)

    # # shuffle rows
    # df2 = df2.sample(frac=1)

    # separate out the data, buys & sells
    b = df2['%temp_buy'].copy()
    s = df2['%temp_sell'].copy()
    df2.drop('%temp_buy', axis=1, inplace=True)
    df2.drop('%temp_sell', axis=1, inplace=True)
    df2.reindex()

    # print("     df2:", df2.shape, " b:", b.shape, " s:", s.shape)

    return df2, b, s


# map column into [0,1]
def get_binary_labels(col):
    binary_encoder = LabelEncoder().fit([min(col), max(col)])
    result = binary_encoder.transform(col)
    # print ("label input:  ", col)
    # print ("label output: ", result)
    return result


# convert dataframe to 3D tensor (for use with keras models)
def df_to_tensor(df, seq_len):

    if not isinstance(df, type([np.ndarray, np.array])):
        data = np.array(df)
    else:
        data = df

    nrows = np.shape(data)[0]
    nfeatures = np.shape(data)[1]
    tensor_arr = np.zeros((nrows, seq_len, nfeatures), dtype=float)
    zero_row = np.zeros((nfeatures), dtype=float)
    # tensor_arr = []

    # print("data:{} tensor:{}".format(np.shape(data), np.shape(tensor_arr)))
    # print("nrows:{} nfeatures:{}".format(nrows, nfeatures))

    reverse = True

    # fill the first part (0..seqlen rows), which are only sparsely populated
    for row in range(seq_len):
        for seq in range(seq_len):
            if seq >= (seq_len - row - 1):
                src_row = (row + seq) - seq_len + 1
                tensor_arr[row][seq] = data[src_row]
            else:
                tensor_arr[row][seq] = zero_row
        if reverse:
            tensor_arr[row] = np.flipud(tensor_arr[row])

    # fill the rest
    # print("Data:{}, len:{}".format(np.shape(data), seq_len))
    for row in range(seq_len, nrows):
        tensor_arr[row] = data[(row - seq_len) + 1:row + 1]
        if reverse:
            tensor_arr[row] = np.flipud(tensor_arr[row])

    # print("data: ", data)
    # print("tensor: ", tensor_arr)
    # print("data:{} tensor:{}".format(np.shape(data), np.shape(tensor_arr)))
    return tensor_arr