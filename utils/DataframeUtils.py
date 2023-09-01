# utility set of funcs for manipulating dataframes & tensors
# Note: this is a class, so you need to instantiate an object to use the functions here. The reason for this is
# that we can then run multiple strategies simultaneously - if everything were to be static, one strat could reset
# state needed by another strat. Also, you may sometimes need to process multiple dataframes at the same time

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
from enum import Enum

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from pandas import DataFrame, Series
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'


class ScalerType(Enum):
    NoScaling = 0
    Standard = 1
    Robust = 2
    MinMax = 3


class DataframeUtils():

    #################################
    # scaling utilities etc.

    scaler = None
    scaler_type:ScalerType = ScalerType.NoScaling
    scaler_fitted = False


    # sets the type of scaler desired, and initialises associated vars
    def set_scaler_type(self, type:ScalerType):
        self.scaler_type = type
        self.scaler_fitted = False
        self.scaler = None
        self.scaler = self.get_scaler()
        # print(f"    Scaler set to: {type}")

    # get a scaler for scaling/normalising the data (in a func because I change it routinely)
    def get_scaler(self):

        # only create if it doesn't yet exist
        if self.scaler is None:
            self.scaler = self.make_scaler()
            self.scaler_fitted = False

        return self.scaler

    # make a scaler that matches the type set with self.scaler_type
    def make_scaler(self):
        scaler = None
        if (self.scaler_type == ScalerType.NoScaling):
            print("    Data will not be scaled")
        elif self.scaler_type == ScalerType.Standard:
            scaler = StandardScaler()
        elif self.scaler_type == ScalerType.MinMax:
            scaler = MinMaxScaler()
        elif self.scaler_type == ScalerType.Robust:
            scaler = RobustScaler()
        else:
            print(f"    Unknown scaler type: {self.scaler_type}")

        return scaler

    def fit_scaler(self, dataframe: DataFrame):
        if self.scaler is not None:
            # if self.scaler_fitted:
            #     print("    Warning: re-fitting scaler")
            self.scaler = self.scaler.fit(dataframe)
            self.scaler_fitted = True
        else:
            print("    WARN: fit_scaler() called, but scaler has not been assigned")
        return

    ###################################
    # debug utilities

    def check_nan(self, dataframe) -> bool:
        found = False
        col_name = dataframe.columns[dataframe.isna().any()].tolist()
        if len(col_name) > 0:
            print("*** NaN in cols: ", col_name)
            found = True
        return found

    def check_inf(self, dataframe) -> bool:
        found = False
        col_name = dataframe.columns.to_series()[np.isinf(dataframe).any()]
        if len(col_name) > 0:
            print("*** Infinity in cols: ", col_name)
            found = True
        return found


    def remove_debug_columns(self, dataframe: DataFrame) -> DataFrame:
        drop_list = dataframe.filter(regex='^%').columns
        if len(drop_list) > 0:
            for col in drop_list:
                dataframe = dataframe.drop(col, axis=1)
            dataframe.reindex()
        return dataframe


    ###################################

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:

        self.check_inf(dataframe)

        df = dataframe.copy()


        # if scaler not created, then do so
        if self.scaler is  None:
            self.scaler = self.get_scaler()

        # convert date column so that it can be scaled.
        # Also, add in date components (maybe there's a pattern, who knows?)
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

        df = self.remove_debug_columns(df)

        df.set_index('date')
        df.reindex()

        cols = df.columns


        # if no scaling then don't transform the dataframe (still need the date conversion though)
        if self.scaler_type != ScalerType.NoScaling:

            # fit, if not already done
            # Note that fitting is only done once, then reused on subsequent calls to norm/denorm.
            # Call set_scaler() to reset
            if not self.scaler_fitted:
                self.fit_scaler(df)
            # self.fit_scaler(df)

            df = self.scaler.transform(df)

        df = pd.DataFrame(df, columns=cols)

        # print("norm_dataframe()") # debug

        return df


    # De-Normalise a dataframe - note this relies on the scaler still being valid
    def denorm_dataframe(self, dataframe: DataFrame) -> DataFrame:


        if self.scaler_type == ScalerType.NoScaling:
            return dataframe
        
        if self.scaler is  None:
            print("   WARN: scaler is None")

        if not self.scaler_fitted:
            print("   WARN: scaler has not been fitted")

        cols = dataframe.columns

        # print(f"Scaler type:{self.scaler_type}  fitted:{self.scaler_fitted}")
        df = pd.DataFrame(self.scaler.inverse_transform(dataframe), columns=cols)

        # print("denorm_dataframe()") # debug

        return df


    ###################################


    # map column into [0,1]
    def get_binary_labels(self, col):
        binary_encoder = LabelEncoder().fit([min(col), max(col)])
        result = binary_encoder.transform(col)
        # print ("label input:  ", col)
        # print ("label output: ", result)
        return result

    # remove outliers from normalised dataframe
    def remove_outliers(self, df_norm: DataFrame, buys, sells):
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

    ###################################
    # train/test dataset utilities

    # build a 'viable' dataframe sample set. Needed because the positive labels are sparse
    def build_viable_dataset(self, size: int, df_norm: DataFrame, buys, sells):
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


    # build a dataset that mimics 'live' runs
    def build_standard_dataset(self, size: int, df_norm: DataFrame, buys, sells, lookahead, seq_len):

        # constrain size to what will be available in run modes
        # df_size = df_norm.shape[0]
        df_size = df_norm.shape[0]
        # data_size = int(min(975, size))
        data_size = size

        pad = lookahead  # have to allow for future results to be in range

        # trying different test options. For some reason, results vary quite dramatically based on the approach

        test_option = 0
        if test_option == 0:
            # take the end  (better fit for recent data). The most realistic option
            start = int(df_size - (data_size + pad))
        elif test_option == 1:
            # take the middle part of the full dataframe
            start = int((df_size - (data_size + lookahead)) / 2)
        elif test_option == 2:
            # use the front part of the data
            start = 0
        elif test_option == 3:
            # search buys array to find window with most buys? Cheating?!
            start = 0
            num_iter = df_size - data_size
            if num_iter > 0:
                max_buys = 0
                for i in range(num_iter):
                    num_buys = buys[i:i + data_size - 1].sum()
                    if num_buys > max_buys:
                        start = i
        else:
            # take the end  (better fit for recent data)
            start = int(df_size - (data_size + pad))

        result_start = start + lookahead

        # just double-check ;-)
        if (data_size + lookahead) > df_size:
            print("ERR: invalid data size")
            print("     df:{} data_size:{}".format(df_size, data_size))

        # print("    df:[{}:{}] start:{} end:{} length:{}]".format(0, (data_size - 1),
        #                                                          start, (start + data_size), data_size))

        # convert dataframe to tensor before extracting train/test data (avoid edge effects)
        tensor = self.df_to_tensor(df_norm, seq_len)
        buy_tensor = self.df_to_tensor(np.array(buys).reshape(-1, 1), seq_len)
        sell_tensor = self.df_to_tensor(np.array(sells).reshape(-1, 1), seq_len)

        # extract desired rows
        t = tensor[start:start + data_size]
        b = buy_tensor[start:start + data_size]
        s = sell_tensor[start:start + data_size]

        return t, b, s

    ###################################
    # Utilities for 'splitting' various datastructures

    # slit a dataframe into two, based on the supplied ratio
    def split_dataframe(self, dataframe: DataFrame, ratio: float) -> (DataFrame, DataFrame):
        split_row = int(ratio * dataframe.shape[0])
        df1 = dataframe.iloc[0:split_row].copy()
        df2 = dataframe.iloc[split_row + 1:].copy()
        return df1, df2


    # slit an array into two, based on the supplied ratio
    def split_array(self, array, ratio: float) -> (DataFrame, DataFrame):
        split_row = int(ratio * np.shape(array)[0])
        a1 = array[0:split_row].copy()
        a2 = array[split_row + 1:].copy()
        return a1, a2

    # splits the tensor, buys & sells into train & test
    # this sort of emulates train_test_split, but with different options for selecting data
    #TODO: make this varable argument, i.e. any number of tensors
    def split_tensor(self, tensor, buys, sells, ratio, lookahead):

        # constrain size to what will be available in run modes
        # df_size = df_norm.shape[0]
        data_size = int(np.shape(tensor)[0])

        pad = lookahead  # have to allow for future results to be in range
        train_ratio = ratio
        test_ratio = 1.0 - train_ratio
        train_size = int(train_ratio * (data_size - pad)) - 1
        test_size = int(test_ratio * (data_size - pad)) - 1

        # trying different test options. For some reason, results vary quite dramatically based on the approach

        # test_option = 1
        test_option = 3
        if test_option == 0:
            # take the middle part of the full dataframe
            train_start = int((data_size - (train_size + test_size + lookahead)) / 2)
            test_start = train_start + train_size + 1
        elif test_option == 1:
            # take the end for training (better fit for recent data), earlier section for testing
            train_start = int(data_size - (train_size + pad))
            test_start = 0
        elif test_option == 2:
            # use the whole dataset for training, last section for testing (yes, I know this is not good)
            train_start = 0
            train_size = data_size - pad - 1
            test_start = data_size - (test_size + pad)
        else:
            # the 'classic' - first part train, last part test
            train_start = 0
            test_start = data_size - (test_size + pad) - 1

        train_result_start = train_start + lookahead
        test_result_start = test_start + lookahead

        # just double-check ;-)
        if (train_size + test_size + lookahead) > data_size:
            print("ERR: invalid train/test sizes")
            print("     train_size:{} test_size:{} data_size:{}".format(train_size, test_size, data_size))

        if (train_result_start + train_size) > data_size:
            print("ERR: invalid train result config")
            print("     train_result_start:{} train_size:{} data_size:{}".format(train_result_start,
                                                                                 train_size, data_size))

        if (test_result_start + test_size) > data_size:
            print("ERR: invalid test result config")
            print("     test_result_start:{} train_size:{} data_size:{}".format(test_result_start,
                                                                                test_size, data_size))

        # print("    data:[{}:{}] train:[{}:{}] train_result:[{}:{}] test:[{}:{}] test_result:[{}:{}] "
        #       .format(0, data_size - 1,
        #               train_start, (train_start + train_size),
        #               train_result_start, (train_result_start + train_size),
        #               test_start, (test_start + test_size),
        #               test_result_start, (test_result_start + test_size)
        #               ))

        # extract desired rows
        train_tensor = tensor[train_start:train_start + train_size]
        train_buys_tensor = buys[train_result_start:train_result_start + train_size]
        train_sells_tensor = sells[train_result_start:train_result_start + train_size]

        test_tensor = tensor[test_start:test_start + test_size]
        test_buys_tensor = buys[test_result_start:test_result_start + test_size]
        test_sells_tensor = sells[test_result_start:test_result_start + test_size]

        num_buys = train_buys_tensor[:, 0].sum()
        num_sells = train_sells_tensor[:, 0].sum()
        if (num_buys <= 2) or (num_sells <= 2):
            print("   WARNING - low number of buys/sells in training data")
            print("   training #buys:{} #sells:{} ".format(num_buys, num_sells))

        return train_tensor, test_tensor, train_buys_tensor, test_buys_tensor, train_sells_tensor, test_sells_tensor

    # convert dataframe to 3D tensor (for use with keras models)
    def df_to_tensor(self, df, seq_len):

        if self.is_dataframe(df):
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

    # utility to check whether an object is a Dataframe
    def is_dataframe(self, data) -> bool:
        ctype = str(type(data)).lower()
        return True if ('dataframe' in ctype) else False

    # utility to check whether an object is a tensor
    def is_tensor(self, data) -> bool:
        ctype = str(type(data)).lower()
        return True if ('array' in ctype) else False
