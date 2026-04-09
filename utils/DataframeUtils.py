# utility set of funcs for manipulating dataframes & tensors
# Note: this is a class, so you need to instantiate an object to use the functions here. The reason for this is
# that we can then run multiple strategies simultaneously - if everything were to be static, one strat could reset
# state needed by another strat. Also, you may sometimes need to process multiple dataframes at the same time

# pragma pylint: disable=W0105, C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413, W0611, W1203, W291

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# from pandas.core.generic import DataFrameFormatter

sys.path.append(str(Path(__file__).parent))

import logging
import warnings
from enum import Enum
from typing import Dict, Tuple

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import tensorflow as tf

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
    def set_scaler_type(self, stype:ScalerType):
        self.scaler_type = stype
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

    def reset_scaler(self):
        """
        Reset the scaler to force re-fitting on next use
        """
        self.scaler_fitted = False

    def remove_debug_columns(self, dataframe: DataFrame) -> DataFrame:
        drop_list = dataframe.filter(regex="^%").columns
        if len(drop_list) > 0:
            for col in drop_list:
                dataframe = dataframe.drop(col, axis=1)
            dataframe.reindex()
        return dataframe

    def one_hot_encode(self, labels, num_classes):
        """
        Convert integer labels to one-hot encoded format
        
        Args:
            labels: numpy array of integer labels
            num_classes: number of classes (should be max label + 1)
            
        Returns:
            numpy array of shape (len(labels), num_classes) with one-hot encoding
        """
        if isinstance(labels, pd.Series):
            labels = labels.values

        # Ensure labels are integers
        labels = labels.astype(int)

        # Guard against invalid classes (negative) and mismatched num_classes
        if len(labels) == 0:
            return np.zeros((0, max(num_classes, 0)), dtype=np.float32)

        if np.any(labels < 0):
            log.warning("one_hot_encode: negative labels found, clipping to 0")
            labels = np.clip(labels, 0, None)

        max_label = int(labels.max())
        if num_classes <= max_label:
            num_classes = max_label + 1

        # Create one-hot encoding
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1.0

        return one_hot

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

    ###################################

    # Normalise a dataframe
    def norm_dataframe(self, dataframe: DataFrame) -> DataFrame:

        self.check_nan(dataframe)
        self.check_inf(dataframe)

        df = dataframe.copy()

        # if scaler not created, then do so
        if self.scaler is  None:
            self.scaler = self.get_scaler()

        # convert date column so that it can be scaled.
        # Also, add in date components (maybe there's a pattern, who knows?)
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], utc=True)
            # start_date = datetime(2020, 1, 1).astimezone(timezone.utc)
            df['date'] = dates.astype(int) / 10**9
            # df['days_from_start'] = (dates - start_date).dt.days
            # df['day_of_week'] = dates.dt.dayofweek
            # df['day_of_month'] = dates.dt.day
            # df['week_of_year'] = dates.dt.isocalendar().week
            # df['month'] = dates.dt.month
            # df['year'] = dates.dt.year

        df = self.remove_debug_columns(df)

        if 'date' in df.columns:
            df.set_index('date')
            df.reindex()

        cols = df.columns

        # if no scaling then don't transform the dataframe (still need the date conversion though)
        if self.scaler_type != ScalerType.NoScaling:

            # Clip extreme values before scaling to prevent numerical instability
            df = df.clip(-100, 100)  # Clip to reasonable range

            # fit, if not already done
            # Note that fitting is only done once, then reused on subsequent calls to norm/denorm.
            # Call set_scaler() to reset
            if not self.scaler_fitted:
                self.fit_scaler(df)
            # self.fit_scaler(df)

            df = self.scaler.transform(df)
            df = pd.DataFrame(df, columns=cols)

            # Clip extreme values after normalization to prevent numerical instability
            df = df.clip(-10, 10)  # More conservative clipping after normalization

        # print("norm_dataframe()") # debug

        self.check_nan(df)
        self.check_inf(df)

        return df

    # De-Normalise a dataframe - note this relies on the scaler still being valid
    def denorm_dataframe(self, dataframe: DataFrame) -> DataFrame:

        if self.scaler_type == ScalerType.NoScaling:
            return dataframe

        if self.scaler is None:
            log.warning("denorm_dataframe() called without a fitted scaler; returning original dataframe")
            return dataframe

        if not self.scaler_fitted:
            log.warning("denorm_dataframe() called before scaler was fitted; returning original dataframe")
            return dataframe

        cols = dataframe.columns

        # print(f"Scaler type:{self.scaler_type}  fitted:{self.scaler_fitted}")
        df = pd.DataFrame(self.scaler.inverse_transform(dataframe), columns=cols)

        # print("denorm_dataframe()") # debug

        return df

    # Converts the date index into a numeric version, to allow scaling
    def convert_index(self, dataframe: DataFrame) -> DataFrame:

        self.check_inf(dataframe)

        df = dataframe.copy()

        # convert date column so that it can be scaled.
        # Also, add in date components (maybe there's a pattern, who knows?)
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'], utc=True)
            # start_date = datetime(2020, 1, 1).astimezone(timezone.utc)
            df['date'] = dates.astype(int) / 10**9
            # df['days_from_start'] = (dates - start_date).dt.days
            # df['day_of_week'] = dates.dt.dayofweek
            # df['day_of_month'] = dates.dt.day
            # df['week_of_year'] = dates.dt.isocalendar().week
            # df['month'] = dates.dt.month
            # df['year'] = dates.dt.year

        df = self.remove_debug_columns(df)

        if 'date' in df.columns:
            df.set_index('date')
            df.reindex()

        # cols = df.columns

        # df = pd.DataFrame(df, columns=cols)

        # print("norm_dataframe()") # debug

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
    def split_dataframe(self, dataframe: DataFrame, ratio: float) -> Tuple[DataFrame, DataFrame]:
        split_row = int(ratio * dataframe.shape[0])
        df1 = dataframe.iloc[0:split_row].copy()
        df2 = dataframe.iloc[split_row + 1:].copy()
        return df1, df2

    # slit an array into two, based on the supplied ratio
    def split_array(self, array, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
        split_row = int(ratio * np.shape(array)[0])
        a1 = array[0:split_row].copy()
        a2 = array[split_row + 1:].copy()
        return a1, a2

    # splits the tensor, buys & sells into train & test
    # this sort of emulates train_test_split, but with different options for selecting data
    # TODO: make this varable argument, i.e. any number of tensors
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
    def df_to_tensor(self, df, seq_len, method=1):
        """
        Convert DataFrame to 3D tensor for sequence models.
        
        Args:
            df: Input DataFrame or numpy array
            seq_len: Length of each sequence
            method: Conversion method (default=1)
                - method=0: TensorFlow approach - reduces sample count by (seq_len-1)
                - method=1: Fast version - preserves sample count, pads with zeros (default)
                - method=2: Manual approach - preserves sample count, different padding pattern
        """
        if self.is_dataframe(df):
            data = np.array(df)
        else:
            data = df

        if method==0:
            # using tensorflow utilities
            tensor = tf.convert_to_tensor(data, dtype=tf.float32)

            # Reshape for sequences
            num_features = tensor.shape[1]
            total_sequences = tensor.shape[0] - seq_len + 1
            # reshaped_tensor = tf.stack([tensor[i:i+seq_len] for i in range(total_sequences)]) # type: ignore

            # reshaped_tensor = tf.tile(tf.expand_dims(tensor, axis=1), [1, seq_len, 1])

            # Create a dataset from the input tensor
            dataset = tf.data.Dataset.from_tensor_slices(tensor)

            # Create windowed sequences
            windowed_dataset = dataset.window(seq_len, shift=1, drop_remainder=True)

            # Convert the windowed dataset into a dataset of sequences
            sequence_dataset = windowed_dataset.flat_map(lambda window: window.batch(seq_len))

            # Batch the sequences (optional, adjust batch_size as needed)
            batch_size = 1000  # Adjust based on your memory constraints and dataset size
            batched_dataset = sequence_dataset.batch(batch_size)

            # Prefetch for better performance (optional)
            prefetched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

            # Convert the dataset to a tensor
            reshaped_tensor = tf.concat(list(prefetched_dataset), axis=0)

            tensor_arr = reshaped_tensor.numpy()
            # print(f'data:{data.shape} tensor_with_seq:{tensor_with_seq.shape} tensor_arr:{tensor_arr.shape}')

        elif method==1:
            # Vectorized Numpy version - much faster than Python loop
            num_samples, num_features = data.shape
            padded_data = np.pad(data, ((seq_len - 1, 0), (0, 0)), mode='constant')
            
            # Use sliding_window_view for O(1) memory view or fast copy
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                # sliding_window_view along axis 0 returns (num_samples, num_features, seq_len)
                tensor_arr = sliding_window_view(padded_data, seq_len, axis=0)
                # Transpose to (num_samples, seq_len, num_features)
                tensor_arr = tensor_arr.transpose(0, 2, 1).copy()
            except (ImportError, AttributeError):
                # Fallback for old numpy
                tensor_arr = np.zeros((num_samples, seq_len, num_features))
                for i in range(num_samples):
                    start_idx = max(0, i - seq_len + 1)
                    tensor_arr[i, -min(i + 1, seq_len):, :] = data[start_idx:i + 1, :]

        elif method==3:
            # MLX accelerated version - extremely fast on Apple Silicon
            try:
                import mlx.core as mx
                num_samples, num_features = data.shape
                
                # 1. Convert to MLX array (if not already)
                data_mx = mx.array(data, dtype=mx.float32)
                
                # 2. Pad with zeros for the beginning of the sequence
                padded = mx.concatenate([mx.zeros((seq_len - 1, num_features)), data_mx], axis=0)
                
                # 3. Use vectorized indexing to create the 3D tensor
                idx = mx.arange(num_samples)[:, None] + mx.arange(seq_len)
                
                # 4. Return as MLX array directly (fastest for MLX models)
                tensor_arr = padded[idx]
            except (ImportError, AttributeError):
                # Fallback to vectorized Numpy if MLX is missing
                return self.df_to_tensor(data, seq_len, method=1)

        else:
            # slower, more 'manual' approach
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

    ########################################################

    # the following are intended to help with normalising a dataframe where only some columns need to be
    # handled, and those will be handlded in a rolling fashion to avoid lookahead bias

    # debug function to suggest which columns might be unbounded (i.e. should be normalised)
    def detect_unbounded_columns(self, 
        df, sample_fraction=0.1, threshold_range=10, threshold_quantile=0.98, threshold_abs_value=2
    ):
        """
        Heuristic to detect likely unbounded columns in a DataFrame.

        Parameters:
            df: pd.DataFrame - DataFrame with time series features
            sample_fraction: float - fraction of rows to sample for speed (default 10%)
            threshold_range: float - min range (max-min) to consider unbounded
            threshold_quantile: float - quantile to check extreme values (e.g. 0.98)

        Returns:
            list of column names that are likely unbounded
        """
        # Sample to speed up if large
        if sample_fraction < 1.0:
            df_sample = df.sample(frac=sample_fraction, random_state=42)
        else:
            df_sample = df

        unbounded_columns = []
        for col in df_sample.columns:
            series = df_sample[col].dropna()
            if (
                series.empty
                or (not pd.api.types.is_numeric_dtype(series))
                or pd.api.types.is_bool_dtype(series)
            ):
                continue

            col_range = series.max() - series.min()

            # Compute high quantile absolute values (to capture extremes)
            high_val = np.abs(series.quantile(threshold_quantile))

            # Heuristics:
            # If range > threshold_range OR absolute value > threshold_abs_value,
            # consider it unbounded
            if col_range > threshold_range or high_val > threshold_abs_value:
                unbounded_columns.append(col)

        return unbounded_columns

    # 'standard' list of unbounded identifiers in TA-lib

    # rolling version of robust normalisation

    def rolling_robust_scale(self, df: pd.DataFrame, window: int = 100, min_iqr: float = 1e-6, clip_range: tuple = (-10, 10)) -> pd.DataFrame:
        """
        Applies rolling robust scaling to numeric columns in a DataFrame.
        Normalization is: (x - median) / max(IQR, min_iqr), optionally clipped.
        Non-numeric columns (like datetime index or string IDs) are untouched.
        
        Args:
            df: Input DataFrame with time series data.
            window: Rolling window size.
            min_iqr: Minimum IQR to avoid division instability.
            clip_range: Tuple to clip scaled values (set to None to disable).

        Returns:
            Scaled DataFrame with same shape and index.
        """
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            rolling_median = df[col].rolling(window, min_periods=1).median()
            q75 = df[col].rolling(window, min_periods=1).quantile(0.75)
            q25 = df[col].rolling(window, min_periods=1).quantile(0.25)
            iqr = (q75 - q25).clip(lower=min_iqr)  # clamp to avoid div-by-zero

            scaled = (df[col] - rolling_median) / iqr

            if clip_range:
                scaled = scaled.clip(*clip_range)

            df_scaled[col] = scaled

        return df_scaled

    # rolling normalise a dataframe using supplied column names
    def normalize_selected_columns(self, df: DataFrame, column_list=[], window=100):
        if len(column_list) > 0: # check that column list is supplied
            # get the columns that are in the dataframe numeric and not boolean
            actual_cols = [
                col
                for col in column_list
                if (col in df.columns and self.is_strictly_numeric(df[col]))
            ]

            if len(actual_cols) > 0:
                df = df.copy()
                # Apply rolling robust scaler only on identified columns
                df[column_list] = self.rolling_robust_scale(df[column_list], window=window)
        return df

    def is_strictly_numeric(self, series: pd.Series) -> bool:
        return (
            pd.api.types.is_numeric_dtype(series) and 
            not pd.api.types.is_bool_dtype(series)
        )

    def check_ranges(self, df: DataFrame, iqr_multiplier: float = 1.5, print_results: bool = False) -> dict:
        """
        Calculate upper and lower bounds for each column using robust scaling approach.
        Uses median and interquartile range (IQR) to identify outliers.
        
        Parameters:
            df: DataFrame to analyze
            iqr_multiplier: Multiplier for IQR to define bounds (default 1.5)
            print_results: Whether to print the results (default False)
            
        Returns:
            dict with column names and their bounds: {'col_name': {'lower': float, 'upper': float, 'outliers': int}}
        """
        results: Dict[str, Dict[str, float]] = {}

        for col in df.columns:
            if col == 'date' or col == 'index':
                continue

            series = df[col].dropna()

            # Skip non-numeric columns
            if not self.is_strictly_numeric(series):
                continue

            if series.empty:
                continue

            # Calculate robust statistics
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            # Calculate bounds using IQR method (similar to robust scaling)
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)

            # Count outliers
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()

            results[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'outliers': outliers,
                'median': median,
                'q1': q1,
                'q3': q3,
                'iqr': iqr
            }
        sorted_results: Dict[str, Dict[str, float]] = dict(
            sorted(results.items(), key=lambda item: item[1]['upper'], reverse=True)
        )

        # Print results if requested
        if print_results and len(sorted_results) > 0:
            print("\n    DEBUG: Range issues found:")
            print("    " + "-" * 50)
            print("    {:<30} {:<10} {:<10} {:<10}".format("Column", "Min", "Max", "Outliers"))
            print("    " + "-" * 50)
            for col, data in sorted_results.items():
                print("    {:<30} {:<10.2f} {:<10.2f} {:<10}".format(
                    col, data['lower'], data['upper'], data['outliers']
                ))
            print("    " + "-" * 50 + "\n")

        return sorted_results
