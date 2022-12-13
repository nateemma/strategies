# class that implements an Anomaly detector using Principal Component Analysis
# This works similarly to an autoencoder, where we use PCA to reduce the dimensions of the input data, then
# reverse the algorithm to reproduce the input. The initial PCA algorithm is trained on 'clean' data, so data
# containing an anomaly (buy/sell) should cause reconstruction errors


import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from sklearn.metrics import f1_score

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

import keras
from keras import layers
from sklearn.decomposition import PCA



import h5py

class AnomalyDetector_PCA():

    classifier: PCA = None
    clean_data_required = True # training data should not contain anomalies
    num_features = 0
    num_columns = 0
    data_cols = None
    find_thresholds = False
    mad_threshold = 5.0
    var_threshold = 0.95

    def __init__(self, num_features, tag=""):
        super().__init__()
        self.num_features = num_features
        self.classifier = None # created in train()  func


    # update training using the suplied (normalised) dataframe. Training is cumulative
    # the 'labels' args should contain 0.0 for normal results, '1.0' for anomalies (buy or sell)
    def train(self, df_train_norm: DataFrame, df_test_norm: DataFrame, train_labels, test_labels, force_train=False):

        if self.is_trained and not force_train:
            return


        if np.shape(df_train_norm)[0] != np.shape(train_labels)[0]:
            print("    *** ERR: dataframe and labels do not not match:")
            print("    df_train_norm:{} train_labels:{}".format(np.shape(df_train_norm), np.shape(train_labels)))

        if not self.clean_data_required:
            df1 = df_train_norm.copy()
            df1['%labels'] = train_labels
            df1 = df1[(df1['%labels'] < 0.1)]
            df_train = df1.drop('%labels', axis=1)
        else:
            df_train = df_train_norm.copy()

        # save the columns names for later
        self.data_cols = df_train.columns

        print("    fitting classifier: ", self.__class__.__name__)

        if self.find_thresholds:

            #TODO: save best results to array, then save/load
            best_vthreshold = 0.0
            top_score = 0.0
            best_th = 0.0

            print("    Searching for best variance & MAD thresholds...")
            for vth in np.arange(0.70, 0.99, 0.05):
                self.classifier = self.get_pca(df_train, vth)

                # test against original training dataframe & labels
                for mth in np.arange(2.0, 6.0, 0.2):
                    self.mad_threshold = mth
                    score = self.get_pca_score(self.classifier, df_train_norm, train_labels)
                    # print("    threshold:{} F1 score: {:.3f}".format(th, score))
                    print("    Variance threshold:{:.3f} MAD Threshold: {:.2f} ({:.4f})".format(vth,
                                                                                                mth,
                                                                                                score))
                    if score > top_score:
                        top_score = score
                        best_th = mth
                        best_vthreshold = vth
                print("")
            self.mad_threshold = best_th
            self.var_threshold = best_vthreshold
            print("    Selected: Variance threshold:{:.3f} MAD Threshold: {:.2f} ({:.4f})".format(self.var_threshold,
                                                                                                  self.mad_threshold,
                                                                                                  top_score))

        self.classifier = self.get_pca(df_train, self.var_threshold)

        return

    def get_pca(self, dataframe: DataFrame, variance_threshold: float) -> PCA:

        # do an initial fit on the full dataframe
        # classifier = PCA(n_components=self.num_features, whiten=True, svd_solver='full').fit(dataframe)
        classifier = PCA(n_components=self.num_features).fit(dataframe)
        var_ratios = classifier.explained_variance_ratio_

        # include columns until sum of contributions reaches threshold
        ncols = 0
        var_sum = 0.0

        while ((var_sum < variance_threshold) & (ncols < len(var_ratios))):
            var_sum = var_sum + var_ratios[ncols]
            ncols = ncols + 1

        self.num_columns = ncols

        # if necessary, re-calculate compressor with reduced column set
        if (ncols != dataframe.shape[1]):
            # classifier = PCA(n_components=ncols, whiten=True, svd_solver='full').fit(dataframe)
            classifier = PCA(n_components=ncols).fit(dataframe)

        return classifier

    def get_pca_score(self, classifier, dataframe, labels) -> float:

        predictions = self.predict(dataframe)
        score = f1_score(labels, predictions, average='macro')
        return score

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):
        transformed = self.classifier.transform(df_norm)
        tensor = np.array(df_norm).reshape(df_norm.shape[0], df_norm.shape[1])
        loss = tf.keras.metrics.mean_squared_error(tensor, transformed)
        loss = np.array(loss[0])
        print("    loss:")
        print("        sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(loss.sum(),
                                                                                       loss.min(), loss.max(),
                                                                                       loss.mean(), loss.std()))
        return

    # 'reconstruct' a dataframe by passing it through the classifier then reversing that
    def reconstruct(self, df_norm:DataFrame) -> DataFrame:
        transformed = self.classifier.transform(df_norm)
        recon = self.classifier.inverse_transform(transformed)
        df_pred = pd.DataFrame(recon, columns=self.data_cols)
        return df_pred

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:
        transformed = self.classifier.transform(df_norm)
        df_tran = pd.DataFrame(transformed, columns=self.data_cols)
        return df_tran.fillna()


    # only need to override/define the predict function
    def predict(self, df_norm: DataFrame):

        transformed = self.classifier.transform(df_norm)
        recon = self.classifier.inverse_transform(transformed)
        tensor = np.array(df_norm).reshape(df_norm.shape[0], df_norm.shape[1])

        # not sure why, but predict sometimes returns an odd length
        if len(recon) != np.shape(tensor)[0]:
            print("    ERR: prediction length mismatch ({} vs {})".format(len(recon), np.shape(tensor)[0]))
            transformed = np.zeros(df_norm.shape[0], dtype=float)
        else:
            # get losses by comparing input to output
            msle = tf.keras.losses.msle(recon, tensor)

            # # mean + stddev method
            # # threshold for anomaly scores
            # loss = np.array(msle)
            # threshold = np.mean(loss) + 2 * np.std(loss)
            #
            # # anything anomylous results in a '1'
            # predictions = np.where(msle > threshold, 1.0, 0.0)

             # Median Absolute Deviation method
            z_scores = self.mad_score(msle)
            predictions = np.where(z_scores > self.mad_threshold, 1.0, 0.0)

            #
            # print("    loss:")
            # print("        sum:{:.3f} min:{:.3f} max:{:.3f} mean:{:.3f} std:{:.3f}".format(loss.sum(),
            #                                                                                loss.min(), loss.max(),
            #                                                                                loss.mean(), loss.std()))
            # print(msle)

        return predictions

    def save(self, path=""):
        return

    def load(self, path=""):
        return self.classifier

    def model_is_trained(self) -> bool:
        return False

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required

    # Median Absolute Deviation
    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)
        return 0.6745 * ad / mad