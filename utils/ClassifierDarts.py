# base class that implements an Anomaly detector using Darts models
# You should not really use this directly, use one of the specific subclasses instead, depending on the type
# of model (linear, binary, encoder etc.)

# NOTE: this only works for models that support multivariate prediction with past observed covariants
# (technical indicators in this case)

# For a list of viable algorithms, see: https://unit8co.github.io/darts/README.html (Forecasting Models section)

# darts uses pytorch for Neural Network-based algorithms, which is why you see a lot of pytorch code here
# Also, pytorch works very differently from tf.keras. I have maintained the same interfaces across sklearn, keras and darts
# which is why I have to update some parameters via a call interface rather than adding a parameter

# specific model subclasses (linear etc) should override create_model

# run: "pip install darts" to get the darts library
import multiprocessing

import traceback

import torch

import darts
import pytorch_lightning

from pytorch_lightning import Trainer
import numpy as np
import numpy
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mase
from darts.models import NBEATSModel, TFTModel
from pandas import DataFrame, Series
import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from torchmetrics import MeanAbsolutePercentageError

# from torchinfo import summary

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

logging.getLogger("lightning").setLevel(logging.WARN)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", ".*MPS available but not used.*")

import random

import os
import multiprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# not all layers are supported on the GPU yet, so fallback to CPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

from DataframeUtils import DataframeUtils


# ---------------------------

class ClassifierDarts():
    num_features = 64
    lookback = 12
    lookahead = 12
    batch_size = 1024

    model = None
    is_trained = False
    category = ""
    model_name = ""
    model_path = ""
    model_ext = ".pt"
    checkpoint_path = "/tmp/model" + model_ext
    target_column = 'gain'

    loaded_from_file = False
    contamination = 0.01  # ratio of signals to samples. Used in several algorithms, so saved

    clean_data_required = False  # train with positive rows removed
    model_per_pair = False  # set to False to combine across all pairs
    new_model = False  # May not wrok for darts-based strats, so leave at False

    dataframeUtils = None
    requires_dataframes = True  # set to True if classifier takes dataframes rather than tensors
    prescale_dataframe = False  # set to True if algorithms need dataframes to be pre-scaled
    single_prediction = False  # True if algorithm only produces 1 prediction (not entire data array)

    trainer = None
    trainer_args = {}
    # num_cpus = 1
    use_gpu = True  # Note: not all classifiers can use the GPU, and some are slower when they do

    train_cols = []  # used for debug

    # ---------------------------

    # Note: pair is needed because we cannot combine model across pairs because of huge price differences

    def __init__(self, pair, lookback, num_features, tag="", use_gpu=True):
        super().__init__()

        # set seeds so that runs are reproducable
        self.set_all_seeds()

        self.loaded_from_file = False
        self.lookback = lookback
        self.num_features = num_features

        self.use_gpu = use_gpu
        self.num_cpus = multiprocessing.cpu_count()
        print(f"    CPUs:{self.num_cpus} GPU:{self.is_gpu_available()}  use_gpu:{self.use_gpu}")

        if self.model_per_pair:
            pair_suffix = "_" + pair.split("/")[0]
        else:
            pair_suffix = ""

        if tag == "":
            tag_suffix = ""
        else:
            tag_suffix = "_" + tag

        self.category = self.__class__.__name__
        self.model_name = self.category + pair_suffix + tag_suffix

        # set model name via function, so that programs have an option to override this (e.g. add program name)
        self.set_model_name(self.category, self.model_name)

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        # set pytorch Trainer args. Ref: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
        # Annoyingly, trainer args need to be specified in the constructor

        # Early stop callback
        early_callback = EarlyStopping(
            monitor=self.get_loss_metric(),
            patience=8,
            min_delta=0.001,
            mode='min',
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.get_checkpoint_dir(),
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor=self.get_loss_metric(),
            mode="min")

        self.trainer_args["callbacks"] = [early_callback, checkpoint_callback]
        # self.trainer_args["deterministic"] = True
        # self.trainer_args["auto_lr_find"] = True
        self.trainer_args["benchmark"] = True
        self.trainer_args["enable_model_summary"] = True
        # self.trainer_args["auto_scale_batch_size"] = True
        self.trainer_args["enable_checkpointing"] = True
        self.trainer_args["enable_progress_bar"] = True
        self.trainer_args["min_epochs"] = 6

        # self.trainer_args["devices"] = "auto"

        if self.is_gpu_available():
            # self.trainer_args['precision'] = 32
            self.trainer_args['precision'] = '32-true'
            # self.trainer_args['precision'] = '64-true'
            # # the following should turn on hardware acceleration, if suported
            # torch.device("mps")
            devices = 1
            accelerator = "mps"
        else:
            self.trainer_args['precision'] = '64-true'
            devices = 'auto'
            accelerator = "cpu"

        self.trainer_args["devices"] = devices
        self.trainer_args["accelerator"] = accelerator

        print(f'    self.trainer_args: {self.trainer_args}')

        # set up the equivalent Trainer object for later use
        self.trainer = Trainer(**self.trainer_args)

    # ---------------------------

    # set the path for the model (dir + file name)
    # need this because utility function cannot figure out naming if multiple directories are used
    def set_model_path(self, path):

        if '.keras' in path:
            path = path.replace('.keras', self.model_ext)

        self.model_path = path
        filename = path.split("/")[-1]
        self.name = filename.split(".")[0]
        self.root_dir = os.path.dirname(path)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        return
   
    # ---------------------------
 

    # set model name - this overrides the default naming. This allows the strategy to set the naming convention
    # directory and extension are handled, just need to supply the category (e.g. the strat name) and main file name
    # caller will have to take care of adding pair names, tag etc.
    def set_model_name(self, category, model_name):

        if '.keras' in model_name:
            model_name = model_name.replace('.keras', self.model_ext)
    
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + category + '/'
        file_path = save_dir + model_name + self.model_ext

        # update tracking vars (need to override defaults)
        self.category = category
        self.model_path = file_path
        self.model_name = model_name
        # print(f"    Set model path:{self.model_path}")

        return self.model_path

    def get_model_name(self):
        return self.model_name

    # ---------------------------

    def set_combine_models(self, combine_models):
        # self.combine_models = combine_models
        pass
        return

    # ---------------------------
    
 
    def set_lookahead(self, lookahead):
        self.lookahead = lookahead
        self.lookback = int(max(self.lookback, 4 * self.lookahead))
        return

    def get_lookahead(self):
        return self.lookahead

    # ---------------------------

    def set_target_column(self, target_column):
        self.target_column = target_column
        return

    # ---------------------------

    # the following are intended to be overridden by  the subclass, if necessary

    # return loss metric for fitting. Can vary by model, hence it's a function. Override in subclass if necessary
    def get_loss_metric(self):
        return 'train_loss'

    # This implementation covers many models, but if the signature is different then override this call in the subclass
    def model_fit(self, model, train_target_series, train_covariate_series, test_target_series, test_covariate_series):
        model = model.fit(train_target_series,
                          past_covariates=train_covariate_series,
                          val_series=test_target_series,
                          val_past_covariates=test_covariate_series,
                          verbose=True
                          )

        # load best checkpoint
        # model = self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)

        return model

    def model_historical_forecasts(self, model, target_series, covariate_series):
        preds = model.historical_forecasts(target_series,
                                           past_covariates=covariate_series,
                                           forecast_horizon=64,
                                           stride=64,
                                           last_points_only=False,
                                           retrain=False,
                                           verbose=True)

        predictions = []
        for p in preds:
            arr = np.squeeze(np.array(p.data_array()))
            # print(f"arr: {arr}")
            predictions = np.concatenate((predictions, arr))
            predictions = np.where(((predictions > 5.0) | (predictions < -5.0)), 0.0, predictions)
            # predictions = predictions.clip(-10.0, 10.0)

        # print(f"    predictions:{np.shape(predictions)}")
        return predictions

    def model_predict(self, model, target_series, covariate_series):
    
        preds = model.predict(#n=self.lookahead,
                              n=1,
                              series=target_series,
                              past_covariates=covariate_series,
                            #   batch_size=self.batch_size,
                              # trainer=self.trainer,
                              # num_loader_workers=self.num_cpus,
                              verbose=True)

        # predictions = np.squeeze(preds.values())
        predictions = preds.values()
        print(f"    model_predict() predictions: {predictions}")
        predictions = np.where(((predictions > 5.0) | (predictions < -5.0)), 0.0, predictions)
        # predictions = predictions.clip(-10.0, 10.0)
        return predictions

    # create model - subclasses should override this
    def create_model(self, lookback, num_features):

        model = None

        print("    WARN: subclass must override create_model(). Using NBeats prediction model as reference")

        # use NBeats because it is used as an example in Darts documentation
        model = NBEATSModel(input_chunk_length=lookback, output_chunk_length=self.lookahead)

        return model

    # ---------------------------

    # update training using the supplied (NON-normalised) dataframe. Training is cumulative
    # Note: darts does not work well with pre-normlaised dataframes, so we need the raw data
    def train(self, df_train, df_test, train_results, test_results, force_train=False):

        # print(f"    train() df_train: {np.shape(df_train)} train_results: {np.shape(train_results)}")

        self.train_cols = df_train.columns.values  # save for later debug

        # lazy loading because params can change up to this point
        if self.model is None:
            # load saved model if present
            self.model = self.load()

        # just return if model has already been trained, unless force_train is set, or this was a new model
        if self.model_is_trained() and (not force_train) and (not self.new_model_created()):
            return

        # no model? Create it from scratch
        if self.model is None:
            self.model = self.create_model(self.lookback, self.num_features)
            if self.model is None:
                print("")
                print("    ERR: model not created")
                print("")
                return

        if self.dataframeUtils.is_tensor(df_train):
            # we have been passed a pandas array,should be a dataframe
            print("")
            print("    ERR: require DataFrame, not Pandas array")
            print("")
            return

        # check lengths
        if (np.shape(df_train)[0] != np.shape(train_results)[0]) or (np.shape(df_test)[0] != np.shape(test_results)[0]):
            print("")
            print("     WARN: lengths do not match")
            print(f'    df_train:{np.shape(df_train)} train_results:{np.shape(train_results)}')
            print(f'    df_test:{np.shape(df_test)} test_results:{np.shape(test_results)}')
            print("")

        # convert time formats
        train_times = pd.to_datetime(df_train.date).dt.tz_localize(None)
        test_times = pd.to_datetime(df_test.date).dt.tz_localize(None)

        df_train['date'] = train_times
        df_test['date'] = test_times

        df_train = df_train.replace([np.nan, np.inf, -np.inf], 0.0)
        df_test = df_test.replace([np.nan, np.inf, -np.inf], 0.0)

        # convert to timeseries
        train_time_series = darts.TimeSeries.from_dataframe(df_train, time_col="date", fillna_value=0.0)
        test_time_series = darts.TimeSeries.from_dataframe(df_test, time_col='date', fillna_value=0.0)

        # convert results. Put into dataframe format first, because we need to match the date index
        df2 = df_train.copy()
        df2['temp'] = train_results
        df2 = df2.replace([np.nan, np.inf], 0.0)
        train_gain_series = darts.TimeSeries.from_dataframe(df2, time_col='date', value_cols='temp',
                                                             fillna_value=0)
        
        # train_gain_series = darts.TimeSeries.from_values(train_results, train_times)

        df3 = df_test.copy()
        df3['temp'] = test_results 
        df3 = df3.replace([np.nan, np.inf], 0.0)
        test_gain_series = darts.TimeSeries.from_dataframe(df3, time_col='date', value_cols='temp',
                                                            fillna_value=0)
        
        # test_gain_series = darts.TimeSeries.from_values(test_results, test_times)

        # scale the dataframes
        # df_scaler = Scaler(RobustScaler())
        df_scaler = Scaler(MinMaxScaler())
        df_scaler = df_scaler.fit(train_time_series)
        train_covariate_series = df_scaler.transform(train_time_series)
        test_covariate_series = df_scaler.transform(test_time_series)


        # convert to 32-bit (allows use of GPU)
        if self.is_gpu_available():
            print("    Converting to 32-bit to allow GPU usage...")
            train_covariate_series = train_covariate_series.astype(np.float32)
            test_covariate_series = test_covariate_series.astype(np.float32)
            train_gain_series = train_gain_series.astype(np.float32)
            test_gain_series = test_gain_series.astype(np.float32)

        # gain_series = df_train_norm[self.target_column]
        # train_gain_scaler = Scaler(RobustScaler())
        # train_gain_scaler = train_gain_scaler.fit(train_gain_series)
        # train_target_series = train_gain_scaler.transform(train_gain_series)
        # test_target_series = train_gain_scaler.transform(test_gain_series)

        train_target_series = train_gain_series
        test_target_series = test_gain_series

        # fit the model against the training data

        # print(f'df_train: {np.shape(df_train)} train_results:{np.shape(train_results)}')

        # print(f"    train() train_target_series: {len(train_target_series)} train_covariate_series: {len(train_covariate_series)}")
        # print(f"train_target_series: {train_target_series}")

        # fit the model
        self.model = self.model_fit(self.model,
                                    train_target_series, train_covariate_series,
                                    test_target_series, test_covariate_series)

        self.save()

        self.is_trained = True

        return

    # ---------------------------

    # backtest across the supplied dataframe. Should be faster than iteratively calling predict()
    def backtest(self, dataframe: DataFrame):

        if self.model is None:
            print("    ERR: no model")
            return np.zeros(np.shape(dataframe)[0])

        # DEBUG: check predict dataframe columns against training dataframe columns
        predict_cols = dataframe.columns.values
        col_diffs = list(set(predict_cols) - set(self.train_cols))
        if len(predict_cols) != len(self.train_cols):
            print("ERR: mismatching columns")
            print(f'diff:{col_diffs}')
            print(f"  train_cols:{self.train_cols}")
            print(f"  predict_cols:{predict_cols}")

        # use the whole dataframe the 'covariate' series
        df = dataframe.copy()
        df['date'] = pd.to_datetime(df.date).dt.tz_localize(None)

        # convert closing price column to time series & scale
        gain_series = darts.TimeSeries.from_dataframe(df, time_col='date', value_cols=self.target_column)
        # gain_col = np.array(df[self.target_column]).reshape(-1,1)
        # gain_scaler = Scaler(RobustScaler())
        # gain_scaler = RobustScaler()
        # gain_scaler = gain_scaler.fit(gain_col)
        # gain_series = gain_scaler.transform(gain_col)

        # convert dataframe to timeseries
        df_time_series = darts.TimeSeries.from_dataframe(df, time_col='date')

        # convert to 32-bit (allows use of GPU)
        if self.is_gpu_available():
            gain_series = gain_series.astype(np.float32)
            df_time_series = df_time_series.astype(np.float32)

        # scale the dataframe
        # df_scaler = Scaler(RobustScaler())
        df_scaler = Scaler(MinMaxScaler())
        covariate_series = df_scaler.fit_transform(df_time_series)
        # covariate_series = df_time_series

        # print(f'    dataframe:{np.shape(dataframe)}')
        # print(f'    covariate_series:{covariate_series.n_samples}, {covariate_series.n_timesteps}, {covariate_series.n_components}')
        # print(f'    gain_series:{gain_series.n_samples}, {gain_series.n_timesteps}, {gain_series.n_components}')

        time_est = dataframe.shape[0] / (200.0 * 60.0)  # ~200 it/sec
        print(f"    backtesting {dataframe.shape[0]} samples. Estimated time:{time_est:.2f} (mins)")
        # run backtesting

        with torch.inference_mode():
            preds = self.model_historical_forecasts(self.model, gain_series, covariate_series)

        # reverse scaling
        # preds2 = gain_scaler.inverse_transform(preds)

        # get the underlying dataframe and column
        # df = preds2.pd_dataframe()
        # df = preds.pd_dataframe()
        # scaled_preds = np.array(df[self.target_column])
        scaled_preds = preds
        # print(f"scaled_preds: {scaled_preds}")

        predictions = np.zeros(np.shape(dataframe)[0], dtype=float)
        predictions = np.array(dataframe[self.target_column])

        # predictions are usually shorter than the original data (need some values to feed the pipeline)
        start = len(predictions) - len(scaled_preds)

        # if start > 0:
        #     predictions[0:start-1] = np.array(dataframe[self.target_column].iloc[0:start-1]) # use original data to pre-populate and size

        # should this be placed at the start or the end?!
        predictions[start:] = np.array(scaled_preds)

        return predictions

    # ---------------------------

    # get a prediction based on the supplied dataframe. Returns an array of predictions, length self.lookahead
    def predict(self, dataframe: DataFrame):


        # print(f"    predict() dataframe: {np.shape(dataframe)}")

        if self.model is None:
            print("    ERR: no model")
            return np.zeros(np.shape(dataframe)[0])

        # DEBUG: check predict dataframe columns against training dataframe columns
        predict_cols = dataframe.columns.values
        col_diffs = list(set(predict_cols) - set(self.train_cols))
        if len(predict_cols) != len(self.train_cols):
            print("ERR: mismatching columns")
            print(f'diff:{col_diffs}')
            print(f"  train_cols:{self.train_cols}")
            print(f"  predict_cols:{predict_cols}")

        # use the whole dataframe the 'covariate' series
        df = dataframe.copy()
        df['date'] = pd.to_datetime(df.date).dt.tz_localize(None)

        # convert target column to time series & scale
        gain_series = darts.TimeSeries.from_dataframe(df, time_col='date', value_cols=self.target_column)
        # gain_scaler = Scaler(RobustScaler())
        # gain_scaler = gain_scaler.fit(gain_series)
        # gain_series = gain_scaler.transform(gain_series)

        # convert dataframe to timeseries
        df_time_series = darts.TimeSeries.from_dataframe(df, time_col='date')

        # # convert to 32-bit (allows use of GPU)
        if self.is_gpu_available():
            gain_series = gain_series.astype(np.float32)
            df_time_series = df_time_series.astype(np.float32)

        # workaround for GPU bug: always convert to 32-bit
        # gain_series = gain_series.astype(np.float32)
        # df_time_series = df_time_series.astype(np.float32)

        # scale the dataframe
        # df_scaler = Scaler(RobustScaler())
        df_scaler = Scaler(MinMaxScaler())
        covariate_series = df_scaler.fit_transform(df_time_series)


        # print(f"    predict() gain_series: {len(gain_series)} covariate_series: {len(covariate_series)}")

        # self.trainer = Trainer(accelerator='mps', devices=1)
        # print(f'Prediction data size: {np.shape(df)}')
        # with torch.no_grad():
        with torch.inference_mode():
            preds = self.model_predict(self.model, gain_series, covariate_series)

        # print (preds)
        # convert to dataframe so that we cann access the predictions
        # df = preds.pd_dataframe()
        # scaled_preds = np.array(df[self.target_column])
        # preds_series = darts.TimeSeries.from_series(scaled_preds)

        # reverse scaling
        # preds2 = gain_scaler.inverse_transform(preds)
        # preds2 = preds

        # get the underlying dataframe and column
        # df = preds2.pd_dataframe()
        # scaled_preds = np.array(df[self.target_column])

        # predictions = scaled_preds

        # predictions = np.array(preds.data_array())
        predictions = preds

        return predictions

    # ---------------------------

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, data, results):

        # self.model.fit(data)
        preds = self.backtest(data)

        acc = np.mean((preds.round() == results))
        acc = float(acc)
        print(f"    accuracy: {acc}")

        return

    # ---------------------------

    # 'reconstruct' a dataframe by passing it through the model
    def reconstruct(self, df_norm: DataFrame) -> DataFrame:

        return df_norm

    # ---------------------------

    # transform supplied (normalised) dataframe into a lower dimension version
    def transform(self, df_norm: DataFrame) -> DataFrame:

        return df_norm

    # ---------------------------

    # set values of various random seeds so that we get repeatable performance
    def set_all_seeds(self):
        seed = 42
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ---------------------------

    # returns path to the root directory used for storing models
    def get_model_root_dir(self):
        # set as subdirectory of location of this file (so that it can be included in the repository)
        file_dir = os.path.dirname(str(Path(__file__)))
        root_dir = file_dir + "/models/"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        return root_dir

    # ---------------------------

    # returns path to 'full' model file
    def get_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.model_name + self.model_ext
        return model_path

    # ---------------------------

    # returns path to the coreml model file
    def get_coreml_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.model_name + '.coreml'
        return model_path

    # ---------------------------

    def get_checkpoint_dir(self):
        checkpoint_dir = '/tmp' + "/" + self.__class__.__name__ + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        return checkpoint_dir

    def get_checkpoint_path(self):
        checkpoint_dir = self.get_checkpoint_dir()
        model_path = checkpoint_dir + self.model_name + self.model_ext
        return model_path

    # ---------------------------

    def save(self, path=""):

        if len(path) == 0:
            # self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        print("    saving model to: ", path)
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save is built in to the model
        print("    saving to: ", self.model_path)
        self.model.save(self.model_path)
        # torch.save(self.model.state_dict(), self.model_path)

        return

    # ---------------------------

    def save_as_coreml(self):
        return

    # ---------------------------

    def load(self, path=""):

        if len(path) == 0:
            # self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        if os.path.exists(path):
            # use joblib to reload model state
            print("    loading from: ", self.model_path)
            # self.model = joblib.load(self.model_path)
            self.model = self.load_from_file(self.model_path, use_gpu=self.is_gpu_available())
            self.loaded_from_file = True
            self.is_trained = True
            print(f'Model: {self.model_path}')
            # summary(self.model, input_size=(self.batch_size, self.lookback, self.num_features))
        else:
            print("    model not found ({})...".format(path))
            # flag this as a new model. Note that this is a class global variable because we need to track this
            # across multiple instances (e.g. if we are combining all pairs into one model)
            ClassifierDarts.new_model = True

        return self.model

    # ---------------------------

    # subclasses should override this, because data format is class-specific in darts/pytorch
    def load_from_checkpoint(self, path):
        print("    *** ERR: subclass must override load_from_checkpoint()")
        return self.model

    # ---------------------------

    # subclasses should override this, because data format is class-specific in darts/pytorch
    def load_from_file(self, model_path, use_gpu=True):
        print("    *** ERR: subclass must override load_from_file()")
        return self.model

    # ---------------------------

    def model_exists(self) -> bool:
        path = self.model_path
        return os.path.exists(path)

    # ---------------------------

    def model_is_trained(self) -> bool:
        return self.is_trained

    # ---------------------------

    def needs_clean_data(self) -> bool:
        # print("    clean_data_required: ", self.clean_data_required)
        return self.clean_data_required

    # ---------------------------

    def needs_dataframes(self) -> bool:
        return self.requires_dataframes

    # ---------------------------

    def prescale_data(self) -> bool:
        return self.prescale_dataframe

    # ---------------------------

    def returns_single_prediction(self) -> bool:
        return self.single_prediction

    # ---------------------------

    def new_model_created(self) -> bool:
        return ClassifierDarts.new_model  # note use of class-level variable

    # ---------------------------

    def is_gpu_available(self) -> bool:
        return torch.backends.mps.is_available() and self.use_gpu

    # ---------------------------

    def get_trainer_args(self):
        print(f"     Trainer args: {self.trainer_args}")
        return self.trainer_args

    # ---------------------------

    # Median Absolute Deviation
    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad
