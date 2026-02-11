# base class that implements a classifier using PyTorch models
# You should not really use this directly, use one of the specific subclasses instead, depending on the type
# of model (linear, binary, encoder etc.)

# NOTE: this only works for models that support multivariate prediction with past observed covariants
# (technical indicators in this case)

# For a list of viable algorithms, see: https://unit8co.github.io/darts/README.html (Forecasting Models section)

# pytorch works very differently from tf.keras. I have maintained the same interfaces across sklearn, keras, darts and
# pytorch, which is why I have to update some parameters via a call interface rather than adding a parameter

# specific model subclasses (linear etc) should override create_model

# To install, run:
#     conda install pytorch torchvision -c pytorch
import multiprocessing

import torch
import pytorch_lightning

from pytorch_lightning import Trainer
import numpy as np
from pandas import DataFrame, Series
import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from torchmetrics import MeanAbsolutePercentageError

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

class ClassifierPyTorch():
    num_features = 64
    seq_len = 12
    lookahead = 12
    batch_size = 1024

    model = None
    is_trained = False
    category = ""
    name = ""
    model_path = ""
    model_ext = ".pt"
    checkpoint_path = "/tmp/model" + model_ext # will be overwritten later

    loaded_from_file = False
    contamination = 0.01  # ratio of signals to samples. Used in several algorithms, so saved

    clean_data_required = False  # train with positive rows removed
    model_per_pair = False  # set to False to combine across all pairs
    new_model = False  # True if a new model was created this run

    dataframeUtils = None
    requires_dataframes = True  # set to True if classifier takes dataframes rather than tensors
    prescale_dataframe = False  # set to True if algorithms need dataframes to be pre-scaled
    single_prediction = True  # True if algorithm only produces 1 prediction (not entire data array)

    trainer = None
    num_cpus = 1
    use_gpu = True # Note: not all classifiers can use the GPU, and some are slower when they do

    train_cols = []  # used for debug

    # ---------------------------

    # Note: pair is needed because we cannot combine model across pairs because of huge price differences

    def __init__(self, pair, seq_len, num_features, tag="", use_gpu=True):
        super().__init__()

        # set seeds so that runs are reproducable
        self.set_all_seeds()

        self.loaded_from_file = False
        self.seq_len = seq_len
        self.num_features = num_features

        self.use_gpu = use_gpu

        if self.model_per_pair:
            pair_suffix = "_" + pair.split("/")[0]
        else:
            pair_suffix = ""

        if tag == "":
            tag_suffix = ""
        else:
            tag_suffix = "_" + tag

        self.category = self.__class__.__name__
        self.name = self.category + pair_suffix + tag_suffix

        # self.model_path = self.get_model_path()
        self.set_model_name(self.category, self.name)

        if self.dataframeUtils is None:
            self.dataframeUtils = DataframeUtils()

        # the following should turn on hardware acceleration, if suported
        torch.device("mps")
        self.num_cpus = multiprocessing.cpu_count()

        # set pytorch Trainer args. Ref: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

        # Early stop callback
        early_callback = EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=0.001,
            mode='min',
        )

        accel = 'auto' if self.is_gpu_available() else 'cpu'

        self.trainer = Trainer(
            accelerator=accel,
            devices="auto",
            callbacks=[early_callback],
            auto_lr_find=True,
            benchmark=True,
            auto_scale_batch_size=True
        )

        print(f"    CPUs:{self.num_cpus} GPU:{self.is_gpu_available()}")

    # ---------------------------

    # set model name - this overrides the default naming. This allows the strategy to set the naming convention
    # directory and extension are handled, just need to supply the category (e.g. the strat name) and main file name
    # caller will have to take care of adding pair names, tag etc.
    def set_model_name(self, category, model_name):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + category + '/'
        file_path = save_dir + model_name + self.model_ext

        # update tracking vars (need to override defaults)
        self.category = category
        self.model_path = file_path
        self.name = model_name
        # print(f"    Set model path:{self.model_path}")

        return self.model_path

    # ---------------------------

    def set_lookahead(self, lookahead):
        self.lookahead = lookahead

    # ---------------------------

    # create model - subclasses should overide this
    def create_model(self, seq_len, num_features):

        model = None

        print("    WARN: subclass must override create_model(). Using NBeats prediction model as reference")

        # use NBeats because it is used as an example
        model = NBEATSModel(input_chunk_length=seq_len, output_chunk_length=self.lookahead)

        return model

    # ---------------------------

    # update training using the supplied (NON-normalised) dataframe. Training is cumulative
    # Note: pytorch does not work well with pre-normalised dataframes, so we need the raw data
    def train(self, df_train, df_test, train_results, test_results, force_train=False):

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
            self.model = self.create_model(self.seq_len, self.num_features)
            if self.model is None:
                print("    ERR: model not created")
                return

        if self.dataframeUtils.is_tensor(df_train):
            # we have been passed a pandas array,should be a dataframe
            print("    ERR: require DataFrame, not Pandas array")
            return

        if self.clean_data_required:
            # only use entries that do not have buy/sell signal
            df1 = df_train.copy()
            df1['%labels'] = train_results
            df1 = df1[(df1['%labels'] < 0.1)]
            results = df1['%labels']
            df_train = df1.drop('%labels', axis=1)
        else:
            # df_train = df_train.copy()
            results = train_results

        # convert various arrays to format expected by PyTorch

        # convert time formats
        df_train['date'] = pd.to_datetime(df_train.date).dt.tz_localize(None)
        df_test['date'] = pd.to_datetime(df_test.date).dt.tz_localize(None)

        # convert to timeseries
        train_time_series = darts.TimeSeries.from_dataframe(df_train, time_col='date', fillna_value=0)
        test_time_series = darts.TimeSeries.from_dataframe(df_test, time_col='date', fillna_value=0)

        # convert results. Put into dataframe format first, because we need to match the date index
        df2 = df_train.copy()
        df2['close'] = train_results
        train_price_series = darts.TimeSeries.from_dataframe(df2, time_col='date', value_cols='close', fillna_value=0)

        df3 = df_test.copy()
        df3['close'] = test_results
        test_price_series = darts.TimeSeries.from_dataframe(df3, time_col='date', value_cols='close', fillna_value=0)

        # convert to 32-bit (allows use of GPU)
        if self.is_gpu_available():
            print("    Converting to 32-bit to allow GPU usage...")
            train_time_series = train_time_series.astype(np.float32)
            test_time_series = test_time_series.astype(np.float32)
            train_price_series = train_price_series.astype(np.float32)
            test_price_series = test_price_series.astype(np.float32)

        # scale the dataframes
        df_scaler = Scaler(RobustScaler())
        df_scaler = df_scaler.fit(train_time_series)
        train_covariate_series = df_scaler.transform(train_time_series)
        test_covariate_series = df_scaler.transform(test_time_series)

        # price_series = df_train_norm['close']
        train_price_scaler = Scaler(RobustScaler())
        train_price_scaler = train_price_scaler.fit(train_price_series)
        train_target_series = train_price_scaler.transform(train_price_series)
        test_target_series = train_price_scaler.transform(test_price_series)

        # check for nans


        # print (f'covariate_series:{covariate_series}')
        # print (f'price_series:{price_series}')

        # fit the model against the training data
        # epochs = 6  # debug
        epochs = 64

        # A TorchMetric or val_loss can be used as the monitor
        torch_metrics = MeanAbsolutePercentageError()

        # print(f'df_train: {np.shape(df_train)} train_results:{np.shape(train_results)}')
        # print(f'train_covariate_series: {train_covariate_series.n_samples} train_target_series:{train_target_series.n_samples}')

        self.model = self.model.fit(train_target_series,
                                    past_covariates=train_covariate_series,
                                    val_series=test_target_series,
                                    val_past_covariates=test_covariate_series,
                                    epochs=epochs,
                                    # num_loader_workers=self.num_cpus,
                                    verbose=True)

        # only save if this is the first time training
        if not self.is_trained:
            self.save()
            print(f'Model: {self.model_path}')
            # summary(self.model, input_size=(self.batch_size, self.seq_len, self.num_features))

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
        price_series = darts.TimeSeries.from_dataframe(df, time_col='date', value_cols='close')
        price_scaler = Scaler(RobustScaler())
        price_scaler = price_scaler.fit(price_series)
        price_series = price_scaler.transform(price_series)

        # convert dataframe to timeseries
        df_time_series = darts.TimeSeries.from_dataframe(df, time_col='date')

        # convert to 32-bit (allows use of GPU)
        if self.is_gpu_available():
            price_series = price_series.astype(np.float32)
            df_time_series = df_time_series.astype(np.float32)

        # scale the dataframe
        df_scaler = Scaler(RobustScaler())
        covariate_series = df_scaler.fit_transform(df_time_series)

        time_est = dataframe.shape[0] / 600.0 # ~10 it/sec
        print(f"    backtesting {dataframe.shape[0]} samples. Estimated time:{time_est:.2f} (mins)")
        # run backtesting
        # with torch.no_grad():
        with torch.inference_mode():
            preds = self.model.historical_forecasts(price_series,
                                                    past_covariates=covariate_series,
                                                    forecast_horizon=self.lookahead,
                                                    last_points_only=True,
                                                    retrain=False,
                                                    verbose=False)

        # reverse scaling
        preds2 = price_scaler.inverse_transform(preds)

        # get the underlying dataframe and column
        df = preds2.pd_dataframe()
        scaled_preds = np.array(df['close'])

        # predictions = np.zeros(np.shape(dataframe)[0])
        predictions = np.array(dataframe['close'])

        # predictions are usually shorter than the original data (need some values to feed the pipeline)
        start = len(predictions) - len(scaled_preds)

        # if start > 0:
        #     predictions[0:start-1] = np.array(dataframe['close'].iloc[0:start-1]) # use original data to pre-populate and size

        # should this be placed at the start or the end?!
        predictions[start:] = np.array(scaled_preds)

        # print(f'predictions2:{predictions}')
        # print(f'scaled_preds:{scaled_preds}')

        return predictions

    # ---------------------------

    # get a prediction based on the supplied dataframe. Returns an array of predictions, length self.lookahead
    def predict(self, dataframe: DataFrame):

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
        price_series = darts.TimeSeries.from_dataframe(df, time_col='date', value_cols='close')
        price_scaler = Scaler(RobustScaler())
        price_scaler = price_scaler.fit(price_series)
        price_series = price_scaler.transform(price_series)

        # convert dataframe to timeseries
        df_time_series = darts.TimeSeries.from_dataframe(df, time_col='date')

        # # convert to 32-bit (allows use of GPU)
        # if self.is_gpu_available():
        #     price_series = price_series.astype(np.float32)
        #     df_time_series = df_time_series.astype(np.float32)

        # workaround for GPU bug: always convert to 32-bit
        price_series = price_series.astype(np.float32)
        df_time_series = df_time_series.astype(np.float32)

        # scale the dataframe
        df_scaler = Scaler(RobustScaler())
        covariate_series = df_scaler.fit_transform(df_time_series)

        self.trainer = Trainer(accelerator='mps', devices=1)
        # print(f'Prediction data size: {np.shape(df)}')
        # with torch.no_grad():
        with torch.inference_mode():
            preds = self.model.predict(n=self.lookahead,
                                       series=price_series,
                                       past_covariates=covariate_series,
                                       batch_size=self.batch_size,
                                       # num_loader_workers=self.num_cpus,
                                       verbose=False)

        # print (preds)
        # convert to dataframe so that we cann access the predictions
        # df = preds.pd_dataframe()
        # scaled_preds = np.array(df['close'])
        # preds_series = darts.TimeSeries.from_series(scaled_preds)

        # reverse scaling
        preds2 = price_scaler.inverse_transform(preds)

        # get the underlying dataframe and column
        df = preds2.pd_dataframe()
        scaled_preds = np.array(df['close'])

        predictions = scaled_preds

        return predictions

    # ---------------------------

    # evaluate model using the supplied (normalised) dataframe as test data.
    def evaluate(self, df_norm: DataFrame):

        self.model.fit(df_norm)
        forecast = self.model.predict(self.lookahead)

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
        model_path = save_dir + self.name + self.model_ext
        return model_path

    # ---------------------------

    # returns path to the coreml model file
    def get_coreml_model_path(self):
        root_dir = self.get_model_root_dir()
        save_dir = root_dir + self.category + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir + self.name + '.coreml'
        return model_path

    # ---------------------------

    def get_checkpoint_path(self):
        checkpoint_dir = '/tmp' + "/" + self.name + "/"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_path = checkpoint_dir + "checkpoint" + self.model_ext
        return model_path

    # ---------------------------

    def save(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        print("    saving model to: ", path)
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save is built in to the model
        print("    saving to: ", self.model_path)
        # joblib.dump(self.model, self.model_path)
        self.model.save(self.model_path)

        return

    # ---------------------------

    def save_as_coreml(self):
        return

    # ---------------------------


    def load(self, path=""):

        if len(path) == 0:
            self.model_path = self.get_model_path()
            path = self.model_path
        else:
            self.model_path = path

        if os.path.exists(path):
            # use joblib to reload model state
            print("    loading from: ", self.model_path)
            # self.model = joblib.load(self.model_path)
            self.model = self.load_from_file(self.model_path)
            self.loaded_from_file = True
            self.is_trained = True
            print(f'Model: {self.model_path}')
            # summary(self.model, input_size=(self.batch_size, self.seq_len, self.num_features))
        else:
            print("    model not found ({})...".format(path))
            # flag this as a new model. Note that this is a class global variable because we need to track this
            # across multiple instances (e.g. if we are combining all pairs into one model)
            ClassifierDarts.new_model = True

        return self.model

    # ---------------------------

    # subclasses should override this, because data format is calss-specific in darts/pytorch
    def load_from_file(self, model_path):
        return darts.models.forecasting.torch_forecasting_model.PastCovariatesTorchModel.load(model_path)

    # ---------------------------

    def model_exists(self) -> bool:
        path = self.get_model_path()
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
