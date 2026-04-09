import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
import tensorflow.experimental.numpy as tnp # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.layers import Concatenate # type: ignore
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape, Layer # type: ignore
# from tensorflow.keras.models import Model # type: ignore

# NOTE: requires package keract

def smape_loss(y_true, y_pred):
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    :return: Loss value
    """
    # mask=tf.where(y_true,1.,0.)
    mask = tf.cast(y_true, tf.bool)
    mask = tf.cast(mask, tf.float32)
    sym_sum = tf.abs(y_true) + tf.abs(y_pred)
    condition = tf.cast(sym_sum, tf.bool)
    weights = tf.where(condition, 1. / (sym_sum + 1e-8), 0.0)
    return 200 * tnp.nanmean(tf.abs(y_pred - y_true) * weights * mask)

# @tf.keras.utils.register_keras_serializable
class NBeatsLayer(Layer):
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 exo_dim=0,
                 backcast_length=10,
                 forecast_length=1,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None,
                 **kwargs):

        # initialize parent class
        super().__init__(**kwargs)

        # set global vars
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.exo_dim = exo_dim
        self.in_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.out_shape = (self.forecast_length, self.output_dim)
        self.layer_weights = {}
        self.nb_harmonics = nb_harmonics
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []
        assert len(self.stack_types) == len(self.thetas_dim)

        self.layer_dict={} # used to hold layer definitions

        # initialise layers (can only do this once)
        self.build_layers()

    def get_config(self):
        """
        Returns the config of a layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer

        NOTE: only save items passed in constructor (__init__)
        """
        config = super().get_config()

        config['stack_types'] = self.stack_types
        config['nb_blocks_per_stack'] = self.nb_blocks_per_stack
        config['thetas_dim'] = self.thetas_dim
        # config['units'] = self.units
        config['share_weights_in_stack'] = self.share_weights_in_stack
        config['backcast_length'] = self.backcast_length
        config['forecast_length'] = self.forecast_length
        config['input_dim'] = self.input_dim
        config['output_dim'] = self.output_dim
        config['exo_dim'] = self.exo_dim
        # config['in_shape'] = self.in_shape
        # config['exo_shape'] = self.exo_shape
        # config['out_shape'] = self.out_shape
        # config['layer_weights'] = self.layer_weights
        config['nb_harmonics'] = self.nb_harmonics
        # config['_gen_intermediate_outputs'] = self._gen_intermediate_outputs
        # config['_intermediary_outputs'] = self._intermediary_outputs

        # config['layer_dict'] = self.layer_dict # should we save this?!

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def save_layer(self, name, layer):
        if name in self.layer_dict:
            print(f'WARN: layer already saved ({name})')
        self.layer_dict[name] = layer

    def get_layer(self, name) -> Layer:
        if name in self.layer_dict:
            return self.layer_dict[name]
        else:
            raise IndexError(f'ERR: get_layer() - layer not saved ({name})')
            return None

    # build has to set up the layers (can only be created once). The layers are not called here
    def build_layers(self):
        # x = inputs

        self.cast_type = self._FORECAST

        x_ = {}
        for k in range(self.input_dim):
            layer_name = f'lambda_x_{k}'
            layer = Lambda(lambda z: z[..., k], name=layer_name)
            self.save_layer(layer_name, layer)
            # x_[k] = layer(x)

        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                layer_name = f'lambda_e_{k}'
                layer = Lambda(lambda z: z[..., k], name=layer_name)
                self.save_layer(layer_name, layer)
                # e_[k] = layer(e)
        else:
            e = None

        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                self.build_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    layer_name = f'subtract_{stack_id}_{block_id}_{k}'
                    self.save_layer(layer_name, Subtract(name=layer_name))

                    layer_name = f'stack_{stack_id}-{stack_type.title()}Block_{block_id}'
                    if self.input_dim >= 1:
                        layer_name += f'_Dim_{k}'
                    # rename.

                    layer = Lambda(function=lambda _x: _x, name=layer_name)
                    self.save_layer(layer_name, layer)
                    # forecast[k] = layer(forecast[k])
                    if stack_id == 0 and block_id == 0:
                        # y_[k] = forecast[k]
                        pass
                    else:
                        layer_name += f'_Add_{k}'
                        layer = Add(name=layer_name)
                        self.save_layer(layer_name, layer)
                        # y_[k] = layer([y_[k], forecast[k]])

        for k in range(self.input_dim):
            layer_name = f'Reshape_y_{k}'
            layer = Reshape(target_shape=(self.forecast_length, 1), name=layer_name)
            self.save_layer(layer_name, layer)
            # y_[k] = layer(y_[k])

            layer_name = f'Reshape_x_{k}'
            layer = Reshape(target_shape=(self.forecast_length, 1), name=layer_name)
            self.save_layer(layer_name, layer)
            # y_[k] = layer(x_[k])

        if self.input_dim > 1:
            layer_name = 'Concat_x'
            layer = Concatenate(name=layer_name)
            self.save_layer(layer_name, layer)
            # y_ = layer([y_[ll] for ll in range(self.input_dim)])

            layer_name = 'Concat_y'
            layer = Concatenate(name=layer_name)
            self.save_layer(layer_name, layer)
            # x_ = layer([x_[ll] for ll in range(self.input_dim)])
        # else:
        #     y_ = y_[0]
        #     x_ = x_[0]

        if self.input_dim != self.output_dim:
            layer_name = 'reg_y'
            layer = Dense(self.output_dim, activation='linear', name=layer_name)
            self.save_layer(layer_name, layer)
            # y_ = layer(y_)

            layer_name = 'reg_x'
            layer = Dense(self.output_dim, activation='linear', name=layer_name)
            self.save_layer(layer_name, layer)
            # x_ = layer(x_)

        return

    def compute_out_shape(self, in_shape):
        out_shape = (in_shape[0], self.forecast_length, self.forecast_length)
        # print(f'out_shape: {out_shape}')
        return out_shape

    def build(self, inputs):
        print('build() called')


    # call() runs the data through the layers
    def call(self, inputs):
        x = inputs

        # print(f'call() Available layers: {self.layer_dict.keys()}')

        self.cast_type = self._FORECAST

        x_ = {}
        for k in range(self.input_dim):
            layer_name = f'lambda_x_{k}'
            layer = self.get_layer(layer_name)
            x_[k] = layer(x)

        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                layer_name = f'lambda_e_{k}'
                layer = self.get_layer(layer_name)
                e_[k] = layer(e)
        else:
            e = None

        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.call_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    layer_name = f'subtract_{stack_id}_{block_id}_{k}'
                    layer = self.get_layer(layer_name)
                    # print(f'{layer_name}: {layer.get_config()}')
                    x_[k] = layer([x_[k], backcast[k]])
                    layer_name = f'stack_{stack_id}-{stack_type.title()}Block_{block_id}'
                    if self.input_dim >= 1:
                        layer_name += f'_Dim_{k}'
                    # rename.

                    layer = self.get_layer(layer_name)
                    forecast[k] = layer(forecast[k])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        layer_name += f'_Add_{k}'
                        layer = self.get_layer(layer_name)
                        y_[k] = layer([y_[k], forecast[k]])

        for k in range(self.input_dim):
            layer_name = f'Reshape_y_{k}'
            layer = self.get_layer(layer_name)
            y_[k] = layer(y_[k])

            layer_name = f'Reshape_x_{k}'
            layer = self.get_layer(layer_name)
            y_[k] = layer(x_[k])

        if self.input_dim > 1:
            layer_name = 'Concat_x'
            layer = self.get_layer(layer_name)
            y_ = layer([y_[ll] for ll in range(self.input_dim)])

            layer_name = 'Concat_y'
            layer = self.get_layer(layer_name)
            x_ = layer([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.input_dim != self.output_dim:
            layer_name = 'reg_y'
            layer = self.get_layer(layer_name)
            y_ = layer(y_)

            layer_name = 'reg_x'
            layer = self.get_layer(layer_name)
            x_ = layer(x_)

        # return Reshape((self.forecast_length, len(y_)))(y_)
        # print(f'y_: {y_.shape}')
        return y_

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' in a['layer'].lower()])
        i_pred = sum([a['value'][0] for a in self._intermediary_outputs if 'generic' not in a['layer'].lower()])
        outputs = {o['layer']: o['value'][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('_')[-1]
            try:
                reused_weights = self.layer_weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.layer_weights:
                self.layer_weights[stack_id] = {}
            self.layer_weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    # register weights (useful when share_weights_in_stack=True)
    def reg(self, layer, stack_id,):
        return self._r(layer, stack_id)

    # update name (useful when share_weights_in_stack=True)
    def n(self, layer_name, stack_id, block_id, stack_type):
        return '_'.join([str(stack_id), str(block_id), stack_type, layer_name])

    # builds and saves the layers needed for an NBeats block
    def build_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        backcast_ = {}
        forecast_ = {}

        layer_name = self.n('d1', stack_id, block_id, stack_type)
        self.save_layer(layer_name, Dense(self.units, activation='relu', name=layer_name))
        layer_name = self.n('d2', stack_id, block_id, stack_type)
        self.save_layer(layer_name, Dense(self.units, activation='relu', name=layer_name))
        layer_name = self.n('d3', stack_id, block_id, stack_type)
        self.save_layer(layer_name, Dense(self.units, activation='relu', name=layer_name))
        layer_name = self.n('d4', stack_id, block_id, stack_type)
        self.save_layer(layer_name, Dense(self.units, activation='relu', name=layer_name))

        if stack_type == 'generic':
            layer_name = self.n('theta_b', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(nb_poly, activation='linear', use_bias=False, name=layer_name))
            layer_name = self.n('theta_f', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(nb_poly, activation='linear', use_bias=False, name=layer_name))
            layer_name = self.n('backcast', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(self.backcast_length, activation='linear', name=layer_name))
            layer_name = self.n('forecast', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(self.forecast_length, activation='linear', name=layer_name))

        elif stack_type == 'trend':
            layer_name = self.n('theta_f_b', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(nb_poly, activation='linear', use_bias=False, name=layer_name))

            layer_name = self.n('lambda_t_b', stack_id, block_id, stack_type)
            self.save_layer(layer_name, 
                Lambda(
                    trend_model,
                    arguments={
                        "is_forecast": False,
                        "backcast_length": self.backcast_length,
                        "forecast_length": self.forecast_length,
                    },
                    name=layer_name,
                )
            ) 

            layer_name = self.n('lambda_t_f', stack_id, block_id, stack_type)
            self.save_layer(layer_name, 
                Lambda(
                    trend_model,
                    arguments={
                        "is_forecast": False,
                        "backcast_length": self.backcast_length,
                        "forecast_length": self.forecast_length,
                    },
                    name=layer_name,
                )
            )

        else:  # 'seasonality'

            layer_name = self.n('theta_b', stack_id, block_id, stack_type)
            if self.nb_harmonics:
                self.save_layer(layer_name, Dense(self.nb_harmonics, activation='linear', use_bias=False, name=layer_name))
            else:
                self.save_layer(layer_name, Dense(self.forecast_length, activation='linear', use_bias=False, name=layer_name))

            layer_name = self.n('theta_f', stack_id, block_id, stack_type)
            self.save_layer(layer_name, Dense(self.forecast_length, activation='linear', use_bias=False, name=layer_name))

            layer_name = self.n('lambda_s_b', stack_id, block_id, stack_type)
            self.save_layer(layer_name, 
                Lambda(
                    seasonality_model,
                    arguments={
                        "is_forecast": False,
                        "backcast_length": self.backcast_length,
                        "forecast_length": self.forecast_length,
                    },
                    name=layer_name,
                )
            ) 

            layer_name = self.n('lambda_s_f', stack_id, block_id, stack_type)
            self.save_layer(layer_name, 
                Lambda(
                    seasonality_model,
                    arguments={
                        "is_forecast": False,
                        "backcast_length": self.backcast_length,
                        "forecast_length": self.forecast_length,
                    },
                    name=layer_name,
                )
            )


        for k in range(self.input_dim):
            if self.has_exog():
                layer_name = f'exog_concat_{k}'
                self.save_layer(layer_name, Concatenate())

        return

    # calls an NBeats block using the saved layers
    def call_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        backcast_ = {}
        forecast_ = {}

        layer_name = self.n('d1', stack_id, block_id, stack_type)
        d1 = self.reg(self.get_layer(layer_name), stack_id)

        layer_name = self.n('d2', stack_id, block_id, stack_type)
        d2 = self.reg(self.get_layer(layer_name), stack_id)

        layer_name = self.n('d3', stack_id, block_id, stack_type)
        d3 = self.reg(self.get_layer(layer_name), stack_id)

        layer_name = self.n('d4', stack_id, block_id, stack_type)
        d4 = self.reg(self.get_layer(layer_name), stack_id)

        if stack_type == 'generic':
            theta_b = self.reg(self.get_layer(self.n('theta_b', stack_id, block_id, stack_type)), stack_id)
            theta_f = self.reg(self.get_layer(self.n('theta_f', stack_id, block_id, stack_type)), stack_id)
            backcast = self.reg(self.get_layer(self.n('backcast', stack_id, block_id, stack_type)), stack_id)
            forecast = self.reg(self.get_layer(self.n('forecast', stack_id, block_id, stack_type)), stack_id)

        elif stack_type == 'trend':
            theta_f = theta_b = self.reg(self.get_layer(self.n('theta_f_b', stack_id, block_id, stack_type)), stack_id)

            backcast = self.get_layer(self.n('lambda_t_b', stack_id, block_id, stack_type))

            forecast = self.get_layer(self.n('lambda_t_f', stack_id, block_id, stack_type))

        else:  # 'seasonality'
            theta_b = self.reg(self.get_layer(self.n('theta_b', stack_id, block_id, stack_type)), 
                                              stack_id)

            theta_f = self.reg(self.get_layer(name=self.n('theta_f', stack_id, block_id, stack_type)), 
                                    stack_id)

            backcast = self.get_layer(self.n('lambda_s_b', stack_id, block_id, stack_type))

            forecast = self.get_layer(self.n('lambda_s_b', stack_id, block_id, stack_type))

        for k in range(self.input_dim):
            if self.has_exog():
                layer_name = f'exog_concat_{k}'
                layer = self.get_layer(layer_name)
                d0 = layer([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    # the original code for reference (from the standalone model)
    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):

        backcast_ = {}
        forecast_ = {}
        d1 = self.reg(Dense(self.units, activation='relu', name=self.n('d1', stack_id, block_id, stack_type)), stack_id)
        d2 = self.reg(Dense(self.units, activation='relu', name=self.n('d2', stack_id, block_id, stack_type)), stack_id)
        d3 = self.reg(Dense(self.units, activation='relu', name=self.n('d3', stack_id, block_id, stack_type)), stack_id)
        d4 = self.reg(Dense(self.units, activation='relu', name=self.n('d4', stack_id, block_id, stack_type)), stack_id)
        if stack_type == 'generic':
            theta_b = self.reg(Dense(nb_poly, activation='linear', use_bias=False, name=self.n('theta_b', stack_id, block_id, stack_type)), stack_id)
            theta_f = self.reg(Dense(nb_poly, activation='linear', use_bias=False, name=self.n('theta_f', stack_id, block_id, stack_type)), stack_id)
            backcast = self.reg(Dense(self.backcast_length, activation='linear', name=self.n('backcast', stack_id, block_id, stack_type)), stack_id)
            forecast = self.reg(Dense(self.forecast_length, activation='linear', name=self.n('forecast', stack_id, block_id, stack_type)), stack_id)
        elif stack_type == 'trend':
            theta_f = theta_b = self.reg(Dense(nb_poly, activation='linear', use_bias=False, name=self.n('theta_f_b', stack_id, block_id, stack_type)), stack_id)
            backcast = Lambda(trend_model, arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
            forecast = Lambda(trend_model, arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = self.reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=self.n('theta_b', stack_id, block_id, stack_type)), stack_id)
            else:
                theta_b = self.reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=self.n('theta_b', stack_id, block_id, stack_type)), stack_id)
            theta_f = self.reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=self.n('theta_f', stack_id, block_id, stack_type)), stack_id)
            backcast = Lambda(seasonality_model,
                              arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_


def linear_space(backcast_length, forecast_length, is_forecast=True):
    # ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    # return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))
    horizon = forecast_length if is_forecast else backcast_length
    return K.arange(0, horizon) / horizon


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))
