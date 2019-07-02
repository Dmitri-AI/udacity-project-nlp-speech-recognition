from abc import abstractmethod
from enum import Enum
from typing import Optional

from keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional,
                          SimpleRNN,  # GRU, LSTM,
                          CuDNNGRU, CuDNNLSTM, Dropout, concatenate,
                          MaxPooling2D, Reshape)
from keras.models import Model


class ModelBuilder(object):
    @abstractmethod
    def model(self, input_shape, output_dim: int) -> Model:
        pass


class CNNConfig:
    filters: int
    kernel_size: int
    conv_stride: int
    conv_border_mode: str
    dilation: int
    cnn_layers: int
    cnn_activation: str
    cnn_activation_before_bn_do: bool
    cnn_dropout_rate: float
    cnn_do_bn_order: bool
    cnn_bn: bool
    cnn_dense: bool

    def __init__(self, filters=200,
                 kernel_size=11, conv_stride=2,
                 kernel_2d=None, conv_stride_2d=None,
                 conv_border_mode='valid', dilation: int = 1,
                 cnn_layers: int = 1, cnn_activation="relu", cnn_activation_before_bn_do: bool = True,
                 cnn_dropout_rate: float = None,
                 cnn_bn: bool = True, cnn_do_bn_order: bool = True, cnn_dense: bool = False):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.kernel_2d = kernel_2d
        self.conv_stride_2d = conv_stride_2d
        self.conv_border_mode = conv_border_mode
        self.dilation = dilation
        self.cnn_activation = cnn_activation
        self.cnn_activation_before_bn_do = cnn_activation_before_bn_do
        self.cnn_layers = cnn_layers
        self.cnn_dropout_rate = cnn_dropout_rate
        self.cnn_bn = cnn_bn
        self.cnn_do_bn_order = cnn_do_bn_order
        self.cnn_dense = cnn_dense


class BidirectionalMerge(Enum):
    concat = "concat"
    ave = "ave"
    sum = "sum"
    mul = "mul"


class RNNType(Enum):
    SimpleRNN = SimpleRNN
    GRU = CuDNNGRU
    LSTM = CuDNNLSTM


class RNNModel(ModelBuilder):
    rnn_dense: bool
    cnn_config: Optional[CNNConfig]
    bd_merge: Optional[BidirectionalMerge]
    rnn_type: RNNType
    rnn_units: int
    rnn_layers: int
    rnn_activation: str
    rnn_bn: bool
    rnn_dropout_rate: Optional[float]
    rnn_do_bn_order: bool
    rnn_activation_before_bn_do: bool
    time_distributed_dense: bool
    activation: str
    activation_before_bn_do: bool
    dropout_rate: float
    do_bn_order: bool
    bn: bool
    name_suffix: Optional[str]

    def __init__(self, cnn_config: CNNConfig = None, bd_merge: BidirectionalMerge = BidirectionalMerge.concat,
                 rnn_type: RNNType = RNNType.GRU, rnn_units: int = 200, rnn_layers: int = 1, rnn_dense: bool = False,
                 rnn_activation: str = None,
                 rnn_bn: bool = True,
                 rnn_dropout_rate: float = None,
                 rnn_activation_before_bn_do: bool = False,
                 rnn_do_bn_order: bool = False,
                 activation_before_bn_do: bool = False,
                 do_bn_order: bool = False,
                 time_distributed_dense: bool = True,
                 activation: str = "relu",
                 dropout_rate: float = 0.2,
                 bn: bool = True,
                 name_suffix: str = None) -> None:
        self.cnn_config = cnn_config
        self.rnn_type = rnn_type
        self.bd_merge = bd_merge
        self.rnn_layers = rnn_layers
        self.rnn_dense = rnn_dense
        self.rnn_units = rnn_units
        self.rnn_activation = rnn_activation if rnn_activation else activation
        self.rnn_bn = rnn_bn
        self.rnn_dropout_rate = rnn_dropout_rate if rnn_dropout_rate is not None else dropout_rate
        self.rnn_do_bn_order = rnn_do_bn_order
        self.rnn_activation_before_bn_do = rnn_activation_before_bn_do
        self.time_distributed_dense = time_distributed_dense
        self.activation = activation
        self.do_bn_order = do_bn_order
        self.activation_before_bn_do = activation_before_bn_do
        self.bn = bn
        self.dropout_rate = dropout_rate
        self.name_suffix = name_suffix
        if self.cnn_config:
            if self.cnn_config.cnn_dropout_rate is None:
                self.cnn_config.cnn_dropout_rate = self.dropout_rate

    def model(self, input_shape, output_dim: int):
        """ Build a recurrent network for speech
        """
        # Main acoustic input

        input_data = Input(name='the_input', shape=input_shape)
        # print("input shape", input_shape)
        x = input_data

        if self.cnn_config:
            if self.cnn_config.kernel_2d is not None:
                reshape_up = Reshape((*input_shape, 1))
                x = reshape_up(x)
                # x = K.expand_dims(input_data, -1)

        z = None
        if self.cnn_config:
            dil = 1
            if self.cnn_config.dilation < -1 and self.cnn_config.cnn_layers > 1:
                dil = (-self.cnn_config.dilation) ** (self.cnn_config.cnn_layers - 1)

            for layer_i in range(0, self.cnn_config.cnn_layers):
                in_layer_activation = self.cnn_config.cnn_activation
                if not self.cnn_config.cnn_activation_before_bn_do or self.cnn_config.cnn_dense:
                    in_layer_activation = None
                if self.cnn_config.kernel_2d is not None:
                    conv = Conv2D(self.cnn_config.filters, self.cnn_config.kernel_2d,
                                  strides=self.cnn_config.conv_stride_2d,
                                  padding="same",
                                  activation=in_layer_activation)
                else:
                    conv = Conv1D(self.cnn_config.filters, self.cnn_config.kernel_size,
                                  strides=self.cnn_config.conv_stride,
                                  padding=self.cnn_config.conv_border_mode,
                                  dilation_rate=dil,
                                  activation=in_layer_activation,
                                  name='conv1d' + str(layer_i + 1))
                if self.cnn_config.cnn_dense:
                    if layer_i == 0:
                        z = x
                    else:
                        if self.cnn_config.kernel_2d is not None:
                            # print(layer_i, x.shape, z.shape)
                            if layer_i % (self.cnn_config.cnn_layers // 5) == 0 and layer_i > 1:
                                z = x
                            else:
                                z = concatenate([z, x], axis=-1)
                        else:
                            z = concatenate([z, x], axis=-1)
                    x = conv(z)
                    if (
                            layer_i < self.cnn_config.cnn_layers - 1 or self.rnn_layers > 0) and self.cnn_config.kernel_2d is None:
                        if self.cnn_config.cnn_bn:
                            x = BatchNormalization()(x)
                        if not self.cnn_config.cnn_activation_before_bn_do:
                            x = Activation(self.cnn_config.cnn_activation, name=self.activation + "C" + str(layer_i))(x)
                        if self.cnn_config.cnn_dropout_rate > 0.01:
                            x = Dropout(rate=self.cnn_config.cnn_dropout_rate)(x)
                else:
                    # print("Before x = conv(x)", x.shape)
                    x = conv(x)
                    # print("After x = conv(x)", x.shape)
                    if (
                            layer_i < self.cnn_config.cnn_layers - 1 or self.rnn_layers > 0) and self.cnn_config.kernel_2d is None:
                        if self.cnn_config.cnn_dropout_rate is None:
                            self.cnn_config.cnn_dropout_rate = self.dropout_rate

                        if self.cnn_config.cnn_do_bn_order:
                            if self.cnn_config.cnn_dropout_rate > 0.01:
                                x = Dropout(rate=self.cnn_config.cnn_dropout_rate)(x)
                            if not self.cnn_config.cnn_activation_before_bn_do:
                                x = Activation(self.cnn_config.cnn_activation,
                                               name=self.activation + "C" + str(layer_i))(x)
                            if self.cnn_config.cnn_bn:
                                x = BatchNormalization()(x)
                        else:
                            if self.cnn_config.cnn_bn:
                                x = BatchNormalization()(x)
                            if not self.cnn_config.cnn_activation_before_bn_do:
                                x = Activation(self.cnn_config.cnn_activation,
                                               name=self.activation + "C" + str(layer_i))(x)
                            if self.cnn_config.cnn_dropout_rate > 0.01:
                                x = Dropout(rate=self.cnn_config.cnn_dropout_rate)(x)

                    if self.cnn_config.dilation < -1:
                        dil = dil // (-self.cnn_config.dilation)
                    if self.cnn_config.dilation > 1:
                        dil *= self.cnn_config.dilation
                if self.cnn_config.kernel_2d is not None:
                    # if self.cnn_config.kernel_2d is not None and (layer_i+1) % (self.cnn_config.cnn_layers // 5) == 0:
                    if self.cnn_config.cnn_bn:
                        x = BatchNormalization()(x)
                    # if not self.cnn_config.cnn_activation_before_bn_do:
                    x = Activation(self.cnn_config.cnn_activation, name=self.activation + "C" + str(layer_i))(x)
                    if not self.cnn_config.cnn_dense:
                        pool = MaxPooling2D(pool_size=(1, 2))
                        x = pool(x)
                    elif (layer_i + 1) % (self.cnn_config.cnn_layers // 5) == 0:
                        # elif layer_i == self.cnn_config.cnn_layers - 1:
                        pool = MaxPooling2D(pool_size=(1, 2))
                        x = pool(x)
                    if self.cnn_config.cnn_dropout_rate > 0.01:
                        x = Dropout(rate=self.cnn_config.cnn_dropout_rate)(x)

        if self.cnn_config and self.cnn_config.kernel_2d is not None:
            # print("Before reshape", x.shape, type(x))
            reshape = Reshape((input_shape[0], -1))
            x = reshape(x)
            # x = K.reshape(x, (x.shape[0], x.shape[1], -1))
            # print("After reshape", x.shape)

        z = None
        for layer_i in range(0, self.rnn_layers):
            # noinspection PyCallingNonCallable
            rnn = self.rnn_type.value(self.rnn_units, return_sequences=True, name='rnn' + str(layer_i + 1))
            if self.bd_merge and layer_i == 0:
                rnn = Bidirectional(rnn, merge_mode=self.bd_merge.name)
            if self.rnn_dense and layer_i > 0:
                z = concatenate([z, x], axis=-1)
            else:
                z = x
            x = rnn(z)
            if layer_i < self.rnn_layers - 1:
                if self.rnn_bn:
                    x = BatchNormalization(name="R_BN_" + str(layer_i))(x)
                x = Activation(self.rnn_activation, name=self.activation + "R" + str(layer_i))(x)
                if self.rnn_dropout_rate > 0.01:
                    x = Dropout(rate=self.rnn_dropout_rate, name="R_DO_" + str(layer_i))(x)

        if self.time_distributed_dense:
            if self.activation_before_bn_do:
                x = Activation(self.activation, name=self.activation)(x)
            if self.do_bn_order:
                if self.dropout_rate > 0.01:
                    x = Dropout(rate=self.dropout_rate, name="TDD_DO")(x)
                if not self.activation_before_bn_do:
                    x = Activation(self.activation, name=self.activation)(x)
                if self.bn:
                    x = BatchNormalization(name="TDD_BN")(x)
            else:
                if self.bn:
                    x = BatchNormalization(name="TDD_BN")(x)
                if not self.activation_before_bn_do:
                    x = Activation(self.activation, name=self.activation)(x)
                if self.dropout_rate > 0.01:
                    x = Dropout(rate=self.dropout_rate, name="TDD_DO")(x)
            x = TimeDistributed(Dense(output_dim))(x)

        # Add softmax activation layer
        x = Activation('softmax', name='softmax')(x)
        # Specify the model
        # print("After activation")
        model = Model(inputs=input_data, outputs=x)
        # print("After model")
        model.name = self.model_name()
        if self.cnn_config:
            # Cannot pass self.field to lambda function as self gets included in the function and the function
            # becomes not serializable since self is not serializable and then the model does not serialize
            kernel_size = self.cnn_config.kernel_2d[
                0] if self.cnn_config.kernel_2d is not None else self.cnn_config.kernel_size
            conv_border_mode = "same" if self.cnn_config.kernel_2d is not None else self.cnn_config.conv_border_mode
            conv_stride = self.cnn_config.conv_stride_2d[
                0] if self.cnn_config.conv_stride_2d is not None else self.cnn_config.conv_stride
            dilation = abs(self.cnn_config.dilation)
            cnn_layers = self.cnn_config.cnn_layers

            model.output_length = lambda input_length: cnn_output_length(input_length, kernel_size,
                                                                         conv_border_mode, conv_stride,
                                                                         dilation,
                                                                         cnn_layers)
        else:
            model.output_length = lambda input_length: input_length

        model.summary()
        return model

    def model_name(self):
        name = []
        if self.cnn_config:
            name += "CNN"
            if self.cnn_config.cnn_dense:
                name += "_DENSE"
            name += ["(", self.cnn_config.filters]
            if self.cnn_config.kernel_2d is not None:
                name += [" (", self.cnn_config.kernel_2d, ",", self.cnn_config.conv_stride_2d, ")"]
            else:
                name += [" (", self.cnn_config.kernel_size, ",", self.cnn_config.conv_stride, ")"]
            if self.cnn_config.cnn_activation_before_bn_do and not self.cnn_config.cnn_dense:
                name += [" ", self.cnn_config.cnn_activation]

            if not (
                    self.cnn_config.cnn_do_bn_order or self.cnn_config.cnn_dense) or self.cnn_config.kernel_2d is not None:
                if self.cnn_config.cnn_bn:
                    name += " BN"
                if not self.cnn_config.cnn_activation_before_bn_do:
                    name += [" ", self.cnn_config.cnn_activation]
                if self.cnn_config.cnn_dropout_rate > 0.01:
                    name += [" DO(", self.cnn_config.cnn_dropout_rate, ")"]
            else:
                if self.cnn_config.cnn_dropout_rate > 0.01:
                    name += [" DO(", self.cnn_config.cnn_dropout_rate, ")"]
                if not self.cnn_config.cnn_activation_before_bn_do or self.cnn_config.cnn_dense:
                    name += [" ", self.cnn_config.cnn_activation]
                if self.cnn_config.cnn_bn:
                    name += " BN"
            name += ")"

            if self.cnn_config.cnn_layers > 1:
                name += ["x", self.cnn_config.cnn_layers]
                if self.cnn_config.dilation > 1 or self.cnn_config.dilation < -1:
                    name += [",d=", self.cnn_config.dilation]
            name += " "

        if self.rnn_layers > 0:
            if self.bd_merge:
                name += ["BD(", self.bd_merge.name, ") "]
            name += self.rnn_type.value.__name__
            if self.rnn_dense:
                name += "_DENSE"
            name += ["(", self.rnn_units, " x", self.rnn_layers]
            if self.rnn_layers > 1:
                # name += " DO(0.2)(:-1)"
                if self.rnn_bn:
                    name += " BN"
                name += " ", self.rnn_activation
                if self.rnn_dropout_rate > 0.01:
                    name += " DO(", self.rnn_dropout_rate, ")"
                if self.rnn_bn or self.rnn_activation or self.rnn_dropout_rate > 0.01:
                    name += "(:-1)"
            name += ")"

        if self.time_distributed_dense:
            if self.activation_before_bn_do:
                name += " ", self.activation
            if self.do_bn_order:
                if self.dropout_rate > 0.01:
                    name += " DO(", self.dropout_rate, ")"
                if not self.activation_before_bn_do:
                    name += " ", self.activation
                if self.bn:
                    name += " BN"
            else:
                if self.bn:
                    name += " BN"
                if not self.activation_before_bn_do:
                    name += " ", self.activation
                if self.dropout_rate > 0.01:
                    name += " DO(", self.dropout_rate, ")"
            name += " TD(D)"
        if self.name_suffix:
            name += [" ", self.name_suffix]
        return "".join([e if type(e) == str else repr(e) for e in name])


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1, cnn_layers=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    # print("cnn_output_length: input_length", input_length[1][0], "cnn_layers", cnn_layers)
    if input_length is None:
        return None
    if cnn_layers > 1:
        input_length = cnn_output_length(input_length, filter_size, border_mode, stride,
                                         dilation=dilation,
                                         cnn_layers=cnn_layers - 1)
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation ** (cnn_layers - 1) - 1)
    output_length = input_length
    if border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    output_length = (output_length + stride - 1) // stride
    # print("cnn_output_length: output_length", output_length[1][0])
    return output_length
