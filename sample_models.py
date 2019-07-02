import models as M


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    return M.RNNModel(bd_merge=None,
                      rnn_type=M.RNNType.SimpleRNN,
                      time_distributed_dense=False).model(input_shape=(None, input_dim), output_dim=output_dim)


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    return M.RNNModel(bd_merge=None,
                      rnn_type=M.RNNType.LSTM,
                      rnn_units=units,
                      activation=activation).model(input_shape=(None, input_dim), output_dim=output_dim)


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    return M.RNNModel(cnn_config=M.CNNConfig(filters=filters, kernel_size=kernel_size, conv_stride=conv_stride,
                                             conv_border_mode=conv_border_mode),
                      bd_merge=None,
                      rnn_type=M.RNNType.LSTM,
                      rnn_units=units).model(input_shape=(None, input_dim), output_dim=output_dim)


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    return M.RNNModel(bd_merge=None,
                      rnn_type=M.RNNType.LSTM,
                      rnn_layers=recur_layers,
                      rnn_units=units).model(input_shape=(None, input_dim), output_dim=output_dim)


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    return M.RNNModel(bd_merge=M.BidirectionalMerge.concat,
                      rnn_type=M.RNNType.LSTM,
                      rnn_units=units).model(input_shape=(None, input_dim), output_dim=output_dim)


def final_model():
    """ Build a deep network for speech 
    """
    return M.RNNModel(cnn_config=M.CNNConfig(kernel_size=3, conv_stride=1, conv_border_mode="same",
                                             cnn_layers=12, cnn_dropout_rate=0.25,
                                             cnn_activation_before_bn_do=True,
                                             cnn_do_bn_order=True),
                      bd_merge=M.BidirectionalMerge.concat,
                      rnn_type=M.RNNType.GRU,
                      rnn_dense=True, rnn_units=250, rnn_layers=4, rnn_dropout_rate=0.2,
                      dropout_rate=0.3, name_suffix="Final").model(input_shape=(None, 26), output_dim=29)
