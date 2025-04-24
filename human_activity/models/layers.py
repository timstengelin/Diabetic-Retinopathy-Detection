"""This file implements all layers for the model architectures."""
import gin
import tensorflow as tf


def basic_lstm_layer(inputs, units, return_sequences=True, batch_norm=True,
                     activation="tanh"):
    """Implement a basic single LSTM layer.

    Parameters:
        inputs (Tensor): input of the LSTM layer
        units (int): number of filters used for the LSTM layer
        return_sequences (bool): whether output is sequence or not
                                (Default is True)
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the single LSTM layer
    """
    out = tf.keras.layers.LSTM(units=units,
                               return_sequences=return_sequences)(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


@gin.configurable
def basic_dense_layer(inputs, units, batch_norm=True, activation="tanh"):
    """Implement a basic single dense layer.

    Parameters:
        inputs (Tensor): input of the dense layer
        units (int): number of filters used for the dense layer
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (bool): activation function to use (Default is "tanh")

    Returns:
        (Tensor): output of the single dense layer
    """
    out = tf.keras.layers.Dense(units, activation='linear')(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


@gin.configurable
def basic_gru_layer(inputs, units, return_sequences=True,
                    batch_norm=True, activation="tanh"):
    """Implement a basic GRU layer.

    Parameters:
        inputs (Tensor): input of the GRU layer
        units (int): number of filters used for the GRU layer
        return_sequences (bool): whether output is sequence or not
                                (Default is True)
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the single GRU layer
    """
    out = tf.keras.layers.GRU(units, return_sequences=return_sequences)(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


@gin.configurable
def basic_rnn_layer(inputs, units, return_sequences=True,
                    batch_norm=True, activation="tanh"):
    """Implement a basic RNN layer.

    Parameters:
        inputs (Tensor): input of the RNN layer
        units (int): number of filters used for the RNN layer
        return_sequences (bool): whether output is sequence or not
                                (Default is True)
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the single RNN layer
    """
    out = tf.keras.layers.SimpleRNN(units,
                                    return_sequences=return_sequences)(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


def bidirectional_lstm_layer(inputs, units, return_sequences=True,
                             batch_norm=True, activation="tanh"):
    """Implement a single bidirectional LSTM layer.

    Parameters:
        inputs (Tensor): input of the LSTM layer
        units (int): number of filters used for the LSTM layer
        return_sequences (bool): whether output is sequence or not
                                (Default is True)
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the bidirectional LSTM layer
    """
    out = tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(units=units,
                                return_sequences=return_sequences))(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


def conv1d_layer(inputs, filters, batch_norm=True, activation="tanh"):
    """Implement a basic single CNN layer.

    Parameters:
        inputs (Tensor): input of the LSTM layer
        filters (int): number of filters used for the layer
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the single Conv1d layer
    """
    out = tf.keras.layers.Conv1D(filters=filters, kernel_size=6,
                                 padding="same")(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out


def max_pool1d_layer(inputs, pool_size, batch_norm=True, activation="tanh"):
    """Implement a max pool layer.

    Parameters:
        inputs (Tensor): input of the LSTM layer
        pool_size (int): number of pool size used for the layer
        batch_norm (bool): whether the batch normalization is used or not
                            (Default is True)
        activation (string): which activation function to use
                            (Default is "tanh")

    Returns:
        (Tensor): output of the single max pool layer
    """
    out = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=1,
                                       padding="valid",
                                       data_format="channels_first")(inputs)
    if batch_norm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation(activation)(out)
    return out
