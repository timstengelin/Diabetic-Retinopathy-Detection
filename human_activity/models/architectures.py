"""This file implements the architectures for the model."""
import gin
import tensorflow as tf
import models.layers as custom_layers


@gin.configurable
def lstm_architecture(input_shape, n_classes, dropout_rate, units):
    """Implement a basic LSTM architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.basic_lstm_layer(inputs=inputs, units=int(units*2),
                                         return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_lstm_layer(inputs=inputs, units=int(units),
                                         return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/2))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_lstm_layer(inputs=out, units=int(units/2),
                                         return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/4))
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation="softmax")

    return tf.keras.Model(inputs=inputs, outputs=out, name='LSTM_model')


@gin.configurable
def gru_architecture(input_shape, n_classes, dropout_rate, units):
    """Implement a basic GRU architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.basic_gru_layer(inputs=inputs, units=int(units * 2),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_gru_layer(inputs=out, units=int(units),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/2))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_gru_layer(inputs=out, units=int(units / 2),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/4))
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation='softmax')

    return tf.keras.Model(inputs=inputs, outputs=out, name='GRU_model')


@gin.configurable
def rnn_architecture(input_shape, n_classes, dropout_rate, units):
    """Implement a basic RNN architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.basic_rnn_layer(inputs=inputs, units=int(units * 2),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_rnn_layer(inputs=out, units=int(units),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/2))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_rnn_layer(inputs=out, units=int(units / 2),
                                        return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/4))
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation='softmax')

    return tf.keras.Model(inputs=inputs, outputs=out, name='RNN_model')


@gin.configurable
def bidi_lstm_architecture(input_shape, n_classes, dropout_rate, units):
    """Implement a bidirectional LSTM architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.bidirectional_lstm_layer(inputs=inputs,
                                                 units=int(units*2),
                                                 return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.bidirectional_lstm_layer(inputs=inputs,
                                                 units=int(units),
                                                 return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/2))
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.bidirectional_lstm_layer(inputs=out,
                                                 units=int(units/2),
                                                 return_sequences=True)
    out = custom_layers.basic_dense_layer(inputs=out, units=int(units/4))
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation="softmax")

    return tf.keras.Model(inputs=inputs, outputs=out, name='bidi_LSTM_model')


@gin.configurable
def conv1d_architecture(input_shape, n_classes, dropout_rate, filters, units):
    """Implement a conv1d architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units
        filters (int): number of convolutional filters

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = custom_layers.conv1d_layer(inputs=inputs, filters=filters)
    out = custom_layers.conv1d_layer(inputs=out, filters=filters*2)
    out = custom_layers.max_pool1d_layer(inputs=out, pool_size=3)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_lstm_layer(inputs=out, units=int(units),
                                         return_sequences=True)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_lstm_layer(inputs=out, units=int(units/2),
                                         return_sequences=True)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation="softmax")

    return tf.keras.Model(inputs=inputs, outputs=out, name='bidi_LSTM_model')


@gin.configurable
def variable_lstm_architecture(input_shape, n_classes, dropout_rate, units,
                               n_layers):
    """Implement a variable LSTM architecture.

    Parameters:
        input_shape (list): input dimensions of the model
        n_classes (int): number of one-hot encoded classes
        dropout_rate (float): dropout rate
        units (int): number of hidden units
        n_layers (int): number of LSTM layers

    Returns:
        (model): tensorflow keras model with given dimensions
    """
    inputs = tf.keras.Input(input_shape)
    out = inputs
    for i in range(n_layers):
        out = custom_layers.basic_lstm_layer(
                        inputs=out,
                        units=int(units * (2 ** (n_layers-i-1))),
                        return_sequences=True)
        out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes*2,
                                          activation="tanh")
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = custom_layers.basic_dense_layer(inputs=out, units=n_classes,
                                          activation="softmax")

    return tf.keras.Model(inputs=inputs, outputs=out,
                          name='variable_LSTM_model')
