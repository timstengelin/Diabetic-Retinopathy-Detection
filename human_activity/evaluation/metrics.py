"""This file implements all custom metrics."""

import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):
    """Represents the custom confusion matrix metric."""

    def __init__(self, name='confusion_matrix', n_classes=12, **kwargs):
        """Initialize the confusion matrix.

        Parameters:
            name (string): name of metric
            n_classes (int): number of classes for confusion matrix
        """
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.binary_confusion_matrix = (
            self.add_weight(name='confusion_matrix',
                            shape=(self.n_classes, self.n_classes),
                            initializer='zeros'))

    def update_state(self, labels, predictions, *args, **kwargs):
        """Update the state of the metric.

        Parameters:
            labels (Tensor): data with true classes
            predictions (Tensor): data with predicted classes
        """
        # calculate sum over one-hot-coding to get indices of
        # all all-zero entrie
        labels_sum = tf.reshape(tf.reduce_sum(labels, axis=-1), [-1])
        index = tf.where(labels_sum != 0)

        # Convert predictions to a 1D array of predicted class indices
        predictions = tf.reshape(tf.math.argmax(predictions, axis=-1),
                                 shape=(-1, 1))
        labels = tf.reshape(tf.math.argmax(labels, axis=-1),
                            [-1])

        # delete the samples without classification
        labels = tf.gather(labels, index)
        predictions = tf.gather(predictions, index)

        # Compute the confusion matrix and update the state
        self.binary_confusion_matrix = tf.math.confusion_matrix(
            tf.squeeze(labels),
            tf.squeeze(predictions),
            num_classes=self.n_classes,
            dtype=tf.float32)

    def result(self):
        """Return confusion matrix.

        Parameters:
            - None -

        Output:
            binary_confusion_matric (Tensor): confusion matrix
        """
        return self.binary_confusion_matrix


class CategoricalAccuracy(tf.keras.metrics.Metric):
    """Represent the custom accuracy metric."""

    def __init__(self, name='accuracy', **kwargs):
        """Initialize the confusion matrix.

        Parameters:
            name (string): name of metric
        """
        super(CategoricalAccuracy, self).__init__(name=name, **kwargs)
        # initialize parameters
        self.accuracy = self.add_weight(name='categorical_accuracy',
                                        initializer='zeros')
        self.count = self.add_weight(name='count',
                                     initializer='zeros')

    def update_state(self, labels, predictions, *args, **kwargs):
        """Update the state of the metric.

        Parameters:
            labels (Tensor): data with true classes
            predictions (Tensor): data with predicted classes
        """
        # get highest rated class
        labels = tf.reshape(tf.argmax(tf.cast(labels,
                                              dtype=tf.float32),
                                      axis=-1), [-1])
        predictions = tf.reshape(tf.argmax(tf.cast(predictions,
                                                   dtype=tf.float32),
                                           axis=-1), [-1])

        # delete the samples without classification
        index = tf.where(labels != 0)
        label_clean = tf.gather(labels, index)
        prediction_clean = tf.gather(predictions, index)

        # compare true and predicted
        acc = tf.cast(tf.equal(label_clean, prediction_clean),
                      dtype=tf.float32)

        # reduce to one value entry
        self.accuracy.assign_add(tf.reduce_mean(acc))
        self.count.assign_add(1)

    def result(self):
        """Return confusion matrix.

        Parameters:
            - None -

        Output:
            binary_confusion_matric (Tensor): confusion matrix
        """
        # return parameters
        return self.accuracy/self.count


def crossentropy(labels, predictions, weights, *args, **kwargs):
    """Return crossentropy loss for given data.

    Parameters:
        labels (Tensor): data with true classes
        predictions (Tensor): data with predicted classes
        weights (Array): weights for each class

    Output:
        loss (int): calculated loss
    """
    # scale predictions so that the class probas of each sample sum to 1
    predictions /= tf.keras.backend.sum(predictions, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    predictions = tf.keras.backend.clip(predictions,
                                        tf.keras.backend.epsilon(),
                                        1 - tf.keras.backend.epsilon())
    # calc
    loss = labels * tf.keras.backend.log(predictions) * weights
    loss = -tf.keras.backend.sum(loss, -1)

    return loss
