"""This file impellements the whole training process class for the model."""
import gin
import tensorflow as tf
import logging
import wandb
import evaluation.metrics as metrics
import numpy as np
from utils import utils_misc
from input_pipeline import datasets
import models.architectures as architectures
from evaluation.metrics import crossentropy


@gin.configurable
class Trainer(object):
    """Create an object Trainer for model training."""

    def __init__(self, model, ds_train, ds_val, learning_rate, run_paths,
                 total_steps, log_interval, ckpt_interval, class_weight,
                 tuning=False):
        """Initialize the Trainer.

        Parameters:
            model (tf.model): model which should be trained
            ds_train (tf.data.Dataset): training dataset
            ds_val (tf.data.Dataset): validation dataset
            learning_rate (float): learning rate
            run_paths (dict): run paths with pathes
            total_steps (int): total number of training steps
            log_interval (int): log interval
            ckpt_interval (int): checkpoint interval
            class_weight (array): class weights for balancing
            tuning (bool): whether the model is trained within tuning process
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = metrics.CategoricalAccuracy()

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = metrics.CategoricalAccuracy()

        # attributes
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.tuning = tuning
        self.class_weight = np.array(class_weight)

        # Checkpoint Manager
        if not tuning:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.optimizer,
                                            net=self.model)
            self.ckpt_manager = (
                tf.train.CheckpointManager(
                    checkpoint=self.ckpt,
                    directory=self.run_paths["path_ckpts_train"],
                    max_to_keep=10))

        # Summary Writer
        if not tuning:
            self.train_summary_writer = (
                tf.summary.create_file_writer(
                    self.run_paths['path_board_train']))
            self.val_summary_writer = (
                tf.summary.create_file_writer(
                    self.run_paths['path_board_val']))

    @tf.function
    def train_step(self, data, labels):
        """Train a step with the model.

        Parameters:
            data (Tensor): input data for train step
            labels (int): true labels for train step
        """
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = crossentropy(labels, predictions, self.class_weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, data, labels):
        """Validate a step with the model.

        Parameters:
            data (Tensor): input data for validation step
            labels (int): true labels for validation step
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        t_loss = crossentropy(labels, predictions, self.class_weight)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        """Train model with function.

        Parameters:
            none
        Returns:
            none
        """
        # resume the model by continuing training if model is available
        if not self.tuning:
            logging.info("Starting training for a new model...")

        for idx, (data, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(data, labels)

            # Write summary to tensorboard
            if not self.tuning:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(),
                                      step=step)
                    tf.summary.scalar('accuracy',
                                      self.train_accuracy.result() * 100,
                                      step=step)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_data, val_labels in self.ds_val:
                    self.val_step(val_data, val_labels)

                template = ('Step {}, Loss: {:.6f}, Accuracy: {:.6f}, ' +
                            'Validation Loss: {:.6f}, ' +
                            'Validation Accuracy: {:.6f}')
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                if not self.tuning:
                    with self.val_summary_writer.as_default():
                        tf.summary.scalar('loss', self.val_loss.result(),
                                          step=step)
                        tf.summary.scalar('accuracy',
                                          self.val_accuracy.result() * 100,
                                          step=step)

                if self.tuning:
                    wandb.log({"acc_train": self.train_accuracy.result()*100,
                               "loss_train": self.train_loss.result(),
                               "acc_val": self.val_accuracy.result()*100,
                               "loss_val": self.val_loss.result()})

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                # Save checkpoint
                if not self.tuning:
                    self.ckpt_manager.save()
                    logging.info(f'Saving checkpoint to ' +
                                 f'{self.run_paths["path_ckpts_train"]}.')

            if step % self.total_steps == 0:
                # Save final checkpoint
                if not self.tuning:
                    self.ckpt_manager.save()
                    logging.info(f'Finished training after {step} steps.')
                return self.val_accuracy.result().numpy()


def create_model(model_name, feature_shape, label_shape):
    """Create a model for the given name.

    Parameters:
        model_name (string): name of requested model
        feature_shape (Tensor): desired shape of model
        label_shape (Tensor): desired shape of labels

    Returns:
        model (tf.keras.Model): model according to name
    """
    if model_name == "LSTM_model":
        model = architectures.lstm_architecture(input_shape=feature_shape,
                                                n_classes=label_shape[-1])
    elif model_name == "bidi_LSTM_model":
        model = architectures.bidi_lstm_architecture(input_shape=feature_shape,
                                                     n_classes=label_shape[-1])
    elif model_name == "GRU_model":
        model = architectures.gru_architecture(input_shape=feature_shape,
                                               n_classes=label_shape[-1])
    elif model_name == "RNN_model":
        model = architectures.rnn_architecture(input_shape=feature_shape,
                                               n_classes=label_shape[-1])
    elif model_name == "Conv1d_model":
        model = architectures.conv1d_architecture(input_shape=feature_shape,
                                                  n_classes=label_shape[-1])
    elif model_name == "variable_LSTM_model":
        model = (architectures.
                 variable_lstm_architecture(input_shape=feature_shape,
                                            n_classes=label_shape[-1]))
    return model


@gin.configurable
def training(run_paths, model_name):
    """Create a model for the given name.

    Parameters:
        run_paths (dict): dict with pathes
        model_name (string): desired name of model
    """
    # set logger for training
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # get shape from actual dataset
    feature_shape = None
    label_shape = None
    for data, label in ds_train:
        feature_shape = data.shape[1:]
        label_shape = label.shape[1:]
        break

    model = create_model(model_name=model_name, feature_shape=feature_shape,
                         label_shape=label_shape)

    # initialize Trainer class based on given model and datasets
    trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val,
                      run_paths=run_paths)
    for _ in trainer.train():
        continue
