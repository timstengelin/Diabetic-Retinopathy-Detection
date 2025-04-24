"""This is the general file for all evaluation steps within the project."""

import tensorflow as tf
import gin
import evaluation.metrics as metrics
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
from utils import utils_misc
from evaluation.visualization import visualization
from train import create_model
from input_pipeline import datasets


@gin.configurable
def evaluate(model, ds_test, run_paths, n_classes):
    """Calculate test accuracy and confusion matrix.

    Parameters:
        model (keras.Model): keras model object for evaluation
        ds_test (tf.data.Dataset): test set
        model_name (string): name of the model
        run_paths (dictionary): storage path of model information
        n_classes (int): number of one hot encoded labels
    """
    logging.info(f'Starting evaluation via metrics of following model' +
                 f' from {run_paths["path_ckpts_train"]}.')
    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     net=model)
    checkpoint_manager = tf.train.CheckpointManager(
                                    checkpoint, run_paths["path_ckpts_train"],
                                    max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored model from " +
                     "{}".format(checkpoint_manager.latest_checkpoint))

    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(
                                                        from_logits=False))

    accuracy = metrics.CategoricalAccuracy()
    conf_matrix = metrics.ConfusionMatrix(n_classes=n_classes)

    for data, label in ds_test:
        y_pred = model(data)
        accuracy.update_state(label, y_pred)
        conf_matrix.update_state(label, y_pred)
        break
    logging.info(f'Calculated categorical accuracy for test dataset:' +
                 f' {accuracy.result().numpy()}')

    classes = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING",
               "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND",
               "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    # plot the confusion matrix
    array = conf_matrix.result().numpy()
    # normalize confusion matrix
    array = np.around(array.astype('float') /
                      (array.sum(axis=1))[:, np.newaxis], decimals=2)
    df_cm = pd.DataFrame(array, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    plt.title("Confusion matrix")
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(run_paths['path_board_val'],
                             'confusion_matrix.png'))


@gin.configurable
def evaluation(run_paths, model_name):
    """Do the managing of all general evaluation steps.

    Parameters:
        model_name (string): name of the model
        run_paths (dictionary): storage path of model information
    """
    utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

    # call of data pipeline to retrieve train, validation and test dataset
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # get shape from actual dataset
    feature_shape = None
    label_shape = None
    for data, label in ds_train:
        feature_shape = data.shape[1:]
        label_shape = label.shape[1:]
        break

    model = create_model(model_name=model_name,
                         feature_shape=feature_shape,
                         label_shape=label_shape)

    evaluate(model=model,
             ds_test=ds_test,
             run_paths=run_paths,
             n_classes=label_shape[-1])

    visualization(model=model,
                  run_paths=run_paths,
                  dataset=ds_test)
