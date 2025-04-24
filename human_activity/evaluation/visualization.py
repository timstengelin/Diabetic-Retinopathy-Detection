"""This file implements the sequence visualization functionality."""

import gin
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import tensorflow as tf
import logging
import matplotlib
import os


@gin.configurable
def visualization(model, run_paths, dataset, range_time):
    """Visualize a sequence.

    Parameters:
        model (keras.Model): keras model object to be evaluated
        run_paths (dictionary): storage path of model information
        dataset (tf.data.Dataset): dataset to be visualized
        range (array): if test-dataset (one batch), size can be controlled
    """
    logging.info(f'Starting evaluation via sequence visualization of ' +
                 f'following model from {run_paths["path_ckpts_train"]}.')
    # set up the model and load the checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     optimizer=tf.keras.optimizers.Adam(),
                                     net=model)
    checkpoint_manager = tf.train.CheckpointManager(
                                            checkpoint,
                                            run_paths["path_ckpts_train"],
                                            max_to_keep=10)
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    # prepare prediction, label and data for visualization
    for idx, (window, label) in enumerate(dataset):
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy())
        # get prediction from dataset/batch (None,250,12)
        prediction_raw = model(window)
        label_raw = label

        # get predicted/true class for each timestep -> (None,250)
        prediction = tf.argmax(prediction_raw, axis=-1)
        label = tf.argmax(label_raw, axis=-1)
        label_sum = tf.reduce_sum(label_raw, axis=-1)

        # append all classes and data from dataset/batch to have one timeline
        prediction = np.concatenate(prediction.numpy()[0::2])
        label = np.concatenate(label.numpy()[0::2])
        label_sum = np.concatenate(label_sum.numpy()[0::2])

        window = np.concatenate(window.numpy()[0::2])

        # prepare separate signals for the sensor data
        acc_x = window[:, 0]
        acc_y = window[:, 1]
        acc_z = window[:, 2]
        gyro_x = window[:, 3]
        gyro_y = window[:, 4]
        gyro_z = window[:, 5]
        break

    # Filtering of the classes to smooth out small mistakes
    prediction = median_filter(prediction, size=140)

    def legending(labels, axis):
        """Define the legend for the class colouring.

        Parameters:
            labels (array): labels to visualize in the legend
            axis (plt.axs): axis that gives the size of legend
        """
        label_color = ['white', 'dimgrey', 'darkorange', 'limegreen',
                       'royalblue', 'lightcoral', 'gold', 'aquamarine',
                       'mediumslateblue', 'saddlebrown', 'chartreuse',
                       'skyblue', 'violet']
        start = 0
        for i in range(1, int(labels.size)+1):
            axis.axvspan(i-1.5, i-0.5,
                         facecolor=label_color[labels[i-1]], alpha=0.5)

    def labeling(labels, labels_sum):
        """Define the color map corresponding to each label.

        Parameters:
            labels (array): labels to visualize in the labeling
            labels_argmin (array): inverted labels to visualize in the labeling
        """
        label_color = ['dimgrey', 'darkorange', 'limegreen', 'royalblue',
                       'lightcoral', 'gold', 'aquamarine', 'mediumslateblue',
                       'saddlebrown', 'chartreuse', 'skyblue', 'violet']
        start = 0
        non_labeled = True
        for i in range(1, int(labels.size)):
            if labels_sum[i] != 0:
                if labels[i] != labels[i - 1] and not non_labeled:
                    end = i - 1
                    plt.axvspan(start, end,
                                facecolor=label_color[labels[i - 1]],
                                alpha=0.5)
                    start = i
                elif non_labeled:
                    end = i - 1
                    plt.axvspan(start, end,
                                facecolor='white',
                                alpha=0.5)
                    start = i
                non_labeled = False
            else:
                if not non_labeled:
                    end = i - 1
                    plt.axvspan(start, end,
                                facecolor=label_color[labels[i - 1]],
                                alpha=0.5)
                    start = i
                non_labeled = True
        plt.axvspan(start, int(labels.size) - 1,
                    facecolor=label_color[labels[-1]], alpha=0.5)

    # limit sequence, that is shown from available dataset
    acc_x, acc_y, acc_z = (acc_x[range_time[0]:range_time[1]],
                           acc_y[range_time[0]:range_time[1]],
                           acc_z[range_time[0]:range_time[1]])
    gyro_x, gyro_y, gyro_z = (gyro_x[range_time[0]:range_time[1]],
                              gyro_y[range_time[0]:range_time[1]],
                              gyro_z[range_time[0]:range_time[1]])
    prediction = prediction[range_time[0]:range_time[1]]
    label_sum = label_sum[range_time[0]:range_time[1]]
    label = label[range_time[0]:range_time[1]]

    # plot the data as well as  their predictions and labels
    # (the first figure is generated from predicted value,
    # the second is the true distribution)
    plt.figure(figsize=(12, 9), dpi=300)

    # plot prediction coming from model
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 9),
                                        height_ratios=[3, 3, 1])
    ax1.set_title(label='PREDICTIONS',
                  fontdict={'weight': 'normal', 'size': 'x-large'})
    plt.tick_params(labelsize='x-small')
    ax1.set_xlabel("TIME SEQUENCE")
    ax1.set_ylabel("NORMALIZED LINEAR ACCELERATION")
    # same x-axis for other y-data
    ax12 = ax1.twinx()
    ax12.set_ylabel("NORMALIZED AUGULAR VELOCITY")
    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)
    labeling(prediction, label_sum)
    ax2.set_title('TRUE VALUES',
                  fontdict={'weight': 'normal', 'size': 'x-large'})
    plt.tick_params(labelsize='x-small')
    ax2.set_xlabel("TIME SEQUENCE")
    ax2.set_ylabel("NORMALIZED LINEAR ACCELERATION")
    ax22 = ax2.twinx()
    ax22.set_ylabel("NORMALIZED AUGULAR VELOCITY")
    plt.plot(acc_x, label='acc_x', linewidth=1)
    plt.plot(acc_y, label='acc_y', linewidth=1)
    plt.plot(acc_z, label='acc_z', linewidth=1)
    plt.plot(gyro_x, label='gyro_x', linewidth=1)
    plt.plot(gyro_y, label='gyro_y', linewidth=1)
    plt.plot(gyro_z, label='gyro_z', linewidth=1)
    labeling(label, label_sum)
    legending(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), ax3)
    ax3.set_yticks([])
    positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    labels = ["NO-LABEL", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND",
              "SIT_TO_LIE", "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    ax3.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(positions))
    ax3.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
    ax3.tick_params(axis='x', labelrotation=45)
    ax1.margins(0, 0)
    ax2.margins(0, 0)
    ax3.margins(0, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(run_paths['path_board_val'],
                             'sequence_visualization.png'), dpi=150)
    plt.show()
