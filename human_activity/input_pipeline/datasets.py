import os
import gin
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt


@gin.configurable
def load(
        name,
        data_dir,
        window_size,
        window_shift,
        tfrecord_files_exist,
        batch_size):
    '''
    Loads and prepares data from files

    Args:
        name (string): abbreviation of the data set name
        data_dir (string): directory where the original raw data is stored
        window_size (int): length of the sliding window
        window_shift (int): amount of shift/slide of the sliding window
        batch_size (int): size of the batch

    Returns:
        ds_train (tf.data.Dataset): training dataset
        ds_val (tf.data.Dataset): validation dataset
        ds_test (tf.data.Dataset): test dataset
        ds_info (tf.data.Dataset): dataset info and structure

    '''

    if name != 'hapt':
        raise ValueError(
            'Unsupported dataset with name \'{}\'. Currently, only \'hapt\' '
            'is supported.'.format(name))

    logging.info('Loading and preparing dataset \'{}\'...'.format(name))

    # Define directory of TFRecord files
    data_dir_tfrecords = ('./input_pipeline/tfrecord_files/window_size_{}_'
                          'shift_{}').format(
        window_size, window_shift)
    os.makedirs(data_dir_tfrecords, exist_ok=True)

    # Create new TFRecord files if they do not exist
    if not tfrecord_files_exist:
        create_tfrecord_files(data_dir, window_size, window_shift)

    # Dataset file paths
    ds_train_path = os.path.join(data_dir_tfrecords, 'train.tfrecord')
    ds_val_path = os.path.join(data_dir_tfrecords, 'validation.tfrecord')
    ds_test_path = os.path.join(data_dir_tfrecords, 'test.tfrecord')

    # Define parsing structure
    ds_info = {
        'window_features': tf.io.FixedLenFeature([], tf.string),
        'window_label': tf.io.FixedLenFeature([], tf.string)
    }

    def parse_example(example):
        '''
        Parses TFRecord example

        Args:
            example (tf.Tensor): A serialized example from a TFRecord file

        Returns:
            window_feature (tf.Tensor): The deserialized tensor of feature data
            window_label (tf.Tensor): The deserialized tensor of label data
        '''

        # Parse serialized example to extract the byte-encoded tensors
        parsed_features = tf.io.parse_single_example(example, ds_info)

        # Deserialize the byte-encoded tensors back to their original tensor
        # formats
        window_feature = tf.io.parse_tensor(
            parsed_features['window_features'], out_type=tf.float64)
        window_label = tf.io.parse_tensor(
            parsed_features['window_label'], out_type=tf.int32)

        # Convert labels to one-hot encoding
        window_label_one_hot = tf.one_hot(window_label - 1, depth=12)

        return window_feature, window_label_one_hot

    # Load datasets
    ds_train = tf.data.TFRecordDataset(ds_train_path).map(parse_example)
    ds_val = tf.data.TFRecordDataset(ds_val_path).map(parse_example)
    ds_test = tf.data.TFRecordDataset(ds_test_path).map(parse_example)

    # Shuffle, batch, repeat training dataset
    # complete shuffle
    num_train_samples = sum(1 for _ in ds_train)
    ds_train = ds_train.shuffle(num_train_samples).batch(
        batch_size, drop_remainder=True).repeat().prefetch(
        tf.data.AUTOTUNE)   #
    # Batch the validation and test datasets
    num_val_samples = sum(1 for _ in ds_val)
    ds_val = ds_val.batch(num_val_samples).prefetch(tf.data.AUTOTUNE)
    # all test data processed in single batch
    num_test_samples = sum(1 for _ in ds_test)
    ds_test = ds_test.batch(num_test_samples).prefetch(tf.data.AUTOTUNE)

    logging.info('Dataset \'{}\' loaded and prepared.'.format(name))

    return ds_train, ds_val, ds_test, ds_info


@gin.configurable
def create_tfrecord_files(data_dir, window_size, window_shift, balance):
    '''
    Creates TFRecord files of original files

    Args:
        data_dir (string): directory where the original data is stored
        window_size: length of the sliding window
        window_shift: amount of shift/slide of the sliding window
    '''

    def load_labels(data_dir):
        '''
        Loads and organizes labels

        Args:
            data_dir (string): directory where the original data is stored

        Returns:
            experiment_labels_list (list):
                A list of 61 sublists of multiple subsublists,
                where each sublist represents the labels/activities of an
                experiment, and each subsublist representing a label/activity
        '''

        # Load the labels file using pandas
        labels_dataframe = pd.read_csv(
            "{}/HAPT_dataset/RawData/labels.txt".format(data_dir),
            sep=' ',
            header=None)

        # Create empty list
        experiment_labels_list = [[] for _ in range(61)]

        # Iterate over each label entry and append it to the corresponding
        # experiment's sublist
        for _, label in labels_dataframe.iterrows():
            experiment_number_id = label[0]  # Extract the experiment ID
            idx = experiment_number_id - 1  # Convert 1-based ID to 0-based idx
            experiment_labels_list[idx].append(label.tolist())

        return experiment_labels_list

    def load_features_filenames(data_dir):
        '''
        Retrieves accelerometer and gyroscope file names

        Args:
            data_dir (string): directory where the original data is stored

        Returns:
            acc_filenames (list):
                A list of 61 elements, where each element is an accelerometer
                file name for an experiment
            gyro_filenames (list):
                A list of 61 element, where each element is a gyroscope file
                name for an experiment
        '''

        # Get a list of all files in the raw data directory that start with
        # 'acc' (for accelerometer)
        acc_filenames = sorted([f for f in os.listdir(
            '{}/HAPT_dataset/'
            'RawData/'.format(data_dir)) if f.startswith('acc')])

        # Get a list of all files in the raw data directory that start with
        # 'gyro' (for gyroscope)
        gyro_filenames = sorted([f for f in os.listdir(
            '{}/HAPT_dataset/'
            'RawData/'.format(data_dir)) if f.startswith('gyro')])

        return acc_filenames, gyro_filenames

    def normalize_data(acc_data, gyro_data):
        '''
        Normalizes accelerometer and gyroscope data using Z-score normalization

        Args:
            acc_data (pandas.DataFrame): A DataFrame containing accelerometer
                data with columns 'x', 'y', 'z'
            gyro_data (pandas.DataFrame): A DataFrame containing gyroscope data
                with columns 'x', 'y', 'z'

        Returns:
            (numpy.ndarray): 2D array with 6 columns,
                    each column representing one of the 6 features,
                    and each row representing a data point in an experiment
        '''

        # Normalize accelerometer data for each axis ('x', 'y', 'z') using
        # Z-score
        normalized_acc_x = stats.zscore(acc_data['x'])
        normalized_acc_y = stats.zscore(acc_data['y'])
        normalized_acc_z = stats.zscore(acc_data['z'])

        # Normalize gyroscope data for each axis ('x', 'y', 'z') using Z-score
        normalized_gyro_x = stats.zscore(gyro_data['x'])
        normalized_gyro_y = stats.zscore(gyro_data['y'])
        normalized_gyro_z = stats.zscore(gyro_data['z'])

        # Combine the normalized accelerometer and gyroscope data into a single
        # 2D array
        return np.array([normalized_acc_x,
                         normalized_acc_y,
                         normalized_acc_z,
                         normalized_gyro_x,
                         normalized_gyro_y,
                         normalized_gyro_z]).T

    def assign_activitynumberids_to_array(
            experiment_length, experiment_labels):
        '''
        Assigns activity number ids to an array with the length of the
            number of data points in an experiment

        Args:
            experiment_length (int): The number of data points in an experiment
            experiment_labels (list):
                A list of multiple sublists,
                where each sublist represents a label/activity

        Returns:
            labels_sequence (ndarray): An 1D-array with the length of the
                number of data points in an experiment,
                where each element represents the current activity for this
                datapoint with its activity number id
        '''

        labels_sequence = np.zeros(experiment_length, dtype=np.int32)

        # Loop over all experiment labels
        for experiment_label in experiment_labels:
            label_start_point, label_end_point, activity_number_id\
                = experiment_label[3], experiment_label[4], experiment_label[2]

            # Assign the activity number id to the positions in 'labels' from
            # label start point to label end point
            labels_sequence[label_start_point:label_end_point +
                            1] = activity_number_id
        return labels_sequence

    def load_split_features_and_labels_for_datasets(
            acc_filenames, gyro_filenames, experiment_labels_list, data_dir):
        '''
        Splits the features and labels for accelerometer and gyroscope data
        based on experiment IDs into
        training, validation, and test sets

        Args:
            acc_filenames (list):
                A list of 61 elements, where each element is an accelerometer
                file name for an experiment
            gyro_filenames (list):
                A list of 61 element, where each element is a gyroscope file
                name for an experiment
            experiment_labels_list (list):
                A list of 61 sublists of multiple subsublists,
                where each sublist represents the labels/activities of an
                experiment, and each subsublist representing a label/activity
            data_dir (string): directory where the original data is stored

        Returns:
            train_features (numpy.ndarray):
                An 2D-array of feature data for training
            train_labels (numpy.ndarray):
                An 1D-array of label data for training
            val_features (numpy.ndarray):
                An 2D-array of feature data for validation
            val_labels (numpy.ndarray):
                An 1D-array of label data for validation
            test_features (numpy.ndarray):
                An 2D-array of feature data for testing
            test_labels (numpy.ndarray):
                An 1D-array of label data for testing
        '''

        def load_features_and_labels_of_experiment(experiment_id):
            '''
            Loads accelerometer and gyroscope data for a specific experiment

            Args:
                experiment_id (int): experiment number ID

            Returns:
                data (numpy.ndarray): 2D array with 6 columns,
                    each column representing one of the 6 features,
                    and each row representing a data point in an experiment
                labels_sequence (ndarray): An 1D-array with the length of the
                    number of data points in an experiment,
                    where each element represents the current activity for
                    this datapoint with its activity number id

            '''

            # Load file names for specific experiment
            acc_file = acc_filenames[experiment_id - 1]
            gyro_file = gyro_filenames[experiment_id - 1]

            # Load data for a specific experiment
            acc_data = pd.read_csv(
                '{}/HAPT_dataset/RawData/{}'.format(data_dir, acc_file),
                sep=' ', names=['x', 'y', 'z'])
            gyro_data = pd.read_csv(
                '{}/HAPT_dataset/RawData/{}'.format(data_dir, gyro_file),
                sep=' ', names=['x', 'y', 'z'])

            # Normalize data
            data = normalize_data(acc_data, gyro_data)

            # Assign activity number ids to an array with the length of the
            # number of data points in an experiment
            labels_sequence = assign_activitynumberids_to_array(
                len(acc_data), experiment_labels_list[experiment_id - 1])

            return data, labels_sequence

        def load_features_and_labels_for_dataset(experiment_ids):
            '''
            Splits data and labels according to experiment ranges for
            predefined split of the dataset

            Args:
                experiment_ids (list): List of experiment IDs to process

            Returns:
                (ndarray): A 2D-array
                    with 6 columns,
                    and [sum of number of data points for every experiment
                    belonging to the experiment range] rows
                    each column representing one of the 6 features,
                    and each row representing a data point
                (ndarray): An 1D-array
                    with [sum of number of data points for every experiment
                    belonging to the experiment range] elements
                    where each element represents the current activity for
                    this datapoint with its activity number id
            '''

            data_part, labels_sequence_part = (
                zip(*[load_features_and_labels_of_experiment(experiment_id)
                      for experiment_id in experiment_ids]))
            return np.concatenate(data_part), np.concatenate(
                labels_sequence_part)

        return (load_features_and_labels_for_dataset(train_experiment_ids),
                load_features_and_labels_for_dataset(
            val_experiment_ids),
                load_features_and_labels_for_dataset(test_experiment_ids))

    def create_tfrecord_dataset(features, labels, window_size, window_shift):
        '''
        Creates a TensorFlow dataset from features and labels using a sliding
            window approach

        Args:
            features (numpy.ndarray): An array of feature data
            labels (numpy.ndarray): An array of label data
            window_size (int): Length of the sliding window
            window_shift (int): Amount of shift/slide of the sliding window

        Returns:
            (tf.data.Dataset):
            A TensorFlow dataset with sliding windows of features and labels
        '''

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # Apply sliding window technique to the dataset
        dataset_windows = dataset.window(
            size=window_size,
            shift=window_shift,
            drop_remainder=True)

        # Flatten the dataset of batching (flatten each window into a single
        # batch of data)
        dataset_flattened_windows = dataset_windows.flat_map(
            lambda x, y: tf.data.Dataset.zip(
                (x.batch(window_size), y.batch(window_size))))

        return dataset_flattened_windows

    def create_tfrecord_file(dataset, filepath):
        '''
        Writes a TensorFlow dataset to a TFRecord file

        Args:
            dataset (tf.data.Dataset): A TensorFlow dataset
            filepath (): The path to save the TFRecord file

        References:
            Website: Tensorflow
            Title: TFRecord and tf.train.Example
            URL: https://www.tensorflow.org/tutorials/load_data/tfrecord
        '''

        # Open a TFRecordWriter
        with tf.io.TFRecordWriter(filepath) as writer:

            # Iterate through each (feature_window, label_window) pair in the
            # dataset
            for window_feature, window_label in dataset:
                serialized_window_feature = tf.io.serialize_tensor(
                    window_feature).numpy()
                serialized_window_label = tf.io.serialize_tensor(
                    window_label).numpy()

                # Create a TFRecord Example containing the serialized feature
                # and label tensors
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'window_features': tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[serialized_window_feature])),
                            'window_label': tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[serialized_window_label]))}))

                # Write serialized Example to the TFRecord file
                writer.write(example.SerializeToString())

    # Set experiment ranges for predefined split of dataset
    # Train dataset: (user 01 to user 21) leads to (experiment 01 to
    # experiment 43)
    train_experiment_ids = range(1, 43 + 1)
    # Validation dataset: (user 28 to user 30) leads to (experiment 56 to
    # experiment 61)
    val_experiment_ids = range(56, 61 + 1)
    # Test dataset: (user 22 to user 27) leads to (experiment 44 to experiment
    # 55)
    test_experiment_ids = range(44, 55 + 1)
    # Total dataset: (user 01 to user 30) leads to (experiment 01 to
    # experiment 61)
    total_experiment_ids = range(1, 61 + 1)

    # Load and organize labels
    experiment_labels_list = load_labels(data_dir)

    # Retrieve accelerometer and gyroscope file names
    acc_filenames, gyro_filenames = load_features_filenames(data_dir)

    # Split the features and labels for accelerometer and gyroscope data
    # based on experiment IDs into
    # training, validation, and test sets
    ((train_features, train_labels), (val_features, val_labels),
     (test_features, test_labels))\
        = load_split_features_and_labels_for_datasets(acc_filenames,
                                                      gyro_filenames,
                                                      experiment_labels_list,
                                                      data_dir)

    # Define directory of TFRecord files
    data_dir_tfrecords = ('./input_pipeline/tfrecord_files/'
                          'window_size_{}_shift_{}').format(
        window_size, window_shift)

    # Count labels in train dataset -> Balance
    histo = np.histogram(train_labels, bins=np.arange(1, 14), density=True)
    dist, classes = histo
    dist = np.round(dist, decimals=3)
    logging.info(f'Distribution of activity classes: {dist}')

    # print histogram if configured
    if balance:
        labels = [train_labels, val_labels, test_labels]
        plt.hist(labels, bins=np.arange(1,14), density=True,
                 label=["train", "validation", "test"])
        plt.xlabel("Activity class")
        plt.ylabel("Class share")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(data_dir_tfrecords, 'histogram.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()

    # Create a TensorFlow dataset from features and labels using a sliding
    # window approach
    ds_train = create_tfrecord_dataset(
        train_features,
        train_labels,
        window_size,
        window_shift)
    ds_val = create_tfrecord_dataset(
        val_features, val_labels, window_size, window_shift)
    ds_test = create_tfrecord_dataset(
        test_features,
        test_labels,
        window_size,
        window_shift)

    # Write a TensorFlow dataset to a TFRecord file
    create_tfrecord_file(
        ds_train,
        '{}/train.tfrecord'.format(data_dir_tfrecords))
    create_tfrecord_file(
        ds_val, '{}/validation.tfrecord'.format(data_dir_tfrecords))
    create_tfrecord_file(
        ds_test,
        '{}/test.tfrecord'.format(data_dir_tfrecords))
