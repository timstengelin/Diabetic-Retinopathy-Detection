"""Implements the whole input pipeline for dataset."""
import gin
import logging
import tensorflow as tf
import os
import pandas as pd


def create_record(img_dir, csv_dir, filename_record, resampling):
    """Create tfrecord for given data.

    Parameters:
        img_dir (str): Directory containing images
        csv_dir (str): Directory containing csv files (label)
        filename_record (str): Desired name of tfrecord file
        resampling (bool): Resampling for tfrecord
    """
    # copied from tensorflow.org (wiki)
    def _int64_feature(value):
        """Return an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # copied from tensorflow.org (wiki)
    def _bytes_feature(value):
        """Return a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _preprocess_image(image_path):
        """Load and preprocess one image.

        Parameters:
            image_path (str): Path to image

        Output:
            img (Tensor): JPEG encoded image
        """
        img = tf.io.read_file(image_path)
        # directly decode from autodetect format to uint8 scaling
        # representation
        img = tf.io.decode_image(img, dtype=tf.uint8)
        # crop images to remove black boxes in right and left part
        # left box width = 266
        # right box width = 596
        # new target width with just eye: 4288-266-596 = 3426
        img = tf.image.crop_to_bounding_box(img, offset_height=0,
                                            offset_width=266,
                                            target_height=2848,
                                            target_width=3426)
        # still no quadratic view of eye -> padding needed
        # use padding to get desired shape of 3426x3426
        # new padding offset: (3426-2848)/2 = 289
        img = tf.image.pad_to_bounding_box(img, offset_height=289,
                                           offset_width=0, target_height=3426,
                                           target_width=3426)
        # resize the images to smaller size according to exercise
        img = tf.image.resize(img, size=(256, 256))
        # ensure range of uint8
        img = tf.cast(img, tf.uint8)
        # encode img with jpeg format to tensor
        img = tf.io.encode_jpeg(img, quality=100, format='rgb')

        return img

    logging.info('  Starting generation of new tf record from ' +
                 'following path: {}'.format(img_dir))

    # check, whether records folder exists -> if not, create one
    directory = os.path.dirname(filename_record)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with tf.io.TFRecordWriter(filename_record) as writer:
        # read image name + label (retinopathy grade) from csv file
        df = pd.read_csv(csv_dir)

        # separate teh different grades into separate dataframes
        df_grade_rg_0 = df[df['Retinopathy grade'] == 0]
        df_grade_rg_1 = df[df['Retinopathy grade'] == 1]
        df_grade_rg_2 = df[df['Retinopathy grade'] == 2]
        df_grade_rg_3 = df[df['Retinopathy grade'] == 3]
        df_grade_rg_4 = df[df['Retinopathy grade'] == 4]

        # create two different dataframes for classification of no dr or dr
        df_no_dr = pd.concat([df_grade_rg_0, df_grade_rg_1])
        df_dr = pd.concat([df_grade_rg_2, df_grade_rg_3, df_grade_rg_4])

        # visualize the number of available samples per class
        logging.info('  Samples imported with No DR: {}'.format(len(df_no_dr)))
        logging.info('  Samples imported with DR: {}'.format(len(df_dr)))

        if resampling:
            logging.info('  Resampling is activated...')
            if len(df_no_dr) > len(df_dr):
                df_dr = df_dr.sample(n=len(df_no_dr), replace=True)
            else:
                df_no_dr = df_no_dr.sample(n=len(df_dr), replace=True)
        else:
            logging.info('  Resampling is deactivated...')

        # define a new column/feature for dr
        df_dr['dr'] = 1
        df_no_dr['dr'] = 0
        df_sum = pd.concat([df_no_dr, df_dr])
        logging.info('  Resampling completed, total ' +
                     'samples: {}'.format(len(df_sum)))
        logging.info('  Distribution between classes ' +
                     '(DR/nonDR): {}'.format(len(df_dr)/len(df_no_dr)))

        logging.info('  Starting import of all images')
        # read images -> preprocess -> save to tfrecord
        for image in df_sum["Image name"]:
            image_path = os.path.join(img_dir, image + '.jpg')
            img = _preprocess_image(image_path)
            label = df_sum.loc[df_sum['Image name'] == image, 'dr'].values[0]
            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(img)}

            # Create sample for record
            sample = tf.train.Example(
                features=tf.train.Features(feature=feature))
            # Add sample to open record
            writer.write(sample.SerializeToString())
        logging.info('  Finished import of all images')


def read_record(record_filename, train_val_split=1.0):
    """Read tfrecord file.

    Parameters:
        record_filename (str): Path to tfrecord file
        train_val_split (float): Split ratio for training and validation

    Output:
        dataset (tf.data.TFRecordDataset): TFRecord dataset (2x with split)
    """
    logging.info('  Loading from following tf ' +
                 'record: {}'.format(record_filename))
    input_dataset = tf.data.TFRecordDataset(record_filename)
    feature_description = {'label':
                               tf.io.FixedLenFeature([], tf.int64,
                                                     default_value=0),
                           'image':
                               tf.io.FixedLenFeature([], tf.string,
                                                     default_value='')
                           }

    def _parse_function(example):
        """Parse single example.

        Parameters:
            example (bytes): Example.

        Output:
            example (Tensor): Parsed example
        """
        return tf.io.parse_single_example(example, feature_description)

    parsed_dataset = input_dataset.map(_parse_function)
    logging.info('  Dataset imported')

    # path for training set with given parameter factor
    if train_val_split < 1.0:
        logging.info('  Splitting train set into train-/val-set ' +
                     'with factor: {}'.format(train_val_split))
        count = 0
        for _ in parsed_dataset:
            count += 1
        # decode jpeg again
        # shuffling, to prevent bias after splitting in train and val
        dataset = parsed_dataset.map(lambda x:
                                     (tf.io.decode_jpeg(x['image']),
                                      x['label'])).shuffle(buffer_size=count)
        # split dataset according to given split
        train_dataset = dataset.take(int(count*train_val_split))
        val_dataset = dataset.skip(int(count*train_val_split))
        return train_dataset, val_dataset

    else:
        logging.info('  Just test set without any splitting')
        # decode jpeg again
        dataset = parsed_dataset.map(lambda x:
                                     (tf.io.decode_jpeg(x['image']),
                                      x['label']))
        return dataset


def prepare_dataset(dataset, augmentation, batch_size, caching, repeat):
    """Prepare the given dataset for usage.

    Parameters:
        dataset (tf.data.TFRecordDataset): TFRecord dataset
        augmentation (bool): Whether to apply augmentation
        batch_size (int): Batch size
        caching (bool): Whether to cache dataset
        repeat (bool): Whether to repeat the dataset

    Output:
        dataset (tf.data.TFRecordDataset): TFRecord dataset
    """
    def _normalize(image):
        """Normalize image."""
        return tf.cast(image, tf.float32) / 255.

    class RandomChance(tf.keras.layers.Layer):
        """Implement a layer with random chance."""

        def __init__(self, layer, probability, **kwargs):
            super(RandomChance, self).__init__(**kwargs)
            self.layer = layer
            self.probability = probability

        def call(self, inputs, training=True):
            apply_layer = tf.random.uniform([]) < self.probability
            outputs = tf.cond(
                pred=tf.logical_and(apply_layer, training),
                true_fn=lambda: self.layer(inputs),
                false_fn=lambda: inputs,
            )
            return outputs

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "layer": tf.keras.layers.serialize(self.layer),
                    "probability": self.probability,
                }
            )
            return config

    # keras model for data augmentation
    data_augmentation = tf.keras.Sequential([
        RandomChance(tf.keras.layers.RandomFlip("horizontal_and_vertical",
                                                dtype='uint8'), 0.75),
        RandomChance(tf.keras.layers.RandomRotation(0.5, dtype='uint8'),
                     0.5)
    ])

    if caching:
        dataset = dataset.cache()
    # used for training set
    if augmentation:
        logging.info('  Augmenting images for training dataset...')
        dataset = dataset.map(lambda image,
                                     label: (data_augmentation(image), label))

    # normalize whole dataset
    logging.info('  Normalizing images of dataset...')
    dataset = dataset.map(lambda image, label: (_normalize(image), label))

    count = 0
    for _ in dataset:
        count += 1
    dataset = dataset.shuffle(buffer_size=count,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


@gin.configurable
def load(load_record, img_dir, csv_dir, resampling, train_val_split,
         caching, batch_size, augmentation):
    """Load the given dataset.

    Parameters:
        load_record (bool): Whether to load the dataset from tfrecord.
        img_dir (str): Directory containing the images.
        csv_dir (str): Directory containing the labels.
        resampling (bool): Whether to resampling images.
        train_val_split (float): Train/val split.
        caching (bool): Whether to cache dataset
        batch_size (int): Batch size.
        augmentation (bool): Whether to apply augmentation

    Output:
        train_set (tf.data.TFRecordDataset): training dataset
        val_set (tf.data.TFRecordDataset): validation dataset
        test_set (tf.data.TFRecordDataset): testing dataset
    """
    train_record_filename = './input_pipeline/records/train.tfrecord'
    test_record_filename = './input_pipeline/records/test.tfrecord'
    if not load_record:
        logging.info('Creation of new tensorflow records from dataset started')

        # Read both train and test set separately
        train_img_dir = img_dir + '/train'
        test_img_dir = img_dir + '/test'
        train_csv_dir = csv_dir + '/train.csv'
        test_csv_dir = csv_dir + '/test.csv'
        create_record(img_dir=train_img_dir, csv_dir=train_csv_dir,
                      filename_record=train_record_filename,
                      resampling=resampling)
        create_record(img_dir=test_img_dir, csv_dir=test_csv_dir,
                      filename_record=test_record_filename,
                      resampling=False)

        logging.info('Creation of new record files from dataset finished')

    logging.info('Loading dataset from tensorflow records started')
    train_set, val_set = read_record(record_filename=train_record_filename,
                                     train_val_split=train_val_split)
    test_set = read_record(record_filename=test_record_filename)
    logging.info('Loading dataset from tensorflow records finished')

    # Preparation and augmentation (only for training data)
    logging.info('Starting preparation (and augmentation) of datasets...')
    train_set = prepare_dataset(train_set, augmentation=augmentation,
                                batch_size=batch_size, caching=caching,
                                repeat=True)
    val_set = prepare_dataset(val_set, augmentation=False,
                              batch_size=batch_size, caching=caching,
                              repeat=False)
    test_set = prepare_dataset(test_set, augmentation=False,
                               batch_size=batch_size, caching=caching,
                               repeat=False)
    logging.info('Finished preparation (and augmentation) of datasets...')

    return train_set, val_set, test_set
