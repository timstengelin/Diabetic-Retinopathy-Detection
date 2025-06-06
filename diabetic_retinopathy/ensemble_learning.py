import os
import tensorflow as tf
import logging
import gin


@gin.configurable
def create_ensemble_model(models, run_paths, model_names):

    '''
    Combines the given models into an ensemble model

    Args:
        model (keras.Model): Neural network to analyze
        run_paths (dict): Dictionary containing directories for saving outputs

    '''

    path_model_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
    helper_path_list = [
        os.path.join(path_model_root, model_names[0], 'ckpts'),
        os.path.join(path_model_root, model_names[1], 'ckpts'),
        os.path.join(path_model_root, model_names[2], 'ckpts'),
        os.path.join(path_model_root, model_names[3], 'ckpts')]

    # Log message
    logging.info('Start combining the given models into an ensemble model')

    # Create empty lists
    ckpts = []
    ckpt_managers = []

    # Restore Checkpoints for models
    for idx, model in enumerate(models):
        ckpts.append(tf.train.Checkpoint(step=tf.Variable(1),
                                         optimizer=tf.keras.optimizers.Adam(),
                                         net=model))
        ckpt_managers.append(
            tf.train.CheckpointManager(checkpoint=ckpts[-1],
                                       directory=helper_path_list[idx],
                                       max_to_keep=1))
        ckpts[-1].restore(ckpt_managers[-1].latest_checkpoint).expect_partial()

        logging.info("Restored from"
                     "{}".format(ckpt_managers[-1].latest_checkpoint))

    # Input layer of ensemble model
    ensemble_input = tf.keras.Input(shape=(256, 256, 3))

    # Combine outputs from all models
    model_outputs = [model(ensemble_input) for model in models]

    # Compute the average probabilities across all models
    mean_probs = tf.reduce_mean(model_outputs, axis=0)

    # Determine the final class based on argmax (voting principle)
    final_classes = tf.argmax(mean_probs, axis=1)

    # Convert the final classes to probabilities (100% for the chosen class,
    # 0% for the other class)
    combined_output = tf.one_hot(final_classes, depth=2, dtype=tf.float32)

    # Define the final ensemble model
    ensemble_model = tf.keras.Model(inputs=ensemble_input,
                                    outputs=combined_output)
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Summarize and log the ensemble model
    ensemble_model.summary()

    # Checkpoint manager for ensemble model
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               optimizer=tf.keras.optimizers.Adam(),
                               net=ensemble_model)
    ckpt_manager = (
        tf.train.CheckpointManager(checkpoint=ckpt,
                                   directory=run_paths["path_ckpts_train"],
                                   max_to_keep=1))

    # Save final checkpoint of ensemble model
    ckpt_path = ckpt_manager.save()
    logging.info(f"Final checkpoint of ensemble model saved at: {ckpt_path}")
