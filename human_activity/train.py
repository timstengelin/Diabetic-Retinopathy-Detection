import gin
import tensorflow as tf
import logging

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, learning_rate, run_paths, total_steps, log_interval, ckpt_interval):
        # Summary Writer
        # ....

        # Loss objective
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=100)

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, data, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(data, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):

        # if training is interrupted unexpectedly, resume the model from here and continue training
        # or if it is the first step of training, start training from the beginning
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            logging.info("Restored training data from {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.step.assign_add(1)
        else:
            logging.info("Starting training for a new model...")

        for idx, (data, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(data, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_data, val_labels in self.ds_val:
                    self.val_step(val_data, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))
                
                # Write summary to tensorboard
                # ...

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                # Save checkpoint
                self.ckpt_manager.save()
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')

            if step % self.total_steps == 0:
                # Save final checkpoint
                self.ckpt_manager.save()
                logging.info(f'Finished training after {step} steps.')
                return self.val_accuracy.result().numpy()
