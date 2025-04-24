"""This file iomplements the tuning functionality."""
import logging
import gin
from train import Trainer
import input_pipeline.datasets as datasets
from utils import utils_misc
from train import create_model
import wandb


@gin.configurable
def tune(run_paths, key, sweep_config, parameters_dict, n_runs):
    """Tune models with wandb evaluation.

    Parameters:
        run_paths (string): dict with pathes
        key (string): key for upload in wandb
        sweep_config (dict): sweep config parameters for wandb
        parameters_dict (dict): hyperparameters for tuning
        n_runs (int): number of wandb runs
    """
    utils_misc.set_loggers(run_paths['path_logs_tune'], logging.INFO)
    # wandb login with given key
    wandb.login(key=key)

    sweep_config['parameters'] = parameters_dict

    # create wandb conf and obj
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project="Activity Recognition HAPT")

    # actual tuning function
    def tuning():
        """Train one model with tuning parameters."""
        config = {
            'steps': 50,
            'lr_rate': 0.01,
            'drop_rate': 0.25,
            'window_size': 250,
            'window_shift': 125,
            'batch_size': 64,
            'units': 64
        }
        if config:
            wandb.init(config=config)

            config = wandb.config

            # Model training code here ...
            ds_train, ds_val, ds_test, ds_info = (
                datasets.load(window_size=config.window_size,
                              window_shift=config.window_shift,
                              batch_size=config.batch_size,
                              tfrecord_files_exist=False))

            # get shape from actual dataset
            feature_shape = None
            label_shape = None
            for data, label in ds_train:
                feature_shape = data.shape[1:]
                label_shape = label.shape[1:]
                break

            model = create_model(model_name=config.model,
                                 feature_shape=feature_shape,
                                 label_shape=label_shape)

            trainer = Trainer(model=model, ds_train=ds_train, ds_val=ds_val,
                              run_paths=run_paths, total_steps=config.steps,
                              tuning=True, learning_rate=config.lr_rate)
            for _ in trainer.train():
                continue

    # use agent for optimization
    wandb.agent(sweep_id, function=tuning, count=n_runs)

    # clear wandb job
    wandb.finish()
