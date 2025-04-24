"""Main file for human_activity."""
import gin
import sys
from train import training
from evaluation.eval import evaluation
from utils import utils_params
import tune as tuning


def valid_model(model):
    """Implement check for model name.

    Parameters:
        model (string): model name

    Returns:
        True/False (Tensor): valid model
    """
    available_models = ["LSTM_model", "bidi_LSTM_model", "GRU_model",
                        "RNN_model", "Conv1d_model", "variable_LSTM_model"]
    # Check whether a implemented model is choosen and give feedback
    if model in available_models:
        return True
    else:
        print("Invalid model choosen. Choose from the following: " +
              str(available_models))
        return False


def main(argv):
    """Main function.

    Parameters:
        argv (array): array with command line arguments
    """
    # collect data from gin configuration file
    gin.parse_config_files_and_bindings(['configs/config.gin'],
                                        [])

    if "--train" in argv:
        model = argv[argv.index("--train") + 1]
        if valid_model(model):
            # generate folder structures for all model and logging data
            run_paths = utils_params.gen_run_folder(path_model_id=model)
            # set config path
            utils_params.save_config(run_paths['path_gin'], gin.config_str())
            # run training process
            training(run_paths=run_paths, model_name=model)

    elif "--evaluate" in argv:
        model = argv[argv.index("--evaluate") + 1]
        if valid_model(model):
            # generate folder structures for all model and logging data
            run_paths = utils_params.gen_run_folder(path_model_id=model)
            # set config path
            utils_params.save_config(run_paths['path_gin'], gin.config_str())
            # run evaluation process
            evaluation(run_paths=run_paths, model_name=model)

    elif "--tune" in argv:
        # generate folder structures for all model and logging data
        run_paths = utils_params.gen_run_folder(path_model_id="Tuning")
        # set config path
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        # run tuning process
        tuning.tune(run_paths=run_paths)

    else:
        print("Invalid arguments")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
