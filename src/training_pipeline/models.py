import pickle

from pathlib import Path

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

from src.setup.paths import COMET_SAVE_DIR, MODELS_DIR, make_fundamental_paths


def get_model(model_name: str) -> Lasso | LGBMRegressor | XGBRegressor:
    """
    
    Args:
        model_name (str): Capitalised forms of the following names are allowed: 'base' for the base model,
                          'xgboost' for XGBRegressor, 'lightgbm' for LGBMRegressor, and 'lasso' for Lasso.

    Returns:
        Lasso|XGBRegressor|LGBMRegressor: the requested model
    """
    models_and_names = {
        "lasso": Lasso,
        "lightgbm": LGBMRegressor,
        "xgboost": XGBRegressor,
    }

    if model_name.lower() in models_and_names.keys():
        return models_and_names[model_name.lower()]
    else:
        raise Exception("Provided improper model name")


def load_local_model(full_model_name: str) -> Pipeline:
    """
    Allows for model objects that have been downloaded from the model registry, or saved locally to be loaded
    and returned for inference. It was important that the function be global and that it allow models to be 
    loaded from either of the directories that correspond to each model source.

    Args:
        base_name: the base name of the sought model
        scenario: "start" or "end" data
        tuned_or_not: whether we seek the tuned or untuned version of each model. The accepted entries are "tuned" and
                      "untuned".

    Returns:
        Pipeline: the model as an object of the sklearn.pipeline.Pipeline class.
    """
    if not Path(MODELS_DIR).exists():
        make_fundamental_paths()

    model_file_path: Path = COMET_SAVE_DIR.joinpath(f"{full_model_name}")

    with open(model_file_path, "rb") as file:
        return pickle.load(file)


def get_full_model_name(scenario: str, base_name: str, tuned: bool) -> str:
    """
    What we want here is the string that describes what I consider to be the complete name of the
    model. This is the name that will be given to the model on Comet's model registry. It is also 
    the name that is given to the model in my local directories.

    Args:
        base_name: the most simple form of a model's name, such as: "lasso" and "xboost" 
        tuned: a boolean that indicates whether the model in question is tuned. It will be used tuned_or_untuned_string
               construct the string we want 

    Returns:
      str: the name that will be given to the model on Comet
    """
    tuned_string = "tuned" if tuned else "untuned"
    return f"{tuned_string}_{base_name}_for_{scenario}s"

