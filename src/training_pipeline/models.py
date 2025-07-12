import pickle
import numpy as np
import pandas as pd 

from pathlib import Path
from datetime import datetime

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

from src.setup.paths import TRAINING_DATA, MODELS_DIR, make_fundamental_paths


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


def load_local_model(directory: Path, model_name: str, scenario: str, tuned_or_not: str) -> Pipeline:
    """
    Allows for model objects that have been downloaded from the model registry, or saved locally to be loaded
    and returned for inference. It was important that the function be global and that it allow models to be 
    loaded from either of the directories that correspond to each model source.

    Args:
        directory: the directory where the models are being stored
        model_name: the name of the sought model
        scenario: "start" or "end" data
        tuned_or_not: whether we seek the tuned or untuned version of each model. The accepted entries are "tuned" and
                      "untuned".

    Returns:
        Pipeline: the model as an object of the sklearn.pipeline.Pipeline class.
    """
    if not Path(MODELS_DIR).exists():
        make_fundamental_paths()

    tuned_bool = True if tuned_or_not.lower() == "tuned" else False  
    full_model_name = get_full_model_name(
        scenario=scenario, 
        model_name=model_name, 
        tuned=tuned_bool
    )

    model_file_path: Path = MODELS_DIR.joinpath(f"{directory.joinpath(full_model_name)}.pkl")

    with open(model_file_path, "rb") as file:
        return pickle.load(file)


def get_name_of_model_type(model_name: str, tuned: bool) -> str:
    tuned_or_untuned_string = "Tuned" if tuned else "Untuned"
    return model_name.title().replace(f"_{tuned_or_untuned_string}", "")


def get_full_model_name(scenario: str, model_name: str, tuned: bool) -> str:
    """
    What we want here is the string that describes what I consider to be the complete name of the
    model. This is the name that will be given to the model on Comet's model registry. It is also 
    the name that is given to the model in my local directories.

    Args:
        model_name: A truncated form of the model's name. F 
        tuned: a boolean that indicates whether the model in question is tuned. It will be used tuned_or_untuned_string
               construct the string we want 

    Returns:
      str: the name that will be given to the model on Comet
    """
    tuned_or_untuned_string = "Tuned" if tuned else "Untuned"
    name_of_model_type = get_name_of_model_type(model_name=model_name, tuned=tuned)
    return f"{name_of_model_type} ({tuned_or_untuned_string} for {scenario}s)"

