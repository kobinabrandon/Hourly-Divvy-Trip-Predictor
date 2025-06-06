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


class BaseModel:

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.data = pd.read_parquet(TRAINING_DATA/f"{scenario}s.parquet")

    @staticmethod
    def fit(x_train: pd.DataFrame, y_train: pd.Series):
        pass

    def train_test_split(self, cutoff_date: datetime) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        This is just a primitive splitting function that treats all data before a certain date
        as training data, and everything after that as test data.

        Args:
            cutoff_date:

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: the training and test sets  (features and targets)
        """
        # Separate training and test data
        training_data = self.data[self.data[f"{self.scenario}_hour"] < cutoff_date].reset_index(drop=True)
        test_data = self.data[self.data[f"{self.scenario}_hour"] > cutoff_date].reset_index(drop=True)

        # Split training data into features and the target
        x_train = training_data.drop(columns=["trips_next_hour"])
        y_train = training_data["trips_next_hour"]

        # Split testing data into features and the target
        x_test = test_data.drop(columns=["trips_next_hour"])
        y_test = test_data["trips_next_hour"]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def predict(x_test: pd.DataFrame) -> np.array:
        return x_test["trips_previous_1_hour"]

    @staticmethod
    def compute_error(y_true: float, y_pred: float):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def get_model(model_name: str) -> BaseModel | Lasso | LGBMRegressor | XGBRegressor:
    """
    
    Args:
        model_name (str): Capitalised forms of the following names are allowed: 'base' for the base model,
                          'xgboost' for XGBRegressor, 'lightgbm' for LGBMRegressor, and 'lasso' for Lasso.

    Returns:
        BaseModel|XGBRegressor|LGBMRegressor: the requested model
    """
    models_and_names = {
        "lasso": Lasso,
        "lightgbm": LGBMRegressor,
        "xgboost": XGBRegressor,
        "base": BaseModel
    }

    if model_name.lower() in models_and_names.keys():
        return models_and_names[model_name.lower()]
    else:
        raise Exception("Provided improper model name")


def load_local_model(directory: Path, model_name: str, scenario: str, tuned: str) -> Pipeline:
    """
    Allows for model objects that have been downloaded from the model registry, or created locally to be loaded
    and returned for inference or other purpose. It was important that the function be global and that it allow
    models to be loaded from either of the directories that correspond to each model source.

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

    assert model_name.lower() in ["base", "lasso", "lightgbm", "xgboost"], \
        "The requested model is not currently among those implemented"

    model_file_name = f"{model_name.title()} ({tuned_or_not.title()} for {scenario}s).pkl"
    model_file = directory / model_file_name
    with open(model_file, "rb") as file:
        return pickle.load(file)

