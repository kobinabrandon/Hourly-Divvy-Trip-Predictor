"""
Contains code for model training with and without hyperparameter tuning, as well as 
experiment tracking.
"""
import pickle
from pathlib import Path

import pandas as pd
from loguru import logger

from comet_ml import Experiment
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

from src.setup.config import config
from src.feature_pipeline.data_sourcing import load_raw_data
from src.feature_pipeline.preprocessing.core import make_training_data

from src.inference_pipeline.backend.model_registry import push_model
from src.training_pipeline.models import get_full_model_name, get_model 
from src.training_pipeline.hyperparameter_tuning import tune_hyperparameters
from src.setup.paths import TRAINING_DATA, LOCAL_SAVE_DIR, make_fundamental_paths
from src.training_pipeline.cleanup import delete_local_saves, identify_best_model, delete_prior_project_from_comet, delete_best_model_from_previous_run 


def get_or_make_training_data(scenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetches or builds the training data for the starts or ends of trips.

    Returns:
        pd.DataFrame: a tuple containing the training data's features and targets
    """
    assert scenario.lower() in ["start", "end"]
    data_path = TRAINING_DATA.joinpath(f"{scenario}s.parquet")
    
    if Path(data_path).is_file():
        training_data: pd.DataFrame = pd.read_parquet(path=data_path)
        logger.success(f"Fetched saved training data for {config.displayed_scenario_names[scenario].lower()}")
    else:
        logger.warning("No training data in storage. Creating the dataset will take a while.")

        raw_data: pd.DataFrame = load_raw_data()
        training_sets = make_training_data(data=raw_data, for_inference=False, geocode=False)
        training_data = training_sets[0] if scenario.lower() == "start" else training_sets[1]
        logger.success("Training data produced successfully")

    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)
    return features.sort_index(), target.sort_index()


def train(scenario: str, base_name: str, tune: bool, tuning_trials: int | None) -> float:
    """
    The function first checks for the existence of the training data, and builds it if
    it doesn't find it locally. Then it checks for a saved model. If it doesn't find a model,
    it will go on to build one, tune its hyperparameters, save the resulting model.

    Args:
        scenario (str): indicating whether training data on the starts or ends of trips. 
                        The only accepted answers are "start" and "end"

        model_name (str): the name of the model to be trained
        tune (bool | None, optional): whether to tune hyperparameters or not.
        hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters

    Returns:
        float: the error of the chosen model on the test dataset.
    """
    model_fn: object = get_model(model_name=base_name)
    features, target = get_or_make_training_data(scenario=scenario)

    train_sample_size = int(0.9 * len(features))
    x_train, x_test = features[:train_sample_size], features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    experiment = Experiment(
        api_key=config.comet_api_key,
        workspace=config.comet_workspace,
        project_name=config.comet_project_name,
        auto_metric_logging=False, 
        auto_param_logging=False, 
        log_git_metadata=False,
        log_env_details=False,
        log_git_patch=False,
        log_env_disk=False, 
        log_env_host=False,
        log_env_cpu=False,
        log_env_gpu=False,
        log_graph=False, 
        log_code=False,
    )

    model_name = get_full_model_name(scenario=scenario, base_name=base_name, tuned=tune) 
    experiment.set_name(name=model_name)

    if not tune:
        logger.info("Using the default hyperparameters")

        if isinstance(model_fn, XGBRegressor):
            pipeline = make_pipeline(model_fn)
        else:
            pipeline = make_pipeline(model_fn())

    else:
        logger.info(f"Tuning hyperparameters of the {model_name} model.")

        best_model_hyperparameters = tune_hyperparameters(
            model_fn=model_fn,
            tuning_trials=tuning_trials,
            experiment=experiment,
            x=x_train,
            y=y_train
        )

        logger.success(f"Best model hyperparameters {best_model_hyperparameters}")
        pipeline = make_pipeline(  model_fn(**best_model_hyperparameters)  )

    logger.info("Fitting model...")

    pipeline.fit(X=x_train, y=y_train)
    y_pred = pipeline.predict(x_test)
    test_error = mean_absolute_error(y_true=y_test, y_pred=y_pred)

    save_model_locally(model_fn=pipeline, model_name=model_name)
    experiment.log_metric(name="Test MAE", value=test_error)
    experiment.end()
    
    return test_error


def save_model_locally(model_fn: Pipeline, model_name: str):
    """
    Save the trained model locally as a .pkl file

    Args:
        model_fn (Pipeline): the model object to be stored
        model_name (str): the name of the model to be saved
    """
    path_to_pickle_file: Path = LOCAL_SAVE_DIR.joinpath(model_name)

    with open(path_to_pickle_file, mode="wb") as file:
        pickle.dump(obj=model_fn, file=file)

    logger.info(f"Saved {model_name} to disk")


def train_all_models(tuning_trials: int = config.tuning_trials):
    """
    Train the named models, identify the best performer (on the test data) and
    register it to the CometML model registry.

    Args:
        tuning_trials: the number of tuning trials 
    """
    make_fundamental_paths()  # Ensure that all the necessary directories exist.
    delete_prior_project_from_comet() 

    delete_local_saves()
    for scenario in ["start", "end"]:
        models_and_errors: dict[tuple[str, str], float] = {}
        delete_best_model_from_previous_run(scenario=scenario)

        for tune_or_not in [False, True]:
            for base_name in config.model_base_names:
                error = train(scenario=scenario, base_name=base_name, tune=tune_or_not, tuning_trials=tuning_trials)
                tuning_indicator: str = "untuned" if not tune_or_not else "tuned"
                models_and_errors[ (base_name, tuning_indicator) ] = error

        best_model_name: str = identify_best_model(scenario=scenario, models_and_errors=models_and_errors)
        logger.info(f"The best performing model for {scenario}s is {best_model_name} -> Pushing it to the CometML model registry")
        push_model(full_model_name=best_model_name, status="Production", version="1.0.0")


if __name__ == "__main__":
    train_all_models()

