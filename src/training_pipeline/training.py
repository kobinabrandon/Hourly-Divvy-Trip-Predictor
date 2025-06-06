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
from src.training_pipeline.models import get_model
from src.feature_pipeline.preprocessing import DataProcessor
from src.training_pipeline.hyperparameter_tuning import tune_hyperparameters
from src.setup.paths import TRAINING_DATA, LOCAL_SAVE_DIR, make_fundamental_paths
from src.inference_pipeline.backend.model_registry import push_model, get_full_model_name

from src.training_pipeline.cleanup import (
    identify_best_model,
    delete_prior_project_from_comet,
    delete_best_model_from_previous_run, 
)



def get_or_make_training_data(scenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetches or builds the training data for the starts or ends of trips.

    Returns:
        pd.DataFrame: a tuple containing the training data's features and targets
    """
    assert scenario.lower() in ["start", "end"]
    data_path = Path.joinpath(TRAINING_DATA, f"{scenario}s.parquet")
    
    if Path(data_path).is_file():
        training_data: pd.DataFrame = pd.read_parquet(path=data_path)
        logger.success(f"Fetched saved training data for {config.displayed_scenario_names[scenario].lower()}")
    else:
        logger.warning("No training data is storage. Creating the dataset will take a while.")

        processor = DataProcessor(years=config.years, for_inference=False)
        training_sets = processor.make_training_data(geocode=False)
        training_data = training_sets[0] if scenario.lower() == "start" else training_sets[1]
        logger.success("Training data produced successfully")

    target = training_data["trips_next_hour"]
    features = training_data.drop("trips_next_hour", axis=1)
    return features.sort_index(), target.sort_index()


def train(scenario: str, model_name: str, tune: bool, tuning_trials: int | None) -> float:
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
    model_fn: object = get_model(model_name=model_name)
    features, target = get_or_make_training_data(scenario=scenario)

    train_sample_size = int(0.9 * len(features))
    x_train, x_test = features[:train_sample_size], features[train_sample_size:]
    y_train, y_test = target[:train_sample_size], target[train_sample_size:]

    experiment = Experiment(
        api_key=config.comet_api_key,
        workspace=config.comet_workspace,
        project_name=config.comet_project_name
    )

    experiment_name: str = get_full_model_name(scenario=scenario, model_name=model_name, tuned=tune) 
    experiment.set_name(name=experiment_name)

    if not tune:
        logger.info("Using the default hyperparameters")

        if model_name == "base":
            pipeline = make_pipeline( model_fn(scenario=scenario) )
        else:
            if isinstance(model_fn, XGBRegressor):
                pipeline = make_pipeline(model_fn)
            else:
                pipeline = make_pipeline( model_fn() )

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

    save_model_locally(scenario=scenario, tuned=tune, model_fn=pipeline, model_name=model_name)
    experiment.log_metric(name="Test MAE", value=test_error)
    experiment.end()
    
    return test_error


def register_model(scenario: str, model_name: str, status: str, version: str = "1.0.0"):

    assert status.lower() in ["staging", "production"], 'The status must be either "staging" or "production"'
    logger.info(f"The best performing model for {scenario} is {model_name} -> Pushing it to the CometML model registry")
    push_model(scenario=scenario, model_name=model_name, status=status.title(), version=version)


def save_model_locally(scenario, model_fn: Pipeline, model_name: str, tuned: bool):
    """
    Save the trained model locally as a .pkl file

    Args:
        model_fn (Pipeline): the model object to be stored
        model_name (str): the name of the model to be saved
    """
    logger.success("Saving model to disk")

    model_file_name = f"{model_name.title()} ( {"Tuned" if tuned else "Untuned"} for {scenario}s ).pkl"
    with open(LOCAL_SAVE_DIR/model_file_name, mode="wb") as file:
        pickle.dump(obj=model_fn, file=file)


def train_all_models(tuning_trials: int = 10):
    """
    Train the named models, identify the best performer (on the test data) and
    register it to the CometML model registry.

    Args:
        tuning_trials: the number of tuning trials 
    """
    make_fundamental_paths()  # Ensure that all the necessary directories exist.
    delete_prior_project_from_comet() 

    for scenario in ["start", "end"]:
        models_and_errors: dict[tuple[str, str], float] = {}
        delete_best_model_from_previous_run(scenario=scenario)

        for tune_or_not in [False, True]:
            for model_name in config.model_names:
                error = train(scenario=scenario, model_name=model_name, tune=tune_or_not, tuning_trials=tuning_trials)
                tuning_indicator: str = "untuned" if not tune_or_not else "tuned"
                models_and_errors[ (model_name, tuning_indicator) ] = error

        best_model_name: str = identify_best_model(scenario=scenario, models_and_errors=models_and_errors)
        register_model(scenario=scenario, model_name=best_model_name, status="production")


if __name__ == "__main__":
    train_all_models()

