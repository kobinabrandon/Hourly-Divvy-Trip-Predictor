"""
The module starts with a function that provide a set of values of each of the hyperparameters 
that we consider with respect to each of the candidate model architectures. It also contains 
a function that provides hyperparameter tuning during model training.

"""
import numpy as np
import pandas as pd

from loguru import logger
from comet_ml import Experiment

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from src.training_pipeline.models import BaseModel


def sample_hyperparameters(
        model_fn: BaseModel | Lasso | LGBMRegressor | XGBRegressor,
        trial: optuna.trial.Trial
) -> dict[str, str | int | float]:
    """
    Define a range of values of the hyperparameters which we will be looking to optimise. 

    Returns:
        dict: the range of hyperparameter values to be considered
    """
    if model_fn == Lasso:
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 2.0, log=True)
        }

    elif model_fn == LGBMRegressor:
        return {
            "metric": "mean_absolute_error",
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 1),
            "importance_type": trial.suggest_categorical("importance_type", ["split", "gain"]),
            "subsample": trial.suggest_int("subsample", 0.1, 1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1)
        }

    elif model_fn == XGBRegressor:
        return {
            "objective": "reg:absoluteerror",
            "eta": trial.suggest_float("eta", 0.1, 1),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "alpha": trial.suggest_float("alpha", 0, 2),
            "subsample": trial.suggest_int("subsample", 0.1, 1)
        }

    else:
        raise NotImplementedError("This model has not been implemented")


def tune_hyperparameters(
        model_fn: BaseModel | Lasso | LGBMRegressor | XGBRegressor,
        tuning_trials: int,
        experiment: Experiment,
        x: pd.DataFrame,
        y: pd.Series
) -> dict:
    """
    Take a sample of values for each hyperparameter, and define an objective function which is to be
    optimised in an attempt to approximate the minimal MAE (within the set of hyperparameters sampled).

    Args:
        model_fn: the model architecture to be used
        tuning_trials: the number of optuna trials that will be run per mode scenario
        experiment: the CometML experiment object
        x: the dataframe of features
        y: the pandas series which contains the target variable

    Returns:
        dict: the optimal hyperparameters
    """
    models_and_tags: dict[callable, str] = {
        Lasso: "lasso",
        LGBMRegressor: "lightgbm",
        XGBRegressor: "xgboost",
        BaseModel: "base"
    }
    assert model_fn in models_and_tags.keys()
    model_name = models_and_tags[model_fn]

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Perform Time series cross validation, fit a pipeline to it (equipped with the)
        selected model, and return the average error across all cross validation splits.

        Args:
            trial: The optuna trial that's being optimised.

        Returns:
            float: Average error per split.
        """
        error_scores = []
        hyperparameters = sample_hyperparameters(model_fn=model_fn, trial=trial)
        tss = TimeSeriesSplit(n_splits=5)
        pipeline = make_pipeline(model_fn(**hyperparameters))

        logger.warning(f"Starting Trial {trial.number}")
        # Use TSS to split the features and target variables for training and validation
        for split_number, (train_indices, val_indices) in enumerate(tss.split(x)):
            logger.info(f"Performing split number {split_number}")
            x_train, x_val = x.iloc[train_indices], x.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            logger.info("Fitting model...")
            pipeline.fit(x_train, y_train)

            logger.info("Evaluating the performance of the trial...")
            y_pred = pipeline.predict(X=x_val)
            error = mean_absolute_error(y_true=y_val, y_pred=y_pred)
            error_scores.append(error)
            logger.info(f"MAE = {error}")

        avg_score = np.mean(error_scores)
        return avg_score

    logger.info("Beginning hyperparameter search")
    sampler = TPESampler(seed=69)
    study = optuna.create_study(study_name="study", direction="minimize", sampler=sampler, pruner=MedianPruner())
    study.optimize(func=objective, n_trials=tuning_trials)

    # Get the dictionary of the best hyperparameters and the error that they produce
    best_hyperparams = study.best_params
    best_value = study.best_value

    experiment.log_parameters(best_hyperparams)
    experiment.log_metric(name="Best MAE Across Trials", value=best_value)

    logger.info(f"The best hyperparameters for the {model_name} model are: {best_hyperparams}")
    logger.success(f"Best MAE Across Trials: {best_value}")

    return best_hyperparams

