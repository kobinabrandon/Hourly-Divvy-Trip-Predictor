"""
Contains code for model training with and without hyperparameter tuning, as well as 
experiment tracking.
"""
import pickle
from pathlib import Path
from argparse import ArgumentParser

from numpy import delete
import pandas as pd
from loguru import logger

from xgboost import XGBRegressor
from comet_ml import Experiment, API
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

from src.setup.config import config
from src.setup.paths import TRAINING_DATA, LOCAL_SAVE_DIR, make_fundamental_paths
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.training_pipeline.models import get_model
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparameters


def gather_best_model_per_scenario(scenario: str, models_and_errors: dict[str, float] ) -> dict[str, str]:

    best_model = {} 
    for model_name in models_and_errors.keys():
        smallest_test_error = min(models_and_errors.values())
        if models_and_errors[model_name] == smallest_test_error:
            best_model[scenario] = model_name

    return best_model


class Trainer:
    def __init__(
        self,
        scenario: str,
        hyperparameter_trials: int,
        tune_hyperparameters: bool | None = True
    ):
        """
        Args:
            scenario (str): a string indicating whether we are training data on the starts or ends of trips.
                            The only accepted answers are "start" and "end"

            tune_hyperparameters (bool | None, optional): whether to tune hyperparameters or not.

            hyperparameter_trials (int | None): the number of times that we will try to optimize the hyperparameters
        """
        self.scenario: str = scenario
        self.tune_hyperparameters: bool | None = tune_hyperparameters
        self.hyperparameter_trials: int = hyperparameter_trials
        self.tuned_or_not: str = "Tuned" if self.tune_hyperparameters else "Untuned"
        make_fundamental_paths()  # Ensure that all the necessary directories exist.


    @staticmethod
    def delete_any_prior_project(delete_experiments: bool = True):
        try:
            api = API(api_key=config.comet_api_key)
            logger.info("Deleting COMET project...")

            _ = api.delete_project(
                workspace=config.comet_workspace, 
                project_name=config.comet_project_name, 
                delete_experiments=delete_experiments
            )

        except Exception as error:
            logger.error(error)

    def get_or_make_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fetches or builds the training data for the starts or ends of trips.

        Returns:
            pd.DataFrame: a tuple containing the training data's features and targets
        """
        assert self.scenario.lower() in ["start", "end"]
        data_path = Path.joinpath(TRAINING_DATA, f"{self.scenario}s.parquet")
        
        if Path(data_path).is_file():
            training_data = pd.read_parquet(path=data_path)
            logger.success(f"Fetched saved training data for {config.displayed_scenario_names[self.scenario].lower()}")
        else:
            logger.warning("No training data is storage. Creating the dataset will take a while.")

            processor = DataProcessor(years=config.years, for_inference=False)
            training_sets = processor.make_training_data(geocode=False)
            training_data = training_sets[0] if self.scenario.lower() == "start" else training_sets[1]
            logger.success("Training data produced successfully")

        target = training_data["trips_next_hour"]
        features = training_data.drop("trips_next_hour", axis=1)
        return features.sort_index(), target.sort_index()

    def train(self, model_name: str) -> float:
        """
        The function first checks for the existence of the training data, and builds it if
        it doesn't find it locally. Then it checks for a saved model. If it doesn't find a model,
        it will go on to build one, tune its hyperparameters, save the resulting model.

        Args:
            model_name (str): the name of the model to be trained

        Returns:
            float: the error of the chosen model on the test dataset.
        """
        model_fn: object = get_model(model_name=model_name)
        features, target = self.get_or_make_training_data()

        train_sample_size = int(0.9 * len(features))
        x_train, x_test = features[:train_sample_size], features[train_sample_size:]
        y_train, y_test = target[:train_sample_size], target[train_sample_size:]

        experiment = Experiment(
            api_key=config.comet_api_key,
            workspace=config.comet_workspace,
            project_name=config.comet_project_name
        )
        
        if not self.tune_hyperparameters:
            experiment.set_name(name=f"{model_name.title()}(Untuned) model for the {self.scenario}s of trips")
            logger.info("Using the default hyperparameters")

            if model_name == "base":
                pipeline = make_pipeline(
                    model_fn(scenario=self.scenario)
                )
            else:
                if isinstance(model_fn, XGBRegressor):
                    pipeline = make_pipeline(model_fn)
                else:
                    pipeline = make_pipeline(model_fn())
        else:
            experiment.set_name(name=f"{model_name.title()}(Tuned) model for the {self.scenario}s of trips")
            logger.info(f"Tuning hyperparameters of the {model_name} model.")

            best_model_hyperparameters = optimise_hyperparameters(
                model_fn=model_fn,
                hyperparameter_trials=self.hyperparameter_trials,
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

        self.save_model_locally(model_fn=pipeline, model_name=model_name)
        experiment.log_metric(name="Test M.A.E", value=test_error)
        experiment.end()
        
        return test_error

    def save_model_locally(self, model_fn: Pipeline, model_name: str):
        """
        Save the trained model locally as a .pkl file

        Args:
            model_fn (Pipeline): the model object to be stored
            model_name (str): the name of the model to be saved
        """
        model_file_name = f"{model_name.title()} ({self.tuned_or_not} for {self.scenario}s).pkl"
        with open(LOCAL_SAVE_DIR/model_file_name, mode="wb") as file:
            pickle.dump(obj=model_fn, file=file)

        logger.success("Saved model to disk")

    def register_model(self, model_name: str, version: str, status: str):

        assert status.lower() in ["staging", "production"], 'The status must be either "staging" or "production"'
        logger.info(f"The best performing model for {self.scenario} is {model_name} -> Pushing it to the CometML model registry")
        registry = ModelRegistry(model_name=model_name, scenario=self.scenario, tuned_or_not=self.tuned_or_not)
        registry.push_model(status=status.title(), version=version)


    def train_and_register_models(self, model_names: list[str]) -> dict[str, float]:
        """
        Train the named models, identify the best performer (on the test data) and
        register it to the CometML model registry.

        Args:
            model_names: the names of the models under consideration
            version: the version of the registered model on CometML.
            status:  the registered status of the model on CometML.
        """
        models_and_errors = {}
        for model_name in model_names:
            test_error = self.train(model_name=model_name)
            models_and_errors[model_name] = test_error

        return models_and_errors


if __name__ == "__main__":
    parser = ArgumentParser()
    _ = parser.add_argument("--scenario", type=str)
    _ = parser.add_argument("--models", type=str, nargs="+", required=True)
    _ = parser.add_argument("--tune_hyperparameters", action="store_true")
    _ = parser.add_argument("--hyperparameter_trials", type=int, default=15)
    args = parser.parse_args()

    version = "1.0.0"

    trainer = Trainer(
        scenario=args.scenario,
        tune_hyperparameters=args.tune_hyperparameters,
        hyperparameter_trials=args.hyperparameter_trials
    )

    trainer.delete_any_prior_project()
    models_and_errors = trainer.train_and_register_models(model_names=args.models)
    best_model_for_scenario: dict[str, str] = gather_best_model_per_scenario(scenario=args.scenario, models_and_errors=models_and_errors)

    best_model_name = best_model_for_scenario[args.scenario]
    trainer.register_model(model_name=best_model_name, version=version, status="production")
    # api.delete_registry_model(workspace=config.comet_workspace, registry_name=)

