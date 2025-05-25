"""
This module contains all the code that allows interaction with CometML's model registry.
"""
from pathlib import Path

from loguru import logger
from sklearn.pipeline import Pipeline
from comet_ml import ExistingExperiment, get_global_experiment, API

from src.setup.config import config
from src.setup.paths import COMET_SAVE_DIR, LOCAL_SAVE_DIR, make_fundamental_paths
from src.training_pipeline.models import load_local_model


class ModelRegistry:
    def __init__(self, scenario: str, model_name: str, tuned_or_not: str) -> None:
        self.scenario = scenario
        self.model_name = model_name
        self.tuned_or_not = tuned_or_not
        self.registered_name = self._set_registered_name()

    def _set_registered_name(self) -> str:
        return f"{self.model_name.title()} ({self.tuned_or_not.title()} for {self.scenario}s)"

    def get_registered_model_version(self) -> str:
        api = API(api_key=config.comet_api_key)

        model_details: dict[str | Any] | None = api.get_registry_model_details(
            workspace=config.comet_workspace, 
            registry_name=self.registered_name
        )
        
        # This particular choice resulted from an inspection of the model details object
        model_versions = model_details["versions"][0]["version"]
        return model_versions

    def push_model(self, status: str, version: str) -> None:
        """
        Find the model (saved locally), log it to CometML, and register it at the model registry.

        Args:
            status: the status that we want to give to the model during registration.
            version: the version of the model being pushed

        Returns:
            None
        """
        running_experiment = get_global_experiment()
        experiment = ExistingExperiment(api_key=running_experiment.api_key, experiment_key=running_experiment.id)

        logger.info("Logging model to Comet ML")
        registered_name = self._set_registered_name()

        model_file = LOCAL_SAVE_DIR/f"{registered_name}.pkl"
        _ = experiment.log_model(name=registered_name, file_or_folder=str(model_file))
        logger.success(f"Finished logging the {self.model_name} model")

        logger.info(f'Pushing version {version} of the model to the registry under "{status.title()}"...')
        experiment.register_model(model_name=registered_name, status=status, version=version)

    def download_latest_model(self, unzip: bool) -> Pipeline:
        """
        Download the latest version of the requested model to the MODEL_DIR directory,
        load the file using pickle, and return it.

        Args:
            unzip: whether to unzip the downloaded zipfile.

        Returns:
            Pipeline: the original model file
        """
        make_fundamental_paths()
        if not Path(COMET_SAVE_DIR/f"{self.registered_name}.pkl").exists():

            api = API(api_key=config.comet_api_key)

            api.download_registry_model(
                workspace=config.comet_workspace,   
                registry_name=self.registered_name,
                version=self.get_registered_model_version(),
                output_path=COMET_SAVE_DIR,
                expand=unzip
            )

        model: Pipeline = load_local_model(
            directory=COMET_SAVE_DIR,
            model_name=self.model_name,
            scenario=self.scenario,
            tuned_or_not=self.tuned_or_not
        )
        
        return model
