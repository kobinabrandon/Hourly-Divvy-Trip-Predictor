"""
This module contains all the code that allows interaction with CometML's model registry.
"""
from typing import Any
from pathlib import Path

from loguru import logger
from numpy import full
from sklearn.pipeline import Pipeline
from comet_ml import ExistingExperiment, get_global_experiment, API

from src.setup.config import config
from src.training_pipeline.models import load_local_model, get_full_model_name
from src.setup.paths import COMET_SAVE_DIR, LOCAL_SAVE_DIR, make_fundamental_paths


def push_model(scenario: str, model_name: str, status: str, version: str) -> None:
    """
    Find the model (saved locally), log it to CometML, and register it at the model registry.

    Args:
        scenario: 
        model_name: 
        status: the status that we want to give to the model during registration.
        version: the version of the model being pushed

    Returns:
        None
    """
    running_experiment = get_global_experiment()
    experiment = ExistingExperiment(api_key=running_experiment.api_key, experiment_key=running_experiment.id)

    logger.info("Logging model to Comet ML")
    tuned: bool = "_tuned" in model_name

    # Remove the suffix "_tuned" or "_untuned" that was added when we identified the best model 
    corrected_model_name: str = model_name.replace("_tuned" if tuned else "_untuned", "")  

    full_model_name = get_full_model_name(
        scenario=scenario, 
        model_name=corrected_model_name,
        tuned=tuned
    )

    model_file_name: Path = LOCAL_SAVE_DIR.joinpath(f"{full_model_name}.pkl")
    _ = experiment.log_model(name=full_model_name, file_or_folder=str(model_file_name))
    logger.success(f"Finished logging the {model_name} model")

    logger.info(f'Pushing version {version} of the model to the registry under "{status.title()}"...')
    _ = experiment.register_model(model_name=full_model_name, status=status, version=version)


def download_model(scenario: str, unzip: bool, tuned: bool, model_name: str) -> Pipeline:
    """
    Download the latest version of the requested model to the MODEL_DIR directory,
    load the file using pickle, and return it.

    Args:
        tuned: 
        model_name: 
        unzip: whether to unzip the downloaded zipfile.

    Returns:
        Pipeline: the original model file
    """
    make_fundamental_paths()
    full_model_name = get_full_model_name(
        scenario=scenario, 
        model_name=model_name, 
        tuned="tuned" if tuned else "untuned"
    )

    save_path: Path = COMET_SAVE_DIR.joinpath(f"{full_model_name}.pkl")
    registered_model_version = get_registered_model_version(full_model_name=full_model_name)

    if not save_path.exists():

        api = API(api_key=config.comet_api_key)

        api.download_registry_model(
            workspace=config.comet_workspace,   
            registry_name=full_model_name,
            version=registered_model_version,
            output_path=str(COMET_SAVE_DIR),
            expand=unzip
        )

    model: Pipeline = load_local_model(
        directory=COMET_SAVE_DIR,
        model_name=model_name,
        scenario=scenario,
        tuned_or_not="tuned" if tuned else "untuned"
    )
    
    return model



def get_registered_model_version(full_model_name: str) -> str:
    api = API(api_key=config.comet_api_key)

    model_details: dict[str | Any] | None = api.get_registry_model_details(
        workspace=config.comet_workspace, 
        registry_name=full_model_name
    )
    
    # This particular choice resulted from an inspection of the model details object
    model_versions = model_details["versions"][0]["version"]
    return model_versions

