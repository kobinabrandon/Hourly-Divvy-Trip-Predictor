"""
This module contains the code that cleans up project data from Comet's experiment tracker, and 
and model data from its model registry. 
"""
import os
from pathlib import Path
from comet_ml import API
from loguru import logger

from src.setup.config import config
from src.setup.paths import LOCAL_SAVE_DIR, MODELS_DIR
from src.training_pipeline.models import get_full_model_name


def delete_prior_project_from_comet(delete_experiments: bool = True):
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


def identify_best_model(scenario: str, models_and_errors: dict[tuple[str, str], float]) -> str: 
    """
    Go through all the models that have been trained on the named scenario's data, find the best 
    performer, and return its full name

    Args:
        scenario: "start" or "end" 
        models_and_errors: a dictionary with model names as keys and the associated errors as values 

    Returns:
        

    Raises:
        Exception: 
    """
    best_model_name = "" 
    smallest_test_error: float = min(models_and_errors.values())
    path_to_log_of_best_model_name: Path = MODELS_DIR.joinpath(f"best_{scenario}_model.txt")

    for (model_name, tuned_string) in models_and_errors.keys():
        # Stop the loop if a model with the minimum error has been identified. This prevents 
        # the string from being concatenated when there are two models with the same error 

        tuned_bool = False if "untuned" in tuned_string else True
        if len(best_model_name) == 0 and models_and_errors[ (model_name, tuned_string) ] == smallest_test_error:
            best_model_name += get_full_model_name(scenario=scenario, base_name=model_name, tuned=tuned_bool) 

    if len(best_model_name) == 0:
        raise Exception(
            f"""Unable to identify model with best performance for 
            {scenario}s. Did any models train in the first place?"""
        )
            
    with open(path_to_log_of_best_model_name, mode="w") as file:
        file.writelines(best_model_name)

    logger.success(f"Saved {best_model_name} as a pickle file")
    return best_model_name


def delete_best_model_from_previous_run(scenario: str):
    """
    The function attempts to find tuned and untuned versions of the  

    Args:
        scenario: "start" or "end" 
    """
    api = API(api_key=config.comet_api_key)
    name_of_best_model_from_past_run: str| None = retrieve_name_of_best_model_from_previous_run(scenario=scenario) 

    if name_of_best_model_from_past_run is None:
        logger.error("Failed to discover the best model from the previous run")
    else:
        try:
            api.delete_registry_model(workspace=config.comet_workspace, registry_name=name_of_best_model_from_past_run)

        except Exception as error:
            logger.error(error)


def retrieve_name_of_best_model_from_previous_run(scenario: str) -> str | None:
    """
    During the training process, I saved a .txt file which contained the name of the best model. This string followed
    the format "{model_name}_{tuned_or_not}"

    Args:
        scenario: "start" or "end"

    Returns:
        Either the name of the model or None
    """
    path_to_txt_containing_best_model_name = MODELS_DIR.joinpath(f"best_{scenario}_model.txt")

    if path_to_txt_containing_best_model_name.exists():
        with open(path_to_txt_containing_best_model_name, mode="r") as file:
            model_name = file.read()

        return model_name 
    else:
        None


def delete_local_saves():
    logger.warning("Deleting locally saved models")
    for file in os.listdir(LOCAL_SAVE_DIR):
        os.remove(LOCAL_SAVE_DIR.joinpath(file))

