from pathlib import Path
from comet_ml import API
from loguru import logger

from src.setup.config import config
from src.setup.paths import MODELS_DIR
from src.inference_pipeline.backend.model_registry import get_full_model_name


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


def identify_best_model(scenario: str, models_and_errors: dict[str, float]) -> str: 

    best_model_name = "" 
    smallest_test_error = min(models_and_errors.values())

    for (model_name, tuned_or_not) in models_and_errors.keys():
        # Stop the loop if a model with the minimum error has been identified. This prevents the 
        # string from being concatenated when there are two models with the same error 

        if len(best_model_name) > 0:  
            break
        elif models_and_errors[ (model_name, tuned_or_not) ] == smallest_test_error:
            best_model_name += model_name + f"_{tuned_or_not}" 

    if len(best_model_name) == 0:
        raise Exception(f"Unable to identify model with best performance for {scenario}s. Did any models train in the first place?")
            
    path_to_log_of_best_model_name: Path = MODELS_DIR.joinpath(f"best_{scenario}_model.txt")
    with open(path_to_log_of_best_model_name, mode="w") as file:
        file.writelines(best_model_name)

    return best_model_name


def delete_best_model_from_previous_run(scenario: str):

    api = API(api_key=config.comet_api_key)
    name_of_best_model_from_past_run: str = retrieve_best_model_from_previous_run(scenario=scenario) 

    for tuned_or_not in [True, False]:
        full_model_name = get_full_model_name(scenario=scenario, model_name=name_of_best_model_from_past_run, tuned=tuned_or_not)

        try:
            api.delete_registry_model(workspace=config.comet_workspace, registry_name=full_model_name)

        except Exception as error:
            logger.error(error)


def retrieve_best_model_from_previous_run(scenario: str) -> str:

    with open(MODELS_DIR.joinpath(f"best_{scenario}_model.txt"), mode="r") as file:
        model_name = file.read()

    return model_name 

