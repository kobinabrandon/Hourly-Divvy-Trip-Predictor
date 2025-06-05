from comet_ml import API
from loguru import logger

from src.setup.config import config
from src.setup.paths import MODELS_DIR


def gather_best_model_per_scenario(scenario: str, models_and_errors: dict[str, float]) -> dict[str, str]:

    best_model = {} 
    for model_name in models_and_errors.keys():
        smallest_test_error = min(models_and_errors.values())
        if models_and_errors[model_name] == smallest_test_error:
            best_model[scenario] = model_name

    with open(MODELS_DIR.joinpath(f"best_{scenario}_model.txt"), mode="w") as file:
        file.writelines(model_name)

    return best_model


def delete_best_model_from_previous_run(scenario: str):

    try:
        api = API(api_key=config.comet_api_key)
        name_of_best_model_from_past_run: str = retrieve_best_model_from_previous_run(scenario=scenario) 
        api.delete_registry_model(workspace=config.comet_workspace, registry_name=name_of_best_model_from_past_run)
    except Exception as error:
        logger.error(error)


def retrieve_best_model_from_previous_run(scenario: str) -> str:

    with open(MODELS_DIR.joinpath(f"best_{scenario}_model.txt"), mode="r") as file:
        model_name = file.read()

    return model_name 

