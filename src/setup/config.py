import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, UTC
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR


env_file_path = PARENT_DIR.joinpath(".env") 
_ = load_dotenv(env_file_path) 


class GeneralConfig(BaseSettings):

    _ = SettingsConfigDict(
        env_file=str(env_file_path),
        env_file_encoding="utf-8", 
        extra="allow"
    )

    email: str
    n_features: int = 672

    # The number of months in the immediate past for which we will retrieve data 
    offset: int = 6 
    tuning_trials: int = 5

    # Hopsworks
    backfill_days: int = 210 
    feature_group_version: int = 1
    feature_view_version: int = 1

    model_base_names: list[str] = ["lasso", "lightgbm", "xgboost"]
    current_hour: datetime = pd.to_datetime(datetime.now(tz=UTC)).floor("h")
    displayed_scenario_names: dict[str, str] = {"start": "Departures", "end": "Arrivals"} 

    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    hopsworks_api_key: str
    hopsworks_project_name: str

    database_public_url: str


config = GeneralConfig()


def get_proper_scenario_name(scenario: str) -> str:
    return config.displayed_scenario_names[scenario].lower()

