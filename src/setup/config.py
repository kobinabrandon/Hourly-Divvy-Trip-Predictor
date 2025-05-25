import pandas as pd 

from datetime import datetime, UTC
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.setup.paths import PARENT_DIR
from src.feature_pipeline.data_sourcing import Year


env_file_path = PARENT_DIR.joinpath(".env") 
_ = load_dotenv(env_file_path) 


class GeneralConfig(BaseSettings):

    _ = SettingsConfigDict(
        env_file=str(env_file_path),
        env_file_encoding="utf-8", 
        extra="allow"
    )

    years: list[Year] = [
        Year(value=2025, offset=0)
    ]

    email: str
    n_features: int = 672

    # Hopsworks
    backfill_days: int = 210 
    feature_group_version: int = 1
    feature_view_version: int = 1

    current_hour: datetime = pd.to_datetime(datetime.now(tz=UTC)).floor("H")
    displayed_scenario_names: dict[str, str] = {"start": "Departures", "end": "Arrivals"} 

    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    hopsworks_api_key: str
    hopsworks_project_name: str

    database_public_url: str


config = GeneralConfig()

