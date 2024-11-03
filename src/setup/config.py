import pandas as pd 

from datetime import datetime, UTC
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv(".env")


class GeneralConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # Names 
    year: int = 2024
    n_features: int = 672
    email: str

    # CometML
    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    # AWS
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_default_region: str
    aws_arn: str

    backfill_days: int = 120
    current_hour: datetime = pd.to_datetime(datetime.now(tz=UTC)).floor("H")
    displayed_scenario_names: dict = {"start": "Departures", "end": "Arrivals"} 


config = GeneralConfig()
