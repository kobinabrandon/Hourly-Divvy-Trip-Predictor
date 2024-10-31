"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import time
import boto3
import sagemaker 
import pandas as pd

from loguru import logger
from datetime import datetime

from sagemaker.feature_store.feature_group import FeatureGroup

from src.setup.config import config


class FeatureStoreAPI:
    def __init__(self, scenario: str, for_predictions: bool) -> None:
        self.scenario = scenario
        self.for_predictions = for_predictions

        self.session = sagemaker.Session()
        self.model_name = "lightgbm" if scenario == "end" else "xgboost"
        self.feature_group_name = f"{self.model_name}_{scenario}_predictions_feature_group" if for_predictions else f"{scenario}_ts"
        self.feature_group = FeatureGroup(name=self.feature_group_name)

        if self.for_predictions:
            self.description = f"predicting {config.displayed_scenario_names[scenario]} - {tuned_or_not} {model_name}"
        else:
            self.description = f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}"

    def create_feature_group(self, data: pd.DataFrame):

        data[f"{self.scenario}_hour"] = data[f"{self.scenario}_hour"].dt.strftime('%Y-%m-%d %H:%M:%S').astype(str)
        self.feature_group.load_feature_definitions(data_frame=data)

        self.feature_group.create(
            s3_uri=f"s3://{self.session.default_bucket()}/divvy_features",
            enable_online_store=True,
            record_identifier_name=f"{self.scenario}_station_id",
            event_time_feature_name=f"{self.scenario}_hour",
            description=self.description,
            role_arn=config.aws_arn
        )

        return self.feature_group

    def describe_feature_group(self) -> FeatureGroup:
        return self.feature_group.describe() 

    def query_offline_store(self, start_date: datetime, target_date: datetime):

        start_timestamp = int(start_date.timestamp())
        target_timestamp = int(target_date.timestamp())

        query = self.feature_group.athena_query()
        table_name = self.describe_feature_group()['OfflineStoreConfig']['DataCatalogConfig']['TableName']

        query_string = f"""
            SELECT "{self.scenario}_hour", "{self.scenario}_station_id", "trips"
            FROM "sagemaker_featurestore"."{table_name}"
        """

        query.run(query_string=query_string, output_location=f"s3://{self.session.default_bucket()}/'divvy_features'")
        query.wait()

        breakpoint()
        return query.as_dataframe()
