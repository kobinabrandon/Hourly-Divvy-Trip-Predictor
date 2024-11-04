"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import time
import boto3
import sagemaker 
import pandas as pd

from tqdm import tqdm
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
        self.feature_group = FeatureGroup(name=self.feature_group_name, sagemaker_session=self.session)

        if self.for_predictions:
            tuned_or_not = "tuned" if scenario == "end" else "untuned"
            self.description = f"{config.displayed_scenario_names[scenario]} predicted by the {self.model_name} model ({tuned_or_not})"
        else:
            self.description = f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}"
    
    def create_feature_group(self, data: pd.DataFrame):

        self.feature_group.load_feature_definitions(data_frame=data)

        self.feature_group.create(
            s3_uri=f"s3://{self.session.default_bucket()}/divvy_features",
            enable_online_store=True,
            record_identifier_name=f"{self.scenario}_station_id",
            event_time_feature_name="timestamp",
            description=self.description,
            role_arn=config.aws_arn
        )

        return self.feature_group

    def describe_feature_group(self) -> FeatureGroup:
        return self.feature_group.describe() 

    def query_offline_store(self) -> pd.DataFrame:

        query = self.feature_group.athena_query()
        table_name = self.describe_feature_group()['OfflineStoreConfig']['DataCatalogConfig']['TableName']

        if self.for_predictions:
            query_string = f"""
                SELECT "{self.scenario}_hour", "{self.scenario}_station_id", "predicted_{self.scenario}s"
                FROM "sagemaker_featurestore"."{table_name}";
            """
        else:
            query_string = f"""
                SELECT "{self.scenario}_hour", "{self.scenario}_station_id", "trips"
                FROM "sagemaker_featurestore"."{table_name}";
            """

        query.run(query_string=query_string, output_location=f"s3://{self.session.default_bucket()}/'divvy_features'")
        query.wait()

        return query.as_dataframe()

    def split_data_for_pushing(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:

        data_per_station = {}
        for station_id in tqdm(
            iterable=data[f"{self.scenario}_station_id"].unique(),
            desc="Splitting data up by the station"
        ):
            data_per_station[str(station_id)] = data[data[f"{self.scenario}_station_id"] == station_id] 

        return data_per_station

    @staticmethod
    def feature_group_ready(feature_group: FeatureGroup) -> bool:
        status = feature_group.describe()["FeatureGroupStatus"]
        while status == "Creating":
            logger.warning("Waiting for feature group to be created")
            time.sleep(5)
            status = feature_group.describe()["FeatureGroupStatus"]
        else:
            logger.success("Feature group ready for ingestion")
            return True

    def push(self, data: pd.DataFrame) -> None:

        data[f"timestamp"] = data[f"{self.scenario}_hour"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # To conform to AWS' requirements
        data[f"{self.scenario}_hour"] = data[f"{self.scenario}_hour"].astype(str)

        try:
            logger.warning(f"Trying to create feature group for {config.displayed_scenario_names[self.scenario].lower()}")
            feature_group = self.create_feature_group(data=data)
        except Exception as error:
            logger.error(error)
            logger.warning("Failed to create feature group. Attempting to fetch it")
            feature_group = self.feature_group
        
        data_to_push = self.split_data_for_pushing(data=data)
    
        if self.feature_group_ready(feature_group=feature_group):           
            for station_id, station_data in tqdm(
                iterable=data_to_push.items(),
                desc=logger.info(f"Pushing {config.displayed_scenario_names[self.scenario][:-1].lower()} data to the feature store")
            ):  
                feature_group.ingest(data_frame=station_data, max_workers=20)   
    