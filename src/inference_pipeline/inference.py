"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""


import numpy as np
import pandas as pd

from loguru import logger
from argparse import ArgumentParser

from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from sklearn.pipeline import Pipeline

from src.setup.config import FeatureGroupConfig, config

from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import perform_feature_engineering
from src.inference_pipeline.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.model_registry_api import ModelRegistry


class InferenceModule:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.n_features = config.n_features

        self.feature_store_api = FeatureStoreAPI(
            scenario=self.scenario,
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            primary_key=[f"{self.scenario}_station_id"],
            event_time="timestamp"
        )

        self.feature_group_metadata = FeatureGroupConfig(
            name=f"{scenario}_feature_group",
            version=config.feature_group_version,
            primary_key=self.feature_store_api.primary_key,
            event_time=self.feature_store_api.event_time
        )

        self.feature_group: FeatureGroup = self.feature_store_api.get_or_create_feature_group(
            description=f"Hourly time series data showing when trips {self.scenario}s",
            version=self.feature_group_metadata.version,
            name=self.feature_group_metadata.name
        )

    def fetch_time_series_and_make_features(self, target_date: datetime, geocode: bool) -> pd.DataFrame:
        """
        Queries the offline feature store for time series data within a certain timeframe, and creates features
        features from that data. We then apply feature engineering so that the data aligns with the features from
        the original training data.

        Args:
            target_date: the date for which we seek predictions.
            geocode: whether to implement geocoding during feature engineering

        Returns:
            pd.DataFrame:
        """
        fetch_data_from = target_date - timedelta(days=28)
        fetch_data_to = target_date - timedelta(hours=1)

        feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view(
            name=f"{self.scenario}_feature_view",
            feature_group=self.feature_group,
            version=1
        )

        # Following Pau's convention of including additional days to avoid losing valuable observation
        ts_data: pd.DataFrame = feature_view.get_batch_data(
            start_time=fetch_data_from - timedelta(days=1),
            end_time=fetch_data_to + timedelta(days=1)
        )

        ts_data = ts_data.sort_values(
            by=[f"{self.scenario}_station_id", f"{self.scenario}_hour"]
        )

        station_ids = ts_data[f"{self.scenario}_station_id"].unique()

        # assert len(ts_data) == config.n_features * len(station_ids), \
        #   "The time series data is incomplete on the feature store. Please review the feature pipeline."

        features = self.make_features(station_ids=station_ids, ts_data=ts_data, geocode=False)

        # Include the {self.scenario}_hour column and the IDs
        features[f"{self.scenario}_hour"] = target_date
        features[f"{self.scenario}_station_id"] = station_ids

        return features.sort_values(
            by=[f"{self.scenario}_station_id"]
        )

    def make_features(self, station_ids: list[int], ts_data: pd.DataFrame, geocode: bool) -> pd.DataFrame:
        """
        Restructure the time series data into features in a way that aligns with the features 
        of the original training data.

        Args:
            station_ids: the list of unique station IDs
            ts_data: the time series data that is store on the feature store

        Returns:
            pd.DataFrame: the dataframe consisting of the features
        """
        processor = DataProcessor(year=config.year, bypass=True)

        # Perform transformation of the time series data with feature engineering
        data = processor.transform_ts_into_training_data(
            ts_data=ts_data,
            geocode=geocode, 
            scenario=self.scenario, 
            step_size=24,
            input_seq_len=24 * 28 * 1
        )

        return data.drop("trips_next_hour", axis = 1)

    def load_predictions_from_store(
            self,
            model_name: str,
            from_hour: datetime,
            to_hour: datetime
    ) -> pd.DataFrame:
        """
        Load predictions from their dedicated feature group in the offline feature store.

        Args:
            model_name: the model's name is part of the name of the feature view to be queried
            from_hour: the first hour for which we want the predictions
            to_hour: the last hour for would like to receive predictions.

        Returns:
            pd.DataFrame: the dataframe containing predictions.
        """
        predictions_feature_view: FeatureView = self.feature_store_api.get_or_create_feature_view(
            name=f"{model_name}_predictions_from_feature_store",
            version=1,
            feature_group=self.feature_group
        )

        logger.info(f'Fetching predictions for "{self.scenario}_hours" between {from_hour} and {to_hour}')
        predictions = predictions_feature_view.get_batch_data(start_time=from_hour, end_time=to_hour)

        predictions[f"{self.scenario}_hour"] = pd.to_datetime(predictions[f"{self.scenario}_hour"], utc=True)
        from_hour = pd.to_datetime(from_hour, utc=True)
        to_hour = pd.to_datetime(to_hour, utc=True)

        predictions = predictions[
            predictions[f"{self.scenario}_hour"].between(from_hour, to_hour)
        ]

        predictions = predictions.sort_values(
            by=[f"{self.scenario}_hour", f"{self.scenario}_station_id"]
        )

        return predictions

    def get_model_predictions(self, model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
        """
        Simply use the model's predict method to provide predictions based on the supplied features

        Args:
            model: the model object fetched from the model registry
            features: the features obtained from the feature store

        Returns:
            pd.DataFrame: the model's predictions
        """
        predictions = model.predict(features)

        prediction_per_station = pd.DataFrame()
        prediction_per_station[f"{self.scenario}_station_id"] = features[f"{self.scenario}_station_id"].values
        prediction_per_station[f"predicted_{self.scenario}s"] = predictions.round(decimals=0)

        return prediction_per_station
