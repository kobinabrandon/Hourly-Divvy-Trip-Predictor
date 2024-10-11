"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""
import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from loguru import logger
from argparse import ArgumentParser

from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from sklearn.pipeline import Pipeline

from src.setup.config import FeatureGroupConfig, config
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER

from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import finish_feature_engineering
from src.inference_pipeline.backend.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.backend.model_registry_api import ModelRegistry


class InferenceModule:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.n_features = config.n_features          

        self.api = FeatureStoreAPI(
            scenario=self.scenario,
            event_time="timestamp",
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            primary_key=[f"{self.scenario}_station_id", f"{self.scenario}_hour"],
        )

        self.feature_group_metadata = FeatureGroupConfig(
            name=f"{scenario}_feature_group",
            version=config.feature_group_version,
            primary_key=self.api.primary_key,
            event_time=self.api.event_time
        )

        self.feature_group: FeatureGroup = self.api.setup_feature_group(
            description=f"Hourly time series data showing when trips {self.scenario}s",
            version=self.feature_group_metadata.version,
            name=self.feature_group_metadata.name,
            for_predictions=False
        )

    def fetch_time_series_and_make_features(self, start_date: datetime, target_date: datetime, geocode: bool) -> pd.DataFrame:
        """
        Queries the offline feature store for time series data within a certain timeframe, and creates features
        features from that data. We then apply feature engineering so that the data aligns with the features from
        the original training data.

        My initial intent was to fetch time series data the 28 days prior to the target date. However, the class
        method that I am using to convert said data into features requires a larger dataset to work (see the while 
        loop in the get_cutoff_indices method from the preprocessing module). So after some experimentation, I 
        decided to go with 168 days of prior time series data. I will look to play around this number in the future.

        Args:
            target_date: the date for which we seek predictions.
            geocode: whether to implement geocoding during feature engineering
            for_plotting (bool): whether we are producing these features purely for the purpose of plotting historical data.

        Returns:
            pd.DataFrame: time series data 
        """ 
        feature_view: FeatureView = self.api.get_or_create_feature_view(
            name=f"{self.scenario}_feature_view",
            feature_group=self.feature_group,
            version=1   
        )

        logger.warning("Fetching time series data from the offline feature store...")
        ts_data: pd.DataFrame = feature_view.get_batch_data(
            start_time=start_date, 
            end_time=target_date,
            read_options={"use_hive": True}
        )

        ts_data = ts_data.sort_values(
            by=[f"{self.scenario}_station_id", f"{self.scenario}_hour"]
        )

        station_ids = ts_data[f"{self.scenario}_station_id"].unique()
        features = self.make_features(station_ids=station_ids, ts_data=ts_data, geocode=geocode)

        features[f"{self.scenario}_hour"] = target_date
        features = features.sort_values(by=[f"{self.scenario}_station_id"])

        return features


    def make_features(self, station_ids: list[int], ts_data: pd.DataFrame, geocode: bool) -> pd.DataFrame:
        """
        Restructure the time series data into features in a way that aligns with the features 
        of the original training data.

        Args:
            station_ids: the list of unique station IDs.
            ts_data: the time series data that is store on the feature store.

        Returns:
            pd.DataFrame: time series data
        """
        processor = DataProcessor(year=config.year, for_inference=True)

        # Perform transformation of the time series data with feature engineering
        return processor.transform_ts_into_training_data(
            ts_data=ts_data,
            geocode=geocode,
            scenario=self.scenario, 
            input_seq_len=config.n_features,
            step_size=24
        )


    def fetch_predictions_group(self, model_name: str) -> FeatureGroup:
        """
        Return the feature group used for predictions.

        Args:
            model_name (str): the name of the model

        Returns:
            FeatureGroup: the feature group for the given model's predictions.
        """
        if model_name == "lightgbm":
            tuned_or_not = "tuned"
        elif model_name == "xgboost":
            tuned_or_not = "untuned"

        return self.api.setup_feature_group(
            description=f"predictions on {self.scenario} data using the {tuned_or_not} {model_name}",
            name=f"{model_name}_{self.scenario}_predictions_feature_group",
            version=config.feature_group_version,
            for_predictions=True
        )


    def load_predictions_from_store(self, from_hour: datetime, to_hour: datetime, model_name: str) -> pd.DataFrame:
        """
        Load a dataframe containing predictions from their dedicated feature group on the offline feature store.
        This dataframe will contain predicted values between the specified hours. 

        Args:
            model_name: the model's name is part of the name of the feature view to be queried
            from_hour: the first hour for which we want the predictions
            to_hour: the last hour for would like to receive predictions.

        Returns:
            pd.DataFrame: the dataframe containing predictions.
        """
        # Ensure these times are datatimes
        from_hour = pd.to_datetime(from_hour, utc=True)
        to_hour = pd.to_datetime(to_hour, utc=True)
            
        predictions_group = self.fetch_predictions_group(model_name=model_name)

        predictions_feature_view: FeatureView = self.api.get_or_create_feature_view(
            name=f"{model_name}_{self.scenario}_predictions",
            feature_group=predictions_group,
            version=1
        )

        logger.info(
            f"Fetching predicted {config.displayed_scenario_names[self.scenario].lower()} between {from_hour.hour}:00 and {to_hour.hour}:00"
        )

        predictions_df = predictions_feature_view.get_batch_data(
            start_time=from_hour - timedelta(days=1), 
            end_time=to_hour + timedelta(days=1)
        )

        predictions_df[f"{self.scenario}_hour"] = pd.to_datetime(predictions_df[f"{self.scenario}_hour"], utc=True)

        return predictions_df.sort_values(
            by=[f"{self.scenario}_hour", f"{self.scenario}_station_id"]
        )

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
        prediction_per_station[f"{self.scenario}_hour"] = pd.to_datetime(datetime.utcnow()).floor("H")
        prediction_per_station[f"predicted_{self.scenario}s"] = predictions.round(decimals=0)
        
        return prediction_per_station


def rerun_feature_pipeline():
    """
    This is a decorator that provides logic which allows the wrapped function to be run if a certain exception 
    is not raised, and the full feature pipeline if the exception is raised. Generally, the functions that will 
    use this will depend on the loading of some file that was generated during the preprocessing phase of the 
    feature pipeline. Running the feature pipeline will allow for the file in question to be generated if isn't 
    present, and then run the wrapped function afterwards.
    """
    def decorator(fn: callable):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except FileNotFoundError as error:
                logger.error(error)
                message = "The JSON file containing station details is missing. Running feature pipeline again..."
                logger.warning(message)
                st.spinner(message)

                processor = DataProcessor(year=config.year, for_inference=False)
                processor.make_training_data(geocode=False)
                return fn(*args, **kwargs)
        return wrapper
    return decorator


@rerun_feature_pipeline()
def load_raw_local_geodata(scenario: str) -> list[dict]:
    """
    Load the json file that contains the geographical information for 
    each station.

    Args:
        scenario (str): "start" or "end" 

    Raises:
        FileNotFoundError: raised when said json file cannot be found. In that case, 
        the feature pipeline will be re-run. As part of this, the file will be created,
        and the function will then load the generated data.

    Returns:
        list[dict]: the loaded json file as a dictionary
    """
    if len(os.listdir(ROUNDING_INDEXER)) != 0:
        geodata_path = ROUNDING_INDEXER / f"{scenario}_geodata.json"
    elif len(os.listdir(MIXED_INDEXER)) != 0:
        geodata_path = MIXED_INDEXER / f"{scenario}_geodata.json"
    else:
        raise FileNotFoundError("No geographical data has been made. Running the feature pipeline...")

    with open(geodata_path, mode="r") as file:
        raw_geodata = json.load(file)
        
    return raw_geodata 
