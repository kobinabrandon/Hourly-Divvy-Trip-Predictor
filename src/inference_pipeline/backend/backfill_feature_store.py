"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import json 
import pandas as pd
from tqdm import tqdm 
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta

from sagemaker.feature_store.feature_group import FeatureGroup

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER, INFERENCE_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names

from src.inference_pipeline.backend.inference import (
    make_features,
    load_raw_local_geodata, 
    fetch_predictions_group,
    get_feature_group_for_time_series,
    fetch_time_series_and_make_features
)

from src.inference_pipeline.backend.feature_store_api import setup_feature_group
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.inference import get_model_predictions


def split_features_for_pushing(scenario: str, data: pd.DataFrame):

    data_per_station = {}
    for station_id in tqdm(
        iterable=data[f"{scenario}_station_id"].unique(),
        desc="Splitting data by station"
    ):
        data_per_station[str(station_id)] = data[data[f"{scenario}_station_id"] == station_id] 
    
    return data_per_station


def backfill_features(scenario: str) -> None:
    """
    Run the preprocessing script and upload the time series data to the feature store.

    Args:
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

    Returns:    
        None
    """
    event_time = "timestamp"
    primary_key = ["timestamp", f"{scenario}_station_id"]
    processor = DataProcessor(year=config.year, for_inference=False)
    ts_data = processor.make_time_series()[0] if scenario == "start" else processor.make_time_series()[1]
    ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 9

    ts_feature_group = get_feature_group_for_time_series(scenario=scenario, data=ts_data)
    ts_data_per_station = split_features_for_pushing(scenario=scenario, data=ts_data)

    for station_id, station_data in tqdm(
        iterable=ts_data_per_station.items(),
        desc=logger.info(f"Pushing data to the feature store by the station")
    ):  
        ts_feature_group.ingest(data_frame=station_data)  # Push time series data to the feature group


def backfill_predictions(scenario: str, target_date: datetime, using_mixed_indexer: bool = True) -> None:
    """
    Fetch the registered version of the named model, and download it. Then load a batch of ts_data
    from the relevant feature group (whether for arrival or departure data), and make predictions on those 
    ts_data using the model. Then create or fetch a feature group for these predictions and push these  
    predictions. 

    Args:
        target_date (datetime): the date up to which we want our predictions.
        
    """
    start_date = target_date - timedelta(days=config.backfill_days)
    end_date = target_date + timedelta(days=1)
    
    # The best model architectures for arrivals & departures at the moment
    model_name = "lightgbm" if scenario == "end" else "xgboost"
    tuned_or_not = "tuned" if scenario == "end" else "untuned"

    registry = ModelRegistry(scenario=scenario, model_name=model_name, tuned_or_not=tuned_or_not)
    model = registry.download_latest_model(unzip=True)
    
    ts_feature_group = get_feature_group_for_time_series(scenario=scenario, primary_key=primary_key)

    features = fetch_time_series_and_make_features(
        scenario=scenario,
        start_date=start_date,
        target_date=end_date,
        feature_group=ts_feature_group,
        geocode=False
    )

    try:
        features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
    except Exception as error:
        logger.error(error)    

    predictions: pd.DataFrame = get_model_predictions(scenario=scenario, model=model, features=features)
    predictions = predictions.drop_duplicates().reset_index(drop=True)

    # Now to add station names to the predictions
    ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
    predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

    predictions_feature_group = setup_feature_group(
        scenario=scenario,
        primary_key=primary_key,
        description=f"predicting {config.displayed_scenario_names[scenario]} - {tuned_or_not} {model_name}",
        name=f"{model_name}_{scenario}_predictions_feature_group",
        for_predictions=True,
        version=6
    )

    predictions_feature_group.insert(write_options={"wait_for_job": True}, features=predictions)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        if args.target.lower() == "features":
            backfill_features(scenario=scenario)
        elif args.target.lower() == "predictions":
            backfill_predictions(scenario=scenario, target_date=datetime.now())
        else:
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
