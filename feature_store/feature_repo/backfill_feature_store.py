"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import os
import json 
import subprocess
import pandas as pd
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta
from feast.data_format import ParquetFormat
from feast import FileSource, FeatureStore

from src.setup.config import config
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.feature_store_api import get_or_create_feature_view
from src.inference_pipeline.backend.inference import get_model_predictions, load_raw_local_geodata

from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER, TIME_SERIES_DATA, INFERENCE_DATA, FEATURE_REPO


def backfill_features(scenario: str) -> None:
    """
    Run the preprocessing script and upload the time series data to the feature store.

    Args:
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

    Returns:
        None
    """
    processor = DataProcessor(year=config.year, for_inference=False)
    ts_data = processor.make_time_series()[0] if scenario == "start" else processor.make_time_series()[1]
    ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms
    
    source = FileSource(
        path=str(TIME_SERIES_DATA/f"{scenario}_ts.parquet"), 
        description=f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}",
        event_timestamp_column=f"{scenario}_hour", 
        file_format=ParquetFormat()
    )

    feature_view = get_or_create_feature_view(scenario=scenario, for_predictions=False, batch_source=source)

    today = datetime.now().strftime("%Y-%m-%d")
    store = FeatureStore(repo_path=FEATURE_REPO, fs_yaml_file=FEATURE_REPO/"feature_store.yaml")
    store.apply([source, feature_view])    
    
    materialize_command = f"feast materialize-incremental {today}"
    os.system(command=materialize_command)

    store.write_to_online_store(feature_view_name=feature_view.name, allow_registry_cache=True, df=ts_data)


def backfill_predictions(scenario: str, target_date: datetime, using_mixed_indexer: bool = True) -> None:
    """
    Fetch the registered version of the named model, and download it. Then load a source of features
    from the relevant feature group(whether for arrival or departure data), and make predictions on those 
    features using the model. Then create or fetch a feature group for these predictions and push these  
    predictions. 

    Args:    
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".
        target_date (datetime): the date up to which we want our predictions.

    Returns:
        None
    """
    if scenario == "end":
        model_name = "lightgbm"
        tuned_or_not = "tuned"

    elif scenario == "start":
        model_name = "xgboost"
        tuned_or_not = "untuned"

    registry = ModelRegistry(scenario=scenario, model_name=model_name, tuned_or_not=tuned_or_not)
    model = registry.download_latest_model(unzip=True)

    features = fetch_time_series_and_make_features(
        start_date=target_date - timedelta(days=270),
        target_date=datetime.now(),
        geocode=False
    )
    
    try:
        features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
    except Exception as error:
        logger.error(error)    

    predictions: pd.DataFrame = get_model_predictions(model=model, features=features)
    predictions = predictions.drop_duplicates().reset_index(drop=True)

    # Now to add station names to the predictions
    ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
    predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

    logger.info(
        f"There are {len(predictions[f"{scenario}_station_name"].unique())} stations in the predictions \
            for {scenario}s"
    )
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        assert scenario.lower() in ["start", "end"], 'Only "start" or "end" are acceptable values'

        if args.target.lower() == "features":
            backfill_features(scenario=scenario)
        elif args.target.lower() == "predictions":
            backfill_predictions(scenario=scenario, target_date=datetime.now())
        else:
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
    