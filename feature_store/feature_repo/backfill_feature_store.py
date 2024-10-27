import os
import json 
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

from src.inference_pipeline.backend.inference import (
    get_model_predictions, 
    load_raw_local_geodata, 
    fetch_time_series_and_make_features
)

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

    get_and_push_data(scenario=scenario, for_predictions=False, data=ts_data)


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
    

    start_date=target_date - timedelta(days=270),
    
    try:
        features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
    except Exception as error:
        logger.error(error)    

    predictions: pd.DataFrame = get_model_predictions(model=model, features=features)
    predictions = predictions.drop_duplicates().reset_index(drop=True)

    # Now to add station names to the predictions
    ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
    predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

    get_and_push_data(scenario=scenario, for_predictions=True, data=predictions)
    

def get_and_push_data(scenario: str, for_predictions: bool, data: pd.DataFrame) -> None:

    if for_predictions:
        data_path = INFERENCE_DATA/f"predicted_{scenario}s.parquet"
        description = f"Hourly predictions for {config.displayed_scenario_names[scenario].lower()}"
    else:
        data_path = TIME_SERIES_DATA/f"{scenario}_ts.parquet"
        description = f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}"

    source = FileSource(path=str(data_path), file_format=ParquetFormat(), description=description)
    store = FeatureStore(repo_path=FEATURE_REPO, fs_yaml_file=FEATURE_REPO/"feature_store.yaml")
        
    if for_predictions:
        feature_view = store.get_feature_view()
    else:
        feature_view, station_ids = get_or_create_feature_view(scenario=scenario, for_predictions=for_predictions, file_source=source)

        os.chdir(path=FEATURE_REPO)
        store.apply([station_ids, feature_view])  

        materialize_command = f"feast materialize-incremental {datetime.now().strftime('%Y-%m-%d')}"
        os.system(command=materialize_command)

        store.write_to_online_store(feature_view_name=feature_view.name, allow_registry_cache=True, df=data)


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
    