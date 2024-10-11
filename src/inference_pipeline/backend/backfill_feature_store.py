"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import json 
import feast
import subprocess
import pandas as pd
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER, TIME_SERIES_DATA, INFERENCE_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names

from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.feature_store_api import get_or_create_feature_view
from src.inference_pipeline.backend.inference import InferenceModule, load_raw_local_geodata


def backfill_features(scenario: str) -> None:
    """
    Run the preprocessing script and upload the time series data to the feature store.

    Args:
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

    Returns:
        None
    """
    source = feast.FileSource(
        path=TIME_SERIES_DATA/f"{scenario}s.parquet", 
        timestamp_field=f"{scenario}_hour", 
        file_format=feast.data_format.ParquetFormat()
    )

    feature_view = get_or_create_feature_view(scenario=scenario, for_predictions=False, batch_source=source)

    materialize_command = ["feast", "materialize-incremental", f"$(date {config.current_hour})"]
    subprocess.run(materialize_command)
    
    processor = DataProcessor(year=config.year, for_inference=False)
    ts_data = processor.make_time_series()[0] if self.scenario == "start" else processor.make_time_series()[1]
    ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms

    logger.info(
        f"There are {len(ts_data[f"{self.scenario}_station_id"].unique())} stations in the time series data for\
            {config.displayed_scenario_names[self.scenario].lower()}"
    )

    description=f"Hourly time series data for {config.displayed_scenario_names[self.scenario].lower()}"


def backfill_predictions(scenario: str, target_date: datetime, using_mixed_indexer: bool = True) -> None:
    """
    Fetch the registered version of the named model, and download it. Then load a source of features
    from the relevant feature group(whether for arrival or departure data), and make predictions on those 
    features using the model. Then create or fetch a feature group for these predictions and push these  
    predictions. 

    Args:    
        scenario: Determines whether we are looking at arrival or departure data.  Its value must be "start" or "end".
        target_date (datetime): the date up to which we want our predictions.
        
    """
    if scenario == "end":
        model_name = "lightgbm"
        tuned_or_not = "tuned"

    elif scenario == "start":
        model_name = "xgboost"
        tuned_or_not = "untuned"

    inferrer = InferenceModule(scenario=scenario)
    registry = ModelRegistry(scenario=scenario, model_name=model_name, tuned_or_not=tuned_or_not)
    model = registry.download_latest_model(unzip=True)

    features = inferrer.fetch_time_series_and_make_features(
        start_date=target_date - timedelta(days=270),
        target_date=datetime.now(),
        geocode=False
    )
    
    try:
        features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
    except Exception as error:
        logger.error(error)    

    predictions: pd.DataFrame = inferrer.get_model_predictions(model=model, features=features)
    predictions = predictions.drop_duplicates().reset_index(drop=True)

    # Now to add station names to the predictions
    ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
    predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

    logger.info(
        f"There are {len(predictions[f"{self.scenario}_station_name"].unique())} stations in the predictions \
            for {self.scenario}s"
    )
    
    predictions_feature_group = self.api.setup_feature_group(
        description=f"predicting {config.displayed_scenario_names[self.scenario]} - {tuned_or_not} {model_name}",
        name=f"{model_name}_{self.scenario}_predictions_feature_group",
        for_predictions=True,
        version=6
    )

    predictions_feature_group.insert(
        predictions,
        write_options={"wait_for_job": True}
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
    