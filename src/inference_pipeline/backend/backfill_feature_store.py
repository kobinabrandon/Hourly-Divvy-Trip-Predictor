"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import pandas as pd
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta

from src.setup.config import config
from src.feature_pipeline.preprocessing.core import make_time_series 
from src.training_pipeline.cleanup import retrieve_best_model_from_previous_run

# from src.inference_pipeline.backend.inference import (
#     get_feature_group_for_time_series, 
#     fetch_time_series_and_make_features, 
#     get_model_predictions
# )
#
# from src.inference_pipeline.backend.feature_store import setup_feature_group
from src.inference_pipeline.backend.model_registry import download_model, get_name_of_model_type, get_full_model_name 


def backfill_features(scenario: str) -> None:
    """
    Run the preprocessing script and upload the time series data to the feature store.

    Args:
        scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

    Returns:
        None
    """
    primary_key = ["timestamp", f"{scenario}_station_id"]

    raw_data: pd.DataFrame = load_raw_data()
    start_ts, end_ts = make_time_series(data=raw_data)
    ts_data = start_ts if scenario == "start" else end_ts 

    ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms
    ts_feature_group = get_feature_group_for_time_series(scenario=scenario, primary_key=primary_key)
    ts_feature_group.insert(write_options={"wait_for_job": True}, features=ts_data) # Push time series data to the feature group


def backfill_predictions(scenario: str, target_date: datetime) -> None: 
    """
    Fetch the registered version of the named model, and download it. Then load a batch of ts_data
    from the relevant feature group (whether for arrival or departure data), and make predictions on those 
    ts_data using the model. Then create or fetch a feature group for these predictions and push these  
    predictions. 

    Args:
        target_date (datetime): the date up to which we want our predictions.
        
    """
    primary_key = [f"{scenario}_station_id"]
    start_date = target_date - timedelta(days=config.backfill_days)
    end_date = target_date + timedelta(days=1)
    
    # Based on the best models for arrivals & departures at the moment
    model_name: str = retrieve_best_model_from_previous_run(scenario=scenario)
    is_tuned: bool = determine_whether_model_is_tuned_or_not(scenario=scenario, best_model=model_name)

    model = download_model(scenario=scenario, model_name=model_name, tuned=is_tuned, unzip=True)
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

    predictions_feature_group = setup_feature_group(
        primary_key=primary_key,
        description=f"predicting {config.displayed_scenario_names[scenario]} - {tuned_or_not} {model_name}",
        name=f"{model_name}_{scenario}_predictions",
        version=config.feature_group_version
    )

    predictions_feature_group.insert(write_options={"wait_for_job": True}, features=predictions)


def determine_whether_model_is_tuned_or_not(scenario: str, best_model: str):

    for tuned_or_not in [True, False]:
        return False if "untuned" in best_model else True 
            
 

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
