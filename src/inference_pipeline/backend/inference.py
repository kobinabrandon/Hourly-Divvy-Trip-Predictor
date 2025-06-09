"""
This module contains code that:
- fetches time series data from the Hopsworks feature store.
- makes that time series data into features.
- loads model predictions from the Hopsworks feature store.
- performs inference on features
"""
import os
import numpy as np
import pandas as pd

from loguru import logger

from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta

from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView

from src.setup.config import config
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.backend.feature_store import setup_feature_group, get_or_create_feature_view


def get_feature_group_for_time_series(scenario: str, primary_key: list[str]) -> FeatureGroup:

    return setup_feature_group(
        primary_key=primary_key,
        description=f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}",
        name=f"{scenario}_feature_group",
        version=config.feature_group_version,
    )


def fetch_time_series_and_make_features(
    scenario: str, 
    start_date: datetime, 
    target_date: datetime,
    feature_group: FeatureGroup, 
    geocode: bool
    ) -> pd.DataFrame:
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

    Returns:
        pd.DataFrame: time series data 
    """ 
    feature_view: FeatureView = get_or_create_feature_view(
        name=f"{scenario}_feature_view",
        feature_group=feature_group,
        version=1   
    )

    logger.warning("Fetching time series data from the feature store...")
    ts_data: pd.DataFrame = feature_view.get_batch_data(
        start_time=start_date, 
        end_time=target_date,
        read_options={"use_hive": True}
    )

    ts_data = ts_data.sort_values(
        by=[f"{scenario}_station_id", f"{scenario}_hour"]
    )

    return make_features(
        scenario=scenario, 
        ts_data=ts_data, 
        geocode=geocode,
        target_date=target_date
    )


def make_features(
    scenario: str, 
    target_date: datetime, 
    ts_data: pd.DataFrame, 
    geocode: bool
    ) -> pd.DataFrame:
    """
    Restructure the time series data into features in a way that aligns with the features 
    of the original training data.

    Args:
        station_ids: the list of unique station IDs.
        ts_data: the time series data that is store on the feature store.

    Returns:
        pd.DataFrame: time series data
    """
    processor = DataProcessor(years=config.years, for_inference=True)
    
    # Perform transformation of the time series data with feature engineering
    features = processor.transform_ts_into_training_data(
        scenario=scenario, 
        ts_data=ts_data,
        geocode=geocode,
        input_seq_len=config.n_features,
        step_size=24
    )

    features[f"{scenario}_hour"] = target_date
    features = features.sort_values(by=[f"{scenario}_station_id"])
    return features


def fetch_predictions_group(scenario: str, model_name: str) -> FeatureGroup:
    """
    Return the feature group used for predictions.

    Args:
        model_name (str): the name of the model

    Returns:
        FeatureGroup: the feature group for the given model's predictions.
    """
    assert model_name == "xgboost", 'The selected model architectures are currently "xgboost" and "lightgbm"'
    tuned_or_not = "tuned" 
       
    return setup_feature_group(
        primary_key=[f"{scenario}_station_id"],
        description=f"predictions on {scenario} data using the {tuned_or_not} {model_name}",
        name=f"{model_name}_{scenario}_predictions",
        version=config.feature_group_version
    )


def load_predictions_from_store(
    scenario: str, 
    from_hour: datetime, 
    to_hour: datetime, 
    model_name: str,
    aggregate_predictions: bool = False, 
    aggregation_method: str = "mean"
    ) -> pd.DataFrame:
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
    assert aggregation_method.lower() in ["sum", "mean"], 'Please specify "sum" or "mean" as aggregation methhods'

    # Ensure these times are datatimes
    from_hour = pd.to_datetime(from_hour, utc=True)
    to_hour = pd.to_datetime(to_hour, utc=True)
        
    predictions_group = fetch_predictions_group(scenario=scenario, model_name=model_name)

    predictions_feature_view: FeatureView = get_or_create_feature_view(
        name=f"{model_name}_{scenario}_predictions",
        feature_group=predictions_group,
        version=config.feature_view_version
    )

    predictions_df = predictions_feature_view.get_batch_data(
        start_time=from_hour - timedelta(days=1), 
        end_time=to_hour + timedelta(days=1)
    )

    predictions_df[f"{scenario}_hour"] = pd.to_datetime(predictions_df[f"{scenario}_hour"], utc=True)
    predictions_df = predictions_df.drop("timestamp", axis=1)

    predictions_df: pd.DataFrame = predictions_df.sort_values(
        by=[f"{scenario}_hour", f"{scenario}_station_id"]
    )
    
    if aggregate_predictions and aggregation_method.lower() in ["sum", "mean"]:
        return get_aggregate_predictions(scenario=scenario, predictions=predictions_df, aggregation_method=aggregation_method)
    elif not aggregate_predictions:
        return predictions_df.reset_index(drop=True)
    

def get_model_predictions(scenario: str, model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """
    Simply use the model's predict method to provide predictions based on the supplied features

    Args:
        model: the model object fetched from the model registry
        features: the features obtained from the feature store

    Returns:
        pd.DataFrame: the model's predictions
    """
    generated_predictions = model.predict(features)

    prediction_per_station = pd.DataFrame()
    prediction_per_station[f"{scenario}_station_id"] = features[f"{scenario}_station_id"].values

    prediction_per_station[f"{scenario}_hour"] = pd.to_datetime(datetime.utcnow()).floor("H")
    prediction_per_station[f"predicted_{scenario}s"] = generated_predictions.round(decimals=0)

    prediction_per_station["timestamp"] = pd.to_datetime(prediction_per_station[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms

    return prediction_per_station


def get_aggregate_predictions(scenario: str, predictions: pd.DataFrame, aggregation_method: str) -> pd.DataFrame:

    if aggregation_method.lower() == "sum":
        predictions[f"predicted_{scenario}s"] = predictions.groupby(f"{scenario}_station_id")[f"predicted_{scenario}s"].transform("sum")
        return predictions.drop_duplicates().reset_index(drop=True)

    elif aggregation_method.lower() == "mean":
        predictions[f"predicted_{scenario}s"] = predictions.groupby(f"{scenario}_station_id")[f"predicted_{scenario}s"].transform("mean")
        predictions[f"predicted_{scenario}s"] = np.ceil(predictions[f"predicted_{scenario}s"])
        return predictions.drop_duplicates().reset_index(drop=True)

    else:
        raise NotImplementedError('The only aggregation methods in use are "sum" and "mean". ')


def round_mean_by_scenario(scenario: str, predicted_values: pd.Series): 
    return np.ceil(predicted_values) if scenario == "start" else None



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

                processor = DataProcessor(years=config.years, for_inference=False)
                processor.make_training_data(geocode=False)
                return fn(*args, **kwargs)
        return wrapper
    return decorator


@rerun_feature_pipeline()
def load_raw_local_geodata(scenario: str) -> pd.DataFrame | None:
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
        geodata_path = ROUNDING_INDEXER.joinpath(f"{scenario}_geodataframe.parquet")
    elif len(os.listdir(MIXED_INDEXER)) != 0:
        geodata_path = MIXED_INDEXER.joinpath(f"{scenario}_geodataframe.parquet")
    else:
        raise FileNotFoundError("No geographical data has been made. Running the feature pipeline...")

    with open(geodata_path, mode="r") as file:
        return pd.read_parquet(geodata_path)

