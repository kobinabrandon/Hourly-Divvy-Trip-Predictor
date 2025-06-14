import os
import pandas as pd
from pathlib import Path
from loguru import logger

from src.setup.config import config
from src.setup.config import get_proper_scenario_name
from src.feature_pipeline.data_sourcing import load_raw_data, Year
from src.setup.paths import TRAINING_DATA, make_fundamental_paths

from src.feature_pipeline.preprocessing.cleaning import clean 
from src.feature_pipeline.preprocessing.transformations.training_data import transform_ts_into_training_data
from src.feature_pipeline.preprocessing.transformations.time_series.core import transform_cleaned_data_into_ts
from src.feature_pipeline.preprocessing.station_indexing.choice import check_if_we_use_custom_station_indexing, check_if_we_tie_ids_to_unique_coordinates 


def retrieve_data(years: list[Year], for_inference: bool) -> pd.DataFrame | None: 
    return load_raw_data(years=years) if not for_inference else None  # Because the data will have been fetched from the feature store instead.


def make_training_data(
    data: pd.DataFrame, 
    for_inference: bool, 
    geocode: bool
) -> list[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract raw data, clean it, transform it into time series data, and transform that time series data into
    training data. 

    Args:
        geocode (bool): whether to geocode as part of feature engineering.

    Returns:
        list[pd.DataFrame]: a list containing the datasets for the starts and ends of trips.
    """
    start_ts, end_ts = make_time_series(data=data, for_inference=for_inference)
    ts_data_per_scenario = { "start": start_ts, "end": end_ts }

    training_sets: list[pd.DataFrame] = []
    for scenario in ts_data_per_scenario.keys():
        path_to_training_data: Path = TRAINING_DATA.joinpath(f"{scenario}s.parquet")

        if path_to_training_data.is_file():
            logger.warning(f"Deleting existing version of the training data for {get_proper_scenario_name(scenario=scenario)}")
            os.remove(path_to_training_data)

        training_data: pd.DataFrame = transform_ts_into_training_data(
            ts_data=ts_data_per_scenario[scenario],
            input_seq_len=config.n_features, 
            for_inference=for_inference,
            scenario=scenario, 
            geocode=geocode,
            step_size=1)

        training_sets.append(training_data)
        
    return training_sets


def make_time_series(data: pd.DataFrame, for_inference: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the transformation of the raw data into time series data 

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the time series datasets for departures and arrivals respectively.
    """
    logger.info("Cleaning downloaded data...")

    cleaned_data: pd.DataFrame = clean(data=data, for_inference=for_inference)
    using_custom_station_indexing: bool = check_if_we_use_custom_station_indexing(data=cleaned_data, for_inference=for_inference) 
    tie_ids_to_unique_coordinates: bool = check_if_we_tie_ids_to_unique_coordinates(data=cleaned_data, for_inference=for_inference)

    start_df_columns = ["started_at", "start_lat", "start_lng", "start_station_id"]
    end_df_columns = ["ended_at", "end_lat", "end_lng", "end_station_id"]

    if using_custom_station_indexing and not tie_ids_to_unique_coordinates:
        start_df_columns.append("start_station_name")
        end_df_columns.append("end_station_name")

    start_df: pd.DataFrame = cleaned_data[start_df_columns]
    end_df: pd.DataFrame = cleaned_data[end_df_columns]

    start_ts, end_ts = transform_cleaned_data_into_ts(
        scenarios=["start", "end"],
        cleaned_start_data=start_df, 
        cleaned_end_data=end_df,
        using_custom_station_indexing=using_custom_station_indexing, 
        tie_ids_to_unique_coordinates=tie_ids_to_unique_coordinates,
    )
    return start_ts, end_ts






                
if __name__ == "__main__":
    make_fundamental_paths()
    raw_data = retrieve_data(years=config.years, for_inference=False)
    training_data = make_training_data(data=raw_data, for_inference=False, geocode=False) 

