import numpy as np
import pandas as pd
from tqdm import tqdm 
from loguru import logger

from src.setup.config import get_proper_scenario_name
from src.setup.paths import TRAINING_DATA, INFERENCE_DATA 
from src.feature_pipeline.feature_engineering import finish_feature_engineering
from src.feature_pipeline.preprocessing.transformations.time_series.cutoffs import CutoffIndexer


def transform_ts_into_training_data(
        scenario: str,
        geocode: bool,
        step_size: int,
        input_seq_len: int,
        for_inference: bool,
        ts_data: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Transpose the time series data into a feature-target format.

    Args:
        scenario: a string that indicates whether we are dealing with the starts or ends of trips
        geocode: whether to use geocoding during feature engineering 
        step_size: the step size to be used by the standard cutoff indexer.
        input_seq_len: the input sequence length to be used to construct the training data
        ts_data: the full time series dataset for arrivals and departures

    Returns:
        pd.DataFrame: the training data for arrivals or departures
    """
    if for_inference and "timestamp" in ts_data.columns:
        ts_data = ts_data.drop("timestamp", axis=1)

    # Ensure first that these are the columns of the chosen data set (and they are listed in this order)
    assert set(ts_data.columns) == {f"{scenario}_hour", f"{scenario}_station_id", "trips"}

    features = pd.DataFrame()
    targets = pd.DataFrame()

    for station_id in tqdm(
        iterable=ts_data[f"{scenario}_station_id"].unique(), 
        desc=f"Turning time series data into training data ({get_proper_scenario_name(scenario=scenario)})"
    ):
        
        data_associated_with_the_station_id = ts_data[f"{scenario}_station_id"] == station_id
        
        ts_per_station = ts_data.loc[
            data_associated_with_the_station_id, [f"{scenario}_hour", f"{scenario}_station_id", "trips"]
        ].sort_values(by=[f"{scenario}_hour"])

        cutoff_indexer = CutoffIndexer(ts_data=ts_per_station, input_seq_len=input_seq_len, step_size=step_size)
        use_standard_cutoff_indexer: bool = cutoff_indexer.use_standard_cutoff_indexer()

        indices = cutoff_indexer.indices
        num_indices = len(indices) 

        x = np.empty(shape=(num_indices, input_seq_len), dtype=np.float32)
        y = np.empty(shape=(num_indices, 1), dtype=np.float32)

        hours = []    
        if use_standard_cutoff_indexer:
            for i, index in enumerate(indices):
                hour = ts_per_station.iloc[index[1]][f"{scenario}_hour"]
                x[i, :] = ts_per_station.iloc[index[0]: index[1]]["trips"].values
                y[i] = ts_per_station.iloc[index[2]]["trips"]
                hours.append(hour)
        
        elif not use_standard_cutoff_indexer and len(ts_per_station) == 1:

            x[0, :] = np.full(
                shape=(1, input_seq_len), 
                fill_value=ts_per_station["trips"].iloc[0]
            )

            y[0] = ts_per_station["trips"].iloc[0]
            hour = ts_per_station[f"{scenario}_hour"].values[0]
            hours.append(hour)

        else:
            ts_per_station = ts_per_station.reset_index(drop=True)
            
            for i, index in enumerate(indices):
                x[i, :] = ts_per_station.iloc[index[0]: index[1], 2].values
                y[i] = ts_per_station[index[1]:index[2]]["trips"].values[0]
                hour = ts_per_station.iloc[index[1]][f"{scenario}_hour"]
                hours.append(hour)

        features_per_station = pd.DataFrame(
            data=x, 
            columns=[ f"trips_previous_{i + 1}_hour" for i in reversed(range(input_seq_len)) ]
        )
        
        features_per_station[f"{scenario}_hour"] = hours  
        features_per_station[f"{scenario}_station_id"] = station_id
        targets_per_station = pd.DataFrame(data=y, columns=["trips_next_hour"])

        features = pd.concat([features, features_per_station], axis=0)
        targets = pd.concat([targets, targets_per_station], axis=0)

    features = features.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    engineered_features = finish_feature_engineering(features=features, scenario=scenario, geocode=geocode)

    training_data = pd.concat( 
        list(engineered_features, targets["trips_next_hour"]), axis=1 
    )

    logger.success("Saving the data so we (hopefully) won't have to do that again...")
    final_data_path = INFERENCE_DATA.joinpath(f"{scenario}s.parquet") if for_inference else TRAINING_DATA.joinpath(f"{scenario}s.parquet") 
    training_data.to_parquet(final_data_path)

    return training_data

