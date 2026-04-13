"""
This module contains code responsible for loading the various pieces of data 
that will be used to deliver the predictions to the streamlit interface.
"""
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from loguru import logger

from src.setup.config import config
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER
from src.inference_pipeline.backend.inference import load_raw_local_geodata


@st.cache_data()
def make_geodataframes(using_mixed_indexer: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a dataframe containing the geographical details of each station using both
    arrival and departure data, and return them

    Returns:
        scenario (str)
        tuple[pd.DataFrame, pd.DataFrame]: geodataframes for arrivals and departures
    """
    geo_dataframes = []
    geodataframe_directory = MIXED_INDEXER if using_mixed_indexer else ROUNDING_INDEXER
    for scenario in config.displayed_scenario_names.keys():
        file_path = geodataframe_directory/f"{scenario}_geodataframe.parquet"
        if Path(file_path).exists():
            geo_dataframe = pd.read_parquet(file_path)
        else:
            geo_dataframe: pd.DataFrame | None = load_raw_local_geodata(scenario=scenario)
        
        geo_dataframe.drop("station_id", axis=1)
        geo_dataframes.append(geo_dataframe)
    
    return geo_dataframes[0], geo_dataframes[1]


def reconcile_geodata(start_geodataframe: pd.DataFrame, end_geodataframe: pd.DataFrame) -> pd.DataFrame:
    """
    To avoid redundancy, and provide a consistent experience, we will render a single map. Consequently, I can 
    only use stations that are common to both arrival and departure datasets. 

    Returns:
        pd.DataFrame: 
    """
    larger_dataframe = start_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else end_geodataframe
    smaller_dataframe = end_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else start_geodataframe

    shared_stations_bool = np.isin(element=larger_dataframe["station_name"], test_elements=smaller_dataframe["station_name"])
    common_data = larger_dataframe.loc[shared_stations_bool, :]

    logger.warning(
        f"{len(larger_dataframe) - len(common_data)} stations were discarded because they were not common to both datasets"
    )

    return common_data

