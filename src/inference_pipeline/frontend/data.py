"""
This module contains code responsible for loading the various pieces of data 
that will be used to deliver the predictions to the streamlit interface.
"""
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from zipfile import ZipFile
from datetime import datetime

from src.setup.config import config
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER, GEOGRAPHICAL_DATA, FRONTEND_DATA

from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.feature_engineering import ReverseGeocoder
from src.inference_pipeline.backend.inference import InferenceModule, rerun_feature_pipeline, load_raw_local_geodata


@st.cache_data
def make_geodataframes() -> pd.DataFrame:
    """
    Create a dataframe containing the geographical details of each station using both
    arrival and departure data, and return them

    Returns:
        scenario (str)
        tuple[pd.DataFrame, pd.DataFrame]: geodataframes for arrivals and departures
    """
    geo_dataframes = []
    for scenario in config.displayed_scenario_names.keys():
        file_path = FRONTEND_DATA/f"{scenario}_geodataframe.parquet"
        if Path(file_path).exists():
            geo_dataframe = pd.read_parquet(file_path)
            geo_dataframes.append(geo_dataframe)
        else:
            coordinates, station_names = [], []
            unique_coordinates, unique_station_names = set(), set()
            station_details: list[dict] = load_raw_local_geodata(scenario=scenario)

            for detail in tqdm(
                iterable=station_details, 
                desc=f"Collecting station details for {config.displayed_scenario_names[scenario].lower()}"
            ):  
                coordinate = tuple(detail["coordinates"][::-1])  # Reverse the order of the coordinates per pydeck's requirements
                station_name = detail["station_name"]

                # To prevent duplication of coordinates and names in the DataFrame. Sets also reduce time complexity, massively speeding things up
                if coordinate not in unique_coordinates and station_name not in unique_station_names:
                    unique_coordinates.add(coordinate)
                    unique_station_names.add(station_name)

                    coordinates.append(coordinate)
                    station_names.append(station_name)

            geo_dataframe = pd.DataFrame(
                data={f"station_name": station_names, f"coordinates": coordinates}
            )

            geo_dataframe.to_parquet(FRONTEND_DATA/f"{scenario}_geodataframe.parquet")
            geo_dataframes.append(geo_dataframe)
    
    start_geodataframe, end_geodataframe = geo_dataframes[0], geo_dataframes[1]  # For readability
    return start_geodataframe, end_geodataframe


def reconcile_geodata(start_geodataframe: pd.DataFrame, end_geodataframe: pd.DataFrame) -> pd.DataFrame:
    """
    To avoid redundancy, and provide a consistent experience, we will render a single map. Consequently, I can 
    only use stations that are common to both arrival and departure datasets. This function finds the stations 
    common to the

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
