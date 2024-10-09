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
from src.setup.paths import ROUNDING_INDEXER, MIXED_INDEXER, GEOGRAPHICAL_DATA

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

        coordinates = []
        station_names = []
        station_details: list[dict] = load_raw_local_geodata(scenario=scenario)

        for detail in tqdm(
            iterable=station_details, 
            desc=f"Collecting station details for {config.displayed_scenario_names[scenario].lower()}"
        ):  
            coordinate = detail["coordinates"][::-1]  # Reverse the order of the coordinates per pydeck's requirements
            station_name = detail["station_name"]

            # ALERT: To prevent duplication of coordinates and names in the pd.DataFrame
            if coordinate not in coordinates and station_name not in station_names:
                coordinates.append(coordinate)
                station_names.append(station_name)

        geo_dataframe = pd.DataFrame(
            data={f"station_name": station_names, f"coordinates": coordinates}
        )

        geo_dataframes.append(geo_dataframe)
    
    start_geodataframe, end_geodataframe = geo_dataframes[0], geo_dataframes[1]  # For readability
    return start_geodataframe, end_geodataframe


def reconcile_geodata(start_geodataframe: pd.DataFrame, end_geodataframe: pd.DataFrame) -> pd.DataFrame:
    """
    To avoid redundancy, and provide a consistent experience, we will render a single map. Consequently, I can 
    only use stations that are common to both arrival and departure datasets. This function finds the stations 
    common to the

    Returns:

    """
    larger_dataframe = start_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else end_geodataframe
    smaller_dataframe = end_geodataframe if len(start_geodataframe) >= len(end_geodataframe) else start_geodataframe

    shared_stations_bool = np.isin(element=larger_dataframe["station_name"], test_elements=smaller_dataframe["station_name"])
    common_data = larger_dataframe.loc[shared_stations_bool, :]

    logger.warning(
        f"{len(larger_dataframe) - len(common_data)} stations were discarded because they were not common to both datasets"
    )

    return common_data


class ExternalShapeFile:
    """
    Allows us to download shapefiles and load them for later processing.
    """
    def __init__(self, map_type: str):
        """
        Args:
            map_type: a string that specifies the type of map whose shapefile we would like.
        """
        urls = {
            "pedestrian": "https://data.cityofchicago.org/api/geospatial/v6kn-gc9b?fourfour=v6kn-gc9b&cacheBust=\
                            1712775948&date=20240920&accessType=DOWNLOAD&method=export&format=Shapefile",

            "divvy_stations": "https://data.cityofchicago.org/api/geospatial/bbyy-e7gq?fourfour=bbyy-e7gq&cacheBust=\
                              1726743616&date=20240920&accessType=DOWNLOAD&method=export&format=Shapefile"
        }

        self.file_names = {
            "pedestrian": "pedestrian_streets_shapefile.zip",
            "divvy_stations": "divvy_stations_shapefile.zip"
        }

        self.map_type = map_type
        self.url = urls[map_type]
        self.zipfile_path = GEOGRAPHICAL_DATA / self.file_names[self.map_type]

    def download_archive(self) -> None:
        """
        Download the zipfile that contains the shapefile for the given map type.

        Returns:
            None
        """
        logger.info(f"Downloading shapefile for {self.map_type}s")
        response = requests.get(url=self.url)
        if response.status_code == 200:
            open(self.zipfile_path, mode="wb").write(response.content)
        else:
            raise Exception(f"The URL for {self.map_type}s is not available")

    def load_data_from_shapefile(self) -> pd.DataFrame:
        """
        Extract the contents of the downloaded archive to access the shapefile within, and then deliver it as a
        geo-dataframe.

        Returns:
            pd.DataFrame: the contents of the shapefile, rendered as a geo-dataframe.
        """
        if Path(self.zipfile_path).is_file():
            logger.success(f"The shapefile for {self.map_type}s is already saved to disk.")
        else:
            self.download_archive()

        with ZipFile(self.zipfile_path, mode="r") as zipfile:
            zipfile.extractall(GEOGRAPHICAL_DATA/self.file_names[:-4])

        return gpd.read_file(filename=self.zipfile_path / f"{self.file_names}[:-4].shp").to_crs("epsg:4326")
