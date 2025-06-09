import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.feature_pipeline.data_sourcing import Year
from src.setup.config import get_proper_scenario_name 


def cleaned_data_needs_update(scenario: str, path: Path, year_of_interest: Year) -> bool:
    cleaned_data: pd.DataFrame = pd.read_parquet(path=path)
    most_recent_date_in_data: pd.Series = cleaned_data[f"{scenario}ed_at"][-1]
    data_is_from_at_least_a_month_ago: bool = most_recent_date_in_data.month() < datetime.now().month() 

    new_data_url: str = f"https://divvy-tripdata.s3.amazonaws.com/{year_of_interest}{most_recent_date_in_data.month():02d}-divvy-tripdata.zip"
    new_data_is_available: bool = requests.get(new_data_url).status_code == 200
    proper_scenario_name: str = get_proper_scenario_name(scenario=scenario)

    if not data_is_from_at_least_a_month_ago:
        logger.info(f"The cleaned data for {proper_scenario_name} is up to date")
        return False 

    elif (data_is_from_at_least_a_month_ago and new_data_is_available):
        logger.info(f"The cleaned data for {proper_scenario_name} is out of date, and new data is available")
        return True

    else: 
        logger.info(f"The cleaned data for {proper_scenario_name} is out of date, but new data is unavailable")
        return False 


def delete_rows_with_missing_station_names_and_coordinates(data: pd.DataFrame) -> pd.DataFrame: 
    """
    There are rows with missing latitude and longitude values for the various
    stations. If any of these rows have available station names, then geocoding
    can be used to get the coordinates. 

    At the current time however, all rows with
    missing coordinates also have missing station names, rendering these rows
    irreparably lacking. We locate and delete these points with this function.

    Returns:
        pd.DataFrame: the data, absent the aforementioned rows.
    """
    for scenario in ["start", "end"]:
        lats = data.columns.get_loc(f"{scenario}_lat")
        longs = data.columns.get_loc(f"{scenario}_lng")
        station_names_col = data.columns.get_loc(f"{scenario}_station_name")

        logger.info(
            f"Deleting rows with missing station names & coordinates ({get_proper_scenario_name(scenario=scenario)})"
        )

        where_missing_latitudes = data.iloc[:, lats].isnull()
        where_missing_longitudes = data.iloc[:, longs].isnull()
        where_missing_station_names = data.iloc[:, station_names_col].isnull()

        all_missing_mask = where_missing_station_names & where_missing_latitudes & where_missing_longitudes
        data_to_delete = data.loc[all_missing_mask, :]

        data = data.drop(data_to_delete.index, axis=0)

    return data

