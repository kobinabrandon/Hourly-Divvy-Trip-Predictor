import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.setup.paths import CLEANED_DATA 
from src.feature_pipeline.data_sourcing import Year
from src.setup.config import get_proper_scenario_name 
from src.feature_pipeline.preprocessing.station_indexing.choice import check_if_we_use_custom_station_indexing, check_if_we_tie_ids_to_unique_coordinates 


def clean(data: pd.DataFrame, for_inference: bool, save: bool = True) -> pd.DataFrame:
    """

    Args:
        data: 
        for_inference: 
        save: 

    Returns:
        

    Raises:
        NotImplementedError: 
    """
    if for_inference:
        path_to_cleaned_data =  CLEANED_DATA.joinpath("partially_cleaned_data_for_inference.parquet")

    else:
        tie_ids_to_unique_coordinates: bool = check_if_we_tie_ids_to_unique_coordinates(data=data, for_inference=for_inference)
        using_custom_station_indexing: bool = check_if_we_use_custom_station_indexing(data=data, for_inference=for_inference)  

        match (using_custom_station_indexing, tie_ids_to_unique_coordinates):
            case (True, True):
                path_to_cleaned_data = CLEANED_DATA.joinpath("data_with_newly_indexed_stations (rounded_indexer).parquet")
            case (True, False):
                path_to_cleaned_data = CLEANED_DATA.joinpath("data_with_newly_indexed_stations (mixed_indexer).parquet")
            case (False, _):
                raise NotImplementedError("The majority of Divvy's IDs weren't numerical and valid during initial development.")

    # Will think of a more elegant solution in due course. This only serves my current interests.
    if path_to_cleaned_data.is_file():
        logger.success("There is already some cleaned data. Fetching it...")
        # cleaned_data: pd.DataFrame = pd.read_parquet(path=path_to_cleaned_data)

        # if cleaned_data_needs_update(scenario=scenario, cleaned_data=cleaned_data, year_of_interest)





    else:
        data["started_at"] = pd.to_datetime(data["started_at"], format="mixed")
        data["ended_at"] = pd.to_datetime(data["ended_at"], format="mixed")

        features_to_drop = ["ride_id", "rideable_type", "member_casual"]
        data_with_missing_details_removed: pd.DataFrame = delete_rows_with_missing_station_names_and_coordinates(data=data)

        using_custom_station_indexing = check_if_we_tie_ids_to_unique_coordinates(data=data, for_inference=for_inference)

        if using_custom_station_indexing and tie_ids_to_unique_coordinates: 
            features_to_drop.extend(
                ["start_station_id", "start_station_name", "end_station_name"]
            )

        data_with_missing_details_removed = data_with_missing_details_removed.drop(columns=features_to_drop)

        if save:
            data_with_missing_details_removed.to_parquet(path=path_to_cleaned_data)

        return data_with_missing_details_removed



def cleaned_data_needs_update(scenario: str, cleaned_data: pd.DataFrame, years_of_interest: list[Year]) -> bool:

    this_month: datetime = datetime.now().month()
    most_recent_date_in_data: pd.Series = cleaned_data[f"{scenario}ed_at"][-1]

    # Data is deemed to be if it is from prior month regardless of how many days have passed 
    data_is_old: bool = most_recent_date_in_data.month() < this_month 

    # This is a link (according to Lyft's link structure) to the data for the most 
    this_year: int = get_latest_year(years_of_interest=years_of_interest)
    new_data_url: str = f"https://divvy-tripdata.s3.amazonaws.com/{this_year}{this_month:02d}-divvy-tripdata.zip"
    new_data_is_available: bool = requests.get(new_data_url).status_code == 200
    proper_scenario_name: str = get_proper_scenario_name(scenario=scenario)

    match (data_is_old, new_data_is_available):
        case (False, _):
            logger.info(f"Cleaned data for {proper_scenario_name} is up to date")
            return False 

        case (True, True):
            logger.info(f"Cleaned data for {proper_scenario_name} is out of date, and new data is available")
            return True

        case (True, False):
            logger.info(f"Cleaned data for {proper_scenario_name} is out of date, but new data is not available")
            return False 


def get_latest_year(years_of_interest: list[Year]) -> int:

    retrieved_year: list[int] = []
    for year in years_of_interest:
        if datetime.now().year() == year.value:
            retrieved_year.append(year.value) 

    if len(retrieved_year) == 0:
        raise Exception("Unable to check age of cleaned data.") 
    else:
        return retrieved_year[0]

    # The years that you are scanning do not in ")




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

