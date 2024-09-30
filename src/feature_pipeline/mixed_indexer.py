import json

from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, CLEANED_DATA, ROUNDING_INDEXER
from src.feature_pipeline.feature_engineering import ReverseGeocoder
from src.feature_pipeline.rounding_indexer import add_column_of_rounded_coordinates


def find_rows_with_either_missing_ids_or_names(scenario: str, data: pd.DataFrame) -> bool:
    """
    Search the dataset for rows which have either a missing station name or a missing 
    station ID. In the version of the data that we are currently using, there no such rows.

    Args:
        scenario (str): "start" or "end"
        data (pd.DataFrame): the dataset to search

    Returns:
        bool: a truth value indicating the existence or lack thereof of such rows. 
    """
    logger.info("Checking for rows that either have missing station names or IDs")
    
    station_ids = data[f"{scenario}_station_id"]
    station_names = data[f"{scenario}_station_name"]
    
    only_missing_id = station_names.notnull() & station_ids.isnull()
    only_missing_names = station_names.isnull() & station_ids.notnull()

    target_condition = only_missing_id | only_missing_names
    return True if target_condition.sum() > 0 else False


def find_rows_with_missing_ids_and_names(
    data: pd.DataFrame, 
    scenario: str, 
    first_time: bool, 
    return_indices: bool
    ) -> list[int]:
    """
    Search for rows with both IDs and names missing.

    Args:
        data (pd.DataFrame): the dataset we want to search
        scenario (str): "start" or "end"
        first_time (bool): whether this function is being run for the first time
        return_indices (bool): whether to return the indices of the rows found. If False, all the data contained
                               in those rows will be returned.

    Returns:
        list[int]: the indices of the rows we found.
    """
    logger.info(f"Searching for rows that{"" if first_time else " still"} have missing station names and IDs.")

    missing_station_ids = data[f"{scenario}_station_id"].isnull()
    missing_station_names = data[f"{scenario}_station_name"].isnull()
    mask_of_problem_rows = missing_station_ids & missing_station_names
    problem_data = data.loc[mask_of_problem_rows, :]

    logger.warning(f"{len(problem_data)} rows{"" if first_time else " still"} have missing station names and IDs.")
    return problem_data.index if return_indices else problem_data


def find_rows_with_known_ids_and_names(scenario: str, data: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """
    Find all the coordinates which have a known ID and known station name, and provide a dictionary 
    of the respective rows and their associated coordinates.

    Returns:
        dict[str, tuple[float]]: pairs consisting of row numbers and their respective coordinates.
    """
    logger.info("Looking for rows that have both station names and IDs...")

    present_station_ids = data[f"{scenario}_station_id"].notnull()
    present_station_names = data[f"{scenario}_station_name"].notnull()
    complete_rows_mask = present_station_ids & present_station_names
    complete_rows = data.loc[complete_rows_mask, :]
    
    latitudes_of_complete_rows = data.loc[complete_rows_mask, f"{scenario}_lat"]
    longitudes_of_complete_rows = data.loc[complete_rows_mask,  f"{scenario}_lng"]

    # The indices of the rows that are without issue will be the keys.
    rows_and_coordinates_with_known_ids_names = dict(
        zip(
            complete_rows.index, zip(latitudes_of_complete_rows, longitudes_of_complete_rows)
        )
    )

    return rows_and_coordinates_with_known_ids_names


def match_names_and_ids_by_station_proximity(scenario: str, data: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """
    Based on common sense, and a perfunctory look at https://account.divvybikes.com/map, it looks like there 
    are (knock on wood) no two stations that are within 10m of each other. On those grounds, we can say with some 
    confidence that any two station coordinates which are within 10m of each other must belong to the same station.

    Suppose we have a given coordinate (which we'll call the target coordinate), and we round it down from 6 to
    4 decimal places. If both coordinates of this rounded target coordinate are equal to the rounded version of 
    some other coordinate (on some other row) which has a known ID and known station name (we've confirmed that 
    it can't be one or the other), then the row of the target coordinate will be associated with the ID and station
    name of the coordinate we found.

    Following this, we take the indices of the rows whose associated IDs and names were just discovered, and fill
    the missing station names and IDs in these rows with their new values.

    Returns:
        pd.DataFrame: the dataset after filling of the empty IDs and names with those that were discovered.
    """
    assert not find_rows_with_either_missing_ids_or_names(scenario=scenario, data=data), 'There is now a row which \
    contains a missing station ID or a station name (not both). This will have occurred due to a change in the data'

    logger.warning("Starting the matching process...")
    problem_row_indices = find_rows_with_missing_ids_and_names(
        data=data, 
        scenario=scenario, 
        first_time=True,
        return_indices=True
    )

    complete_rows_and_their_original_coordinates = find_rows_with_known_ids_and_names(scenario=scenario, data=data)
    coordinates_of_complete_rows = np.array(list(complete_rows_and_their_original_coordinates.values()))
    rounded_coordinates_of_complete_rows = np.round(coordinates_of_complete_rows, decimals=4)

    latitudes_index = data.columns.get_loc(f"{scenario}_lat")
    longitudes_index = data.columns.get_loc(f"{scenario}_lng")

    rounded_problem_lats = np.round(data.iloc[problem_row_indices, latitudes_index].values, decimals=5)
    rounded_problem_lngs = np.round(data.iloc[problem_row_indices, longitudes_index].values, decimals=5)
    rounded_problem_coordinates = list(zip(rounded_problem_lats, rounded_problem_lngs))

    # Get a boolean array of the indices of rounded coordinates
    rounded_coordinates_match = np.isin(
        element=rounded_problem_coordinates, 
        test_elements=rounded_coordinates_of_complete_rows
    ).all(axis=1)
                
    is_problem_row = np.isin(
        element=problem_row_indices, 
        test_elements=np.arange(len(data))
    )      

    rows_to_be_targeted = np.where(is_problem_row & rounded_coordinates_match)[0]

    found_ids = data.iloc[rows_to_be_targeted, data.columns.get_loc(f"{scenario}_station_id")]
    found_names = data.iloc[rows_to_be_targeted, data.columns.get_loc(f"{scenario}_station_name")]

    problem_rows_and_their_discovered_names_and_ids = {
        int(index): (code, name) for index, code, name in zip(rows_to_be_targeted, found_ids, found_names)
    }

    logger.success(f"Found new names and IDs for {len(problem_rows_and_their_discovered_names_and_ids)} rows.")

    rows_and_new_ids = {
        int(row): new_id for row, (new_id, new_name) in problem_rows_and_their_discovered_names_and_ids.items()
    }

    rows_and_new_names = {
        int(row): new_name for row, (new_id, new_name) in problem_rows_and_their_discovered_names_and_ids.items()
    }

    index = pd.Series(data.index)
    # Perform the replacement on the target column by mapping the index to the new values using the dictionaries
    data[f"{scenario}_station_id"] = data[f"{scenario}_station_id"].fillna(index.map(rows_and_new_ids))
    data[f"{scenario}_station_name"] = data[f"{scenario}_station_name"].fillna(index.map(rows_and_new_names))

    return data

def save_geodata(data: pd.DataFrame, scenario: str, for_plotting: bool) -> None:
    """
    Saves the station ID, mame, and coordinates for use in the frontend
    """
    station_names = data[f"{scenario}_station_name"].values
    station_ids = [int(id_number) for id_number in data[f"{scenario}_station_id"].values]
    longitudes = data[f"{scenario}_lng"].values
    latitudes = data[f"{scenario}_lat"].values

    if for_plotting:
        file_path = MIXED_INDEXER / f"{scenario}_geojson.geojson"

        geodata = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",

                    "geometry": {
                        "type": "Point",
                        "coordinate": [longitude, latitude]  # Apparently, this reversal is standard for geojson files
                    },

                    "properties": {
                        "station_id": station_id,
                        "station_name": station_name
                    }
                } 
                for (latitude, longitude, station_id, station_name) in tqdm(
                    iterable=zip(latitudes, longitudes, station_ids, station_names),
                    desc="Saving the geodata in each row"
                )     
            ] 
        }

    else:
        file_path = MIXED_INDEXER / f"{scenario}_geodata.json"
        geodata = [
            {   
                "coordinates": [latitude, longitude],
                "station_id": station_id,
                "station_name": station_name    
            }
            for latitude, longitude, station_id, station_name in zip(latitudes, longitudes, station_ids, station_names)
        ]

    with open(file_path, mode="w") as file:
        json.dump(geodata, file)


def make_json_of_names_and_ids(scenario: str, using_mixed_indexer: bool = True) -> None:
    """
    Extract the names and IDs that have been created, and save them in a new json file. This json file will later be
    used to backfill predictions to the feature store. This function will need to be used regardless of which of the 
    custom indexers is triggered, but I am storing it in this script because I consider the mixed indexer to be the 
    "preferred" choice.

    Args:
        scenario (str): "start" or "end"
        using_mixed_indexer (bool, optional): whether we will be using the mixed indexer or not. Defaults to True.
    """

    save_path = MIXED_INDEXER if using_mixed_indexer else ROUNDING_INDEXER

    with open(save_path / f"{scenario}_geodata.json", mode="r") as file:
        geodata = json.load(file)

    names_and_ids = {detail["station_id"]: detail["station_name"] for detail in geodata}
    with open(save_path / f"{scenario}_stations_and_ids.json", mode="w") as file:
        json.dump(names_and_ids, file)


def run_mixed_indexer(scenario: str, data: pd.DataFrame, delete_leftover_rows: bool, save: bool = True) -> pd.DataFrame:
    """
    Execute the full chain of functions in this module that culminates in the following outcomes:
        - Make a replacement for every existing ID because many of the IDs are long strings (see the preprocessing
        script for details).
        -

    Args:
        scenario (str): 
        data:
        delete_leftover_rows:
        save:

    Returns:
        pd.DataFrame: the data, but with all the station IDs re-indexed
    """
    logger.info("Initiating reindexing procedure for the station IDs...")
    data_with_replaced_ids_and_names = match_names_and_ids_by_station_proximity(scenario=scenario, data=data)

    leftover_row_indices: list[int] = find_rows_with_missing_ids_and_names(
        scenario=scenario,
        data=data_with_replaced_ids_and_names,
        return_indices=True,
        first_time=False 
    )

    is_an_unproblematic_row = np.isin(
        element=data_with_replaced_ids_and_names.index, test_elements=leftover_row_indices, invert=True
    )

    unproblematic_data = data_with_replaced_ids_and_names.loc[is_an_unproblematic_row, :]

    if delete_leftover_rows:
        logger.warning(f"Discarding the {len(leftover_row_indices)} rows that still have no station IDs and names.")
        logger.info("Providing new indices to each station in the rest of the data")

        station_id_index = unproblematic_data.columns.get_loc(f"{scenario}_station_id")
        station_ids = unproblematic_data.iloc[:, station_id_index]
        unique_old_ids = station_ids.unique()
        
        # Use the indices of this enumeration as the new station IDs
        old_and_new_ids = {old_id: index for index, old_id in enumerate(unique_old_ids)}
        data.iloc[:, station_id_index] = station_ids.map(old_and_new_ids)
        unproblematic_data = data.reset_index(drop=True)

        for column in unproblematic_data.select_dtypes(include=["datetime64[ns]"]):
            unproblematic_data[column] = unproblematic_data[column].astype(str)

        save_geodata(data=unproblematic_data, scenario=scenario, for_plotting=False)
        save_geodata(data=unproblematic_data, scenario=scenario, for_plotting=True)

        unproblematic_data = unproblematic_data.drop(
            columns=[f"{scenario}_lat", f"{scenario}_lng", f"{scenario}_station_name"]
        )

        if save:
            unproblematic_data.to_parquet(path=CLEANED_DATA / f"fully_cleaned_and_indexed_{scenario}_data.parquet")

        return unproblematic_data

    else:
        logger.warning("Initiating a reverse geocoding procedure to save the leftover rows from deletion")

        remaining_problem_data: list[int] = find_rows_with_missing_ids_and_names(
            scenario=scenario,
            data=data_with_replaced_ids_and_names,
            return_indices=False,
            first_time=False
        )

        problem_data_with_rounded_coordinates: pd.DataFrame = add_column_of_rounded_coordinates(
            scenario=scenario,
            data=remaining_problem_data,
            drop_original_coordinates=False,
            decimal_places=6  # No rounding. 
        )

        unproblematic_data = add_column_of_rounded_coordinates(
            scenario=scenario,
            data=unproblematic_data,
            drop_original_coordinates=False,
            decimal_places=6
        )

        geocoder = ReverseGeocoder(scenario=scenario, data=problem_data_with_rounded_coordinates)
        data_with_new_names = geocoder.reverse_geocode_rounded_coordinates(using_mixed_indexer=True)

        all_data = pd.concat(
            [unproblematic_data, data_with_new_names], axis=0
        )
        
        station_names_and_new_ids = {
            station_name: new_id for new_id, station_name in enumerate(all_data[f"{scenario}_station_name"].unique())
        }

        all_data[f"{scenario}_station_id"] = all_data[f"{scenario}_station_name"].map(station_names_and_new_ids)

        save_geodata(data=all_data, scenario=scenario, for_plotting=False)
        save_geodata(data=all_data, scenario=scenario, for_plotting=True)
        make_json_of_names_and_ids(scenario=scenario)

        all_data = all_data.drop(
            [f"{scenario}_lat", f"{scenario}_lng", f"{scenario}_station_name"], axis=1
        )

        if save:
            all_data.to_parquet(path=CLEANED_DATA / f"fully_cleaned_and_indexed_{scenario}_data.parquet")

        return all_data
