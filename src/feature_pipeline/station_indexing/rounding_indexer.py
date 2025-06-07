"""
The contents of this module are only to be used if the data being processed is so voluminous that it
poses a problem (in terms of memory and time), such that the compromise of geographical accuracy
resulting from its use can be justified.
"""
import json
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from loguru import logger

from src.setup.paths import ROUNDING_INDEXER


def add_column_of_rounded_coordinates(
    scenario: str,
    data: pd.DataFrame,
    decimal_places: int | None,
    drop_original_coordinates: bool
) -> pd.DataFrame:
    """
    This function takes the latitude and longitude columns of a dataframe, rounds them down to a specified 
    number of decimal places, and makes a column which consists of points containing the rounded latitudes 
    and longitudes.

    Args:
    decimal_places (int): the number of decimal places to which we will round the coordinates. The original 
                        coordinates are written in 6 decimal places. For each decimal place that is lost, 
                        the accuracy of the coordinates degrades by a factor of 10 meters.
                        

    scenario (str): whether we are looking at departures ("start") or arrivals ("end").
    drop_original_coordinates (bool): whether to delete the columns that contain the original coordinates
    """
    logger.info(f"Approximating the coordinates of the location where each trip {scenario}s...")
    rounded_latitudes = np.round(data[f"{scenario}_lat"], decimals=decimal_places)
    rounded_longitudes = np.round(data[f"{scenario}_lng"], decimals=decimal_places)
    rounded_coordinates = [coordinate for coordinate in zip(rounded_latitudes, rounded_longitudes)]
    
    data.insert(
        loc=data.shape[1],
        column=f"rounded_{scenario}_coordinates",
        value=rounded_coordinates,
        allow_duplicates=False
    )

    if drop_original_coordinates:
        data = data.drop(
            columns=[f"{scenario}_lat", f"{scenario}_lng"]
        )
    
    return data


def make_station_ids_from_unique_coordinates(scenario: str, data: pd.DataFrame) -> dict[float, int]:
    """
    This function makes a list of random numbers for each unique point, and 
    associates each point with a corresponding number. This effectively creates new 
    IDs for each location.
    """
    logger.info("Matching up approximate locations with generated IDs...")

    unique_coordinates = data[f"rounded_{scenario}_points"].unique()
    num_unique_points = len(unique_coordinates)

    # Set a seed to ensure reproducibility. Come on...what other number would I choose?
    random.seed(69)

    # Make a random mixture of the numbers from 0 to len(num_unique_points) 
    new_station_ids = random.sample(population=range(num_unique_points), k=num_unique_points)

    # Make a dictionary of points
    points_and_new_ids = {}

    for point, new_station_id in tqdm(zip(unique_coordinates, new_station_ids)):
        points_and_new_ids[point] = new_station_id

    # Because tuples can't be keys of a dictionary
    swapped_dict = {station_id: point for point, station_id in points_and_new_ids.items()}
    with open(ROUNDING_INDEXER / f"rounded_{scenario}_points_and_new_ids.json", mode="w") as file:
        json.dump(swapped_dict, file)

    return points_and_new_ids


def run_rounding_indexer(scenario: str, data: pd.DataFrame, decimal_places: int) -> pd.DataFrame:
    """
    Take each point, and the ID which corresponds to it, and put those IDs in the
    relevant dataframe (in a manner that matches each point with its ID row-wise).
    """
    data = data.drop(f"{scenario}_station_id", axis=1)
    data = add_column_of_rounded_coordinates(   
        data=data, 
        scenario=scenario, 
        decimal_places=decimal_places, 
        drop_original_coordinates=True
    )

    points_and_ids = make_station_ids_from_unique_coordinates(scenario=scenario, data=data)

    new_station_ids = [
        points_and_ids[point] for point in list(data.loc[:, f"rounded_{scenario}_points"]) if
        point in points_and_ids.keys()
    ]

    data.insert(
        loc=data.shape[1],
        column=f"{scenario}_station_id",
        value=pd.Series(new_station_ids),
        allow_duplicates=False
    )

    return data
