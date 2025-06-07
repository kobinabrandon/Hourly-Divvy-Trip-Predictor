import os
import requests
from pathlib import Path

import pandas as pd
from loguru import logger
from datetime import datetime

from src.setup.config import config
from src.feature_pipeline.mixed_indexer import run_mixed_indexer
from src.feature_pipeline.data_sourcing import load_raw_data, Year
from src.feature_pipeline.rounding_indexer import run_rounding_indexer

from src.setup.paths import CLEANED_DATA, TRAINING_DATA, make_fundamental_paths


def investigate_making_new_station_ids(cleaned_data: pd.DataFrame, start_or_end: str) -> pd.DataFrame:
    """
    In an earlier version of the project, I ran into memory issues for two reasons:
        1) I was dealing with more than a year's worth of data. 
        2) there were bugs embedded in the feature pipeline.

    As a result, I decided to match approximations of each coordinate with an ID of 
    my own making, thereby reducing the size of the dataset during the aggregation
    stages of the creation of the training data. This worked well. However, I am no
    longer in need of any memory conservation measures because I have reduced the size of 
    the dataset.

    So why is the code still being used? Why not simply use the original IDs? You 
    may be thinking that perchance there weren't even many missing values, and that 
    perhaps all this could have been avoided.

    There's code in what immediately follows that checks for the presence of both long
    string indices (see the very first method of the class) and missing ones. If the proportion
    of rows that feature such indices exceeds a certain hardcoded threshold (I chose 50%),
    we will use the custom procedures (again see the aforementioned class method).

    As of late July 2024, 60% of the IDs (for origin and destination stations) have long strings
    or missing indices. It is therefore unlikely that we will need an alternative method. However, 
    I will write one eventually. Most likely it will involve simply applying the custom procedure to only
    that problematic minority of indices, to generate new integer indices that aren't already in the column.

    Args:
        cleaned_data (pd.DataFrame): the version of the dataset that has been cleaned
        start_or_end (str): whether we are looking at arrivals or departures.

    Returns:
        pd.DataFrame:
    """

    interim_dataframes: list[pd.DataFrame] = []

    logger.info(f"Recording the hour during which each trip {start_or_end}s...")
    cleaned_data.insert(
        loc=cleaned_data.shape[1],
        column=f"{start_or_end}_hour",
        value=cleaned_data.loc[:, f"{start_or_end}ed_at"].dt.floor("h"),
        allow_duplicates=False
    )

    cleaned_data = cleaned_data.drop(f"{start_or_end}ed_at", axis=1)
    logger.info("Determining the method of dealing with invalid station indices...")

    if self.use_custom_station_indexing(scenarios=[start_or_end], data=self.data) and \
    self.tie_ids_to_unique_coordinates(data=self.data):

        logger.warning("Custom station indexer required: tying new station IDs to unique coordinates")
        interim_data = run_rounding_indexer(data=cleaned_data, scenario=start_or_end, decimal_places=6)
        interim_dataframes.append(interim_data)

        return interim_data

    elif self.use_custom_station_indexing(scenarios=[start_or_end], data=self.data) and \
            not self.tie_ids_to_unique_coordinates(data=self.data):

        logger.warning("Custom station indexer required: NOT tying new IDs to unique coordinates")
        interim_data = run_mixed_indexer(
            scenario=start_or_end,
            data=cleaned_data,
            delete_leftover_rows=False
        )

        interim_dataframes.append(interim_data)
        return interim_data

    else:
        raise NotImplementedError(
            "The majority of Divvy's IDs weren't numerical and valid during initial development."
        )





def get_proper_scenario_name(scenario: str) -> str:
    return config.displayed_scenario_names[scenario].lower()


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
            

    
def retrieve_data(years: list[Year], for_inference: bool) -> pd.DataFrame | None: 
    return load_raw_data(years=years) if not for_inference else None  # Because the data will have been fetched from the feature store instead.



def make_training_data(geocode: bool) -> list[pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract raw data, clean it, transform it into time series data, and transform that time series data into
    training data. 

    Args:
        geocode (bool): whether to geocode as part of feature engineering.
        
    Returns:
        list[pd.DataFrame]: a list containing the datasets for the starts and ends of trips.
    """
    start_ts, end_ts = self.make_time_series()
    ts_data_per_scenario = { "start": start_ts, "end": end_ts }

    training_sets: list[pd.DataFrame] = []
    for scenario in ts_data_per_scenario.keys():
        path_to_training_data = TRAINING_DATA.joinpath(f"{scenario}s.parquet")

        if path_to_training_data.is_file():
            logger.warning(f"Found an existing version of the training data for {config.displayed_scenario_names[scenario]} -> Deleting it")
            os.remove(path_to_training_data)

        training_data: pd.DataFrame = self.transform_ts_into_training_data(
            ts_data=ts_data_per_scenario[scenario],
            geocode=geocode,
            scenario=scenario, input_seq_len=config.n_features, step_size=1)

        training_sets.append(training_data)
        
    return training_sets


def make_time_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the transformation of the raw data into time series data 

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the time series datasets for departures and arrivals respectively.
    """
    logger.info("Cleaning downloaded data...")

    # if cleaned_data_needs_update(scenario=self.scenario, path: Path, year_of_interest: Year)

    self.data = self.clean()

    start_df_columns = ["started_at", "start_lat", "start_lng", "start_station_id"]
    end_df_columns = ["ended_at", "end_lat", "end_lng", "end_station_id"]

    if self.use_custom_station_indexing(scenarios=["start", "end"], data=self.data) and not \
    self.tie_ids_to_unique_coordinates(data=self.data):

        start_df_columns.append("start_station_name")
        end_df_columns.append("end_station_name")

    start_df: pd.DataFrame = self.data[start_df_columns]
    end_df: pd.DataFrame = self.data[end_df_columns]

    start_ts, end_ts = self.transform_cleaned_data_into_ts_data(start_df=start_df, end_df=end_df)
    return start_ts, end_ts


def clean(self, save: bool = True) -> pd.DataFrame:

    tie_ids_to_unique_coordinates: bool = self.tie_ids_to_unique_coordinates(data=self.data)
    using_custom_station_indexing: bool = self.use_custom_station_indexing(scenarios=self.scenarios, data=self.data)  

    if using_custom_station_indexing and tie_ids_to_unique_coordinates:
        cleaned_data_file_path = CLEANED_DATA.joinpath("data_with_newly_indexed_stations (rounded_indexer).parquet")

    elif using_custom_station_indexing and not self.tie_ids_to_unique_coordinates(data=self.data):
        cleaned_data_file_path = CLEANED_DATA.joinpath("data_with_newly_indexed_stations (mixed_indexer).parquet")

    # Will think of a more elegant solution in due course. This only serves my current interests.
    elif self.for_inference:
        cleaned_data_file_path = CLEANED_DATA.joinpath("partially_cleaned_data_for_inference.parquet")

    else:
        raise NotImplementedError(
            "The majority of Divvy's IDs weren't numerical and valid during initial development."
        )

    if not cleaned_data_file_path.is_file():

        self.data["started_at"] = pd.to_datetime(self.data["started_at"], format="mixed")
        self.data["ended_at"] = pd.to_datetime(self.data["ended_at"], format="mixed")

        def delete_rows_with_missing_station_names_and_coordinates() -> pd.DataFrame:
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
            for scenario in self.scenarios:
                lats = self.data.columns.get_loc(f"{scenario}_lat")
                longs = self.data.columns.get_loc(f"{scenario}_lng")
                station_names_col = self.data.columns.get_loc(f"{scenario}_station_name")

                logger.info(
                    f"Deleting rows with missing station names & coordinates ({config.displayed_scenario_names[scenario]})"
                )

                where_missing_latitudes = self.data.iloc[:, lats].isnull()
                where_missing_longitudes = self.data.iloc[:, longs].isnull()
                where_missing_station_names = self.data.iloc[:, station_names_col].isnull()

                all_missing_mask = where_missing_station_names & where_missing_latitudes & where_missing_longitudes
                data_to_delete = self.data.loc[all_missing_mask, :]

                self.data = self.data.drop(data_to_delete.index, axis=0)

            return self.data

        self.data = delete_rows_with_missing_station_names_and_coordinates()
        features_to_drop = ["ride_id", "rideable_type", "member_casual"]

        if self.use_custom_station_indexing(data=self.data, scenarios=self.scenarios) and \
        self.tie_ids_to_unique_coordinates(data=self.data):

            features_to_drop.extend(
                ["start_station_id", "start_station_name", "end_station_name"]
            )

        self.data = self.data.drop(columns=features_to_drop)

        if save:
            self.data.to_parquet(path=cleaned_data_file_path)

        return self.data

    else:
        breakpoint()
        logger.success("There is already some cleaned data. Fetching it...")
        return pd.read_parquet(path=cleaned_data_file_path)


    
    
    
            
if __name__ == "__main__":
    make_fundamental_paths()
    processor= DataProcessor(years=config.years, for_inference=False)
    training_data = processor.make_training_data(geocode=False) 

