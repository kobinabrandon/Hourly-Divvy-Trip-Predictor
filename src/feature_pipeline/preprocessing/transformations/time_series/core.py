import json
import pandas as pd
from pathlib import Path
from loguru import logger

from src.setup.config import get_proper_scenario_name
from src.setup.paths import START_TS_PATH, END_TS_PATH, MIXED_INDEXER, TIME_SERIES_DATA
from src.feature_pipeline.preprocessing.station_indexing.choice import investigate_making_new_station_ids 


def transform_cleaned_data_into_ts(
        scenarios: list[str] | None, 
        cleaned_start_data: pd.DataFrame,
        cleaned_end_data:pd.DataFrame,
        using_custom_station_indexing: bool,
        tie_ids_to_unique_coordinates: bool,
        save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
        Converts cleaned data into time series data.

        In addition to the putting the arrival and departure times in hourly form, we approximate
        the latitudes and longitudes of each point of origin or destination (we are targeting
        no more than a 100m radius of each point, but ideally we would like to maintain a 10m
        radius), and use these to construct new station IDs.

        Args:
            scenarios: 
            cleaned_start_data (pd.DataFrame): dataframe of departure data
            cleaned_end_data (pd.DataFrame): dataframe of arrival data
            using_custom_station_indexing: 
            tie_ids_to_unique_coordinates: 
            save (bool): whether we wish to save the generated time series data

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on arrivals or departures
    """
    interim_dataframes: list[pd.DataFrame] = []
    match (using_custom_station_indexing, tie_ids_to_unique_coordinates):

        case (True, True):

            if scenarios == ["start", "end"]:
                
                for data, scenario in zip( [cleaned_start_data, cleaned_end_data], scenarios ):
                    # The coordinates are in 6 dp, so no rounding is happening here.
                    interim_data: list[pd.DataFrame] = investigate_making_new_station_ids(
                        cleaned_data=data, 
                        scenario=scenario,
                        using_custom_station_indexing=using_custom_station_indexing,
                        tie_ids_to_unique_coordinates=tie_ids_to_unique_coordinates
                    )

                    interim_dataframes.extend(interim_data)
                
                path_to_rounded_points_for_starts: Path = MIXED_INDEXER.joinpath("rounded_start_points_and_new_ids.json")
                path_to_rounded_points_for_ends: Path = MIXED_INDEXER.joinpath("rounded_ends_points_and_new_ids.json")

                with open(path_to_rounded_points_for_starts, mode="r") as file:
                    rounded_start_points_and_ids = json.load(file)

                with open(path_to_rounded_points_for_ends, mode="r") as file:
                    rounded_end_points_and_ids = json.load(file)

                # Get all the coordinates that are common to both dictionaries
                common_points: list[tuple[float, float]] = [
                    point for point in rounded_start_points_and_ids.keys() if point in
                    rounded_end_points_and_ids.keys()
                ]

                # Ensure that these common points have the same IDs in each dictionary.
                for point in common_points:
                    rounded_start_points_and_ids[point] = rounded_end_points_and_ids[point]

                start_ts: pd.DataFrame = aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                end_ts: pd.DataFrame = aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                if save:
                    start_ts.to_parquet(TIME_SERIES_DATA.joinpath("start_ts.parquet"))
                    end_ts.to_parquet(TIME_SERIES_DATA.joinpath("end_ts.parquet"))

                return start_ts, end_ts

            elif scenarios == ["start"] or ["end"]:

                scenario: str = scenarios[0]

                interim_data = investigate_making_new_station_ids(
                    scenario=scenario,
                    cleaned_data=cleaned_start_data if scenario == "start" else cleaned_end_data,
                    using_custom_station_indexing=using_custom_station_indexing,
                    tie_ids_to_unique_coordinates=tie_ids_to_unique_coordinates
                )

                ts_data = aggregate_final_ts(interim_data=interim_data, start_or_end=scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA.joinpath(f"{scenario}s_ts.parquet"))

                return ts_data

        case (True, False):

            if scenarios == ["start", "end"]:
                for data, scenario in zip( [cleaned_start_data, cleaned_end_data], ["start", "end"] ):
                    interim_data = investigate_making_new_station_ids(
                        cleaned_data=data, 
                        scenario=scenario,
                        using_custom_station_indexing=using_custom_station_indexing,
                        tie_ids_to_unique_coordinates=tie_ids_to_unique_coordinates
                    )
                    
                    interim_dataframes.extend(interim_data)

                start_ts: pd.DataFrame = aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                end_ts: pd.DataFrame = aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                if save:
                    start_ts.to_parquet(TIME_SERIES_DATA.joinpath("start_ts.parquet"))
                    end_ts.to_parquet(TIME_SERIES_DATA.joinpath("end_ts.parquet"))

                return start_ts, end_ts

            elif scenarios == ["start"] or ["end"]:

                data = cleaned_start_data if scenario == "start" else cleaned_end_data
                data = investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)
                ts_data = aggregate_final_ts(interim_data=data, start_or_end=scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA.joinpath(f"{scenario}_ts.parquet"))

                return ts_data



def get_ts_or_transform_cleaned_data_into_ts() -> tuple[pd.DataFrame, pd.DataFrame]:

    match (START_TS_PATH.exists(), END_TS_PATH.exists()):

        case (True, True):
            logger.success("Both time series datasets are already present")
            start_ts: pd.DataFrame = pd.read_parquet(path=START_TS_PATH)
            end_ts: pd.DataFrame = pd.read_parquet(path=END_TS_PATH)
            return start_ts, end_ts

        case (True, False):
            logger.warning("No time series dataset for arrivals has been made")
            start_ts: pd.DataFrame = pd.read_parquet(path=START_TS_PATH)
            end_ts: pd.DataFrame = transform_cleaned_data_into_ts(scenarios=["end"])
            return start_ts, end_ts

        case (False, True):
            logger.warning("No time series dataset for departures has been made")
            start_ts: pd.DataFrame = transform_cleaned_data_into_ts(scenarios=["start"])
            end_ts: pd.DataFrame = pd.read_parquet(path=END_TS_PATH)
            return start_ts, end_ts
        
        case (False, False):
            logger.warning("Neither time series dataset exists")
            start_ts, end_ts = transform_cleaned_data_into_ts(scenarios=["start", "end"])
            return start_ts, end_ts

       
def aggregate_final_ts(interim_data: pd.DataFrame, start_or_end: str) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:
 
    logger.info(f"Aggregating the final time series data for the {get_proper_scenario_name(scenario=start_or_end)}...")

    columns_to_group_by = [f"{start_or_end}_hour", f"{start_or_end}_station_id"]
    agg_data = interim_data.groupby(columns_to_group_by).size().reset_index()
    agg_data = agg_data.rename(columns={0: "trips"})
    return agg_data

