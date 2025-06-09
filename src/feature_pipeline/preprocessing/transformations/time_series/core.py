import json
import pandas as pd
from loguru import logger

from src.setup.config import get_proper_scenario_name
from src.setup.paths import START_TS_PATH, END_TS_PATH, MIXED_INDEXER, TIME_SERIES_DATA

from src.feature_pipeline.preprocessing.station_indexing.choice import investigate_making_new_station_ids 
from src.feature_pipeline.preprocessing.station_indexing.choice import check_if_we_tie_ids_to_unique_coordinates, check_if_we_use_custom_station_indexing


def transform_cleaned_data_into_ts(
        data: pd.DataFrame, 
        scenarios: list[str] | None, 
        start_df: pd.DataFrame,
        end_df:pd.DataFrame,
        save: bool
) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:

    match ( 
        check_if_we_tie_ids_to_unique_coordinates(data=data),
        check_if_we_use_custom_station_indexing(scenarios=scenarios, data=data)
    ):

        case (True, True):

            if scenarios == ["start", "end"]:

                for data, scenario in zip( [start_df, end_df], scenarios ):
                    # The coordinates are in 6 dp, so no rounding is happening here.
                    investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                with open(MIXED_INDEXER / "rounded_start_points_and_new_ids.json", mode="r") as file:
                    rounded_start_points_and_ids = json.load(file)

                with open(MIXED_INDEXER / "rounded_end_points_and_new_ids.json", mode="r") as file:
                    rounded_end_points_and_ids = json.load(file)

                # Get all the coordinates that are common to both dictionaries
                common_points: list[tuple[float, float]] = [
                    point for point in rounded_start_points_and_ids.keys() if point in
                    rounded_end_points_and_ids.keys()
                ]

                # Ensure that these common points have the same IDs in each dictionary.
                for point in common_points:
                    rounded_start_points_and_ids[point] = rounded_end_points_and_ids[point]

                start_ts = aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                end_ts = aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                if save:
                    start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                    end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                return start_ts, end_ts

            elif scenario == "start" or "end":
                
                interim_data = investigate_making_new_station_ids(
                    cleaned_data=start_df if scenario == "start" else end_df, 
                    start_or_end=scenario
                )

                ts_data = aggregate_final_ts(interim_data=interim_data, start_or_end=scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA / f"{scenario}s_ts.parquet")

                return ts_data

        case (True, False):

            if scenario == "both":
                for data, scenario in zip( [start_df, end_df], ["start", "end"] ):
                    interim_dataframes = investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                start_ts = aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                end_ts = aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                if save:
                    start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                    end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                return start_ts, end_ts

            elif scenario == "start" or "end":

                data = start_df if scenario == "start" else end_df
                data = investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)
                ts_data = aggregate_final_ts(interim_data=data, start_or_end=scenario)

                if save:
                    ts_data.to_parquet(TIME_SERIES_DATA / f"{scenario}_ts.parquet")

                return ts_data



def get_ts_or_transform_cleaned_data_into_ts() -> tuple[pd.DataFrame, pd.DataFrame]:

    match (START_TS_PATH.exists(), END_TS_PATH.exists()):

        case (True, True):
            logger.success("Both time series datasets are already present")
            start_ts = pd.read_parquet(path=START_TS_PATH)
            end_ts = pd.read_parquet(path=END_TS_PATH)
            return start_ts, end_ts

        case (True, False):
            logger.warning("No time series dataset for arrivals has been made")
            start_ts = pd.read_parquet(path=START_TS_PATH)
            end_ts = transform_cleaned_data_into_ts(scenarios=["end"])
            return start_ts, end_ts

        case (False, True):
            logger.warning("No time series dataset for departures has been made")
            start_ts = transform_cleaned_data_into_ts(scenarios=["start"])
            end_ts = pd.read_parquet(path=END_TS_PATH)
            return start_ts, end_ts
        
        case (False, False):
            logger.warning("Neither time series dataset exists")
            start_ts, end_ts = transform_cleaned_data_into_ts(scenarios=["start", "end"])
            return start_ts, end_ts

       
# def transform_cleaned_data_into_ts_data(
#         self,
#         start_df: pd.DataFrame,
#         end_df: pd.DataFrame,
#         save: bool = True,
#     ) -> tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Converts cleaned data into time series data.
#
#         In addition to the putting the arrival and departure times in hourly form, we approximate
#         the latitudes and longitudes of each point of origin or destination (we are targeting
#         no more than a 100m radius of each point, but ideally we would like to maintain a 10m
#         radius), and use these to construct new station IDs.
#
#         Args:
#             start_df (pd.DataFrame): dataframe of departure data
#
#             end_df (pd.DataFrame): dataframe of arrival data
#
#             save (bool): whether we wish to save the generated time series data
#
#         Returns:
#             tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on arrivals or departures
#         """
#

        
def aggregate_final_ts(interim_data: pd.DataFrame, start_or_end: str) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:

    #  if self.use_custom_station_indexing(data=self.data, scenarios=[start_or_end]) and \
    #        self.tie_ids_to_unique_coordinates(data=self.data):

    #    interim_data = interim_data.drop(f"rounded_{start_or_end}_points", axis=1)

    logger.info(f"Aggregating the final time series data for the {get_proper_scenario_name(scenario=start_or_end)}...")

    agg_data = interim_data.groupby(
        [f"{start_or_end}_hour", f"{start_or_end}_station_id"]).size().reset_index()

    agg_data = agg_data.rename(columns={0: "trips"})
    return agg_data




