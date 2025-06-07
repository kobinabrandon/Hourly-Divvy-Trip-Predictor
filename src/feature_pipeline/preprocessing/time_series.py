import pandas as pd
from pathlib import Path
from loguru import logger


def transform_cleaned_data_into_ts_data(
            self,
            start_df: pd.DataFrame,
            end_df: pd.DataFrame,
            save: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Converts cleaned data into time series data.

        src.feature_pipeline.In addition to the putting the arrival and departure times in hourly form, we approximate
        the latitudes and longitudes of each point of origin or destination (we are targeting
        no more than a 100m radius of each point, but ideally we would like to maintain a 10m
        radius), and use these to construct new station IDs.

        Args:
            start_df (pd.DataFrame): dataframe of departure data

            end_df (pd.DataFrame): dataframe of arrival data

            save (bool): whether we wish to save the generated time series data

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: the time series datasets on arrivals or departures
        """

        def _get_ts_or_begin_transformation(start_ts_path: str, end_ts_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

            if Path(start_ts_path).exists() and Path(end_ts_path).exists():
                logger.success("Both time series datasets are already present")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif not Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("Neither time series dataset exists")
                start_ts, end_ts = _begin_transformation(missing_scenario="both")
                return start_ts, end_ts

            elif not Path(start_ts_path).exists() and Path(end_ts_path).exists():
                logger.warning("No time series dataset for departures has been made")
                start_ts = _begin_transformation(missing_scenario="start")
                end_ts = pd.read_parquet(path=end_ts_path)
                return start_ts, end_ts

            elif Path(start_ts_path).exists() and not Path(end_ts_path).exists():
                logger.warning("No time series dataset for arrivals has been made")
                start_ts = pd.read_parquet(path=start_ts_path)
                end_ts = _begin_transformation(missing_scenario="end")
                return start_ts, end_ts


def __aggregate_final_ts(interim_data: pd.DataFrame, start_or_end: str) -> pd.DataFrame | list[pd.DataFrame, pd.DataFrame]:

    #  if self.use_custom_station_indexing(data=self.data, scenarios=[start_or_end]) and \
    #        self.tie_ids_to_unique_coordinates(data=self.data):

    #    interim_data = interim_data.drop(f"rounded_{start_or_end}_points", axis=1)

    logger.info(f"Aggregating the final time series data for the {config.displayed_scenario_names[start_or_end].lower()}...")

    agg_data = interim_data.groupby(
        [f"{start_or_end}_hour", f"{start_or_end}_station_id"]).size().reset_index()

    agg_data = agg_data.rename(columns={0: "trips"})
    return agg_data




