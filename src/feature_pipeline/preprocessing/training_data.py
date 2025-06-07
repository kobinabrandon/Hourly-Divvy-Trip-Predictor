import pandas as pd
from tqdm import tqdm 
from loguru import logger

from src.setup.config import config
from src.feature_pipeline.cutoff import CutoffIndexer
from src.feature_pipeline.feature_engineering import finish_feature_engineering

from src.setup.paths import (
    TRAINING_DATA, 
    MIXED_INDEXER, 
    INFERENCE_DATA, 
    TIME_SERIES_DATA, 
)

def transform_ts_into_training_data(
            self,
            geocode: bool,
            scenario: str,
            step_size: int,
            input_seq_len: int,
            ts_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transpose the time series data into a feature-target format.

        Args:
            scenario: a string that indicates whether we are dealing with the starts or ends of trips
            geocode: whether to use geocoding during feature engineering 
            step_size: the step size to be used by the standard cutoff indexer.
            input_seq_len: the input sequence length to be used to construct the training data
            ts_data: the full time series dataset for arrivals and departures

        Returns:
            pd.DataFrame: the training data for arrivals or departures
        """
        if self.for_inference and "timestamp" in ts_data.columns:
            ts_data = ts_data.drop("timestamp", axis=1)

        # Ensure first that these are the columns of the chosen data set (and they are listed in this order)
        assert set(ts_data.columns) == {f"{scenario}_hour", f"{scenario}_station_id", "trips"}

        features = pd.DataFrame()
        targets = pd.DataFrame()

        for station_id in tqdm(
            iterable=ts_data[f"{scenario}_station_id"].unique(), 
            desc=f"Turning time series data into training data ({config.displayed_scenario_names[scenario].lower()})"
        ):
            ts_per_station = ts_data.loc[
                ts_data[f"{scenario}_station_id"] == station_id, [f"{scenario}_hour", f"{scenario}_station_id", "trips"]
            ].sort_values(by=[f"{scenario}_hour"])

            cutoff_indexer = CutoffIndexer(ts_data=ts_per_station, input_seq_len=input_seq_len, step_size=step_size)
            use_standard_cutoff_indexer: bool = cutoff_indexer.use_standard_cutoff_indexer()
            indices = cutoff_indexer.indices
            num_indices = len(indices) 

            x = np.empty(shape=(num_indices, input_seq_len), dtype=np.float32)
            y = np.empty(shape=(num_indices, 1), dtype=np.float32)

            hours = []    
            if use_standard_cutoff_indexer:
                for i, index in enumerate(indices):
                    hour = ts_per_station.iloc[index[1]][f"{scenario}_hour"]
                    x[i, :] = ts_per_station.iloc[index[0]: index[1]]["trips"].values
                    y[i] = ts_per_station.iloc[index[2]]["trips"]
                    hours.append(hour)
            
            elif not use_standard_cutoff_indexer and len(ts_per_station) == 1:

                x[0, :] = np.full(
                    shape=(1, input_seq_len), 
                    fill_value=ts_per_station["trips"].iloc[0]
                )

                y[0] = ts_per_station["trips"].iloc[0]
                hour = ts_per_station[f"{scenario}_hour"].values[0]
                hours.append(hour)

            else:
                ts_per_station = ts_per_station.reset_index(drop=True)
                
                for i, index in enumerate(indices):
                    x[i, :] = ts_per_station.iloc[index[0]: index[1], 2].values
                    y[i] = ts_per_station[index[1]:index[2]]["trips"].values[0]
                    hour = ts_per_station.iloc[index[1]][f"{scenario}_hour"]
                    hours.append(hour)

            features_per_station = pd.DataFrame(
                data=x, 
                columns=[f"trips_previous_{i + 1}_hour" for i in reversed(range(input_seq_len))]
            )
            
            features_per_station[f"{scenario}_hour"] = hours  
            features_per_station[f"{scenario}_station_id"] = station_id
            targets_per_station = pd.DataFrame(data=y, columns=["trips_next_hour"])

            features = pd.concat([features, features_per_station], axis=0)
            targets = pd.concat([targets, targets_per_station], axis=0)

        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)

        engineered_features = finish_feature_engineering(features=features, scenario=scenario, geocode=geocode)
        training_data = pd.concat([engineered_features, targets["trips_next_hour"]], axis=1)

        logger.success("Saving the data so we (hopefully) won't have to do that again...")
        final_data_path = INFERENCE_DATA if self.for_inference else TRAINING_DATA
        training_data.to_parquet(final_data_path / f"{scenario}s.parquet")

        return training_data


def _begin_transformation(missing_scenario: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:

            
            indexer_two_scenarios = self.scenarios if missing_scenario == "both" else [missing_scenario]

            if self.use_custom_station_indexing(scenarios=indexer_two_scenarios, data=self.data) and \
                    self.tie_ids_to_unique_coordinates(data=self.data):

                if missing_scenario == "both":

                    for data, scenario in zip([start_df, end_df], self.scenarios):
                        # The coordinates are in 6 dp, so no rounding is happening here.
                        __investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                    with open(MIXED_INDEXER / "rounded_start_points_and_new_ids.json", mode="r") as file:
                        rounded_start_points_and_ids = json.load(file)

                    with open(MIXED_INDEXER / "rounded_end_points_and_new_ids.json", mode="r") as file:
                        rounded_end_points_and_ids = json.load(file)

                    # Get all the coordinates that are common to both dictionaries
                    common_points = [
                        point for point in rounded_start_points_and_ids.keys() if point in
                        rounded_end_points_and_ids.keys()
                    ]

                    # Ensure that these common points have the same IDs in each dictionary.
                    for point in common_points:
                        rounded_start_points_and_ids[point] = rounded_end_points_and_ids[point]

                    start_ts = __aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                    end_ts = __aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                    if save:
                        start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                        end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                    return start_ts, end_ts

                elif missing_scenario == "start" or "end":
                    
                    data = start_df if missing_scenario == "start" else end_df
                    data = __investigate_making_new_station_ids(cleaned_data=data, start_or_end=missing_scenario)
                    ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                    if save:
                        ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}s_ts.parquet")

                    return ts_data

            elif self.use_custom_station_indexing(scenarios=indexer_two_scenarios, data=self.data) and \
                    not self.tie_ids_to_unique_coordinates(data=self.data):

                if missing_scenario == "both":
                    for data, scenario in zip( [start_df, end_df], ["start", "end"] ):
                        __investigate_making_new_station_ids(cleaned_data=data, start_or_end=scenario)

                    start_ts = __aggregate_final_ts(interim_data=interim_dataframes[0], start_or_end="start")
                    end_ts = __aggregate_final_ts(interim_data=interim_dataframes[1], start_or_end="end")

                    if save:
                        start_ts.to_parquet(TIME_SERIES_DATA / "start_ts.parquet")
                        end_ts.to_parquet(TIME_SERIES_DATA / "end_ts.parquet")

                    return start_ts, end_ts

                elif missing_scenario == "start" or "end":

                    data = start_df if missing_scenario == "start" else end_df
                    data = __investigate_making_new_station_ids(cleaned_data=data, start_or_end=missing_scenario)
                    ts_data = __aggregate_final_ts(interim_data=data, start_or_end=missing_scenario)

                    if save:
                        ts_data.to_parquet(TIME_SERIES_DATA / f"{missing_scenario}_ts.parquet")

                    return ts_data

        return _get_ts_or_begin_transformation(start_ts_path=self.start_ts_path, end_ts_path=self.end_ts_path)


