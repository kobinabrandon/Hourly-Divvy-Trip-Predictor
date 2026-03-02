"""
Plot time series data for a given number of top stations.
"""

import numpy as np 
import pandas as pd 
import streamlit as st 
import plotly.express as px 

from tqdm import tqdm 
from plotly.graph_objects import Figure
from datetime import datetime, timedelta, UTC
from streamlit_extras.colored_header import colored_header

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, INFERENCE_DATA, GEOGRAPHICAL_DATA
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names
from src.inference_pipeline.backend.inference import fetch_time_series_and_make_features, get_feature_group_for_time_series


def get_station_name(station_id: int) -> str:
    """
    Look up the name of a station given that ID that it carries in the dataset in question.

    Args:
        station_id (int): the ID of the station of interest.

    Returns:
        str: the name of the station
    """
    with open(MIXED_INDEXER/"end_geodata.json", mode="r") as file:
        geodata = json.load(file)

    for station in tqdm(iterable=geodata, desc="Looking up the name of the station"):
        if station["station_id"] == station_id:
            return station["station_name"]


@st.cache_data
@st.cache_resource
def plot_for_one_station(
    scenario: str,
    station_name: int,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    display_title: bool | None = True,
    targets: pd.Series | None = None
) -> Figure:
    """
    Plot the time series data for the given station.

    Args:
        station_name (int): the name of the station whose data we want to plot 
        features (pd.DataFrame): _description_
        predictions (pd.DataFrame): _description_
        display_title (str, optional): _description_. Defaults to None.
        targets (pd.Series | None, optional): _description_. Defaults to None.
    """
    station_features: pd.DataFrame = features[features[f"{scenario}_station_name"] == station_name]
    station_targets = targets[targets["station_name"] == station_name] if targets is not None else None

    columns_of_past_trips = [column for column in station_features.columns if column.startswith("trips_previous_")]
    cumulative_trips = [station_features[column].iloc[0] for column in columns_of_past_trips] + [station_targets[f"predicted_{scenario}s"].iloc[0]]
    trip_hour = station_features[f"{scenario}_hour"].iloc[0]

    all_dates = pd.date_range(
        start=trip_hour - timedelta(hours=len(columns_of_past_trips)),
        end=trip_hour,
        freq="h" 
    )

    st.subheader(f":blue[Station] Location: {station_name}" if display_title else None)
    figure = px.line(x=all_dates, y=cumulative_trips, markers=True)

    if targets is not None:
        figure.add_scatter(
            x=all_dates[-1:], y=[station_targets], line_color="green", mode="markers", name="Actual value"
        )

    if predictions is not None:
        station_prediction = predictions[predictions["station_name"] == station_name]  

        figure.add_scatter(
            x=all_dates[-1:], 
            y=[station_prediction], 
            line_color="red",
            marker_symbol="x", 
            mode="markers", 
            name="Prediction"
        )

    return figure


@st.cache_data
def load_features(start_date: datetime, target_date: datetime) -> list[pd.DataFrame, pd.DataFrame]:

    start_and_end_features = []
    for scenario in config.displayed_scenario_names.keys():
        primary_key = ["timestamp", f"{scenario}_station_id"]

        feature_group = get_feature_group_for_time_series(scenario=scenario, primary_key=primary_key)

        features = fetch_time_series_and_make_features(
            scenario=scenario,
            start_date=start_date, 
            target_date=target_date, 
            feature_group=feature_group,
            geocode=False
        )

        # Add station names to features
        ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
        features[f"{scenario}_station_name"] = features[f"{scenario}_station_id"].map(ids_and_names)
        start_and_end_features.append(features)

    return start_and_end_features



if __name__ != "__main__":

    colored_header(
        label="Plots of :blue[Trips] Over Time", 
        description="View plots of time series data of trips from the top 10 stations"
    )

    with st.spinner(text="Fetching features to be used for plotting"):        
        all_features = load_features(start_date = datetime.now() - timedelta(days=60), target_date=datetime.now())

    with st.spinner(text="Fetching the geographical data and the predictions for the next hour"):

        geographical_features_and_predictions = pd.read_parquet(
            path=INFERENCE_DATA/"geographical_features_and_predictions.parquet"
        )
    
    with st.spinner(text=" Preparing to generate plots using acquired data"):

        scenarios_and_features = {"start": all_features[0], "end": all_features[1]}
        
        for scenario in config.displayed_scenario_names.keys():
            
            # print(f"{scenario}_hour" in scenarios_and_features[scenario].columns)
            # breakpoint()
        
            row_indices = np.argsort(geographical_features_and_predictions[f"predicted_{scenario}s"].values)[::-1]
        
            for row_id in row_indices[:10]:
                station_name = geographical_features_and_predictions["station_name"].iloc[row_id]
                prediction = geographical_features_and_predictions[f"predicted_{scenario}s"].iloc[row_id]

                st.metric(
                    label=f"Predicted {config.displayed_scenario_names[scenario].lower()}", 
                    value=int(prediction)
                )

                fig = plot_for_one_station(
                    scenario=scenario,
                    station_name=station_name,
                    features=scenarios_and_features[scenario],
                    targets=geographical_features_and_predictions[[f"predicted_{scenario}s", "station_name"]],
                    predictions=geographical_features_and_predictions[[f"predicted_{scenario}s", f"{scenario}_hour", "station_name"]],
                    display_title=False
                )

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
