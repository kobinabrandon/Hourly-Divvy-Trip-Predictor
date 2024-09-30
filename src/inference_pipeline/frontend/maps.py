"""
This experimental module contains code that:
 - displays the locations of the various stations on an interactive map. It has been challenging to create an 
   implementation of this that produces a good experience.
 - loads geospatial data using shapefiles and makes it available for mapping
"""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from loguru import logger
from datetime import timedelta

from src.setup.config import config 
from src.inference_pipeline.frontend.main import ProgressTracker
from src.inference_pipeline.frontend.data import make_geodataframes, reconcile_geodata
from src.inference_pipeline.frontend.predictions import retrieve_predictions, retrieve_predictions_for_this_hour


def restrict_geodataframe_to_stations_with_predictions(
    scenario: str,
    predictions: pd.DataFrame,
    geo_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    
    Args:
        scenario (str): _description_
        geo_dataframe (pd.DataFrame): _description_
        predictions (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    stations_we_have_predictions_for = predictions[f"{scenario}_station_name"].unique()

    predictions_are_present = np.isin(
        element=geo_dataframe[f"station_name"],
        test_elements=stations_we_have_predictions_for
    )

    number_present = [boolean for boolean in predictions_are_present if boolean == True]

    logger.warning(
        f"{len(geo_dataframe) - len(number_present)} stations won't be plotted due to a lack of predictions"
    )

    return geo_dataframe.loc[predictions_are_present, :]


def pseudocolour(
    value: float,
    min_value: float,
    max_value: float,
    start_colour: tuple[int, int, int],
    stop_colour: tuple[int, int, int]
) -> tuple[float, ...]:
    """
    Use linear interpolation to convert a given value into a tuple(representing a colour) in the range between
    start_colour and stop_colour. Credit to https://stackoverflow.com/a/10907855

    Args:
        value: the input value to be converted into a tuple that represents a colour
        min_value: the smallest value in the range of available input values
        max_value: the largest value in the range of available input values
        start_colour: a tuple representing the RGB values of the colour on the extreme left of the colour scale
        stop_colour:a tuple representing the RGB values of the colour on the extreme right of the colour scale

    Returns:

    """
    relative_value =  float(value-min_value)/(max_value-min_value)
    return tuple(
        relative_value*(b-a) + a for (a,b) in zip(start_colour, stop_colour)
    )


def perform_colour_scaling(
    start_geodataframe: pd.DataFrame,
    end_geodataframe: pd.DataFrame,
    predicted_starts: pd.DataFrame, 
    predicted_ends: pd.DataFrame
) -> pd.DataFrame:
    """
    Feed each of the predictions arrivals and departures into the pseudocolour function in order to produce the colour
    scaling effect.

    Args:
        geo_dataframes: a list of which will contain the geodataframes for arrivals and departures.
        predictions: the dataframe of predictions obtained from the feature store.

    Returns:
        list[pd.DataFrame]: a list of dataframes consisting of the merged external geodata (from the shapefile) and 
                            the predictions from the feature store.
    """
    black, green = (0, 0, 0), (0, 255, 0)
    geographical_features_and_predictions = []
    scenarios_and_geodataframes = {"start": start_geodataframe, "end": end_geodataframe}
    scenarios_and_predictions = {"start": predicted_starts, "end": predicted_ends}

    for scenario in config.displayed_scenario_names.keys():
        predictions = scenarios_and_predictions[scenario]
        predictions = predictions.rename(columns={f"{scenario}_station_name": "station_name"})

        geo_dataframe = scenarios_and_geodataframes[scenario]
        logger.info("Merging geographical details and predictions for ", config.displayed_scenario_names[scenario].lower())

        merged_data = pd.merge(left=geo_dataframe, right=predictions, left_on="station_name", right_on="station_name")
        max_prediction, min_prediction = merged_data[f"predicted_{scenario}s"].max(), merged_data[f"predicted_{scenario}s"].min()

        merged_data[f"{scenario}_fill_colour"] = merged_data[f"predicted_{scenario}s"].apply(
            lambda x: pseudocolour(
                value=x,
                min_value=min_prediction,
                max_value=max_prediction,
                start_colour=black,
                stop_colour=green
            )
        )

        merged_data[f"coordinates"] = merged_data[f"coordinates"].apply(tuple)

        from src.setup.paths import DATA_DIR
        merged_data.to_parquet(DATA_DIR/f"merge_{scenario}.parquet")

        geographical_features_and_predictions.append(merged_data)

    complete_merger = pd.merge(
        left=geographical_features_and_predictions[0], 
        right=geographical_features_and_predictions[1], 
        left_on="station_name", 
        right_on="station_name"
    )

    complete_merger["fill_colour"] = complete_merger.apply(
        func=lambda row: tuple((a+b) / 2 for a, b in zip(row["start_fill_colour"], row["end_fill_colour"])), 
        axis=1
    )

    complete_merger = complete_merger.drop(
        ["start_fill_colour", "end_fill_colour", "start_station_id", "end_station_id"], axis=1
    )
        
    return complete_merger
        

@st.cache_resource
def make_map(_geodataframe_and_predictions: pd.DataFrame) -> None:

    initial_view_state = pdk.ViewState(
        latitude=41.872866,
        longitude=-87.63363,
        zoom=12,
        max_zoom=20,
        pitch=45,
        bearing=0
    )

    layer = pdk.Layer(
        data=_geodataframe_and_predictions,
        type="ScatterplotLayer",    
        get_position=f"coordinates",
        get_fill_color=f"fill_colour",
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        get_radius=70,
        auto_highlight=True,
        pickable=True
    )

    tooltip = {
        "style": {"backgroundColor": "steelblue", "color": "white"},
        "html": "<b>Station:</b> {station_name} <br />\
                 <b>Predicted departures in the next hour:</b> {predicted_starts}\
                 <br /> <b>Predicted arrivals in the next hour:</b> {predicted_ends}"
    }

    map = pdk.Deck(
        layers=layer,
        initial_view_state=initial_view_state,
        map_style="mapbox://styles/mapbox/navigation-day-v1",
        tooltip=tooltip
    )

    st.pydeck_chart(pydeck_obj=map)


if __name__ != "__main__":
    tracker = ProgressTracker(n_steps=5)

    with st.spinner(text="Collecting station information"):
        start_geodataframe, end_geodataframe = make_geodataframes()
        tracker.next()

    with st.spinner(text=f"Fetching all predictions from the feature store"):
        predicted_starts, predicted_ends = retrieve_predictions()

        predicted_starts_this_hour, predicted_ends_this_hour = retrieve_predictions_for_this_hour(
            predicted_starts=predicted_starts,
            predicted_ends=predicted_ends,
            from_hour=config.current_hour-timedelta(hours=1),
            to_hour=config.current_hour
        )

        tracker.next()

    with st.spinner(text="Looking up the stations that we have predictions for"):
        start_geodataframe = restrict_geodataframe_to_stations_with_predictions(
            scenario="start", 
            predictions=predicted_starts_this_hour,
            geo_dataframe=start_geodataframe 
        )

        end_geodataframe = restrict_geodataframe_to_stations_with_predictions(
            scenario="end", 
            predictions=predicted_ends_this_hour,
            geo_dataframe=end_geodataframe
        )

        tracker.next()

    with st.spinner(text="Setting up ingredients for the map"):
        
        geographical_features_and_predictions = perform_colour_scaling(
            start_geodataframe=start_geodataframe,
            end_geodataframe=end_geodataframe,
            predicted_starts=predicted_starts,
            predicted_ends=predicted_ends
        )

        breakpoint()
        
        from src.setup.paths import DATA_DIR
        geographical_features_and_predictions.to_parquet(DATA_DIR/"merge.parquet")

        tracker.next()
        
    with st.spinner(text="Generating map of the Chicago area"):
        make_map(_geodataframe_and_predictions=geographical_features_and_predictions)
        tracker.next()

    st.sidebar.write("✅ Map Drawn")
