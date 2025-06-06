"""
This module contains code that creates a new feature view based on the feature groups
for features and predictions. This feature view will be referenced in order to get 
feature and prediction data for the purpose of monitoring model performance inside 
the streamlit frontend.
"""
import pandas as pd 
from datetime import datetime

from src.setup.config import config
from src.inference_pipeline.backend.feature_store import setup_feature_group


def load_predictions_and_historical_trips(
    scenario: str,
    model_name: str,
    from_date: datetime,
    to_date: datetime
) -> pd.DataFrame:
    """


    Args:
        scenario (str): _description_
        model_name (str): _description_
        from_date (datetime): _description_
        to_date (datetime): _description_

    Returns:
        pd.DataFrame: the data to be used fo
    """
    tuned_or_not = "tuned" if scenario == "start" else "untuned"
    from_ts = int(from_date.timestamp() * 1000)
    to_ts = int(to_date.timestamp() * 1000)

    #print(from_date)
    #print(to_ts)
    #breakpoint()

    arrivals_or_departures: str = config.displayed_scenario_names[scenario].lower()

    predictions_fg = setup_feature_group(
        scenario=scenario,
        description=f"predicting {arrivals_or_departures} - {tuned_or_not} {model_name}",
        name=f"{model_name}_{scenario}_predictions_feature_group",
        for_predictions=True,
        primary_key=[f"{scenario}_station_id"],
        version=6
    )

    historical_fg = setup_feature_group(
        description=f"Hourly time series data for {arrivals_or_departures.lower()}",
        primary_key=["timestamp", f"{scenario}_station_id"],
        version=config.feature_group_version,
        name=f"{scenario}_feature_group",
        for_predictions=False
    ) 

    query = (
        predictions_fg
        .select_all()
        .join(
            sub_query=historical_fg.select(features=[f"{scenario}_station_id", f"{scenario}_hour", "trips"]),
            on=[f'{scenario}_station_id'],
            prefix=None
        )
        .filter(predictions_fg[f"{scenario}_hour"] >= str(from_ts)) 
        .filter(predictions_fg[f"{scenario}_hour"] <= str(to_ts))
    )

    monitoring_feature_view = api.get_or_create_feature_view(
        feature_group=historical_fg,
        name=f"monitoring_feature_view_for_{arrivals_or_departures}",
        use_sub_query=True,
        sub_query=query,
        version=1
    )

    monitoring_data = monitoring_feature_view.get_batch_data(
        start_time=from_date, 
        end_time=to_date, 
        read_options={"use_hive": True}
    )


    return monitoring_data[
        monitoring_data[f"{scenario}_hour"].between(from_ts, to_ts)
    ]
  
