import os 
import subprocess
import pandas as pd 
from pathlib import Path 
from loguru import logger
from datetime import timedelta

from feast import Entity, Feature, FeatureView, FileSource, Field
from feast.types import Float32, Int64, String, UnixTimestamp, ValueType
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype, is_datetime64_any_dtype

from src.setup.config import config
from src.setup.paths import FEATURE_REPO_DATA, TIME_SERIES_DATA, INFERENCE_DATA, FEATURE_REPO


def convert_pandas_types(for_schema: bool, dtype: type):
    if is_integer_dtype(dtype):
        return Int64 if for_schema else ValueType.INT64
    elif is_float_dtype(dtype):
        return Float32 if for_schema else ValueType.FLOAT
    elif is_string_dtype(dtype):
        return String if for_schema else ValueType.STRING
    elif is_datetime64_any_dtype(dtype):
        return UnixTimestamp if for_schema else ValueType.UNIX_TIMESTAMP


def get_or_create_feature_view(scenario: str, for_predictions: bool, file_source: FileSource) -> FeatureView:
    """
    Creates or alternatively retrieves a feature view using the provided details. If a sub-query is to be used, 
    that has to be indicated, and the sub-query is to be provided. Otherwise, all features will be selected for 
    retrieval from the associated feature group.

    Args:
        name: the name of the feature view to fetch or create
    
    Returns:
        FeatureView: the desired feature view
    """
    if for_predictions:
        data_path = INFERENCE_DATA/f"{scenario}_predictions.parquet"
        feature_view_name = f"{scenario}_predictions"
    else:       
        data_path = TIME_SERIES_DATA/f"{scenario}_ts.parquet"
        feature_view_name = f"{scenario}_ts"
        
    data: pd.DataFrame = pd.read_parquet(path=data_path)

    station_ids = Entity(name=f"{scenario}_station_id", join_keys=[f"{scenario}_station_id"])
    schema = [Field(name=column, dtype=convert_pandas_types(for_schema=True, dtype=data[column].dtype)) for column in data.columns]

    feature_view = FeatureView(name=feature_view_name, schema=schema, source=file_source, online=True, entities=[station_ids])

    return feature_view, station_ids


if __name__ == "__main__":
    for scenario in config.displayed_scenario_names.keys():
        get_or_create_feature_view(scenario=scenario, for_predictions=True)
        get_or_create_feature_view(scenario=scenario, for_predictions=False)
