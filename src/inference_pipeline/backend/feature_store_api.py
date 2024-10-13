"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import subprocess
import pandas as pd 
from pathlib import Path 
from loguru import logger
from datetime import timedelta

from feast import Entity, Feature, FeatureView, FileSource, Field
from feast.types import Float32, Int64, String, UnixTimestamp
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype, is_datetime64_any_dtype

from src.setup.config import config
from src.setup.paths import FEATURE_REPO_DATA, TIME_SERIES_DATA, INFERENCE_DATA


def convert_pandas_types(dtype: type):
    if is_integer_dtype(dtype):
        return Int64
    elif is_float_dtype(dtype):
        return Float32
    elif is_string_dtype(dtype):
        return String
    elif is_datetime64_any_dtype(dtype):
        return UnixTimestamp 


def get_or_create_feature_view(scenario: str, for_predictions: bool, batch_source: FileSource) -> FeatureView:
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
        feature_view_name = f"{scenario}_features"
        
    data: pd.DataFrame = pd.read_parquet(path=data_path)
    columns_and_dtypes = {col: convert_pandas_types(data[col].dtype) for col in data.columns}

    schema = [Field(name=col, dtype=columns_and_dtypes[col]) for col in data.columns]
    entity = Entity(name=f"{scenario}_station_id", join_keys=[f"{scenario}_station_id"])

    feature_view = FeatureView(
        name=feature_view_name,
        entities=[entity],
        schema=schema,
        ttl=timedelta(days=0),
        source=batch_source,
        online=True
    )

    subprocess.run(["feast", "apply"])
    return feature_view


if __name__ == "__main__":
    for scenario in config.displayed_scenario_names.keys():
        get_or_create_feature_view(scenario=scenario, for_predictions=True)
        get_or_create_feature_view(scenario=scenario, for_predictions=False)
