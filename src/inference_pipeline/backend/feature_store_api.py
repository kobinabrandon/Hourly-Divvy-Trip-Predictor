"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import time
import boto3
import sagemaker 
import pandas as pd
from loguru import logger

from sagemaker.feature_store.feature_group import FeatureGroup

from src.setup.config import config


session = sagemaker.Session()
region = session.boto_region_name
prefix = "divvy_features"   


def setup_feature_group(
    scenario: str, 
    data: pd.DataFrame,
    primary_key: list[str] | None, 
    description: str, 
    for_predictions: bool
    ) -> FeatureGroup:
    """
    Create or connect to a feature group with the specified name, and return an object that represents it.

    Returns:
        FeatureGroup: a representation of the fetched or created feature group
    """
    feature_group = FeatureGroup()
    feature_group_name = f"predictied_{scenario}s" if for_predictions else f"{scenario}_feature_group"
    
    data[f"{scenario}_hour"] = data[f"{scenario}_hour"].dt.strftime('%Y-%m-%d %H:%M:%S').astype(str)
    feature_group.load_feature_definitions(data_frame=data)

    feature_group.create(
        s3_uri=f's3://{session.default_bucket()}/{prefix}',
        enable_online_store=True,
        record_identifier_name=f"{scenario}_station_id",
        event_time_feature_name="timestamp",
        description=description,
        role_arn=sagemaker.get_execution_role()
    )


def check_feature_group_status(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")

    while status == "Creating": 
        logger.warning("Creating feature group")
        time.sleep(secs=5)
    logger.success(f"Feature group {feature_group.feature_group_name} created")
