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


def setup_feature_group(
    scenario: str, 
    data: pd.DataFrame,
    description: str, 
    for_predictions: bool
    ) -> FeatureGroup:
    """
    Create or connect to a feature group with the specified name, and return an object that represents it.

    Returns:
        FeatureGroup: a representation of the fetched or created feature group
    """
    model_name = "lightgbm" if scenario == "end" else "xgboost"
    feature_group_name = f"{model_name}_{scenario}_predictions_feature_group" if for_predictions else f"{scenario}_ts"
    feature_group = FeatureGroup(name=feature_group_name)

    data[f"{scenario}_hour"] = data[f"{scenario}_hour"].dt.strftime('%Y-%m-%d %H:%M:%S').astype(str)
    feature_group.load_feature_definitions(data_frame=data)

    try:
        session = sagemaker.Session()

        feature_group.create(
            s3_uri=f"s3://{session.default_bucket()}/'divvy_features'",
            enable_online_store=True,
            record_identifier_name=f"{scenario}_station_id",
            event_time_feature_name="timestamp",
            description=description,
            role_arn=config.aws_arn
        )
    except:
        feature_group.describe()  # If the feature group already exists:

    return feature_group





def check_feature_group_status(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")

    while status != "Created":
        try:
            status = feature_group.describe()["OfflineStoreStatus"]["Status"]
        except:
            raise Exception

        logger.warning("Offline feature store status", status)    
        time.sleep(secs=15)  

    logger.success(f"Feature group {feature_group.name} created")
