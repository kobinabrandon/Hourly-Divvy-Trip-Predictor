"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import hopsworks
from loguru import logger

from azure.ai.ml import MLClient
from azureml.featurestore import FeatureStoreClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml.entities import FeatureStore, FeatureStoreEntity, FeatureSet
from azure.identity import DefaultAzureCredential

from azureml.featurestore.feature_source.parquet_feature_source import ParquetFeatureSource

from src.setup.config import config


def create_feature_store():

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config.feature_store_subscription_id, 
        resource_group_name=config.feature_store_resource_group
    )

    fs = FeatureStore(name=config.feature_store_name, location=config.feature_store_loction)
    ml_client.feature_stores.begin_create(feature_store=fs)


def setup_feature_group(
    scenario: str, 
    name: str, 
    version: int, 
    primary_key: list[str] | None, 
    description: str, 
    for_predictions: bool
    ) -> FeatureGroup:
    """
    Create or connect to a feature group with the specified name, and 
    return an object that represents it.

    Returns:
        FeatureGroup: a representation of the fetched or created feature group
    """
    store = get_feature_store()
    event_time = f"{scenario}_hour" if for_predictions else None

    return store.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time=event_time
    )
    

def get_or_create_feature_view(
    name: str, 
    version: int, 
    feature_group: FeatureGroup,
    sub_query: Query | None = None,
    use_sub_query: bool = False,
) -> FeatureView:
    """
    Creates or alternatively retrieves a feature view using the provided details. If a sub-query is to be used, 
    that has to be indicated, and the sub-query is to be provided. Otherwise, all features will be selected for 
    retrieval from the associated feature group.

    Args:
        name: the name of the feature view to fetch or create
        version: the version of the feature view
        feature_group: the feature group object that will be queried if the feature view needs to be created
        sub_query: the specific part of the feature data that is to be retrieved.
        use_sub_query (bool, optional): a boolean indicating whether a subquery is to be used. Defaults to False.

    Returns:
        FeatureView: the desired feature view
    """
    store = get_feature_store()
    try:
        query: Query = sub_query if use_sub_query else feature_group.select_all() 
        feature_view = store.create_feature_view(name=name, version=version, query=query)

    except Exception as error:
        logger.exception(error)
        feature_view = store.get_feature_view(name=name, version=version)
        
    return feature_view
