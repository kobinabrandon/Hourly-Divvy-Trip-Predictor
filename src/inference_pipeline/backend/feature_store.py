"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import hopsworks
from loguru import logger
from hsfs.feature_view import FeatureView
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup
from hsfs.constructor.query import Query

from src.setup.config import config


def get_feature_store() -> FeatureStore:
    """
    Login to Hopsworks and return a pointer to the feature store

    Returns:
        FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(project=config.hopsworks_project_name, api_key_value=config.hopsworks_api_key)
    return project.get_feature_store()


def setup_feature_group(
    name: str, 
    version: int, 
    primary_key: list[str] | None, 
    description: str 
    ) -> FeatureGroup:
    """
    Create or connect to a feature group with the specified name, and 
    return an object that represents it.

    Returns:
        FeatureGroup: a representation of the fetched or created feature group
    """
    store = get_feature_store()

    return store.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        event_time="timestamp"
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
        feature_view = store.get_or_create_feature_view(name=name, version=version, query=query)

    except Exception as error:
        logger.exception(error)
        feature_view = store.get_feature_view(name=name, version=version)
        
    return feature_view
