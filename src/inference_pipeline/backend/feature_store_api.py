"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import hopsworks
from pathlib import Path 
from loguru import logger

from feast import Entity, Feature, FeatureView, FileSource, Field
from src.setup.paths import FEATURE_REPO_DATA, TIME_SERIES_DATA

def create_entity(name: str) -> Entity:
    return Entity(name=name, join_keys=[name])


def get_feature_store(path: Path) -> FileSource:

    return FileSource(
        path=path, 
        name="divvy_feature_store",
        description="Feature store for arrivals and departures",
        timestamp_field="timestamp"
    )


def setup_feature_group(self, name: str, version: int, description: str, for_predictions: bool) -> FeatureGroup:
        """
        Create or connect to a feature group with the specified name, and 
        return an object that represents it.

        Returns:
            FeatureGroup: a representation of the fetched or created feature group
        """
        feature_store = self.get_feature_store()
        feature_group = feature_store.get_or_create_feature_group(
            name=name,
            version=version,
            description=description,
            primary_key=self.primary_key,
            event_time=f"{self.scenario}_hour" if for_predictions else self.event_time
        )
        return feature_group
    

def get_or_create_feature_view(scenario: str, name: str, data_path: Path) -> FeatureView:
    """
    Creates or alternatively retrieves a feature view using the provided details. If a sub-query is to be used, 
    that has to be indicated, and the sub-query is to be provided. Otherwise, all features will be selected for 
    retrieval from the associated feature group.

    Args:
        name: the name of the feature view to fetch or create
    
    Returns:
        FeatureView: the desired feature view
    """
    data = pd.read_parquet(path=path)

    divvy_fs = get_feature_store()
    entity = create_entity(name=f"{self.scenario}_id")

    feature_view = FeatureView(
        name=f"{scenario}_feature_view",
        entities=[entity]
    )

    except Exception as error:
        logger.exception(error)
        feature_view = feature_store.get_feature_view(name=name, version=version)
        
    return feature_view
