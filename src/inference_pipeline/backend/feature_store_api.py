"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import hopsworks
from loguru import logger

class FeatureStoreAPI:
    def __init__(
        self,
        api_key: str,
        scenario: str,
        project_name: str,
        event_time: str | None,
        primary_key: list[str] | None
    ) -> None:
        self.api_key = api_key
        self.scenario = scenario
        self.event_time = event_time
        self.primary_key = primary_key
        self.project_name = project_name

    def get_feature_store(self) -> FeatureStore:
        """
        Login to Hopsworks and return a pointer to the feature store

        Returns:
            FeatureStore: pointer to the feature store
        """
        project = hopsworks.login(project=self.project_name, api_key_value=self.api_key)
        return project.get_feature_store()

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
    
    def get_or_create_feature_view(
        self, 
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
        
        feature_store = self.get_feature_store()

        try:
            query: Query = sub_query if use_sub_query else feature_group.select_all() 
            feature_view = feature_store.create_feature_view(name=name, version=version, query=query)

        except Exception as error:
            logger.exception(error)
            feature_view = feature_store.get_feature_view(name=name, version=version)
            
        return feature_view
