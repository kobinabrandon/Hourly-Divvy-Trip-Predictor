import hopsworks

from loguru import logger
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup
from hsfs.feature_view import FeatureView


class FeatureStoreAPI:
    def __init__(
            self,
            api_key: str,
            scenario: str,
            project_name: str,
            feature_view_name: str,
            feature_group_name: str,
            feature_group_version: int,
            feature_view_version: int
    ) -> None:

        self.api_key = api_key
        self.scenario = scenario
        self.project_name = project_name
        self.feature_group_name = feature_group_name
        self.feature_view_version = feature_view_version
        self.feature_group_version = feature_group_version
        self.feature_view_name = feature_view_name

    def login_to_hopsworks(self) -> any:
        project = hopsworks.login(project=self.project_name, api_key_value=self.api_key)
        return project

    def get_feature_store(self) -> FeatureStore:
        """
        Login to Hopsworks and return a pointer to the feature store

        Returns:
            FeatureStore: pointer to the feature store
        """
        project = self.login_to_hopsworks()
        return project.get_feature_store()

    def get_or_create_feature_group(self) -> FeatureGroup:
        feature_store = self.get_feature_store()
        feature_group = feature_store.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            description=f"Hourly time series data showing when trips {self.scenario}",
            primary_key=[f"{self.scenario}_hour", f"{self.scenario}_station_id"],
            event_time=f"{self.scenario}_hour"
        )
        return feature_group

    def get_or_create_feature_view(self) -> FeatureView:
        feature_store = self.get_feature_store()
        feature_group = self.get_or_create_feature_group()

        try:
            feature_view = feature_store.create_feature_view(
                name=self.feature_view_name,
                version=self.feature_view_version,
                query=feature_group.select_all()
            )

        except Exception as error:
            logger.exception(error)
            feature_view = feature_store.get_feature_view(
                name=self.feature_view_name,
                version=self.feature_view_version
            )
        return feature_view