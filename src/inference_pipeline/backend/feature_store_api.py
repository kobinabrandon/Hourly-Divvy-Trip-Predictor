"""
The class in this module and its methods are just wrappers around the existing hopsworks
feature store API. 
"""
import os 
from loguru import logger

from pandas.api.types import is_string_dtype, is_integer_dtype, is_float_dtype, is_datetime64_any_dtype

from azure.identity import DefaultAzureCredential

from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml.entities import (
    FeatureStore, 
    FeatureStoreEntity, 
    FeatureSet, 
    FeatureSetSpecification,
    DataColumn,
    DataColumnType
)

from azureml.featurestore import FeatureStoreClient, create_feature_set_spec
from azureml.featurestore.feature_source.parquet_feature_source import ParquetFeatureSource

from azureml.featurestore.contracts import (
    DateTimeOffset,
    TransformationCode, 
    Column,
    ColumnType,
    SourceType, 
    TimestampColumn
)

from src.setup.config import config
from src.setup.paths import FEATURE_STORE, TIME_SERIES_DATA


class FeatureStoreAPI:
    def __init__(
        self, 
        scenario: str,
        data: pd.DataFrame,
        for_predictions: bool,
        subscription_id: str = config.feature_store_subscription_id,
        resource_group: str = config.feature_store_resource_group, 
        feature_store_location: str = config.feature_store_loction,
        feature_store_name: str = config.feature_store_name
    ) -> None:

        self.scenario = scenario
        self.for_predictions = for_predictions
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.feature_store_name = feature_store_name
        
        self.spec = self.make_spec()    
        self.feature_set_spec_path = FEATURE_STORE/"spec"

        self.fs_client = MLClient(
            AzureMLOnBehalfOfCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group_name
        )

    def create_feature_store(self):

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id, 
            resource_group_name=self.resource_group
        )

        fs = FeatureStore(name=self.feature_store_name, location=self.feature_store_loction)
        ml_client.feature_stores.begin_create(feature_store=fs)

    def get_feature_store_client(self):

        return FeatureStoreClient(
            credential=AzureMLOnBehalfOfCredential(),
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            name=self.feature_store_name
        )

    @staticmethod
    def convert_pandas_type(dtype: type) -> ColumnType:
        if is_integer_dtype(dtype):
            return ColumnType.INTEGER
        elif is_float_dtype(dtype):
            return ColumnType.FLOAT
        elif is_string_dtype(dtype):
            return ColumnType.STRING
        elif is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

    def make_spec(self):

        if self.for_predictions:
            feature_set_path = FEATURE_STORE/f"{self.scenario}_predictions.parquet"
            source_path = INFERENCE_DATA/f"{self.scenario}_predictions.parquet"
        else:
            feature_set_path = FEATURE_STORE/f"{self.scenario}_ts" 
            source_path = TIME_SERIES_DATA/f"{self.scenario}_ts.parquet" 
            
        file_source = ParquetFeatureSource(path=source_path)

        feature_transformation = TransformationCode(
            path=source_path,
            transformer_class="transaction_transform.TransactionFeatureTransformer"
        )

        columns = [
            Column(name=column, type=self.convert_pandas_type(self.data[column].dtype)) for column in self.data.columns
        ]

        feature_set_spec = create_feature_set_spec(
            source=file_source,
            spec_path=source_path,
            feature_transformation=feature_transformation,
            index_columns=columns,
            infer_schema=True
        )

        if not os.path.exists(path=self.feature_set_spec_path):
            os.mkdir(path=self.feature_set_spec_path)
        
        feature_set_spec.dump(feature_set_path, overwrite=True)


    def register_entity(self):

        index_columns = [
            DataColumn(name=col, type=convert_pandas_type(dtype=self.data[col].dtype)) for col in self.data.columns
        ]

        entity_config = FeatureStoreEntity(name="divvy", version=1, index_columns=index_columns)
        self.fs_client.feature_store_entities.begin_create_or_update(feature_store_entity=entity_config)
        

    def register_feature_set(self):

        feature_set_config = FeatureSet(
            name=f"{self.scenario}_features",
            version="1",
            description=f"features for {self.scenario}s",
            specification=FeatureSetSpecification(path=self.feature_set_spec_path)
        )

        self.fs_client.feature_sets.begin_create_or_update(featureset=feature_set_config)


