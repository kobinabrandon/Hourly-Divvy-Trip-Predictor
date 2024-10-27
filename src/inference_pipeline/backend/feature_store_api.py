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


def create_feature_store():

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config.feature_store_subscription_id, 
        resource_group_name=config.feature_store_resource_group
    )

    fs = FeatureStore(name=config.feature_store_name, location=config.feature_store_loction)
    ml_client.feature_stores.begin_create(feature_store=fs)


def get_feature_store_client():
    return FeatureStoreClient(
        credential=AzureMLOnBehalfOfCredential(),
        subscription_id=config.feature_store_subscription_id,
        resource_group_name=config.feature_store_resource_group,
        name=config.feature_store_name
    )


def convert_pandas_type(dtype: type) -> ColumnType:
    if is_integer_dtype(dtype):
        return ColumnType.INTEGER
    elif is_float_dtype(dtype):
        return ColumnType.FLOAT
    elif is_string_dtype(dtype):
        return ColumnType.STRING
    elif is_datetime64_any_dtype(dtype):
        return ColumnType.DATETIME



def make_spec(scenario: str, data: pd.DataFrame, for_predictions: bool):
    
    feature_set_path = FEATURE_STORE/f"{scenario}_predictions" if for_predictions else FEATURE_STORE/f"{scenario}_ts" 
    source_path = INFERENCE_DATA/f"{scenario}_predictions" if for_predictions else TIME_SERIES_DATA/f"{scenario}_ts.parquet" 
    
    file_source = ParquetFeatureSource(path=source_path)
    feature_transformation = TransformationCode(
        path=source_path,
        transformer_class="transaction_transform.TransactionFeatureTransformer"
    )

    columns = [Column(name=col, type=convert_pandas_type(col)) for col in data.columns]

    feature_set_spec = create_feature_set_spec(
        source=file_source,
        spec_path=source_path,
        feature_transformation=feature_transformation,
        index_columns=columns,
        infer_schema=True
    )

    feature_set_path = FEATURE_STORE/"spec"
    if not os.path.exists(path=feature_set_path):
        os.mkdir(path=feature_set_path)
    
    feature_set_spec.dump(feature_set_path, overwrite=True)


def register_entity(scenario: str):

    fs_client = MLClient(
        AzureMLOnBehalfOfCredential(),
        subscription_id=featurestore_subscription_id,
        resource_group_name=featurestore_resource_group_name[.]
    )



def register_feature_set(scenario: str):

    feature_set_config = FeatureSet(
        name=f"{scenario}_features",
        version="1",
        description=f"features for {scenario}s",
        entities=["azureml:account:1"],
        stage="Development",

    )

    



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
