import os
import json 
import pandas as pd
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta
from feast.data_format import ParquetFormat
from feast import FileSource, FeatureStore, FeatureView

from src.setup.config import config
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.feature_store_api import get_or_create_feature_view

from src.inference_pipeline.backend.inference import (
    get_model_predictions, 
    load_raw_local_geodata, 
    fetch_time_series_and_make_features
)

from src.setup.paths import (
    MIXED_INDEXER, 
    ROUNDING_INDEXER, 
    TIME_SERIES_DATA, 
    INFERENCE_DATA, 
    FEATURE_REPO, 
    FEATURE_REPO_DATA
)


class Backfiller:
    def __init__(self, scenario: str, for_predictions: bool):
        self.scenario = scenario    
        self.for_predictions = for_predictions

        if for_predictions:
            self.source_path = INFERENCE_DATA/f"predicted_{scenario}s.parquet"
            self.description = f"Hourly predictions for {config.displayed_scenario_names[scenario].lower()}"
        else:
            self.source_path = TIME_SERIES_DATA/f"{scenario}_ts.parquet"
            self.description = f"Hourly time series data for {config.displayed_scenario_names[scenario].lower()}"
        
        self.source = FileSource(path=str(self.source_path), file_format=ParquetFormat(), description=self.description)
        self.store = FeatureStore(repo_path=FEATURE_REPO, fs_yaml_file=FEATURE_REPO/"feature_store.yaml")


    def backfill_features(self) -> None:
        """
        Run the preprocessing script and upload the time series data to the feature store.

        Args:
            scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

        Returns:
            None
        """
        self.for_predictions = False
        processor = DataProcessor(year=config.year, for_inference=False)
        ts_data = processor.make_time_series()[0] if self.scenario == "start" else processor.make_time_series()[1]
        ts_data["timestamp"] = pd.to_datetime(ts_data[f"{self.scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms

        self.get_and_push_data(scenario=self.scenario, for_predictions=False, data=ts_data)


    def backfill_predictions(self, using_mixed_indexer: bool = True) -> None:
        """
        Fetch the registered version of the named model, and download it. Then load a source of features
        from the relevant feature group(whether for arrival or departure data), and make predictions on those 
        features using the model. Then create or fetch a feature group for these predictions and push these  
        predictions. 

        Args:    
            scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".

        Returns:
            None
        """
        model_name = "xgboost" if self.scenario == "start" else "lightgbm"
        tuned_or_not = "untuned" if self.scenario == "start" else "tuned"

        registry = ModelRegistry(scenario=self.scenario, model_name=model_name, tuned_or_not=tuned_or_not)
        model = registry.download_latest_model(unzip=True)
        
        start_date= datetime.now() - timedelta(days=30)
        end_date = datetime.now() + timedelta(days=1)

        entity_sql = f"""
            SELECT {self.scenario}_station_id, timestamp
            FROM {self.store.get_data_source(name=f"{self.scenario}_ts.parquet").get_table_query_string()}
            WHERE timestamp BETWEEN '{start_date}' and '{end_date}' 
            GROUP by {self.scenario}_station_id
        """

        ts_data = self.store.get_historical_features(entity_df=entity_sql, features=self.store.get_feature_service()).to_df()
        breakpoint()


        try:
            features = features.drop(["trips_next_hour", f"{self.scenario}_hour"], axis=1)
        except Exception as error:
            logger.error(error)    

        predictions: pd.DataFrame = get_model_predictions(model=model, features=features)
        predictions = predictions.drop_duplicates().reset_index(drop=True)

        # Now to add station names to the predictions
        ids_and_names = fetch_json_of_ids_and_names(scenario=scenario, using_mixed_indexer=True, invert=False)
        predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

        breakpoint()

        get_and_push_data(scenario=scenario, for_predictions=True, data=predictions)
    

    @staticmethod
    def get_and_push_data(self, data: pd.DataFrame) -> None:

        if for_predictions:
            feature_view = store.get_feature_view()
        else:
            feature_view, entity = get_or_create_feature_view(scenario=scenario, for_predictions=for_predictions, file_source=source)

            os.chdir(path=FEATURE_REPO)
            store.apply([entity, feature_view])  

            os.system(command=f"feast materialize-incremental {datetime.now().strftime('%Y-%m-%d')}")  
            store.write_to_online_store(feature_view_name=feature_view.name, allow_registry_cache=True, df=data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        assert scenario.lower() in ["start", "end"], 'Only "start" or "end" are acceptable values'

        if args.target.lower() == "features":
            Backfiller(scenario=scenario, for_predictions=False).backfill_features()
        elif args.target.lower() == "predictions":
            Backfiller(scenario=scenario, for_predictions=True).backfill_predictions()
        else:   
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
    