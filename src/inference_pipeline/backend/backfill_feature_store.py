"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import json 
import pandas as pd
from tqdm import tqdm 
from loguru import logger
from argparse import ArgumentParser
from datetime import datetime, timedelta

from sagemaker.feature_store.feature_group import FeatureGroup

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER, INFERENCE_DATA
from src.feature_pipeline.preprocessing import DataProcessor
from src.feature_pipeline.mixed_indexer import fetch_json_of_ids_and_names

from src.inference_pipeline.backend.inference import fetch_time_series_and_make_features

from src.inference_pipeline.backend.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.inference import get_model_predictions


class Backfiller:
    def __init__(self, scenario: str):
        self.scenario = scenario

        self.start_date = datetime.now() - timedelta(days=config.backfill_days) - timedelta(days=config.backfill_days)  # Just for now.
        self.end_date = datetime.now() + timedelta(days=1)

    def push_features(self, local: bool = False) -> None:
        """
        Run the preprocessing script and upload the time series data to local storage or the feature store.
        You'll want to save this data locally this because pushing to AWS will take forever.

        Args:
            scenario: Determines whether we are looking at arrival or departure data. Its value must be "start" or "end".
            local (bool, optional): whether to save the time series data locally. Defaults to True.

        Returns:
            _type_: _description_
        """
        processor = DataProcessor(year=config.year, for_inference=False)
        ts_data = processor.make_time_series()[0] if scenario == "start" else processor.make_time_series()[1]

        if local:
            ts_data.to_parquet(path=INFERENCE_DATA/f"{self.scenario}_ts.parquet")
        else:
            ts_data = ts_data[
                ts_data[f"{self.scenario}_hour"].between(left=self.start_date, right=self.end_date)
            ]

            api = FeatureStoreAPI(scenario=self.scenario, for_predictions=False)
            api.push(data=ts_data)

        return ts_data

    def push_predictions(self, using_mixed_indexer: bool = True) -> None:
        """
        Fetch the registered version of the named model, and download it. Then load a batch of predictions
        from the relevant feature group (whether for arrival or departure data), and make predictions on those 
        predictions using the model. Then create or fetch a feature group for these predictions and push these  
        predictions. 

        Args:
            using_mixed_indexer (bool): whether the mixed indexer was used to index the stations.
        """
        # The best model architectures for arrivals & departures at the moment
        model_name = "lightgbm" if self.scenario == "end" else "xgboost"
        tuned_or_not = "tuned" if self.scenario == "end" else "untuned"

        registry = ModelRegistry(scenario=self.scenario, model_name=model_name, tuned_or_not=tuned_or_not)
        model = registry.download_latest_model(unzip=True)

        features = fetch_time_series_and_make_features(
            scenario=self.scenario,
            start_date=self.start_date,
            target_date=self.end_date,
            geocode=False
        )

        try:
            features = features.drop(["trips_next_hour", f"{self.scenario}_hour"], axis=1)
        except Exception as error:
            logger.error(error)    

        predictions: pd.DataFrame = get_model_predictions(scenario=self.scenario, model=model, features=features)
        predictions = predictions.drop_duplicates().reset_index(drop=True)

        # Now to add station names to the predictions
        ids_and_names = fetch_json_of_ids_and_names(scenario=self.scenario, using_mixed_indexer=True, invert=False)
        predictions[f"{self.scenario}_station_name"] = predictions[f"{self.scenario}_station_id"].map(ids_and_names)

        predictions_api = FeatureStoreAPI(scenario=self.scenario, for_predictions=True)
        predictions_api.push(data=predictions)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        backfiller = Backfiller(scenario=scenario)
        if args.target.lower() == "features":
            backfiller.push_features()
        elif args.target.lower() == "predictions":
            backfiller.push_predictions()
        else:
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
