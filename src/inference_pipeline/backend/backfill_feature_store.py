"""
This module contains the code that is used to backfill feature and prediction 
data.
"""
import json 
import pandas as pd
from loguru import logger
from datetime import datetime
from argparse import ArgumentParser

from src.setup.config import config
from src.setup.paths import MIXED_INDEXER, ROUNDING_INDEXER
from src.feature_pipeline.preprocessing import DataProcessor
from src.inference_pipeline.backend.feature_store_api import FeatureStoreAPI
from src.inference_pipeline.backend.model_registry_api import ModelRegistry
from src.inference_pipeline.backend.inference import InferenceModule, load_raw_local_geodata


class BackFiller:
    def __init__(self, scenario: str) -> None:
        """
        Introduce the scenario variable, and the Feature Store API object.

        Args:
            scenario: Determines whether we are looking at arrival or departure data. 
                    Its value must be "start" or "end".
        """
        self.scenario = scenario.lower()
        assert self.scenario.lower() in ["start", "end"], 'Only "start" or "end" are acceptable values'
        
        self.api = FeatureStoreAPI(
            scenario=self.scenario,
            api_key=config.hopsworks_api_key,
            project_name=config.hopsworks_project_name,
            event_time=None,
            primary_key=None
        )

    def backfill_features(self) -> None:
        """
        Run the preprocessing script and upload the time series data to the feature store.

        Returns:
            None
        """
        self.api.event_time = "timestamp"
        self.api.primary_key = ["timestamp", f"{self.scenario}_station_id"]
        processor = DataProcessor(year=config.year, for_inference=False)
        ts_data = processor.make_time_series()[0] if self.scenario == "start" else processor.make_time_series()[1]

        ts_data["timestamp"] = pd.to_datetime(ts_data[f"{scenario}_hour"]).astype(int) // 10 ** 6  # Express in ms

        logger.info(
            f"There are {len(ts_data[f"{self.scenario}_station_id"].unique())} stations in the time series data for\
                {config.displayed_scenario_names[self.scenario].lower()}"
        )

        feature_group = self.api.setup_feature_group(
            description=f"Hourly time series data for {config.displayed_scenario_names[self.scenario].lower()}",
            name=f"{self.scenario}_feature_group",
            version=config.feature_group_version,
            for_predictions=False
        )

        # Push time series data to the feature group
        feature_group.insert(
            ts_data,
            write_options={"wait_for_job": True}
        )

    def backfill_predictions(
        self, 
        target_date: datetime, 
        model_name: str = "xgboost",
        include_station_names: bool = True,
        using_mixed_indexer: bool = True
    ) -> None:
        """
        Fetch the registered version of the named model, and download it. Then load a batch of features
        from the relevant feature group(whether for arrival or departure data), and make predictions on those 
        features using the model. Then create or fetch a feature group for these predictions and push these  
        predictions. 

        Args:
            target_date (datetime): the date up to which we want our predictions.
            
            model_name (str, optional): the shorthand name of the model we will use. Defaults to "xgboost",
                                        because the best performing models (for arrivals and departures) were
                                        LGBMRegressors.
        """
        self.api.primary_key = [f"{self.scenario}_station_id"]

        # The best models for arrivals & departures are currently tuned XGBRegressors
        tuned_or_not = "tuned"
        
        inferrer = InferenceModule(scenario=self.scenario)
        registry = ModelRegistry(scenario=self.scenario, model_name=model_name, tuned_or_not=tuned_or_not)
        model = registry.download_latest_model(unzip=True)

        features = inferrer.fetch_time_series_and_make_features(target_date=datetime.now(), geocode=False)
        
        try:
            features = features.drop(["trips_next_hour", f"{scenario}_hour"], axis=1)
        except Exception as error:
            logger.error(error)    

        predictions_df: pd.DataFrame = inferrer.get_model_predictions(model=model, features=features)
        predictions_df = predictions_df.drop_duplicates().reset_index(drop=True)

        if include_station_names:
            json_path = MIXED_INDEXER if using_mixed_indexer else ROUNDING_INDEXER
            with open(json_path / f"{scenario}_names_and_ids.json", mode="r") as file:
                ids_and_names = json.load(file)
                
            ids_and_names = {int(code): name for code, name in ids_and_names}  # Make sure the IDs are integers here
            predictions[f"{scenario}_station_name"] = predictions[f"{scenario}_station_id"].map(ids_and_names)

        logger.info(
            f"There are {len(predictions_df[f"{self.scenario}_station_id"].unique())} stations in the predictions \
                for {self.scenario}s"
        )
        
        predictions_feature_group = self.api.setup_feature_group(
            description=f"predicting {config.displayed_scenario_names[self.scenario]} - {tuned_or_not} {model_name}",
            name=f"{model_name}_{self.scenario}_predictions_feature_group",
            for_predictions=True,
            version=6
        )

        predictions_feature_group.insert(
            predictions_df,
            write_options={"wait_for_job": True}
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--target", type=str)
    args = parser.parse_args()    
    
    for scenario in args.scenarios:
        filler = BackFiller(scenario=scenario)

        if args.target.lower() == "features":
            filler.backfill_features()
        elif args.target.lower() == "predictions":
            filler.backfill_predictions(target_date=datetime.now())
        else:
            raise Exception('The only acceptable targets of the command are "features" and "predictions"')
    