import unittest
import pandas as pd 


class CheckForALlStationIDs(unittest.TestCase):

    def test_presence_of_station_ids(self, agg_data: pd.DataFrame, scenario: str):
        possible_station_ids = range(agg_data[f"{scenario}_station_id"].max()+1)
        actual_station_ids = agg_data[f"{scenario}_station_id"].to_list()

        for station_id in possible_station_ids:
            try:
                self.assertIn(member=station_id, container=actual_station_ids)
            except:
                raise AssertionError(
                    f"There is no station with ID {station_id} in the {scenario} data"
                )
        
import unittest
import pandas as pd 


class CheckForALlStationIDs(unittest.TestCase):

    def test_presence_of_station_ids(self, agg_data: pd.DataFrame, scenario: str):
        possible_station_ids = range(agg_data[f"{scenario}_station_id"].max()+1)
        actual_station_ids = agg_data[f"{scenario}_station_id"].to_list()

        for station_id in possible_station_ids:
            try:
                self.assertIn(member=station_id, container=actual_station_ids)
            except:
                raise AssertionError(
                    f"There is no station with ID {station_id} in the {scenario} data"
                )
        