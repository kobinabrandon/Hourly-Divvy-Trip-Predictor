import requests
import pandas as pd
from loguru import logger
from datetime import datetime

from src.setup.config import config
from src.feature_pipeline.data_sourcing import Year


def cleaned_data_needs_update(cleaned_data: pd.DataFrame, years_of_interest: list[Year] = config.years) -> bool:
    """
    The primary purpose of this function is to determine whether a saved version of a cleaned dataset is up to 
    date. It does this by checking whether the last trip is from at least a month ago. If it is, the data will 
    be deemed to be old. Furthermore, we will say that data is available for the following month, if new data 
    (new relative to the cleaned data on file) is available. 

    Since the data from Lyft comes with both start and terminal data per trip, it generally doesn't matter 
    whether I check for the time when the last trip starts or ends. These trips only last minutes to hours anyway, 
    but for the sake of the edge case where someone starts a ride a few minutes before new year's day, and ends it 
    a few minutes into new years' day, I am choosing to use the column that pertains to start times in order to 
    bias earlier times. This bias makes sense when you consider that data is deemed to be if it is from prior month 
    regardless of how many days have passed 

    Args:
        cleaned_data: 
        years_of_interest: 

    Returns:
        
    """
    this_month: datetime = datetime.now().month()

    most_recent_date_in_data: datetime = cleaned_data["started_at"][-1]
    last_month_in_data: int = most_recent_date_in_data.month()  
    last_year_in_data: int = most_recent_date_in_data.year()  
    data_is_old: bool = last_month_in_data < this_month 

    # New data will be deemed to be available if data is available for the current month of the current year 
    # This is a link (according to Lyft's link structure) to the data for the most 
    new_data_url: str = f"https://divvy-tripdata.s3.amazonaws.com/{last_year_in_data}{last_month_in_data + 1:02d}-divvy-tripdata.zip"
    new_data_is_available: bool = requests.get(new_data_url).status_code == 200

    match (data_is_old, new_data_is_available):
        case (False, _):
            logger.info("Cleaned data for is up to date")
            return False 

        case (True, True):
            logger.info("Cleaned data for out of date, and new data is available")
            return True

        case (True, False):
            logger.info("Cleaned data for out of date, but new data is not available")
            return False 

