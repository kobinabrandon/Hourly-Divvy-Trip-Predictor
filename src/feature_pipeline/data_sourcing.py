"""
Much of the code in this module is concerned with downloading the zip files that 
contain raw data, extracting their contents, and loading said contents as 
dataframes.

In an earlier version of this project, I downloaded the data for every year that 
Divvy has been in operation (from 2014 to date). 

I have since decided to restrict my training data to the most recent year of data so
because the most relevant data is obviously valuable, and because I don't want to deal 
with the extra demands on my memory and time that preprocessing that data would have required 
(not to speak of the training and testing of models).    

I embedded an assert statement in the function that performs the downloads to enforce the 
requirement that data not come from prior to 2021. After 2021, the zipfiles were packaged differently. 
Also, data from earlier years would be less useful for my purposes anyway. Plus I only intend to 
download data from at most a year prior anyway. I just want that assert statement to be there as a 
testament to my awareness of the fact that Lyft changed the way they name their files.
"""

import os
import requests
import pandas as pd

from pathlib import Path
from loguru import logger
from zipfile import ZipFile

from src.setup.paths import RAW_DATA_DIR, make_fundamental_paths
from src.feature_pipeline.timing import Period, select_months_of_interest

       
def download_file_if_needed(
        year: int, 
        file_name: str, 
        month: int | None = None, 
        keep_zipfile: bool = False
) -> None:
    """
    Checks for the presence of a file, and downloads it if necessary.

    If the HTTP request for the data is successful, download the zipfile containing the data, 
    and extract the .csv file it contains into a folder of the same name. The zipfile will then
    be deleted by default, unless otherwise specified.    

    Args:
        file_name (str): the name of the file to be saved to disk
        year (int): the year whose data we are looking to potentially download
        month (list[int] | None, optional): the month for which we seek data
    """
    if month is not None:
        local_file: Path = RAW_DATA_DIR.joinpath(file_name)
        if local_file.exists():
            logger.success(f"{file_name}.zip is already saved")
        else:
            try:
                logger.info(f"Downloading and extracting {file_name}.zip")
                assert year >= 2021  # See the module's docstring
                
                zipfile_name: str = f"{year}{month:02d}-divvy-tripdata.zip"
                url = f"https://divvy-tripdata.s3.amazonaws.com/{zipfile_name}"
                response = requests.get(url)

                if response.status_code != 200:
                    logger.error(f"File not found on remote server. Status code: {response.status_code}")
                else:
                    file_name = zipfile_name[:-4]  # Remove ".zip" from the name of the zipfile
                    folder_path = RAW_DATA_DIR.joinpath(file_name)
                    zipfile_path = RAW_DATA_DIR.joinpath(zipfile_name)

                    # Write the zipfile to the disk
                    with open(file=zipfile_path, mode="wb") as zipfile:
                        _ = zipfile.write(response.content)
                    
                    # Extract the contents of the zipfile
                    with ZipFile(file=zipfile_path, mode="r") as zipfile:
                        _ = zipfile.extract(f"{file_name}.csv", folder_path)  # Extract only the .csv file

                    if not keep_zipfile:
                        os.remove(zipfile_path)

            except Exception as error:
                logger.error(error)


def load_raw_data() -> pd.DataFrame:
    """
    For each year, we download or load the data for either the specified months, or 
    for all months up to the present month (if the data being sought is from this year).
    
    Yields:
        pd.DataFrame: a dataframe made that is a concatenation of the downloaded data
    """
    data = pd.DataFrame()
    make_fundamental_paths()
    periods: list[Period] = select_months_of_interest()

    for period in periods:
        period.months.sort()  # Sort the list that contains the months into ascending order 
        for month in period.months:
            file_name = f"{period.year}{month:02d}-divvy-tripdata"
            download_file_if_needed(year=period.year, month=month, file_name=file_name)
            path_to_month_data: Path = RAW_DATA_DIR.joinpath(f"{file_name}").joinpath(f"{file_name}.csv")

            if path_to_month_data.exists():
                month_data: pd.DataFrame = pd.read_csv(path_to_month_data)
                data = pd.concat([data, month_data], axis=0)
            else:
                logger.error(f"Skipping over {file_name} as Lyft hasn't uploaded it yet.")
    
    return data

