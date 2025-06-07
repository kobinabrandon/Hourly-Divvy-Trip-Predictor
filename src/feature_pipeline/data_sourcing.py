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
"""
import os
import requests
import pandas as pd

from pathlib import Path
from loguru import logger
from zipfile import ZipFile
from datetime import datetime as dt

from src.setup.paths import RAW_DATA_DIR, make_fundamental_paths

       
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
        local_file = RAW_DATA_DIR.joinpath(file_name)
        if local_file.exists():
            logger.success(f"{file_name}.zip is already saved")
        else:
            try:
                logger.info(f"Downloading and extracting {file_name}.zip")

                assert year >= 2021; 
                """
                The downloader is currently configured for years after 2021 because the zipfiles were packaged 
                differently for earlier years. Also, data from earlier years would be less useful for my purposes 
                anyway.
                """

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


class Year:
    """
    Attributes: 
        value: the year (as a number) 
        offset: how many months to skip (from the "front"). This argument will allow me to adjust how 
                much data I download and use as new data is released  
    """
    def __init__(self, value: int, offset: int) -> None:
        self.value: int = value 
        self.offset: int = offset


def load_raw_data(years: list[Year]) -> pd.DataFrame:
    """
    For each year, we download or load the data for either the specified months, or 
    for all months up to the present month (if the data being sought is from this year).
    
    Args:
        year (int): the year whose data we want to load

    Yields:
        Iterator[pd.DataFrame]: the requested datasets.
    """
    data = pd.DataFrame()
    make_fundamental_paths()

    for year in years:
        is_current_year = True if year.value == dt.now().year else False
        end_month = dt.now().month if is_current_year else 12 
        months_to_download = range(1 + year.offset, end_month + 1) 

        for month in months_to_download:
            file_name = f"{year.value}{month:02d}-divvy-tripdata"
            download_file_if_needed(year=year.value, month=month, file_name=file_name)
            path_to_month_data: Path = RAW_DATA_DIR.joinpath(f"{file_name}").joinpath(f"{file_name}.csv")

            if path_to_month_data.exists():
                month_data: pd.DataFrame = pd.read_csv(path_to_month_data)
                data = pd.concat([data, month_data], axis=0)
            else:
                logger.error(f"Skipping over {file_name} as Lyft hasn't uploaded it yet.")
    
    return data

