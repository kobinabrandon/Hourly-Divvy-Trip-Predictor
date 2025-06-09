import pandas as pd
from loguru import logger

from src.feature_pipeline.preprocessing.station_indexing.mixed_indexer import run_mixed_indexer
from src.feature_pipeline.preprocessing.station_indexing.rounding_indexer import run_rounding_indexer


def check_if_we_tie_ids_to_unique_coordinates(data: pd.DataFrame, for_inference: bool, threshold: int = 10_000_000) -> bool:
    """
    With a large enough dataset (subjectively defined to be one with more than 10M rows), I found it 
    necessary to round the coordinates of each station to make the preprocessing operations that follow less 
    taxing in terms of time and system memory. Following the rounding operation, a number will be assigned to 
    each unique coordinate which will function as the new ID for that station. 

    In smaller datasets (which are heavily preferred by the author), the new IDs are created with no connection to
    the number of unique coordinates.

    Args:
        data (pd.DataFrame): the dataset to be examined.

    Returns:
        bool: whether the dataset is deemed to be large enough to trigger the above condition 
    """
    assert not for_inference
    return len(data) > threshold 


def check_if_we_use_custom_station_indexing(
        data: pd.DataFrame, 
        for_inference: bool, 
        scenarios: list[str] = ["start", "end"]
) -> bool:
    """
    Certain characteristics of the data will lead to the selection of one of two custom methods of indexing 
    the station IDs. This is necessary because there are several station IDs such as "KA1504000135" (dubbed 
    long IDs) which we would like to be rid of.  We observe that the vast majority of the IDs contain no more 
    than 6 or 7 values, while the long IDs are generally longer than 7 characters. The second group of 
    problematic station IDs are, naturally, the missing ones.
    
    This function starts by checking how many of the station IDs fall into either of these two groups. 
    If there are enough of these (subjectively determined to be at least half the total number of IDs), then 
    the station IDs will have to be replaced with numerical values using one of the two custom indexing methods
    mentioned before. Which of these methods will be used will depend on the result of the function after this 
    one.

    Returns:
        bool: whether a custom indexing method will be used. 
    """
    assert for_inference
    results: list[bool] = []

    for scenario in scenarios:
        long_id_count = 0

        for station_id in data.loc[:, f"{scenario}_station_id"]:
            if len(str(station_id)) >= 7 and not pd.isnull(station_id):
                long_id_count += 1

        number_of_missing_indices: int = data[f"{scenario}_station_id"].isna().sum()
        proportion_of_problem_rows: float = (number_of_missing_indices + long_id_count) / data.shape[0] 
        result: bool = True if proportion_of_problem_rows >= 0.5 else False
        results.append(result)

    return (False not in results) 


def investigate_making_new_station_ids(data: pd.DataFrame, cleaned_data: pd.DataFrame, start_or_end: str) -> list[pd.DataFrame]:
    """
    In an earlier version of the project, I ran into memory issues for two reasons:
        1) I was dealing with more than a year's worth of data. 
        2) there were bugs embedded in the feature pipeline.

    As a result, I decided to match approximations of each coordinate with an ID of 
    my own making, thereby reducing the size of the dataset during the aggregation
    stages of the creation of the training data. This worked well. However, I am no
    longer in need of any memory conservation measures because I have reduced the size of 
    the dataset.

    So why is the code still being used? Why not simply use the original IDs? You 
    may be thinking that perchance there weren't even many missing values, and that 
    perhaps all this could have been avoided.

    There's code in what immediately follows that checks for the presence of both long
    string indices (see the very first method of the class) and missing ones. If the proportion
    of rows that feature such indices exceeds a certain hardcoded threshold (I chose 50%),
    we will use the custom procedures (again see the aforementioned class method).

    As of late July 2024, 60% of the IDs (for origin and destination stations) have long strings
    or missing indices. It is therefore unlikely that we will need an alternative method. However, 
    I will write one eventually. Most likely it will involve simply applying the custom procedure to only
    that problematic minority of indices, to generate new integer indices that aren't already in the column.

    Args:
        cleaned_data (pd.DataFrame): the version of the dataset that has been cleaned
        start_or_end (str): whether we are looking at arrivals or departures.

    Returns:
        pd.DataFrame:
    """
    interim_dataframes: list[pd.DataFrame] = []
    logger.info(f"Recording the hour during which each trip {start_or_end}s...")

    cleaned_data.insert(
        loc=cleaned_data.shape[1],
        column=f"{start_or_end}_hour",
        value=cleaned_data.loc[:, f"{start_or_end}ed_at"].dt.floor("h"),
        allow_duplicates=False
    )

    cleaned_data = cleaned_data.drop(f"{start_or_end}ed_at", axis=1)
    logger.info("Determining the method of dealing with invalid station indices...")

    match (
        check_if_we_use_custom_station_indexing(scenarios=[start_or_end], data=data), 
        check_if_we_tie_ids_to_unique_coordinates(data=data)
    ):
        case (True, True):
            logger.warning("Custom station indexer required: tying new station IDs to unique coordinates")
            interim_data = run_rounding_indexer(data=cleaned_data, scenario=start_or_end, decimal_places=6)
            interim_dataframes.append(interim_data)
            return interim_dataframes

        case (True, False):
            logger.warning("Custom station indexer required: NOT tying new IDs to unique coordinates")
            interim_data = run_mixed_indexer(
                scenario=start_or_end,
                data=cleaned_data,
                delete_leftover_rows=False
            )

            interim_dataframes.append(interim_data)
            return interim_dataframes

        case (False, _):
            raise NotImplementedError(
                "The majority of Divvy's IDs weren't numerical and valid during initial development."
            )




