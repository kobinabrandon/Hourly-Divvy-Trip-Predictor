import pandas as pd


def tie_ids_to_unique_coordinates(data: pd.DataFrame, for_inference: bool, threshold: int = 10_000_000) -> bool:
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


def use_custom_station_indexing(scenarios: list[str], data: pd.DataFrame, for_inference: bool) -> bool:
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

