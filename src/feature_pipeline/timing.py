from datetime import datetime
from src.setup.config import config


class Period:
    """
    This class allows us to specify the year and the months for which data retrieval is to be performed.

    Attributes: 
        year: the year (as a number) 
        months: The months (as integers) that whose data we will download. 
                This argument will allow me to adjust how much data I download and use as new data is released  
    """
    def __init__(self, year: int, months: list[int]) -> None:
        self.year : int = year 
        self.months: list[int] = months


def select_months_of_interest(offset: int = config.offset) -> list[Period]:
    """
    The function gets the current year and month, and compares the current month with the offset.
    Then we gather the previous {offset} months, taking note of which year the various months belong to.
    The function then returns Period objects that correspond to these months. Crucially, it is important
    to note that the offset will over ever be in 1,..., 12, because I have no desire to train models on data 
    that is over a year old.

    Args:
        offset: The number of months in the immediate past that for which we will retrieve data 

    Returns:
        list[Period]: the Period objects that tell us which years and months we want data for. 
    """
    current_year = datetime.now().year
    current_month = datetime.now().month
    month_numbers: list[int] = [month for month in range(13) if month != 0]

    if current_month > offset:
        months_of_interest: list[int] = [ month for month in range(current_month - offset, current_month + 1) ]
        return [ Period(year=current_year, months=months_of_interest) ] 

    elif current_month < offset:
        current_year_months_of_interest: list[int] = [] 
        previous_year_months_of_interest: list[int] = [] 

        # Gather the months from this year
        for index in [month for month in range(current_month + 1 ) if month != 0]: 
            current_year_months_of_interest.append(index) 
            
        # Gather the months from the previous year
        for index in [month for month in range(offset - current_month + 1 ) if month != 0]: 
            previous_year_months_of_interest.append(month_numbers[-index])

        return [
            Period(year=current_year - 1, months=previous_year_months_of_interest),
            Period(year=current_year, months=current_year_months_of_interest)
        ] 

    else:
        months_of_interest = [month for month in range(current_month + 1 ) if month != 0] 
        return [ Period(year=current_year, months=months_of_interest) ] 
        
