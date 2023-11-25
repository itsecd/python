from datetime import datetime, timedelta
from typing import Tuple, Union
from get_data import get_data_for_date


class DateIterator:
    def __init__(self,
                start_date: datetime,
                end_date: datetime,
                input_csv:str,
                dates_file_path: str,
                values_file_path: str
                ) -> None:
        """the function initializing the iterator class"""
        self.start_date = start_date
        self.end_date = end_date
        self.input = input_csv
        self.date_path = dates_file_path
        self.value_path = values_file_path

    def get_next_valid_date(self, current_date: datetime) -> Tuple[Union[datetime, None], Union[str, None]]:
        """the function gets the next valid date and the required data"""
        while current_date <= self.end_date:
            data = get_data_for_date(current_date,self.input,self.date_path,self.value_path)
            current_date += timedelta(days=1)
            if data is not None and data != "data not found":
                return current_date - timedelta(days=1), data
        return None, None

    def __iter__(self) -> "DateIterator":
        """the function returns itself as an iterator"""
        return self

    def __next__(self) -> Tuple[Union[datetime, None], Union[str, None]]:
        """the function returns data for the next date"""
        date, data = self.get_next_valid_date(self.start_date)
        if date is None:
            raise StopIteration
        self.start_date = date + timedelta(days=1)
        return date, data