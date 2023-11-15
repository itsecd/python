from datetime import datetime, timedelta
from get_value_from_date import get_data_for_date
from typing import Tuple, Union

class DateDataIterator:
    def __init__(self, start_date: datetime, end_date: datetime) -> None:
        """Initializing the iterator class"""
        self.current_date = start_date
        self.end_date = end_date

    def get_next_valid_date(self, current_date: datetime) -> Tuple[Union[datetime, None], Union[str, None]]:
        """Gets the next valid date and the corresponding data"""
        while current_date <= self.end_date:
            data = get_data_for_date(current_date)
            current_date += timedelta(days=1)
            if data is not None and data != "Page not found":
                return current_date - timedelta(days=1), data
        return None, None

    def __iter__(self) -> "DateDataIterator":
        """Returns itself as an iterator"""
        return self

    def __next__(self) -> Tuple[Union[datetime, None], Union[str, None]]:
        """Returns data for the next date"""
        date, data = self.get_next_valid_date(self.current_date)
        if date is None:
            raise StopIteration
        self.current_date = date + timedelta(days=1)
        return date, data
    
if __name__ == "__main__": 
    
    iterator = DateDataIterator(datetime(1998, 1, 2), datetime(2023, 10, 14))

    for _ in range(50):
        try:
            next_date, next_data_value = next(iterator)
            if next_date is not None:
                print(f"Date: {next_date}, Value: {next_data_value}")
            else:
                print("No more data")
                break
        except StopIteration:
            print("No more data")
            break