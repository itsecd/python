# -*- coding: utf-8 -*-

from datetime import datetime
from file_manipulation import read_csv


class DateIterator:
    __data_dict = None
    __index = None
    __dates = None

    def __init__(self, file_path: str):
        header, data = read_csv(file_path)
        self.set_data_dict(data)
        self.set_dates(self.get_data_dict())
        self.set_index(0)

    def set_data_dict(self, data):
        self.__data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row] for row in data}

    def get_data_dict(self):
        return self.__data_dict

    def set_dates(self, data_dict):
        self.__dates = sorted(data_dict.keys())

    def get_dates(self):
        return self.__dates

    def set_index(self, index):
        self.__index = index

    def get_index(self):
        return self.__index

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        if self.get_index() < len(self.get_dates()):
            date = self.get_dates()[self.get_index()]
            data = self.get_data_dict()[date]
            self.set_index(self.get_index()+1)
            return date, data
        else:
            raise StopIteration
