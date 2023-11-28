# -*- coding: utf-8 -*-

from datetime import datetime
from file_manipulation import read_csv


class DateIterator:
    def __init__(self, file_path: str):
        header, data = read_csv(file_path)
        self.__data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row[1:]] for row in data}
        self.__dates = sorted(self.__data_dict.keys())
        self.__index = 0
    
    @property
    def data_dict(self):
        return self.__data_dict

    @data_dict.setter
    def data_dict(self, x):
        self.__data_dict = x
    
    @property
    def dates(self):
        return self.__dates

    @dates.setter
    def dates(self, x):
        self.__dates = x

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, index):
        self.__index = index

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        if self.__index < len(self.__dates):
            date = self.__dates[self.__index]
            data = self.__data_dict[date]
            self.__index += 1
            return date, data
        else:
            raise StopIteration
