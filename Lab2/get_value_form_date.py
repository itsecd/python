# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from file_manipulation import read_csv


def read_data_for_date(file_path: str, target_date: datetime) -> dict:
    """
    This function returns data by date
    :param file_path:
    :param target_date:
    :return:
    """
    try:
        header, data = read_csv(file_path)
        data_dict = {datetime.strptime(row[0], '%Y-%m-%d'): [t for t in row] for row in data}
        return data_dict.get(target_date)
    except Exception as e:
        logging.exception(f"Can't read data for date: {e}\n{e.args}\n")
