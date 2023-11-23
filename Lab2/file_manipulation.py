# -*- coding: utf-8 -*-

import logging
import csv


def read_csv(file_path: str) -> tuple:
    """
    This function reads a CSV file
    :param file_path:
    :return tuple(header: list(str), data: list(str)):
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = list(reader)
        return header, data
    except Exception as e:
        logging.exception(f"Can't read data: {e}\n{e.args}\n")


def write_csv(file_path: str, header: list, data: list) -> None:
    """
    This function writes a CSV file
    :param file_path:
    :param header:
    :param data:
    :return:
    """
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
    except Exception as e:
        logging.exception(f"Can't write data: {e}\n{e.args}\n")
