import logging
from file_manipulation import read_csv, write_csv
import os


def split_csv_by_columns(input_file: str, output_file_x: str, output_file_y: str) -> None:
    """
    This function writes a CSV file separated by columns.
    :param input_file:
    :param output_file_x:
    :param output_file_y:
    :return:
    """
    try:
        dir_name_x = os.path.dirname(output_file_x)
        dir_name_y = os.path.dirname(output_file_y)
        os.makedirs(dir_name_x, exist_ok=True)
        os.makedirs(dir_name_y, exist_ok=True)
        header, data = read_csv(input_file)
        x_data = [[row[0]] for row in data]
        y_data = [[t for t in row[1:]] for row in data]
        write_csv(output_file_x, [header[0]], x_data)
        write_csv(output_file_y, [*header[1:]], y_data)
    except Exception as e:
        logging.exception(f"Can't split scv by columns: {e}\n{e.args}\n")
