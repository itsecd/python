import os
import logging


def create_folder(base_folder: str) -> None:
    """The function form a folder"""
    try:
        if not os.path.exists(base_folder):
            os.mkdir(base_folder)
    except Exception as ex:
        logging.exception(f"Can't create folder: {ex}\n{ex.args}\n")