import os
import json
import random
import logging
import csv_make

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    with open(os.path.join("Lab1", "main.json"), "r") as main_file:
        # чтение; указатель на начале файла; вызывается по умолчанию
        main = json.load(main_file)