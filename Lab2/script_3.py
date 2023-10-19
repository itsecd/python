import csv
import json
import os
import time
import random
import logging


if __name__ == '__main__':
    with open(os.path.join("Lab1", "input_data.json"), 'r') as fjson:
        fj = json.load(fjson)

    logging.basicConfig(level=logging.INFO)