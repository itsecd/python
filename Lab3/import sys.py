import sys

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QDateEdit, QPushButton, QFileDialog
from PyQt6.QtCore import Qt, QDate
from datetime import datetime

sys.path.append("D:/python/Lab2")
from get_value_form_date import read_data_for_date
from spliting_into_two_files import split_csv_by_columns
