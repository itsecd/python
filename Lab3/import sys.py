import sys

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QDateEdit, QPushButton, QFileDialog
from PyQt6.QtCore import Qt, QDate
from datetime import datetime

sys.path.append("D:/python/Lab2")
from get_value_form_date import read_data_for_date
from spliting_into_two_files import split_csv_by_columns


def show_data(self) -> None:
        """
        Display data for the selected date.
        """
        selected_date = self.date_edit.date().toString(Qt.DateFormat.ISODate)
        if self.data_path is not None:
            r = read_data_for_date(self.data_path, datetime.strptime(selected_date, '%Y-%m-%d'))
            if r is not None:
                self.result_label.setText(" ".join(r))
            else:
                self.result_label.setText("No data for the selected time")
        else:
            self.result_label.setText("Please select a file")
