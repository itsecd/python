import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow,QPushButton,QMessageBox,QLabel,QFileDialog,QVBoxLayout,QWidget,QGridLayout)
from PyQt6.QtGui import QPixmap

sys.path.insert(1, "D:/AppProgPython/appprog/Lab2")
from iterator import PathIterator
from create_annotation import create_csv_list, write_into_csv
from create_copy_folder import copy_folder
from create_copy_random import copy_random


logging.basicConfig(level=logging.INFO)


