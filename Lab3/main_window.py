import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
)

sys.path.insert(1, "D:\Study\Applied Programming (Python)\Applied-Programming\Lab2")
from csv_build import make_pathlist, write_into_file
from randomize import randomize_dataset
from unify import unify_dataset

logging.basicConfig(level=logging.INFO)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(1000, 200, 1000, 1000)
        self.setMaximumSize(1000, 1000)

        box_layout = QVBoxLayout()

        self.setWindowTitle('Dataset operation')
        self.data_path = os.path.abspath("dataset")
        src = QLabel(f"Base dataset:\n{self.data_path}", self)
        src.setFixedSize(QSize(500,80))
        box_layout.addWidget(src)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()