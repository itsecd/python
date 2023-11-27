import os
import sys
import logging

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)
from PyQt6.QtGui import QPixmap
sys.path.append("C:\\Users\\Yana\\Documents\\python-v6\\Lab2")
from csv_ import write_csv,make_list
from iterator import Iterator
from copy_dataset import copy_dataset



logging.basicConfig(filename="py_log3.log", filemode="a", level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pictures")
        self.setFixedSize(1080,900)
        widget=QWidget()
        buttons_layout=QVBoxLayout()
        layout=QGridLayout()

        self.dataset=os.path.abspath("Lab1\dataset")
        print(self.dataset)
        base=QLabel(f'Basic dataset:{self.dataset}',self)
        base.setFixedSize(QSize(350,50))
        buttons_layout.addWidget(base)

        #buttons
        self.button_annotation=self.add_button("Create annotation",300,50)
        self.button_сopy_dataset=self.add_button("Copy dataset",300,50)
        self.button_сopy_random=self.add_button("Copy dataset with random numbers",300,50)
        buttons_layout.addWidget(self.button_annotation)
        buttons_layout.addWidget(self.button_сopy_dataset)
        buttons_layout.addWidget(self.button_сopy_random)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout,0,1)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.tags=['tiger','leopard']

        #app functions
        self.button_annotation.clicked.connect(self.create_annotation)
        #self.button_сopy_dataset.connect(self.copy_dataset)

        self.show()

    def add_button(self, name:str, size_x:str, size_y:str):
        button=QPushButton(name,self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_x,size_y))
        return button
          
    def create_annotation(self):
        try:
            directory = QFileDialog.getSaveFileName(self,"Select folder - ",
            "","CSV File(*.csv)",)[0]
            if directory == "":
                QMessageBox.information(None, "Path error")
                return
            l=make_list(self.dataset, self.tags)
            print('make list')
            write_csv(directory, l)
            QMessageBox.information(None, "Done", "Annotation created")
        except Exception as e:
            logging.error(f"Annotation not created:{e}")
"""
    def copy_dataset(self):
        try:
            directory = QFileDialog.getSaveFileName(self,"Select folder - ",
            "","CSV File(*.csv)",)[0]
            folder = QFileDialog.getExistingDirectory(self, "Select folder for copy - ")

            if (folder == "") or (directory == ""):
                QMessageBox.information(None, "Path error")
                return
            copy_dataset(self.dataset_path, self.tags,self.new_dir ,folder, option)
            QMessageBox.information(None, "Done", "Dataset copied")
        except Exception as ex:
            logging.error(f"Couldn't create copy: {e}")
"""


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()