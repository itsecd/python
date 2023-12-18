import os
import sys
import logging
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt5.QtGui import QPixmap

sys.path.insert(1, r"C:\Users\Ceh9\PycharmProjects\pythonProject\Lab2")
from create_annotation import create_annotation_file
from copy_dataset import copy_and_rename_dataset, generate_random_set
from iterator import ReviewIterator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dataset Processing App')
        self.setGeometry(300, 300, 200, 200)

        self.folder_label = QtWidgets.QLabel('Select Dataset Folder:')
        self.folder_path = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse_folder)

        self.create_annotation_button = QtWidgets.QPushButton('Create Annotation File')
        self.create_annotation_button.clicked.connect(self.create_annotation)


        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.folder_path)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.create_annotation_button)


        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_path.setText(folder_path)

    def create_annotation(self):
        folder_path = self.folder_path.text()
        QtWidgets.QMessageBox.information(self, 'Select', 'Select Destination File And Name Of Annotation file')
        destination_file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select Destination File', filter='CSV Files (*.csv)')

        if folder_path and destination_file:
            create_annotation_file(folder_path, destination_file)
            QtWidgets.QMessageBox.information(self, 'Success', 'Annotation file created successfully.')



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()