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

        self.copy_dataset_button = QtWidgets.QPushButton('Copy Dataset')
        self.copy_dataset_button.clicked.connect(self.copy_dataset)

        self.random_copy = QtWidgets.QCheckBox('Random')
        self.random_copy.stateChanged.connect(self.get_random)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.folder_path)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.create_annotation_button)
        layout.addWidget(self.copy_dataset_button)
        layout.addWidget(self.random_copy)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


    def browse_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_path.setText(folder_path)

    def create_annotation(self):
        folder_path = self.folder_path.text()
        QtWidgets.QMessageBox.information(self, 'Select', 'Select Destination File And Name Of Annotation file')
        destination_file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select Destination File', filter='(*.csv)')

        if folder_path and destination_file:
            create_annotation_file(folder_path, destination_file)
            QtWidgets.QMessageBox.information(self, 'Success', 'Annotation file created successfully.')

    def get_random(self):
        return self.random_copy.isChecked()

    def copy_dataset(self):
        source_folder = self.folder_path.text()
        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        annotation_file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select Destination File', filter='(*.csv)')
        random_copy =self.random_copy.isChecked()
        if source_folder and destination_folder:
            copy_and_rename_dataset(source_folder, destination_folder, destination_folder, random_copy)
            create_annotation_file(destination_folder, str(annotation_file))
            QtWidgets.QMessageBox.information(self, 'Success', 'Dataset copy successfully.')



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()