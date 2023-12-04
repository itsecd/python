import sys
import os
import logging
from PyQt6.QtWidgets import (
    QWidget, 
    QApplication, QMainWindow, 
    QLabel,QPushButton, 
    QVBoxLayout,QFileDialog, 
    QMessageBox, QTextBrowser, 
    QComboBox, QInputDialog
    )


sys.path.insert(1, "C:\\Users\\ksush\\OneDrive\\Рабочий стол\\python-v8\\Lab2")
from create_annotation import create_annotation_file
from random_dataset import random_dataset
from copy_dataset import copy_dataset
from file_iterator import FileIterator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.iter = None
        self.dataset_path = ""
        self.annotation_file_path = ""
        self.randomized_dataset_path = ""
        self.dataset_iterator = None
        self.classes = ["bad", "good"]
        self.default_size = 1000
        self.combo = QComboBox(self)
        self.combo.addItems(self.classes)
        self.combo.setCurrentIndex(0)
        self.dataset_type = ["copy","random"]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dataset Application')
        self.setGeometry(100, 100, 400, 300)

        self.browse_dataset_btn = QPushButton('Select Data Folder', self)
        self.create_annotation_btn = QPushButton('Create Annotation', self)
        self.create_random_dataset_btn = QPushButton('Create Random Dataset', self)
        self.create_copy_dataset_btn = QPushButton('Create Copy Dataset', self)
        self.next_good_review_btn = QPushButton('Next Positive Review', self)
        self.next_bad_review_btn = QPushButton('Next Negative Review', self)

        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        self.create_annotation_btn.clicked.connect(self.create_annotation)
        self.create_random_dataset_btn.clicked.connect(lambda: self.create_dataset('random'))
        self.create_copy_dataset_btn.clicked.connect(lambda: self.create_dataset('copy'))
        self.next_good_review_btn.clicked.connect(lambda: self.next('good'))
        self.next_bad_review_btn.clicked.connect(lambda: self.next('bad'))

        self.txt_file = QLabel(self)
        self.text_label = QTextBrowser(self)
        self.text_label.setText("Здесь будет отзыв")
        self.text_label.setFixedSize(600, 400)

        layout = QVBoxLayout()
        layout.addWidget(self.browse_dataset_btn)
        layout.addWidget(self.create_annotation_btn)
        layout.addWidget(self.create_random_dataset_btn)
        layout.addWidget(self.create_copy_dataset_btn)
        layout.addWidget(self.next_bad_review_btn)
        layout.addWidget(self.next_good_review_btn)
        layout.addWidget(self.txt_file)
        layout.addWidget(self.text_label)  

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


    def create_annotation(self):
        """Create an annotation file for the selected dataset."""
        try:
            if self.dataset_path:
                self.annotation_file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Annotation File", "", "CSV Files (*.csv)"
                )
                if self.annotation_file_path:
                    create_annotation_file(self.dataset_path, self.annotation_file_path)
        except Exception as ex:
            logging.error(f"Failed to create annotation: {ex}\n")


    def create_dataset(self, dataset_type):
        """Create a dataset based on the given type (copy or random)."""
        if self.dataset_path:
            dataset_path = QFileDialog.getExistingDirectory(
                self, f"Select Folder for {dataset_type.capitalize()} Dataset"
            )
            if dataset_path:
                subfolder_name, _ = QInputDialog.getText(
                    self, 'Subfolder Name', 'Enter Subfolder Name:'
                )
                if subfolder_name:
                    subfolder_path = os.path.join(dataset_path, subfolder_name)
                    if not os.path.exists(subfolder_path):
                        os.makedirs(subfolder_path)

                    annotation_file_path, _ = QFileDialog.getSaveFileName(
                        self, f"Save {dataset_type.capitalize()} Annotation File", "", "CSV Files (*.csv)"
                    )
                    if annotation_file_path:
                        if dataset_type == 'copy':
                            copy_dataset(
                                self.dataset_path,
                                subfolder_path,
                                self.classes,
                                annotation_file_path,
                            )
                            self.copy_dataset_path = dataset_path
                            self.copy_annotation_file_path = annotation_file_path
                        elif dataset_type == 'random':
                            random_dataset(
                                self.dataset_path,
                                subfolder_path,
                                self.default_size,
                                self.classes,
                                annotation_file_path,
                            )
                            self.random_dataset_path = dataset_path
                            self.random_annotation_file_path = annotation_file_path


    def browse_dataset(self):
        """Open dialog to select the data folder."""
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if self.dataset_path:
            dataset_iterator = self.get_dataset_files()
            dataset_files = list(dataset_iterator)  
            self.iter = FileIterator(dataset_files)


    def get_dataset_files(self):
        """Generator to enumerate file paths in the dataset."""
        if self.dataset_path:
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    yield os.path.join(root, file)


    def next(self, review_type):
        """Function returns the path to the next element of the class
        and opens text review in the widget"""
        if self.iter is None:
            QMessageBox.information(None, "File not selected", "No file selected for iteration")
            return

        if review_type == "good":
            element = self.iter.next_good()
        elif review_type == "bad":
            element = self.iter.next_bad()
        else:
            QMessageBox.information(None, "Invalid value", "An invalid value has been selected")
            return

        self.review_path = element

        if self.review_path is None:
            QMessageBox.information(None, "End of class", "No more files for this class")
            return

        self.text_label.update()
        
        with open(self.review_path, 'r', encoding='utf-8') as file:
            self.txt_file.setText(self.review_path)
            self.text_label.setText(file.read())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
