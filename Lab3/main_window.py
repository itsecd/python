import sys
import os
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMessageBox, QTextBrowser, QComboBox

# Importing functions from other modules
sys.path.insert(1, "C:\\Users\\ksush\\OneDrive\\Рабочий стол\\python-v8\\Lab2")
from create_annotation import create_annotation_file
from random_dataset import random_dataset
from file_iterator import FileIterator
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.iter = None
        # Initializing variables
        self.dataset_path = ""
        self.annotation_file_path = ""
        self.randomized_dataset_path = ""
        self.dataset_iterator = None
        self.classes = ["bad", "good"]
        self.default_size = 0
        # Создание ComboBox и добавление вариантов
        self.combo = QComboBox(self)
        self.combo.addItems(self.classes)
        self.combo.setCurrentIndex(0)

        # Initializing the user interface
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dataset Application')
        self.setGeometry(100, 100, 400, 300)

        # Buttons
        self.browse_dataset_btn = QPushButton('Select Data Folder', self)
        self.create_annotation_btn = QPushButton('Create Annotation', self)
        self.create_random_dataset_btn = QPushButton('Create Random Dataset', self)
        self.next_good_review_btn = QPushButton('Next Positive Review', self)
        self.next_bad_review_btn = QPushButton('Next Negative Review', self)

        # Connecting button signals to methods
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        self.create_annotation_btn.clicked.connect(self.create_annotation)
        self.create_random_dataset_btn.clicked.connect(self.create_random_dataset)
        self.next_good_review_btn.clicked.connect(lambda: self.show_next_review('Positive Review'))
        self.next_bad_review_btn.clicked.connect(lambda: self.show_next_review('Negative Review'))

        # Display image
        self.txt_file = QLabel(self)


        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.browse_dataset_btn)
        layout.addWidget(self.create_annotation_btn)
        layout.addWidget(self.create_random_dataset_btn)
        layout.addWidget(self.next_bad_review_btn)
        layout.addWidget(self.next_good_review_btn)
        layout.addWidget(self.txt_file)
        self.setLayout(layout)


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

    def create_random_dataset(self):
        """Create a random dataset."""
        if self.dataset_path:
            self.randomized_dataset_path = QFileDialog.getExistingDirectory(
                self, "Select Folder for Random Dataset"
            )
            if self.randomized_dataset_path:
                self.random_annotation_file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Random Annotation File", "", "CSV Files (*.csv)"
                )
                if self.random_annotation_file_path:
                    random_dataset(
                        self.dataset_path,
                        self.random_annotation_file_path,
                        self.randomized_dataset_path,
                        self.classes,
                        self.default_size,
                    )

    def browse_dataset(self):
        """Open dialog to select the data folder."""
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if self.dataset_path:
            self.dataset_iterator = self.get_dataset_files()

    def get_dataset_files(self):
        """Generator to enumerate file paths in the dataset."""
        if self.dataset_path:
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    yield os.path.join(root, file)

    def next(self):
        """Function returns the path to the next element of the class
        and opens text review in the widget"""
        if self.iter is None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return

        current_text = self.combo.currentText()

        if current_text == "good":
            element = self.iter.next_good()
        elif current_text == "bad":
            element = self.iter.next_bad()
        else:
            QMessageBox.information(None, "Недопустимое значение", "Выбрано недопустимое значение")
            return

        self.review_path = element

        if self.review_path is None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса закончились")
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
