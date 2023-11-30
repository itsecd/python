import sys
import os
import logging
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QFileDialog
)
from PyQt6.QtGui import QPixmap
sys.path.insert(1, "K:/Pyth/PLab1/Lab2")
from generate_annotation import generate_annotation_file
from randomize_dataset import randomize_dataset_with_annotation
from copy_dataset import copy_dataset_with_annotation
from iterator import DatasetIterator


class DatasetApp(QWidget):
    def __init__(self):
        super().__init__()

        self.dataset_path = ""
        self.annotation_file_path = ""
        self.randomized_dataset_path = ""
        self.dataset_iterator = None
        self.classes = ["brown_bear", "polar_bear"]
        self.default_size = 10

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dataset App')
        self.setGeometry(100, 100, 400, 300)

        # Buttons
        self.browse_dataset_btn = QPushButton('Browse Dataset', self)
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)

        self.create_annotation_btn = QPushButton('Create Annotation', self)
        self.create_annotation_btn.clicked.connect(self.create_annotation)

        self.create_randomized_dataset_btn = QPushButton('Create Randomized Dataset', self)
        self.create_randomized_dataset_btn.clicked.connect(lambda: self.create_dataset(randomize=True))

        self.create_copied_dataset_btn = QPushButton('Create Copied Dataset', self)
        self.create_copied_dataset_btn.clicked.connect(lambda: self.create_dataset(randomize=False)) 

        self.next_brown_bear_btn = QPushButton('Next brown bear', self)
        self.next_brown_bear_btn.clicked.connect(lambda: self.show_next_animal('brown_bear'))

        self.next_polar_bear_btn = QPushButton('Next polar bear', self)
        self.next_polar_bear_btn.clicked.connect(lambda: self.show_next_animal('polar_bear'))

        self.brown_iterator = None
        self.polar_iterator = None

        # Image display
        self.image_label = QLabel(self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.browse_dataset_btn)
        layout.addWidget(self.create_annotation_btn)
        layout.addWidget(self.create_randomized_dataset_btn)
        layout.addWidget(self.create_copied_dataset_btn)
        layout.addWidget(self.next_brown_bear_btn)
        layout.addWidget(self.next_polar_bear_btn)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        
    def browse_dataset(self):
        """Open a file dialog to select a dataset folder."""
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if self.dataset_path:
            self.dataset_iterator = iter(self.get_dataset_files())
            

    def create_annotation(self):
        """This function create a csv annotation."""
        try:
            if self.dataset_path:
                self.annotation_file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotation File", "", "CSV Files (*.csv)")
                if self.annotation_file_path:
                    generate_annotation_file(self.dataset_path, self.annotation_file_path)
        except Exception as ex:
            logging.error(f"Couldn't create annotation: {ex.message}\n{ex.args}\n")            

    def create_dataset(self, randomize: bool):
        """Create a randomized or copied dataset and corresponding CSV file using the selected dataset folder."""
        if self.dataset_path:
            if randomize:
                dataset_path = QFileDialog.getExistingDirectory(self, "Select Randomized Dataset Folder")
                annotation_file_path, _ = QFileDialog.getSaveFileName(self, "Save Randomized Annotation File", "", "CSV Files (*.csv)")
                if dataset_path and annotation_file_path:
                    randomize_dataset_with_annotation(self.dataset_path, annotation_file_path, dataset_path, self.classes, self.default_size)
            else:
                dataset_path = QFileDialog.getExistingDirectory(self, "Select Copied Dataset Folder")
                annotation_file_path, _ = QFileDialog.getSaveFileName(self, "Save Copied Annotation File", "", "CSV Files (*.csv)")
                if dataset_path and annotation_file_path:
                    copy_dataset_with_annotation(self.dataset_path, dataset_path, annotation_file_path)

    def get_dataset_files(self):
        """Generator function to yield file paths in the selected dataset folder."""
        if self.dataset_path:
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    yield os.path.join(root, file)

    def show_next_animal(self, animal_type):
        """Display the next image of the specified animal type in the dataset."""
        if self.dataset_iterator:
            try:
                file_path = next(self.dataset_iterator)
                while animal_type not in file_path.lower():
                    file_path = next(self.dataset_iterator)
                pixmap = QPixmap(file_path)
                self.image_label.setPixmap(pixmap)
            except StopIteration:
                print(f"No more {animal_type}s in the dataset.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DatasetApp()
    window.show()
    sys.exit(app.exec())

    