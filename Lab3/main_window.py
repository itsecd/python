import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
sys.path.insert(1, "K:/Pyth/PLab1/Lab2")
from generate_annotation import generate_annotation_file
from randomize_dataset import randomize_dataset_with_annotation


class DatasetApp(QWidget):
    def __init__(self):
        super().__init__()

        self.dataset_path = ""
        self.annotation_file_path = ""
        self.randomized_dataset_path = ""
        self.dataset_iterator = None

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
        self.create_randomized_dataset_btn.clicked.connect(self.create_randomized_dataset)

        self.next_brown_bear_btn = QPushButton('Next brown bear', self)
        self.next_brown_bear_btn.clicked.connect(self.show_next_brown_bear)

        self.next_polar_bear_btn = QPushButton('Next polar bear', self)
        self.next_polar_bear_btn.clicked.connect(self.show_next_polar_bear)

        # Image display
        self.image_label = QLabel(self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.browse_dataset_btn)
        layout.addWidget(self.create_annotation_btn)
        layout.addWidget(self.create_randomized_dataset_btn)
        layout.addWidget(self.next_brown_bear_btn)
        layout.addWidget(self.next_polar_bear_btn)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

    def browse_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if self.dataset_path:
            self.dataset_iterator = iter(self.get_dataset_files())

    def create_annotation(self):
        if self.dataset_path:
            self.annotation_file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotation File", "", "CSV Files (*.csv)")
            if self.annotation_file_path:
                generate_annotation_file(self.dataset_path, self.annotation_file_path)

    def create_randomized_dataset(self):
        if self.dataset_path:
            self.randomized_dataset_path = QFileDialog.getExistingDirectory(self, "Select Randomized Dataset Folder")
            if self.randomized_dataset_path:
                self.randomized_annotation_file_path, _ = QFileDialog.getSaveFileName(self, "Save Randomized Annotation File", "", "CSV Files (*.csv)")
                if self.randomized_annotation_file_path:
                    randomize_dataset_with_annotation(self.dataset_path, self.randomized_annotation_file_path,
                                                      self.randomized_dataset_path, self.classes, self.default_size)

    