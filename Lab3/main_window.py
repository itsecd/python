import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
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

    