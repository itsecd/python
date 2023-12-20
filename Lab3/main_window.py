import os
import sys
import logging
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog
from PyQt5.QtGui import QPixmap
from annotation import create_annotation_file
from dataset_random_copy import copy_dataset
from iterator import ClassIterator, ImageIterator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.folder_path = ""
        self.annotation_file_path = ""
        self.dest_folder_path = ""

        self.class_iterator = None
        self.dataset_iterator = None

        self.init_ui()

    def init_ui(self):
        """
        interface
        """
        layout = QVBoxLayout()

        self.label_folder = QLabel("Select the source dataset folder:")
        layout.addWidget(self.label_folder)

        self.btn_browse_folder = QPushButton("Browse")
        self.btn_browse_folder.clicked.connect(self.browse_folder)
        layout.addWidget(self.btn_browse_folder)

        self.label_annotation = QLabel("Enter the annotation file path:")
        layout.addWidget(self.label_annotation)

        self.edit_annotation = QLineEdit()
        layout.addWidget(self.edit_annotation)

        self.btn_create_annotation = QPushButton("Create Annotation File")
        self.btn_create_annotation.setEnabled(False)
        self.btn_create_annotation.clicked.connect(self.create_annotation)
        layout.addWidget(self.btn_create_annotation)

        self.label_dest_folder = QLabel("Select the destination folder:")
        layout.addWidget(self.label_dest_folder)

        self.btn_browse_dest_folder = QPushButton("Browse")
        self.btn_browse_dest_folder.clicked.connect(self.browse_dest_folder)
        layout.addWidget(self.btn_browse_dest_folder)

        self.btn_copy_dataset = QPushButton("Copy and Rename Dataset")
        self.btn_copy_dataset.setEnabled(False)
        self.btn_copy_dataset.clicked.connect(self.copy_dataset)
        layout.addWidget(self.btn_copy_dataset)

        self.image_label = QLabel("Image will be displayed here.")
        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()

        self.btn_next_cat = QPushButton("Next cat")
        self.btn_next_cat.clicked.connect(lambda: self.show_next_instance("cat"))
        btn_layout.addWidget(self.btn_next_cat)

        self.btn_next_dog = QPushButton("Next dog")
        self.btn_next_dog.clicked.connect(lambda: self.show_next_instance("dog"))
        btn_layout.addWidget(self.btn_next_dog)

        self.btn_next_cat.setEnabled(False)
        self.btn_next_dog.setEnabled(False)

        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setWindowTitle("Work with dataset")
        self.setMinimumSize(400, 400)

    def show_next_instance(self, target_class: str) -> None:
        """
        show the next instance
        Parameters
        ----------
        target_class : str
        """
        if target_class not in ["cat", "dog"]:
            logger.error("Invalid target class. Supported classes: 'cat', 'dog'")
            return

        if self.class_iterator is not None:
            try:
                if target_class == "cat":
                    next_instance = next(self.image_iterators["cat"])
                else:
                    next_instance = next(self.image_iterators["dog"])

                self.display_image(next_instance)
                logger.info(f"Displaying the following {target_class}: {next_instance}")
            except StopIteration:
                logger.warning(f"There are no more instances {target_class}.")
                self.class_iterator = None
        else:
            logger.error("The dataset is not loaded. First, copy the dataset.")

    def display_image(self, image_path: str) -> None:
        """
        display the image at the specified path
        Parameters
        ----------
        image_path : str;
        """
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

    def browse_folder(self) -> None:
        """
        the function allows you to select the destination folder for dataset original
        """
        self.folder_path = QFileDialog.getExistingDirectory(self, 'Select Source Dataset Folder')
        self.label_folder.setText(f"Selected Source Dataset Folder: {self.folder_path}")

        if self.folder_path:
            self.cat_annotation_file = os.path.join(self.folder_path, 'cat_annotation.csv')
            self.dog_annotation_file = os.path.join(self.folder_path, 'dog_annotation.csv')

            self.edit_annotation.setText(self.cat_annotation_file)

            self.btn_create_annotation.setEnabled(True)
            self.btn_copy_dataset.setEnabled(True)
        else:
            self.btn_create_annotation.setEnabled(False)
            self.btn_copy_dataset.setEnabled(False)

    def create_annotation(self) -> None:
        """
       func create annotations
        """
        create_annotation_file(self.folder_path, os.listdir(self.folder_path), self.dest_folder_path)

        self.cat_annotation_file = os.path.join(self.folder_path, 'cat_annotation.csv')
        self.dog_annotation_file = os.path.join(self.folder_path, 'dog_annotation.csv')

        if os.path.exists(self.cat_annotation_file) and os.path.exists(self.dog_annotation_file):
            self.btn_next_cat.setEnabled(True)
            self.btn_next_dog.setEnabled(True)

    def browse_dest_folder(self) -> None:
        """
        the function allows you to select the destination folder
        """
        self.dest_folder_path = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        self.label_dest_folder.setText(f"Selected Destination Folder: {self.dest_folder_path}")

    def copy_dataset(self) -> None:
        """
        the function copies the dataset to the selected directory
        """
        try:
            copy_dataset(self.folder_path, self.dest_folder_path, randomize=False)

            cat_annotation_file = os.path.join(self.folder_path, 'cat_annotation.csv')
            dog_annotation_file = os.path.join(self.folder_path, 'dog_annotation.csv')

            if os.path.exists(cat_annotation_file) and os.path.exists(dog_annotation_file):
                self.image_iterators = {
                    "cat": ImageIterator(cat_annotation_file, "cat", self.dest_folder_path),
                    "dog": ImageIterator(dog_annotation_file, "dog", self.dest_folder_path)
                }

                self.class_iterator = ClassIterator(cat_annotation_file, ["cat", "dog"], self.dest_folder_path)
                self.dataset_iterator = iter(self.class_iterator)
            else:
                logger.error(f"The annotation files for cat and dog were not found.")
        except Exception as e:
            logger.exception(f"Error in copying dataset: {e}")


if __name__ == "__main__":
    print(sys.path)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())