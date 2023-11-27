import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import shutil
import os
import random

sys.path.insert(0, "Lab2")
from create_annotation import get_absolute_paths, get_relative_paths, write_annotation_to_csv
from copy_dataset_in_new_folder import replace_images
from copy_dataset_random_names import process_images
from path_of_next import get_next
from iterator import DirectoryIterator


class MainWindow(QtWidgets.QMainWindow):
    """
    Main window class for the Dataset Annotation App.

    Attributes:
    - dataset_path: The path to the dataset folder.
    - cat_iterator: Iterator for the 'cat' class in the dataset.
    - dog_iterator: Iterator for the 'dog' class in the dataset.
    """
    def __init__(self):
        """
        Initialize the main window and set up the user interface.
        """
        super().__init__()

        self.setWindowTitle("Dataset Annotation App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout()

        self.folder_path_label = QtWidgets.QLabel("Dataset Folder:")
        self.layout.addWidget(self.folder_path_label)

        self.create_annotation_button = QtWidgets.QPushButton("Create Annotation File")
        self.create_annotation_button.clicked.connect(self.create_annotation_file)
        self.layout.addWidget(self.create_annotation_button)

        self.create_dataset_button = QtWidgets.QPushButton("Create Dataset")
        self.create_dataset_button.clicked.connect(self.create_dataset)
        self.layout.addWidget(self.create_dataset_button)

        self.copy_dataset_button = QtWidgets.QPushButton("Copy Dataset")
        self.copy_dataset_button.clicked.connect(self.copy_dataset)
        self.layout.addWidget(self.copy_dataset_button)

        self.copy_random_dataset_button = QtWidgets.QPushButton("Copy Dataset Randomly")
        self.copy_random_dataset_button.clicked.connect(self.copy_random_dataset)
        self.layout.addWidget(self.copy_random_dataset_button)

        self.next_cat_button = QtWidgets.QPushButton("Next Cat")
        self.next_cat_button.clicked.connect(lambda: self.show_next_instance('cat'))        
        self.layout.addWidget(self.next_cat_button)

        self.next_dog_button = QtWidgets.QPushButton("Next Dog")
        self.next_dog_button.clicked.connect(lambda: self.show_next_instance('dog'))
        self.layout.addWidget(self.next_dog_button)

        self.exit_button = QtWidgets.QPushButton("EXIT", self)
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.central_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.central_layout.addLayout(self.layout)
        self.central_layout.addWidget(self.scroll_area)

        self.dataset_path = None
        self.cat_iterator = None
        self.dog_iterator = None

    def get_dataset_path(self):
        """
        Open a dialog to select the dataset folder and set the dataset_path attribute.

        Returns:
        - str: The selected dataset folder path.
        """
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_path_label.setText(f"Dataset Folder: {folder_path}")

        self.dataset_path = folder_path
        self.cat_iterator = DirectoryIterator('cat', folder_path)
        self.dog_iterator = DirectoryIterator('dog', folder_path)

        return folder_path

    def create_annotation_file(self):
        """
        Create an annotation file for the dataset.

        This method prompts the user to select a folder, and then creates an annotation
        file by collecting absolute and relative paths for 'cat' and 'dog' classes.

        Returns:
        - None
        """
        folder_path = self.get_dataset_path()
        annotation_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Annotation File', '', 'CSV Files (*.csv)')

        if annotation_file_path:
            cat_absolute_paths = get_absolute_paths('cat', folder_path)
            cat_relative_paths = get_relative_paths('cat', folder_path)
            dog_absolute_paths = get_absolute_paths('dog', folder_path)
            dog_relative_paths = get_relative_paths('dog', folder_path)

            write_annotation_to_csv(
                annotation_file_path, cat_absolute_paths, cat_relative_paths, 'cat')
            write_annotation_to_csv(
                annotation_file_path, dog_absolute_paths, dog_relative_paths, 'dog')

    def create_dataset(self):
        """
        Create a new dataset by replacing images in the original dataset.

        This method prompts the user to select a destination folder and then replaces
        images for 'cat' and 'dog' classes in the original dataset.

        Returns:
        - None
        """
        folder_path = self.get_dataset_path()
        new_dataset_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Destination Folder')

        if new_dataset_path:
            replace_images('cat', folder_path)
            replace_images('dog', folder_path)

            process_images('cat', folder_path, new_dataset_path)
            process_images('dog', folder_path, new_dataset_path)

    def copy_dataset(self, random_suffix=None):
        """
        Copy the dataset to a new location.
        
        This method prompts the user to select a destination folder and copies the dataset
        to that location. If a random_suffix is provided, it will be added to the folder name.

        Parameters:
        - random_suffix (str or None): Random suffix to be added to the folder name (default is None).
        """
        source_folder = self.get_dataset_path()
        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Destination Folder for Copy')

        if destination_folder:
            if random_suffix is not None:
                destination_folder = os.path.join(destination_folder, f"{os.path.basename(source_folder)}_copy_{random_suffix}")
            else:
                destination_folder = os.path.join(destination_folder, f"{os.path.basename(source_folder)}_copy")

            shutil.copytree(source_folder, destination_folder)
            action = "Random Dataset Copy" if random_suffix else "Dataset Copy"
            QtWidgets.QMessageBox.information(self, action, f"{action} created successfully.")


    def get_next_instance(self, class_name):
        """
        Get the next instance of the specified class from an annotation file.

        This method prompts the user to select an annotation file and displays the path
        of the next instance of the specified class using the DirectoryIterator.

        Parameters:
        - class_name (str): The class name ('cat' or 'dog').

        Returns:
        - None
        """
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Annotation File', '', 'CSV Files (*.csv)')

        if csv_path:
            iterator = DirectoryIterator(class_name, csv_path)
            next_path = next(iterator, None)

            if next_path:
                QtWidgets.QMessageBox.information(
                    self, "Next Instance", f"Next {class_name}: {next_path}")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"No more instances of {class_name}.")

    def show_next_instance(self, class_name):
        """
        Show the next instance of the specified class from an annotation file.

        This method prompts the user to select an annotation file and displays the
        next instance of the specified class using the corresponding iterator.

        Parameters:
        - class_name (str): The class name ('cat' or 'dog').

        Returns:
        - None
        """
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Annotation File', '', 'CSV Files (*.csv)')

        if csv_path:
            iterator = DirectoryIterator(class_name, csv_path)
            class_iterator = self.cat_iterator if class_name == 'cat' else self.dog_iterator
            class_iterator = DirectoryIterator(class_name, csv_path)
            next_path = next(class_iterator, None)

            if next_path:
                self.display_image(next_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"No more instances of {class_name}.")



    def display_image(self, image_path: str) -> None:
        """
        Display the image at the specified path in the scroll area.

        This method loads the image, scales it to fit the width of the scroll area,
        and displays it using a QLabel.

        Parameters:
        - image_path (str): The path of the image to be displayed.

        Returns:
        - None
        """
        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            print("Не удалось загрузить изображение:", image_path)
        else:
            print("Изображение успешно загружено.")
            pixmap = pixmap.scaledToWidth(self.scroll_area.width())

            label = QtWidgets.QLabel()
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(pixmap)

            container_widget = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container_widget)
            container_layout.addWidget(label)
            container_layout.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignTop)

            self.scroll_area.setWidget(container_widget)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
