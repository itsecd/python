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
    def __init__(self):
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
        self.next_cat_button.clicked.connect(lambda: self.get_next_instance('cat'))
        self.layout.addWidget(self.next_cat_button)

        self.next_dog_button = QtWidgets.QPushButton("Next Dog")
        self.next_dog_button.clicked.connect(lambda: self.get_next_instance('dog'))
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
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder_path_label.setText(f"Dataset Folder: {folder_path}")
        return folder_path

    def create_annotation_file(self):
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
        folder_path = self.get_dataset_path()
        new_dataset_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Destination Folder')

        if new_dataset_path:
            replace_images('cat', folder_path)
            replace_images('dog', folder_path)

            process_images('cat', folder_path, new_dataset_path)
            process_images('dog', folder_path, new_dataset_path)

    def copy_dataset(self):
        source_folder = self.get_dataset_path()
        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Destination Folder for Copy')

        if destination_folder:
            shutil.copytree(source_folder, os.path.join(
                destination_folder, os.path.basename(source_folder) + "_copy"))
            QtWidgets.QMessageBox.information(
                self, "Copy Dataset", "Dataset copied successfully.")

    def copy_random_dataset(self):
        source_folder = self.get_dataset_path()
        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Destination Folder for Random Copy')

        if destination_folder:
            random_suffix = str(random.randint(1, 1000))
            shutil.copytree(source_folder, os.path.join(
                destination_folder, os.path.basename(source_folder) + f"_copy_{random_suffix}"))
            QtWidgets.QMessageBox.information(
                self, "Copy Random Dataset", "Random dataset copy created successfully.")

    def get_next_instance(self, class_name):
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

    def show_next_cat(self):
        if self.dataset_path and self.cat_iterator:
            next_path = next(self.cat_iterator)
            if next_path:
                self.show_image(next_path)

    def show_next_dog(self):
        if self.dataset_path and self.dog_iterator:
            next_path = next(self.dog_iterator)
            if next_path:
                self.show_image(next_path)

    def display_image(self, image_path: str) -> None:
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
