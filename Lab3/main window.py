import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
import logging

sys.path.insert(0, "Lab2")
from create_annotation import get_absolute_paths, get_relative_paths, write_annotation_to_csv
from copy_dataset_in_new_folder import replace_images
from copy_dataset_random_names import process_images
from iterator import DirectoryIterator

class AppLogger:
    def __init__(self, log_file="app.log"):
        logging.basicConfig(filename=log_file, level=logging.ERROR)
        self.logger = logging.getLogger()

    def log_error(self, message):
        self.logger.error(message)

logger = AppLogger()

class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Dataset Manager')
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.select_dataset_button = QPushButton('Select Dataset')
        self.layout.addWidget(self.select_dataset_button)
        self.select_dataset_button.clicked.connect(self.select_dataset)

        self.create_annotation_button = QPushButton('Create Annotation')
        self.layout.addWidget(self.create_annotation_button)
        self.create_annotation_button.clicked.connect(self.create_annotation)

        self.create_random_dataset_button = QPushButton('Create Random Dataset')
        self.layout.addWidget(self.create_random_dataset_button)
        self.create_random_dataset_button.clicked.connect(lambda: self.copy_dataset(True))

        self.create_dataset_button = QPushButton('Create No Random Dataset')
        self.layout.addWidget(self.create_dataset_button)
        self.create_dataset_button.clicked.connect(lambda: self.copy_dataset(False))

        self.next_cat_instance_button = QPushButton('Next cat')
        self.layout.addWidget(self.next_cat_instance_button)
        self.next_cat_instance_button.clicked.connect(lambda: self.show_next_instance('cat'))

        self.next_dog_instance_button = QPushButton('Next dog')
        self.layout.addWidget(self.next_dog_instance_button)
        self.next_dog_instance_button.clicked.connect(lambda: self.show_next_instance('dog'))

        self.quit_button = QPushButton('Exit')
        self.layout.addWidget(self.quit_button)
        self.quit_button.clicked.connect(self.close_application)

        self.cat_iterator = None
        self.dog_iterator = None
        self.dataset_path = None
        self.annotation_path = None
        self.destination_path = None

    def select_dataset(self) -> None:
        self.dataset_path = QFileDialog.getExistingDirectory(self, 'Select Dataset Folder')
        if self.dataset_path:
            QMessageBox.about(self, "Directory Selected", f"Selected Directory: {self.dataset_path}")
            self.create_csv(self.dataset_path)
            self.cat_iterator = DirectoryIterator('cat', f'{self.dataset_path}/data.csv')
            self.dog_iterator = DirectoryIterator('dog', f'{self.dataset_path}/data.csv')
        else:
            QMessageBox.about(self, "Error", "Please select a directory")

    def create_csv(self, dataset_folder: str) -> None:
        try:
            cat_absolute_paths = get_absolute_paths('cat', dataset_folder)
            cat_relative_paths = get_relative_paths('cat', dataset_folder)
            dog_absolute_paths = get_absolute_paths('dog', dataset_folder)
            dog_relative_paths = get_relative_paths('dog', dataset_folder)

            annotation_file = os.path.join(dataset_folder, 'data.csv')
            if os.path.exists(annotation_file):
                os.remove(annotation_file)

            write_annotation_to_csv(annotation_file, cat_absolute_paths, cat_relative_paths, 'cat')
            write_annotation_to_csv(annotation_file, dog_absolute_paths, dog_relative_paths, 'dog')

            self.annotation_path = annotation_file
        except Exception as error:
            logger.log_error(f"Failed to create annotation: {error}")
            QMessageBox.critical(self, "Error", f"Failed to create annotation: {error}")

    def create_annotation(self) -> None:
        try:
            if self.dataset_path:
                self.annotation_path, _ = QFileDialog.getSaveFileName(
                    self, 'Specify Annotation File', '', 'CSV Files (*.csv)')

                if self.annotation_path:
                    self.create_annotation_file()
        except Exception as ex:
            logger.log_error(f"Couldn't create annotation: {ex}")
            QMessageBox.critical(self, "Error", f"Couldn't create annotation: {ex}")

    def create_annotation_file(self) -> None:
        try:
            if self.annotation_path and self.dataset_path:
                for class_name in ['cat', 'dog']:
                    abs_paths = get_absolute_paths(class_name, self.dataset_path)
                    rel_paths = get_relative_paths(class_name, self.dataset_path)
                    write_annotation_to_csv(self.annotation_path, abs_paths, rel_paths, class_name)

                QMessageBox.about(self, "Success", "Annotation file successfully created.")
        except Exception as ex:
            logger.log_error(f"Couldn't create annotation: {ex}")
            QMessageBox.critical(self, "Error", f"Couldn't create annotation: {ex}")

    def copy_dataset(self, with_random: bool) -> None:
        try:
            if self.dataset_path:
                if not self.destination_path:
                    self.destination_path = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')

                if self.destination_path:
                    self.create_annotation_file()  # Вызываем создание аннотации перед копированием

                    for class_name in ['cat', 'dog']:
                        if with_random:
                            process_images(class_name, self.dataset_path, self.destination_path)
                        else:
                            replace_images(class_name, self.dataset_path)

                    QMessageBox.about(self, "Success", "Dataset successfully created.")
            else:
                QMessageBox.about(self, "Error", "Please select a directory")
        except Exception as ex:
            logger.log_error(f"Couldn't create dataset: {ex}")
            QMessageBox.critical(self, "Error", f"Couldn't create dataset: {ex}")

    def display_image(self, image_path: str) -> None:
        """
        Display an image in the QLabel.
        """
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            window_width = self.size().width()
            window_height = self.size().height()

            width_factor = window_width / pixmap.width()
            height_factor = window_height / pixmap.height()
            scale_factor = min(width_factor, height_factor)

            scaled_width = int(pixmap.width() * scale_factor)
            scaled_height = int(pixmap.height() * scale_factor)

            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText('Image not found')

    def show_next_instance(self, instance_class: str) -> None:
        """
        Display the next instance of the specified class.

        Args:
        - instance_class (str): The class of the instance to display ('cat' or 'dog').
        """
        iterator = None

        if instance_class == 'cat':
            iterator = self.cat_iterator
        elif instance_class == 'dog':
            iterator = self.dog_iterator

        if self.dataset_path and iterator:
            next_path = next(iterator)
            if next_path:
                self.display_image(next_path)
        else:
            QMessageBox.about(self, "Error", "Please select a directory")

    def close_application(self) -> None:
        choice = QMessageBox.question(
            self, 'Exit', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No
        )
        if choice == QMessageBox.Yes:
            sys.exit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
