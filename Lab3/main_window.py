from create_copy_dataset import copy_dataset, CopyType
from create_annotation import write_annotation_to_csv
from iterator import ClassIterator
import sys
import os
import logging
from PyQt6 import QtWidgets, QtGui, QtCore

sys.path.insert(1, 'Lab2')

logging.basicConfig(level=logging.INFO)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Dataset Annotation App")

        # Create widgets
        self.create_annotation_button = QtWidgets.QPushButton(
            "Создать аннотацию")
        self.copy_dataset_button = QtWidgets.QPushButton("Скопировать датасет")
        self.random_copy_dataset_button = QtWidgets.QPushButton(
            "Скопировать датасет рандомно")
        self.show_tiger_button = QtWidgets.QPushButton(
            "Показать следующего тигра")
        self.show_leopard_button = QtWidgets.QPushButton(
            "Показать следующего леопарда")
        self.exit_button = QtWidgets.QPushButton("Выйти из программы")

        # Create a QLabel to display the images
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Connect buttons to functions
        self.create_annotation_button.clicked.connect(self.create_annotation)
        self.copy_dataset_button.clicked.connect(self.copy_dataset)
        self.random_copy_dataset_button.clicked.connect(
            self.random_copy_dataset)
        self.show_tiger_button.clicked.connect(self.show_tiger)

        # Layout setup
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.create_annotation_button)
        layout.addWidget(self.copy_dataset_button)
        layout.addWidget(self.random_copy_dataset_button)
        layout.addWidget(self.show_tiger_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def create_annotation(self):
        dataset_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для создания аннотации:')

        if dataset_folder:
            annotation_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Выберите папку для сохранения аннотации:', '', 'All Files (*)')

            if annotation_path:
                try:
                    write_annotation_to_csv(dataset_folder, annotation_path)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Annotation file created successfully!')
                except Exception as ex:
                    logging.error(f"Failed to create annotation: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Failed to create annotation: {ex}')

    def copy_dataset(self):
        main_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для копирования датасета:')

        if main_folder:
            new_copy_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Введите имя для новой копии датасета:', 'Новая копия')

            if ok and new_copy_name:
                try:
                    copy_dataset(main_folder, new_copy_name, CopyType.NUMBERED)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Dataset copied successfully!')
                except Exception as ex:
                    logging.error(f"Failed to copy dataset: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Failed to copy dataset: {ex}')

    def random_copy_dataset(self):
        main_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для копирования датасета:')

        if main_folder:
            new_copy_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Введите имя для новой рандом-копии датасета:', 'Новая копия')

            if ok and new_copy_name:
                try:
                    copy_dataset(main_folder, new_copy_name, CopyType.RANDOM)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Dataset copied successfully!')
                except Exception as ex:
                    logging.error(f"Failed to copy dataset: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Failed to copy dataset: {ex}')

    def show_tiger(self):

        if not hasattr(self, 'tiger_iterator') or not self.tiger_iterator:

            annotation_file = 'copy_dataset.csv'
            class_label = ["tiger"]

            self.tiger_iterator = ClassIterator(annotation_file, class_label)

            if not self.tiger_iterator:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', 'Failed to initialize image iterator.')
                return

        tiger_image_path = self.tiger_iterator.next_image()

        if tiger_image_path:
            print("Showing tiger image:", tiger_image_path)
            self.display_image(tiger_image_path)
        else:
            QtWidgets.QMessageBox.information(
                self, 'Information', 'No more tiger images in the dataset.')

    def display_image(self, image_path):
        print("Displaying image:", image_path)
        pixmap = QtGui.QPixmap(image_path)

        if pixmap.isNull():
            print("Failed to load image:", image_path)
        else:
            print("Image loaded successfully.")
            self.image_label.setPixmap(pixmap)
            self.image_label.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
