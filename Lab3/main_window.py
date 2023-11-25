import sys
import os
import logging
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import QFileDialog
sys.path.insert(1, 'Lab2')
from create_copy_dataset import copy_dataset, CopyType
from create_annotation import write_annotation_to_csv
from iterator import ClassIterator


logging.basicConfig(level=logging.INFO)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.recursion_counter = 0

    def init_ui(self) -> None:
        self.setWindowTitle("Dataset Annotation App")
        self.setFixedSize(1000, 540)

        button_size = QtCore.QSize(400, 86)
        self.create_annotation_button = QtWidgets.QPushButton(
            "Создать аннотацию", self)
        self.create_annotation_button.setFixedSize(button_size)

        self.copy_dataset_button = QtWidgets.QPushButton(
            "Скопировать датасет", self)
        self.copy_dataset_button.setFixedSize(button_size)

        self.random_copy_dataset_button = QtWidgets.QPushButton(
            "Скопировать датасет рандомно", self)
        self.random_copy_dataset_button.setFixedSize(button_size)

        self.show_tiger_button = QtWidgets.QPushButton(
            "Показать следующего тигра", self)
        self.show_tiger_button.setFixedSize(button_size)

        self.show_leopard_button = QtWidgets.QPushButton(
            "Показать следующего леопарда", self)
        self.show_leopard_button.setFixedSize(button_size)

        self.exit_button = QtWidgets.QPushButton("Выйти из программы", self)
        self.exit_button.setFixedSize(button_size)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setFixedHeight(524)

        self.create_annotation_button.clicked.connect(self.create_annotation)
        self.copy_dataset_button.clicked.connect(self.copy_dataset)
        self.random_copy_dataset_button.clicked.connect(
            self.random_copy_dataset)
        self.show_tiger_button.clicked.connect(self.show_tiger)
        self.show_leopard_button.clicked.connect(self.show_leopard)
        self.exit_button.clicked.connect(self.close)

        left_layout = QtWidgets.QVBoxLayout(self)
        left_layout.addWidget(self.create_annotation_button)
        left_layout.addWidget(self.copy_dataset_button)
        left_layout.addWidget(self.random_copy_dataset_button)
        left_layout.addWidget(self.show_tiger_button)
        left_layout.addWidget(self.show_leopard_button)
        left_layout.addWidget(self.exit_button)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.scroll_area)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout, 0)
        main_layout.addLayout(right_layout, 5)

        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)

    def create_annotation(self) -> None:
        dataset_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для создания аннотации:')

        if dataset_folder:
            annotation_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Выберите папку для сохранения аннотации:', '', 'All Files (*)')

            if annotation_path:
                try:
                    write_annotation_to_csv(dataset_folder, annotation_path)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Файл аннотация успешно создан!')
                except Exception as ex:
                    logging.error(f"Failed to create annotation: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Не удалось создать аннотацию.: {ex}')

    def copy_dataset(self) -> None:
        main_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для копирования:')

        if main_folder:
            new_copy_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Введите имя для новой копии:', 'Новая копия')

            if ok and new_copy_name:
                try:
                    copy_dataset(main_folder, new_copy_name, CopyType.NUMBERED)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Набор данных успешно скопирован!')
                except Exception as ex:
                    logging.error(f"Failed to copy dataset: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Не удалось скопировать набор данных: {ex}')

    def random_copy_dataset(self) -> None:
        main_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для копирования:')

        if main_folder:
            new_copy_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Введите имя для новой рандом-копии:', 'Новая копия')

            if ok and new_copy_name:
                try:
                    copy_dataset(main_folder, new_copy_name, CopyType.RANDOM)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Набор данных успешно скопирован!')
                except Exception as ex:
                    logging.error(f"Failed to copy dataset: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Не удалось скопировать набор данных: {ex}')

    def show_tiger(self) -> None:
        if not hasattr(self, 'tiger_iterator') or not self.tiger_iterator:
            csv_file, _ = QFileDialog.getOpenFileName(
                self, 'Выберите CSV-файл', '', 'CSV Files (*.csv);;All Files (*)')

            if not csv_file.lower().endswith('.csv'):
                QtWidgets.QMessageBox.critical(
                    self, 'Error', 'Выбран неверный файл. Пожалуйста, выберите файл CSV.')
                return

            class_label = ["tiger"]

            self.tiger_iterator = ClassIterator(csv_file, class_label)

            if not self.tiger_iterator:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', 'Не удалось инициализировать итератор.')
                return

        tiger_image_path = self.tiger_iterator.next_image()

        if tiger_image_path:
            print("Showing tiger image:", tiger_image_path)
            self.display_image(tiger_image_path)
        else:
            QtWidgets.QMessageBox.information(
                self, 'Information', 'В наборе данных больше нет изображений.')

    def show_leopard(self) -> None:
        if not hasattr(self, 'leopard_iterator') or not self.leopard_iterator:

            csv_file, _ = QFileDialog.getOpenFileName(
                self, 'Выберите CSV-файл', '', 'CSV Files (*.csv);;All Files (*)')

            if not csv_file.lower().endswith('.csv'):
                QtWidgets.QMessageBox.critical(
                    self, 'Error', 'Выбран неверный файл. Пожалуйста, выберите файл CSV.')
                return

            class_label = ["leopard"]

            self.leopard_iterator = ClassIterator(csv_file, class_label)

            if not self.leopard_iterator:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', 'В наборе данных больше нет изображений.')
                return

        leopard_image_path = self.leopard_iterator.next_image()

        if leopard_image_path:
            print("Showing leopard image:", leopard_image_path)
            self.display_image(leopard_image_path)
        else:
            QtWidgets.QMessageBox.information(
                self, 'Information', 'В наборе данных больше нет изображений.')

    def display_image(self, image_path: str) -> None:
        print("Displaying image:", image_path)
        pixmap = QtGui.QPixmap(image_path)

        if pixmap.isNull():
            print("Не удалось загрузить изображение:", image_path)
        else:
            print("Изображение успешно загружено.")
            pixmap = pixmap.scaledToWidth(self.scroll_area.width())
            self.image_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignHCenter)
            self.image_label.setPixmap(pixmap)
            self.image_label.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
