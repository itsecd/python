import sys
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QFileDialog,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QPixmap

from iterator import ClassesIterator
from csv_annotation import make_list, write_in_file
from new_name_copy import copy_in_new_directory
from random_of_copy import copy_with_random

logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(400, 100, 1000, 1000)
        self.setWindowTitle("Tooltips")
        main_widget = QWidget()
        button_layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        # кнопки
        self.btn_create_of_annotation = self.add_button("Создать аннотацию", 250, 50)
        self.btn_copy = self.add_button("Копирование датасета", 250, 50)
        self.btn_random = self.add_button("Датасет с рандомными числами", 250, 50)
        self.btn_iterator = self.add_button("Начать итерацию", 150, 50)
        self.btn_next_rose = self.add_button("Следующая роза-->", 150, 50)
        self.btn_next_tulip = self.add_button("Следующий тюльпан-->", 150, 50)
        self.go_to_exit = self.add_button("Выйти из программы", 150, 50)

        # изображение
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setScaledContents(True)

        # делаем виджеты адаптивными по размер окна
        button_layout.addWidget(self.btn_create_of_annotation)
        button_layout.addWidget(self.btn_copy)
        button_layout.addWidget(self.btn_random)
        button_layout.addWidget(self.btn_iterator)
        button_layout.addWidget(self.btn_next_rose)
        button_layout.addWidget(self.btn_next_tulip)
        button_layout.addWidget(self.go_to_exit)
        image_layout.addLayout(button_layout)
        image_layout.addWidget(self.image_label)
        main_widget.setLayout(image_layout)

        self.setCentralWidget(main_widget)

        self.setWindowTitle("Main window")
        self.dataset_path = QFileDialog.getExistingDirectory(
            self, "Путь к папке базового датасет"
        )
        src = QLabel(f"Базовый датасет:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(800, 50))
        self.classes = ["rose", "tulip"]

        self.classes_iterator = None
        self.image_path = None

        # создание аннотации, копия + рандом
        self.btn_create_of_annotation.clicked.connect(self.create_annotation)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_random.clicked.connect(self.random)

        # кнопки итератора
        self.btn_iterator.clicked.connect(self.csv_path)
        self.btn_next_rose.clicked.connect(self.next_first)
        self.btn_next_tulip.clicked.connect(self.next_second)

        # выход из программы
        self.go_to_exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str, size_x: int, size_y: int):
        """The function creates buttons with the specified names and sizes"""
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def create_annotation(self):
        """The function creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите папку для создания файла аннотации:"
            )[0]
            l = make_list(self.dataset_path, self.classes)
            write_in_file(directory, l)
        except Exception as ex:
            logging.error(f"Couldn't create annotation: {ex.message}\n{ex.args}\n")

    def copy(self):
        """The function copy dataset with new name(class_number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите путь для создания файла-аннотации:"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета:"
            )
            copy_in_new_directory(self.dataset_path, self.classes, folder, directory)
        except Exception as ex:
            logging.error(f"Couldn't create copy: {ex.message}\n{ex.args}\n")

    def random(self):
        """The function copy dataset with new name(random number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите путь для создания файла аннотации:"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета аннотации:"
            )
            copy_with_random(self.dataset_path, self.classes, folder, directory)
        except Exception as ex:
            logging.error(f"Couldn't create randon copy: {ex.message}\n{ex.args}\n")

    def csv_path(self):
        """Function requests the path for the iteration file and creates an iterator"""
        try:
            path = QFileDialog.getOpenFileName(self, "Выберите файл для итерации:")[0]
            self.classes_iterator = ClassesIterator(
                path, self.classes[0], self.classes[1]
            )
        except Exception as ex:
            logging.error(f"Incorrect path: {ex.message}\n{ex.args}\n")

    def next_first(self):
        """Function returns the path to the next element of the first class
        and opens this image in the widget"""
        element = self.classes_iterator.next_first()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))

    def next_second(self):
        """Function returns the path to the next element of the second class
        and opens this image in the widget"""
        element = self.classes_iterator.next_second()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
