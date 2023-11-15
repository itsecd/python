import sys
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt6 import QtGui, QtWidgets


from csv_annotation import make_list, write_in_file
from new_name_copy import copy_in_new_directory
from random_of_copy import copy_with_random


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(400, 100, 1000, 1000)
        self.setWindowTitle("Tooltips")

        # кнопки
        self.btn_create_of_annotation = self.add_button(
            "Создать аннотацию", 250, 50, 630, 50
        )
        self.btn_copy = self.add_button("Копирование датасета", 250, 50, 630, 100)
        self.btn_random = self.add_button(
            "Датасет с рандомными числами", 250, 50, 630, 150
        )
        self.btn_next_rose = self.add_button("Следующая роза-->", 150, 50, 630, 250)
        self.btn_next_tulip = self.add_button("Следующий тюльпан-->", 150, 50, 630, 350)
        self.go_to_exit = self.add_button("Выйти из программы", 150, 50, 630, 500)

        self.setWindowTitle("Main window")
        self.dataset_path = QFileDialog.getExistingDirectory(
            self, "Путь к папке базового датасет"
        )
        src = QLabel(f"Базовый датасет:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(800, 50))
        self.classes = ["rose", "tulip"]

        # создание аннотации, копия + рандом
        self.btn_create_of_annotation.clicked.connect(self.create_annotation)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_random.clicked.connect(self.random)

        # выход из программы
        self.go_to_exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str, size_x: int, size_y: int, x: int, y: int):
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.move(x, y)
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def create_annotation(self):
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите папку для создания файла аннотации:"
            )[0]
            l = make_list(self.dataset_path, self.classes)
            write_in_file(directory, l)
        except OSError:
            print("error")

    def copy(self):
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите папку для создания файла аннотации:"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета аннотации:"
            )
            copy_in_new_directory(self.dataset_path, self.classes, folder, directory)

        except OSError:
            print("error")

    def random(self):
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите папку для создания файла аннотации:"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета аннотации:"
            )
            copy_with_random(self.dataset_path, self.classes, folder, directory)
        except OSError:
            print("error")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()
