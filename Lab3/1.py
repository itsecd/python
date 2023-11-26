import sys
import os
import logging
import typing
from PyQt6 import QtCore
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow,QPushButton,QMessageBox,QLabel,QFileDialog,QVBoxLayout,QWidget,QGridLayout)
from PyQt6.QtGui import QPixmap

sys.path.insert(1, "D:/AppProgPython/appprog/Lab2")
from iterator import PathIterator
from create_annotation import create_csv_list, write_into_csv
from create_copy_folder import copy_folder
from create_copy_random import copy_random


logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self)-> None:
        super().__init__()

        self.setGeometry(100, 100, 800, 800)
        self.setWindowTitle("Toolshit")

        main_widget = QWidget()
        box_layout = QVBoxLayout()
        grid_layout = QGridLayout()

        self.setWindowTitle("Main")
        self.data_path = os.path.abspath("dataset")
        src = QLabel(f"Basic dataset:\n{self.data_path}", self)
        src.setFixedSize(QSize(250,40))
        box_layout.addWidget(src)


        self.btn_create = self.add_button("Создать аннотацию", 250, 40)
        self.btn_copy = self.add_button("Копирование папки датасета", 250, 40)
        self.btn_rand = self.add_button("Создать датасет со случ. числами", 250, 40)
        self.btn_iterator = self.add_button("Начать итерацию", 200, 40)
        self.btn_next_1 = self.add_button("Следующий отзыв с рейтингом 1", 200, 30)
        self.btn_next_2 = self.add_button("Следующий отзыв с рейтингом 2", 200, 30)
        self.btn_next_3 = self.add_button("Следующий отзыв с рейтингом 3", 200, 30)
        self.btn_next_4 = self.add_button("Следующий отзыв с рейтингом 4", 200, 30)
        self.btn_next_5 = self.add_button("Следующий отзыв с рейтингом 5", 200, 30)
        self.btn_close = self.add_button("Закрыть программу", 200, 30)

        self.label = QLabel(self)
        self.label.setFixedSize(400, 400)
        self.label.setScaledContents(True)


        box_layout.addWidget(self.btn_create)
        box_layout.addWidget(self.btn_copy)
        box_layout.addWidget(self.btn_rand)
        box_layout.addWidget(self.btn_iterator)
        box_layout.addWidget(self.btn_next_1)
        box_layout.addWidget(self.btn_next_2)
        box_layout.addWidget(self.btn_next_3)
        box_layout.addWidget(self.btn_next_4)
        box_layout.addWidget(self.btn_next_5)
        box_layout.addWidget(self.btn_close)
        box_layout.addStretch()

        grid_layout.addWidget(self.label, 0, 0)
        grid_layout.addLayout(box_layout, 0, 1)

        main_widget.setLayout(grid_layout)

        self.setCentralWidget(main_widget)
        self.classes = ["1", "2", "3", "4", "5"]

        self.classes_iteraor = None
        self.review_path = None

        self.btn_create.clicked.connect(self.create_annotation)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_rand.clicked.connect(self.rand)

        self.btn_close.clicked.connect(self.close)

        self.show()

    def add_button(self, name:str, size_x: int, size_y: int):
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_x, size_y))
        return button


    def create_annotation(self):
        try:
            directory = QFileDialog.getSaveFileName(self, "Выберите папку для создания файла аннотации:", "", "CSV File(*.csv)")[0]
            if directory == "":
                return
            temp = create_csv_list(self.data_path, self.classes)
            write_into_csv(directory, temp)
            QMessageBox.information(None, "Успешно", "Аннотация была создана")
        except Exception as exc:
            logging.error(f"Can not create annotation: {exc.message}\n{exc.args}\n")

    def copy(self):
        try:
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "", "CSV File(*csv)")[0]
            directory = QFileDialog.getExistingDirectory(self, "Выберите папку для копирования датасета")
            if (file == "") or (directory == ""):
                QMessageBox.information(None, "Не указан путь", "Не был выбран файл или папка")
                return
            copy_folder(self.data_path, directory, self.classes, file)
            QMessageBox.information(None, "Успешно", "Датасет скопирован")
        except Exception as exc:
            logging.error(f"Can not create copy or annotation: {exc.message}\n{exc.args}\n")

    def rand(self):
        try:
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "", "CSV File(*csv)")[0]
            directory = QFileDialog.getExistingDirectory(self, "Выберите папку для копирования датасета")
            if (file == "") or (directory == ""):
                QMessageBox.information(None, "Не указан путь", "Не был выбран файл или папка")
                return
            copy_random(self.data_path, directory, self.classes, file, 5000)
            QMessageBox.information(None, "Успешно", "Датасет скопирован")
        except Exception as exc:
                    logging.error(f"Can not create copy or annotation: {exc.message}\n{exc.args}\n")







if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()