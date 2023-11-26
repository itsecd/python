import sys
import os
import logging
import typing
from PyQt6 import QtCore
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QToolBar, QMainWindow,QPushButton,QMessageBox,QLabel,QFileDialog,QVBoxLayout,QWidget,QGridLayout)
from PyQt6.QtGui import QPixmap

sys.path.insert(1, "D:/AppProgPython/appprog/Lab2")
from iterator import ClassIterator
from create_annotation import create_csv_list, write_into_csv
from create_copy_folder import copy_folder
from create_copy_random import copy_random


logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self)-> None:
        super(MainWindow, self).__init__()

        self.setGeometry(100, 100, 800, 800)

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
        self.btn_next_first = self.add_disabled_button("Следующий отзыв с рейтингом 1", 200, 30)
        self.btn_next_second = self.add_disabled_button("Следующий отзыв с рейтингом 2", 200, 30)
        self.btn_next_third = self.add_disabled_button("Следующий отзыв с рейтингом 3", 200, 30)
        self.btn_next_fourth = self.add_disabled_button("Следующий отзыв с рейтингом 4", 200, 30)
        self.btn_next_fifth = self.add_disabled_button("Следующий отзыв с рейтингом 5", 200, 30)

        self.btn_close = self.add_button("Закрыть программу", 200, 30)

        self.text_label = QLabel(self)
        self.text_label.setWordWrap(True)
        self.text_label.setFixedSize(600, 400)

        box_layout.addWidget(self.btn_create)
        box_layout.addWidget(self.btn_copy)
        box_layout.addWidget(self.btn_rand)
    
        box_layout.addStretch()

        box_layout.addWidget(self.btn_iterator)
        box_layout.addWidget(self.btn_next_first)
        box_layout.addWidget(self.btn_next_second)
        box_layout.addWidget(self.btn_next_third)
        box_layout.addWidget(self.btn_next_fourth)
        box_layout.addWidget(self.btn_next_fifth)

        box_layout.addStretch()

        box_layout.addWidget(self.btn_close)

        box_layout.addStretch()
        
        grid_layout.addWidget(self.text_label, 0, 0)
        grid_layout.addLayout(box_layout, 0, 1)

        main_widget.setLayout(grid_layout)

        self.setCentralWidget(main_widget)
        self.classes = ["1", "2", "3", "4", "5"]

        self.classes_iterator = None
        self.review_path = None

        self.btn_create.clicked.connect(self.create_annotation)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_rand.clicked.connect(self.rand)

        self.btn_iterator.clicked.connect(self.csv_path)
        self.btn_next_first.clicked.connect(self.next_first)
        self.btn_next_second.clicked.connect(self.next_second)
        self.btn_next_third.clicked.connect(self.next_third)
        self.btn_next_fourth.clicked.connect(self.next_fourth)
        self.btn_next_fifth.clicked.connect(self.next_fifth)

        self.btn_close.clicked.connect(self.close)

        self.show()

    def add_button(self, name:str, size_x: int, size_y: int):
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def add_disabled_button(self, name:str, size_x: int, size_y: int):
        button = QPushButton(name, self)
        button.setEnabled(False)
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
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "", "CSV File(*.csv)")[0]
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
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "", "CSV File(*.csv)")[0]
            directory = QFileDialog.getExistingDirectory(self, "Выберите папку для копирования датасета")
            if (file == "") or (directory == ""):
                QMessageBox.information(None, "Не указан путь", "Не был выбран файл или папка")
                return
            copy_random(self.data_path, directory, self.classes, file, 5000)
            QMessageBox.information(None, "Успешно", "Датасет скопирован")
        except Exception as exc:
                    logging.error(f"Can not create copy or annotation: {exc.message}\n{exc.args}\n")
                    

    def csv_path(self):
        try:
            path = QFileDialog.getOpenFileName(self, "Выберите файл для итерации:", "", "CSV File(*.csv)")[0]
            if path == "":
                return
            self.classes_iterator = ClassIterator(path, self.classes[0], self.classes[1], self.classes[2], self.classes[3], self.classes[4])
            self.btn_next_first.setEnabled(True)
            self.btn_next_second.setEnabled(True)
            self.btn_next_third.setEnabled(True)
            self.btn_next_fourth.setEnabled(True)
            self.btn_next_fifth.setEnabled(True)
        except Exception as exc:
            logging.error(f"Incorrect path: {exc.message}\n{exc.args}\n")
    
    def next_first(self):
        if self.classes_iterator == None:
            QMessageBox(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_first()
        self.review_path = element
        self.text_label.update()
        print(self.review_path)
        file = open(self.review_path, "r")
        self.text_label.setText(file.read())

    def next_second(self):
        if self.classes_iterator == None:
            QMessageBox(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_second()
        self.review_path = element
        self.text_label.update()
        print(self.review_path)
        file = open(self.review_path, "r")
        self.text_label.setText(file.read())

    def next_third(self):
        if self.classes_iterator == None:
            QMessageBox(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_third()
        self.review_path = element
        self.text_label.update()
        print(self.review_path)
        file = open(self.review_path, "r")
        self.text_label.setText(file.read())

    def next_fourth(self):
        if self.classes_iterator == None:
            QMessageBox(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_fourth()
        self.review_path = element
        self.text_label.update()
        print(self.review_path)
        file = open(self.review_path, "r")
        self.text_label.setText(file.read())

    def next_fifth(self):
        if self.classes_iterator == None:
            QMessageBox(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_fifth()
        self.review_path = element
        self.text_label.update()
        print(self.review_path)
        file = open(self.review_path, "r")
        self.text_label.setText(file.read())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()




#Thanks habr.com that you are exist