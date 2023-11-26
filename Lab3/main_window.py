import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                             QPushButton, QMessageBox,QLabel, 
                             QFileDialog, QVBoxLayout, 
                             QWidget, QGridLayout)

sys.path.insert(1, "D:/AppProgPython/appprog/Lab2")
from iterator import ClassIterator
from create_annotation import create_csv_list, write_into_csv
from create_copy_folder import copy_folder
from create_copy_random import copy_random


logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self)-> None:
        super(MainWindow, self).__init__()

        self.setGeometry(100, 100, 700, 600)
        self.setMaximumSize(1000, 700)

        main_widget = QWidget()
        box_layout = QVBoxLayout()
        grid_layout = QGridLayout()

        self.setWindowTitle("Main")
        self.data_path = os.path.abspath("dataset")
        src = QLabel(f"Basic dataset:\n{self.data_path}", self)
        src.setFixedSize(QSize(250,40))
        box_layout.addWidget(src)

        self.btn_create = self.add_button("Создать аннотацию", 300, 40)
        self.btn_copy = self.add_button("Создать копию датасета и его аннотацию", 300, 40)
        self.btn_rand = self.add_button("Создать датасет со случ. числами и его аннотацию", 300, 40)

        self.btn_iterator = self.add_button("Начать итерацию", 250, 40)
        self.btn_next_first = self.add_disabled_button("Следующий отзыв с рейтингом 1 ==>", 250, 30)
        self.btn_next_second = self.add_disabled_button("Следующий отзыв с рейтингом 2 ==>", 250, 30)
        self.btn_next_third = self.add_disabled_button("Следующий отзыв с рейтингом 3 ==>", 250, 30)
        self.btn_next_fourth = self.add_disabled_button("Следующий отзыв с рейтингом 4 ==>", 250, 30)
        self.btn_next_fifth = self.add_disabled_button("Следующий отзыв с рейтингом 5 ==>", 250, 30)

        self.btn_close = self.add_button("Закрыть программу", 200, 30)

        self.text_label = QLabel(self)
        self.text_label.setText("Здесь будет отзыв")
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
        self.path_label=QLabel("Здесь будет обозачен путь активного файла")
        box_layout.addWidget(self.path_label)

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

    def add_button(self, name:str, size_w: int, size_h: int):
        """This function creates active button with specified name and sizes
        Parametres:
        
            name(str): name of the button

            size_w: width of the button

            size_h: height of the button
            """
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_w, size_h))
        return button

    def add_disabled_button(self, name:str, size_w: int, size_h: int):
        """This function creates inactive button with specified name and sizes
        Parametres:
        
            name(str): name of the button

            size_w: width of the button

            size_h: height of the button
            """
        button = QPushButton(name, self)
        button.setEnabled(False)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_w, size_h))
        return button

    def create_annotation(self):
        """This function creates annotation for default dataset"""
        try:
            directory = QFileDialog.getSaveFileName(self, "Выберите папку для создания файла аннотации:",\
                                                     "", "CSV File(*.csv)")[0]
            if directory == "":
                return
            temp = create_csv_list(self.data_path, self.classes)
            write_into_csv(directory, temp)
            QMessageBox.information(None, "Успешно", "Аннотация была создана")
        except Exception as exc:
            logging.error(f"Can not create annotation: {exc.message}\n{exc.args}\n")

    def copy(self):
        """This function creates copy of default dataset and his annotation"""
        try:
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "",\
                                                "CSV File(*.csv)")[0]
            directory = QFileDialog.getExistingDirectory(self, "Выберите папку для копирования датасета")
            if (file == "") or (directory == ""):
                QMessageBox.information(None, "Не указан путь", "Не был выбран файл или папка")
                return
            copy_folder(self.data_path, directory, self.classes, file)
            QMessageBox.information(None, "Успешно", "Датасет скопирован")
        except Exception as exc:
            logging.error(f"Can not create copy or annotation: {exc.message}\n{exc.args}\n")

    def rand(self):
        """This function creates copy of default dataset with renamed files to random numbers\
            and creates annotation"""
        try:
            file = QFileDialog.getSaveFileName(self, "Выберите файл для создания аннотации:", "",\
                                                "CSV File(*.csv)")[0]
            directory = QFileDialog.getExistingDirectory(self, "Выберите папку для копирования датасета")
            if (file == "") or (directory == ""):
                QMessageBox.information(None, "Не указан путь", "Не был выбран файл или папка")
                return
            copy_random(self.data_path, directory, self.classes, file, 5000)
            QMessageBox.information(None, "Успешно", "Датасет скопирован")
        except Exception as exc:
            logging.error(f"Can not create copy or annotation: {exc.message}\n{exc.args}\n")             

    def csv_path(self):
        """This function opens csv file to read review in window application"""
        try:
            path = QFileDialog.getOpenFileName(self, "Выберите файл для итерации:", "", "CSV File(*.csv)")[0]
            if path == "":
                return
            self.classes_iterator = ClassIterator(path, self.classes[0], self.classes[1],\
                                                   self.classes[2], self.classes[3], self.classes[4])
            self.btn_next_first.setEnabled(True)
            self.btn_next_second.setEnabled(True)
            self.btn_next_third.setEnabled(True)
            self.btn_next_fourth.setEnabled(True)
            self.btn_next_fifth.setEnabled(True)
        except Exception as exc:
            logging.error(f"Incorrect path: {exc.message}\n{exc.args}\n")
    
    def next_first(self):
        """Function returns the path to the next element of the first class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_first()
        self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            self.btn_next_first.setEnabled(False)
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())
        

    def next_second(self):
        """Function returns the path to the next element of the second class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_second()
        self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            self.btn_next_second.setEnabled(False)
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())

    def next_third(self):
        """Function returns the path to the next element of the third class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_third()
        self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            self.btn_next_third.setEnabled(False)
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())

    def next_fourth(self):
        """Function returns the path to the next element of the fourth class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_fourth()
        self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            self.btn_next_fourth.setEnabled(False)
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())

    def next_fifth(self):
        """Function returns the path to the next element of the fifth class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        element = self.classes_iterator.next_fifth()
        self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            self.btn_next_fifth.setEnabled(False)
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()