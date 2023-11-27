import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, 
                             QPushButton, QMessageBox,QLabel, 
                             QFileDialog, QVBoxLayout, 
                             QWidget, QGridLayout,
                             QComboBox)

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

        self.combo = QComboBox(self)
        self.combo.addItems(["Рейтинг 1", "Рейтинг 2", "Рейтинг 3", "Рейтинг 4", "Рейтинг 5"])
        self.combo.setFixedSize(QSize(250, 20))

        self.combo_copy = QComboBox(self)
        self.combo_copy.addItems(["Стандарт", "Случайные числа"])
        self.combo_copy.setFixedSize(QSize(300, 30))

        self.btn_create = self.add_button("Создать аннотацию", 300, 40)
        self.btn_execute = self.add_button("Выполнить создание копии", 300, 40)

        self.btn_iterator = self.add_button("Начать итерацию", 250, 40)
        self.btn_next = self.add_disabled_button("Следующий отзыв", 250, 30)

        self.btn_close = self.add_button("Закрыть программу", 200, 30)

        self.text_label = QLabel(self)
        self.text_label.setText("Здесь будет отзыв")
        self.text_label.setWordWrap(True)
        self.text_label.setFixedSize(600, 400)

        box_layout.addWidget(self.btn_create)
        box_layout.addWidget(self.btn_execute)
        box_layout.addWidget(self.combo_copy)
    
        box_layout.addStretch()

        box_layout.addWidget(self.btn_iterator)
        box_layout.addWidget(self.btn_next)
        box_layout.addWidget(self.combo)
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
        self.btn_execute.clicked.connect(self.copy)

        self.btn_iterator.clicked.connect(self.csv_path)
        self.btn_next.clicked.connect(self.next)

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
            if self.combo_copy.currentText() == "Стандарт":
                copy_folder(self.data_path, directory, self.classes, file)
            if self.combo_copy.currentText() == "Случайные числа":
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
            self.btn_next.setEnabled(True)
        except Exception as exc:
            logging.error(f"Incorrect path: {exc.message}\n{exc.args}\n")
    
    def next(self):
        """Function returns the path to the next element of the class
        and opens text review in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(None, "Не выбран файл", "Не выбран файл для итерации")
            return
        if self.combo.currentText() == "Рейтинг 1":
            element = self.classes_iterator.next_first()
            self.review_path = element
        if self.combo.currentText() == "Рейтинг 2":
            element = self.classes_iterator.next_second()
            self.review_path = element
        if self.combo.currentText() == "Рейтинг 3":
            element = self.classes_iterator.next_third()
            self.review_path = element
        if self.combo.currentText() == "Рейтинг 4":
            element = self.classes_iterator.next_fourth()
            self.review_path = element
        if self.combo.currentText() == "Рейтинг 5":
            element = self.classes_iterator.next_fifth()
            self.review_path = element
        if self.review_path == None:
            QMessageBox.information(None, "Конец класса", "Файлы для класса кончились")
            return
        self.text_label.update()
        file = open(file=self.review_path, mode="r", encoding="utf-8")
        self.path_label.setText(self.review_path)
        self.text_label.setText(file.read())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()