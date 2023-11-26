import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QMessageBox, QLabel, QFileDialog, QVBoxLayout, QWidget, QGridLayout,)
from PyQt6.QtGui import QPixmap

from iterator import ChoiceIterator
from csv_name import make_list, write_in_file
from new import write_in_new
# from random_of_copy import copy_with_random

logging.basicConfig(level=logging.INFO)


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        
        self.setGeometry(700, 200, 600, 350)
        self.setWindowTitle("Lab3-var3 main window")
        main_widget = QWidget()
        box_layout = QVBoxLayout()  # Размещает область в вертикальном столбце
        layout = QGridLayout()  # Упорядочивает в виде сетки из строк и столбцов

        self.dataset_path = os.path.abspath("dataset")
        src = QLabel(f"Путь к папке исходного датасета:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(220, 40))
        box_layout.addWidget(src)
        src.setStyleSheet('color:blue')

        # установим кнопки
        self.annotation = self.add_button("Создать файл аннотацию", 220, 30)
        self.bata_copy = self.add_button("Копирование датасета", 220, 30)
        self.bata_random = self.add_button("Датасет из рандомных чисел", 220, 30)
        self.bata_iterator = self.add_button("Начать итерацию", 220, 30)
        self.next_cat = self.add_button("Следующая кошка", 220, 30)
        self.next_dog = self.add_button("Следующая собака", 220, 30)
        self.exit = self.add_button("Выйти из программы", 220, 30)

        # установим изображение
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(200, 200)
        self.image_label.setScaledContents(True) # масштабирует пиксельное изображение

        # форматируем виджеты по размеру окна
        box_layout.addWidget(self.annotation)
        box_layout.addWidget(self.bata_copy)
        box_layout.addWidget(self.bata_random)
        box_layout.addWidget(self.bata_iterator)
        box_layout.addWidget(self.next_cat)
        box_layout.addWidget(self.next_dog)
        box_layout.addWidget(self.exit)
        box_layout.addStretch() # кнопки вплотную
        layout.addLayout(box_layout, 0, 0)
        layout.addWidget(self.image_label, 0, 1)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.classes = ["cat", "dog"]
        self.classes_iterator = None
        self.image_path = None

        # то, что будет на экране при нажатии кнопки создания аннотации, копии и рандома
        self.annotation.clicked.connect(self.create_annotation)

        # то, что будет на экране при нажатии кнопки выход из программы
        self.exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str, size_x: int, size_y: int) -> QPushButton:
        '''принимает название поля кнопки и ее размеры'''
        button = QPushButton(name, self) # Виджет кнопок, на который пользователь может нажать
        button.resize(button.sizeHint()) # Возвращает измененную копию этого изображения (возвращает объект QSize)
        button.setStyleSheet('color:blue')
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def create_annotation(self) -> None:
        '''создание csv-файла по указанному пути'''
        try:
            directory = QFileDialog.getSaveFileName(
                self,
                "Выберите папку для создания csv-файла аннотации:",
                "datasets",
                "File (*.csv)",
            )[0]
            if directory == "":
                return
            list = make_list(self.dataset_path, self.classes)
            write_in_file(list, directory)
            QMessageBox.information(None, "Результат нажатия кнопки", "CSV-файл аннотации создан")
        except:
            logging.error(f"Error create annotation")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()