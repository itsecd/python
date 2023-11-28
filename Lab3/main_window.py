import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QGridLayout
)
sys.path.insert(1,'C:\Users\ksush\OneDrive\Рабочий стол\python-v8\Lab2')
from copy_dataset import copy_dataset
from create_annotation import create_annotation_file
from random_dataset import random_dataset

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(400, 100, 1000, 1000)
        self.setWindowTitle("Tooltips")
        main_widget = QWidget()
        button_layout = QVBoxLayout()
        layout = QGridLayout()

        self.setWindowTitle("Main window")
        self.dataset_path = os.path.abspath("dataset")
        src = QLabel(f"Базовый датасет:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(250, 40))
        button_layout.addWidget(src)

        # кнопки
        self.btn_create_of_annotation = self.add_button("Создать аннотацию", 250, 40)
        self.btn_copy = self.add_button("Копирование датасета", 250, 40)
        self.btn_random = self.add_button("Датасет с рандомными числами", 250, 40)
        self.btn_iterator = self.add_button("Начать итерацию", 150, 40)
        self.btn_next_rose = self.add_button("Следующая роза-->", 150, 40)
        self.btn_next_tulip = self.add_button("Следующий тюльпан-->", 150, 40)
        self.go_to_exit = self.add_button("Выйти из программы", 150, 40)

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
        button_layout.addStretch()
        layout.addWidget(self.image_label, 0, 0)
        layout.addLayout(button_layout, 0, 1)

        main_widget.setLayout(layout)

        self.setCentralWidget(main_widget)
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
