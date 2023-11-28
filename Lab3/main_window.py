import sys
import os
import logging
from enum import Enum
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                            QMessageBox, QLabel, QFileDialog, QVBoxLayout, QWidget, QGridLayout,)
from PyQt6.QtGui import QPixmap

sys.path.append("C:/Users/user/Desktop/Python2/Lab2")
from iterator import ChoiceIterator
from csv_name import make_list, write_in_file
from new import write_in_new

logging.basicConfig(level=logging.INFO)


class Action(Enum):
    COPY = 0
    RANDOM = 1


class Animal(Enum):
    Cat = 0
    Dog = 1


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(500, 200, 700, 400)
        self.setWindowTitle("Lab3-var3 main window")
        self.setStyleSheet('background-color: #2cf3e2;')

        main_widget = QWidget()
        box_layout = QVBoxLayout()
        layout = QGridLayout()

        self.dataset_path = os.path.abspath("dataset")
        src = QLabel(
            f"Путь к папке исходного датасета:\n{self.dataset_path}", self)
        box_layout.addWidget(src)
        main_widget.setStyleSheet(
            "max-width: 1400%;" "margin: 0 0 0 10%;" "height: auto;" "padding: 5% 40% 5% 40%;")

        # установим кнопки
        self.annotation = self.add_button("Создать файл-аннотацию")
        self.bata_copy = self.add_button("Копирование датасета")
        self.bata_random = self.add_button("Датасет из радомных чисел")
        self.bata_iterator = self.add_button(
            "Выбрать аннотацию для просмотра датасета")
        self.next_cats = self.add_button("Следующий котик")
        self.next_dogs = self.add_button("Следующая собачка")
        self.exit = self.add_button("Выйти из программы")

        # установим изображение
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet(
            "margin: 80% 0 170% 30%;" "max-width: 1600%;" "height: auto;" "padding: 0;")

        # форматируем виджеты по размеру окна
        box_layout.addWidget(self.annotation)
        box_layout.addWidget(self.bata_copy)
        box_layout.addWidget(self.bata_random)
        box_layout.addWidget(self.bata_iterator)
        box_layout.addWidget(self.next_cats)
        box_layout.addWidget(self.next_dogs)
        box_layout.addWidget(self.exit)
        box_layout.addStretch()  # кнопки вплотную
        layout.addLayout(box_layout, 0, 0)
        layout.addWidget(self.image_label, 0, 1)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.classes = ["cat", "dog"]
        self.choice_iterator = None
        self.image_path = None

        # то, что будет на экране при нажатии кнопки создания аннотации, копии и рандома
        self.annotation.clicked.connect(self.create_annotation)
        self.bata_copy.clicked.connect(self.copy)
        self.bata_random.clicked.connect(self.random)

        # то, что будет на экране при нажатии кнопки итератора
        self.bata_iterator.clicked.connect(self.csv_path)
        self.next_cats.clicked.connect(lambda: self.next(Animal.Cat))
        self.next_dogs.clicked.connect(lambda: self.next(Animal.Dog))

        # то, что будет на экране при нажатии кнопки выход из программы
        self.exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str) -> QPushButton:
        '''принимает название поля кнопки и ее размеры'''
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setStyleSheet('background-color: #ee5300;')
        return button

    def create_annotation(self) -> None:
        '''создаем csv-файл по указанному пути'''
        try:
            folder = QFileDialog.getSaveFileName(
                self,
                "Введите название папки для создания csv-файла:",
            )[0]
            if folder == "":
                QMessageBox.information(
                    None, "Ошибка работы программы!", "Не правильно выбрана папка")
                return
            a = make_list(self.dataset_path, self.classes)
            write_in_file(a, folder)
            QMessageBox.information(
                None, "Результат нажатия кнопки", "Действия успешно выполнены!")
        except:
            logging.error(f"Error in cteate_copy_random\n")

    def copy_or_random(self, number: int) -> None:
        '''копируем данные с новым именем'''
        try:
            folder = QFileDialog.getSaveFileName(
                self,
                "Введите название папки для создания csv-файла:",
            )[0]
            if folder == "":
                QMessageBox.information(
                    None, "Ошибка работы программы!", "Не правильно выбрана папка")
                return
            write_in_new(self.dataset_path, self.classes, folder, number)
            QMessageBox.information(
                None, "Результат нажатия кнопки", "Действия успешно выполнены!")
        except:
            logging.error(f"Error in cteate_copy_random\n")

    def copy(self) -> None:
        self.copy_or_random(Action.COPY.value)

    def random(self) -> None:
        self.copy_or_random(Action.RANDOM.value)

    def csv_path(self) -> None:
        '''запрашиваем путь к файлу для итерации и куда итеруем'''
        try:
            path_base = QFileDialog.getOpenFileName(
                self, "Выберите файл для итерации:")[0]
            path_new = QFileDialog.getSaveFileName(
                self, "Выберите файл куда итерируем:")[0]
            if path_new == "" or path_base == "":
                return
            self.choice_iterator = ChoiceIterator(os.path.relpath(path_base.rpartition('.')[0]),
                                                os.path.relpath(path_new), self.classes[0], self.classes[1])
        except:
            logging.error(f"Error in csv_path\n")

    def next(self, number: int) -> None:
        '''возвращаем путь следующего элемента'''
        if self.choice_iterator == None:
            QMessageBox.information(
                None, "Ошибка работы программы!", "Не выбран файл для итерации")
            return
        if number == 0:
            choice = self.choice_iterator.next_cat()
        else:
            choice = self.choice_iterator.next_dog()
        self.image_label.setPixmap(QPixmap(choice))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()
