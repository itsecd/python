# Прикладное программирование

## Лабораторная № 3. Работа с GUI

__Задание:__  
Создать приложение с графическим интерфейсом повторяющее функционал лабораторной работы 2. 

__Задачи:__
1. Приложение должно запрашивать у пользователя путь к папке исходного датасета.  
2. Приложение должно иметь кнопку для создания файла аннотации исходного датасета. Для этого потребуется запросить у пользователя путь к файлу назначения.  
3. Приложение должно иметь кнопку для создания датасета с другой организацией файлов (в соответсвии с пунктами лабораторной работы 2) и файла аннотации создаваемого датасета. Для этого потребуется запросить у пользователя путь к папке назначения.  
4. Задание по вариантам:  
   4.1.  __Вар 1-2:__ Приложение должно иметь поле ввода даты и кнопку "Получить данные". Пользователь вводит дату и видит в интерфейсе данные для этой даты.  
   4.2. __Вар 3-10:__  Приложение должно иметь кнопки для получения следующего экземпляра класса из датасета. Например, если у вас датасет из кошек и собак, то должна быть кнопка "Следующая кошка" и кнопка "Следующая собака". После нажатия на них должен быть получен следующий путь при помощи итератора, а затем отображена картинка новой кошки или собаки из вашего датасета в интерфейсе вашей программы. В случае варианта с текстом отображаем текст.


Ноутбук с заданием и примерами кода доступен по [ссылке](https://colab.research.google.com/drive/1nSL1HwRrn8La732kYB7K0KeBiaaor1fM#scrollTo=1wSBpC7OIR6Y).

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
        box_layout = QVBoxLayout()  # Размещает в вертикальном столбце
        layout = QGridLayout()  # Упорядочивает в виде сетки из строк и столбцов

        self.dataset_path = os.path.abspath("dataset")
        src = QLabel(f"Путь к папке исходного датасета:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(220, 40))
        box_layout.addWidget(src)
        src.setStyleSheet('color:blue')

        # установим кнопки
        self.annotation = self.add_button("Создать файл аннотацию", 220, 30)
        self.btn_copy = self.add_button("Копирование датасета", 220, 30)
        self.btn_random = self.add_button("Датасет из рандомных чисел", 220, 30)
        self.btn_iterator = self.add_button("Начать итерацию", 220, 30)
        self.next_cat = self.add_button("Следующая кошка", 220, 30)
        self.next_dog = self.add_button("Следующая собака", 220, 30)
        self.exit = self.add_button("Выйти из программы", 220, 30)

        # установим изображение
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setScaledContents(True) # масштабирует пиксельное изображение

        # форматируем виджеты по размеру окна
        box_layout.addWidget(self.annotation)
        box_layout.addWidget(self.btn_copy)
        box_layout.addWidget(self.btn_random)
        box_layout.addWidget(self.btn_iterator)
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

        # создание аннотации, копия + рандом
        self.annotation.clicked.connect(self.create_annotation)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_random.clicked.connect(self.random)

        # кнопки итератора
        self.btn_iterator.clicked.connect(self.csv_path)
        self.next_cat.clicked.connect(self.next_first)
        self.next_dog.clicked.connect(self.next_second)

        # выход из программы
        self.exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str, size_x: int, size_y: int):
        ''' принимает название поля кнопки и ее размеры'''
        button = QPushButton(name, self) # Виджет кнопок, на который пользователь может нажать
        button.resize(button.sizeHint()) # Возвращает измененную копию этого изображения (возвращает объект QSize)
        button.setStyleSheet('color:blue')
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def create_annotation(self):
        ''' принимает имя папки, классы, URL и количество файлов'''
        """The function creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self,
                "Выберите папку для создания файла аннотации:",
                "",
                "CSV(*.csv)",
            )[0]
            if directory == "":
                return
            l = make_list(self.dataset_path, self.classes)
            write_in_file(l, directory)
            QMessageBox.information(None, "Успешно", "Аннотация создана!")
        except Exception as ex:
            logging.error(
                f"Couldn't create annotation: {ex.message}\n{ex.args}\n")

    def copy(self):
        """The function copy dataset with new name(class_number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self,
                "Выберите путь для создания файла-аннотации:",
                "",
                "CSV(*.csv)",
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета:"
            )
            if (folder == "") or (directory == ""):
                QMessageBox.information(
                    None, "Ошибка. Не указан путь", "Не был выбран файл или папка"
                )
                return
            write_in_new(self.dataset_path, self.classes, folder, directory, 0)
            QMessageBox.information(None, "Успешно", "Датасет скопирован!")
        except Exception as ex:
            logging.error(f"Couldn't create copy: {ex.message}\n{ex.args}\n")

    def random(self):
        ''' принимает имя папки, классы, URL и количество файлов'''
        """The function copy dataset with new name(random number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите файл для создания аннотации:", "", "CSV(*.csv)"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета:"
            )
            if (folder == "") or (directory == ""):
                QMessageBox.information(
                    None, "Не указан путь", "Не был выбран файл или папка"
                )
                return
            write_in_new(self.dataset_path, self.classes, folder, directory, 1)
            QMessageBox.information(None, "Успешно", "Датасет скопирован!")
        except Exception as ex:
            logging.error(
                f"Couldn't create randon copy: {ex.message}\n{ex.args}\n")

    def csv_path(self):
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function requests the path for the iteration file and creates an iterator"""
        try:
            path = QFileDialog.getOpenFileName(
                self, "Выберите файл для итерации:")[0]
            path_1 = QFileDialog.getOpenFileName(
                self, "Выберите файл для :")[0]
            if path == "":
                return
            self.classes_iterator = ChoiceIterator(
                path_1, path, self.classes[0], self.classes[1]
            )
        except Exception as ex:
            logging.error(f"Incorrect path: {ex.message}\n{ex.args}\n")

    def next_first(self):
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the first class
        and opens this image in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(
                None, "Не выбран файл", "Не выбран файл для итерации"
            )
            return
        element = self.classes_iterator.next_first()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))

    def next_second(self):
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the second class
        and opens this image in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(
                None, "Не выбран файл", "Не выбран файл для итерации"
            )
            return
        element = self.classes_iterator.next_second()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()




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
        self.bata_copy.clicked.connect(self.copy)
        self.bata_random.clicked.connect(self.random)

        # то, что будет на экране при нажатии кнопки итератора
        self.bata_iterator.clicked.connect(self.csv_path)
        self.next_cat.clicked.connect(self.next_first)
        self.next_dog.clicked.connect(self.next_second)

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
            QMessageBox.information(None, "Результат нажатия", "CSV-файл аннотации создан")
        except:
            logging.error(f"Error create annotation")

    def copy(self) -> None:
        """The function copy dataset with new name(class_number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self,
                "Выберите путь для создания файла-аннотации:",
                "",
                "CSV(*.csv)",
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета:"
            )
            if (folder == "") or (directory == ""):
                QMessageBox.information(
                    None, "Ошибка. Не указан путь", "Не был выбран файл или папка"
                )
                return
            write_in_new(self.dataset_path, self.classes, folder, directory, 0)
            QMessageBox.information(None, "Результат нажатия", "Датасет скопирован!")
        except Exception as ex:
            logging.error(f"Couldn't create copy: {ex.message}\n{ex.args}\n")

    def random(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """The function copy dataset with new name(random number)
        and creates a csv file at the specified path"""
        try:
            directory = QFileDialog.getSaveFileName(
                self, "Выберите файл для создания аннотации:", "", "CSV(*.csv)"
            )[0]
            folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку для копирования датасета:"
            )
            if (folder == "") or (directory == ""):
                QMessageBox.information(
                    None, "Не указан путь", "Не был выбран файл или папка"
                )
                return
            write_in_new(self.dataset_path, self.classes, folder, directory, 1)
            QMessageBox.information(None, "Успешно", "Датасет скопирован!")
        except Exception as ex:
            logging.error(
                f"Couldn't create randon copy: {ex.message}\n{ex.args}\n")

    def csv_path(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function requests the path for the iteration file and creates an iterator"""
        try:
            path = QFileDialog.getOpenFileName(
                self, "Выберите файл для итерации:")[0]
            path_1 = QFileDialog.getOpenFileName(
                self, "Выберите файл для :")[0]
            if path == "":
                return
            self.classes_iterator = ChoiceIterator(
                path_1, path, self.classes[0], self.classes[1]
            )
        except Exception as ex:
            logging.error(f"Incorrect path: {ex.message}\n{ex.args}\n")

    def next_first(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the first class
        and opens this image in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(
                None, "Не выбран файл", "Не выбран файл для итерации"
            )
            return
        element = self.classes_iterator.next_first()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))

    def next_second(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the second class
        and opens this image in the widget"""
        if self.classes_iterator == None:
            QMessageBox.information(
                None, "Не выбран файл", "Не выбран файл для итерации"
            )
            return
        element = self.classes_iterator.next_second()
        self.image_path = element
        self.image_label.update()
        print(
            self.image_path
        )  # для проверки на соответсвие пути и картинки по этому пути
        self.image_label.setPixmap(QPixmap(element))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()


import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QMessageBox, QLabel, QFileDialog, QVBoxLayout, QWidget, QGridLayout,)
from PyQt6.QtGui import QPixmap

from iterator import ChoiceIterator
from csv_name import make_list, write_in_file
from new import write_in_new

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
        src = QLabel(
            f"Путь к папке исходного датасета:\n{self.dataset_path}", self)
        src.setFixedSize(QSize(230, 40))
        box_layout.addWidget(src)
        src.setStyleSheet('color:blue')

        # установим кнопки
        self.annotation = self.add_button("Создать файл-аннотацию", 230, 30)
        self.bata_copy = self.add_button("Копирование датасета", 230, 30)
        self.bata_random = self.add_button(
            "Датасет из радомных чисел", 230, 30)
        self.bata_iterator = self.add_button(
            "Получение следующего экземпляра", 230, 30)
        self.next_cats = self.add_button("Следующая кошка", 230, 30)
        self.next_dogs = self.add_button("Следующая собака", 230, 30)
        self.exit = self.add_button("Выйти из программы", 230, 30)

        # установим изображение
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        # масштабирует пиксельное изображение
        self.image_label.setScaledContents(True)

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
        self.next_cats.clicked.connect(self.next_cat)
        self.next_dogs.clicked.connect(self.next_dog)

        # то, что будет на экране при нажатии кнопки выход из программы
        self.exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str, size_x: int, size_y: int) -> QPushButton:
        '''принимает название поля кнопки и ее размеры'''
        button = QPushButton(
            name, self)  # Виджет кнопок, на который пользователь может нажать
        # Возвращает измененную копию этого изображения (возвращает объект QSize)
        button.resize(button.sizeHint())
        button.setStyleSheet('color:blue')
        button.setFixedSize(QSize(size_x, size_y))
        return button

    def create_copy_random(self, number: int) -> None:
        '''создаем csv-файл по указанному пути; копируем данные с новым именем'''
        try:
            directory = QFileDialog.getSaveFileName(
                self,
                "Введите название папки для создания csv-файла:",
            )[0]
            if directory == "":
                QMessageBox.information(
                    None, "Ошибка работы программы!", "Не правильно выбрана папка")
                return
            if number == 0 or number == 1:
                write_in_new(self.dataset_path,
                             self.classes, directory, number)
            else:
                a = make_list(self.dataset_path, self.classes)
                write_in_file(a, directory)
            QMessageBox.information(
                None, "Результат нажатия книпки", "Датасет успешно скопирован!")
        except Exception as ex:
            logging.error(f"Couldn't create copy: {ex.message}\n{ex.args}\n")

    def create_annotation(self) -> None:
        self.create_copy_random(7)

    def copy(self) -> None:
        self.create_copy_random(0)

    def random(self) -> None:
        self.create_copy_random(1)

    def csv_path(self) -> None:
        '''запрашиваем путь к файлу для итерации и создаем итератор'''
        try:
            path = QFileDialog.getOpenFileName(
                self, "Выберите файл для итерации:")[0]
            path_1 = QFileDialog.getSaveFileName(
                self, "Выберите файл куда итеруем:")[0]
            if path == "":
                return
            print(os.path.relpath(path))
            print(os.path.relpath(path_1))
            self.choice_iterator = ChoiceIterator(os.path.relpath(
                path), os.path.relpath(path_1), self.classes[0], self.classes[1])
        except Exception as ex:
            logging.error(f"Incorrect path: {ex.message}\n{ex.args}\n")

    def next_cat(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the first class
        and opens this image in the widget"""
        if self.choice_iterator == None:
            QMessageBox.information(
                None, "Ошибка работы программы!", "Не выбран файл для итерации")
            return
        a = self.choice_iterator.next_cat()
        self.image_path = a
        self.image_label.update()
        # для проверки на соответсвие пути и картинки по этому пути
        print(self.image_path)
        self.image_label.setPixmap(QPixmap(a))

    def next_dog(self) -> None:
        ''' принимает имя папки, классы, URL и количество файлов'''
        """Function returns the path to the next element of the second class
        and opens this image in the widget"""
        if self.choice_iterator == None:
            QMessageBox.information(
                None, "Ошибка работы программы!", "Не выбран файл для итерации")
            return
        a = self.choice_iterator.next_dog()
        self.image_path = a
        self.image_label.update()
        # для проверки на соответсвие пути и картинки по этому пути
        print(self.image_path)
        self.image_label.setPixmap(QPixmap(a))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    app.exec()
