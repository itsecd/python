import sys
import os
import logging
from PyQt6.QtWidgets import (QApplication, 
                             QWidget, 
                             QVBoxLayout, 
                             QLabel, 
                             QPushButton, 
                             QFileDialog, 
                             QComboBox, 
                             QMessageBox, 
                             QDialog, 
                             QDialogButtonBox, 
                             QVBoxLayout)
from PyQt6.QtGui import QPixmap

sys.path.insert(0,"Lab2")
from make_rel_abs_path import get_full_paths, get_rel_paths, write_to_csv
from copy_dataset_no_random import replace_images as replace_images_no_random
from copy_dataset_random import replace_images_and_randomize
from iterator import ElementIterator

logging.basicConfig(level=logging.INFO)

class CreateDatasetDialog(QDialog):
    def __init__(self)-> None:
        super().__init__()
        self.setWindowTitle('Выберите тип датасета')
        self.layout = QVBoxLayout()

        self.combo_box = QComboBox()
        self.combo_box.addItems(['С рандомом', 'Без рандома'])
        self.layout.addWidget(self.combo_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.setLayout(self.layout)


class App(QWidget):
    def __init__(self)-> None:
        super().__init__()
        self.setWindowTitle('Dataset Manager')
        self.layout = QVBoxLayout()

        self.dataset_path = None
        self.annotation_path = None
        self.destination_path = None

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.select_dataset_button = QPushButton('Выбрать папку датасета')
        self.create_annotation_button = QPushButton('Создать файл аннотации')
        self.create_new_dataset_button = QPushButton('Создать новый датасет')
        self.next_brown_bear_button = QPushButton('Следующий бурый медведь')
        self.next_polar_bear_button = QPushButton('Следующий полярный медведь')

        self.layout.addWidget(self.select_dataset_button)
        self.layout.addWidget(self.create_annotation_button)
        self.layout.addWidget(self.create_new_dataset_button)
        self.layout.addWidget(self.next_brown_bear_button)
        self.layout.addWidget(self.next_polar_bear_button)

        self.select_dataset_button.clicked.connect(self.select_dataset_folder)
        self.create_annotation_button.clicked.connect(self.create_annotation)
        self.create_new_dataset_button.clicked.connect(self.show_create_dataset_dialog)
        self.next_brown_bear_button.clicked.connect(lambda: self.show_next_bear('brown bear'))
        self.next_polar_bear_button.clicked.connect(lambda: self.show_next_bear('polar bear'))

        self.brown_iterator = None
        self.polar_iterator = None

        self.setLayout(self.layout)
        self.setFixedSize(320, 400)    


    def select_dataset_folder(self)-> None:
        """function for selecting a directory"""
        self.dataset_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с датасетом')
        if self.dataset_path:
            QMessageBox.about(self, "Выбрана директория", f"Выбрана директория: {self.dataset_path}")
            self.create_csv(self.dataset_path)
            self.brown_iterator = ElementIterator('brown bear', f'{self.dataset_path}/data.csv')
            self.polar_iterator = ElementIterator('polar bear', f'{self.dataset_path}/data.csv')
        else:
            QMessageBox.about(self, "Ошибка", "Пожалуйста, выберите директорию")


    def create_csv(self, 
                   dataset_path:str)-> None:
        """function for creating an annotation for an iterator"""
        try:
            brown_full_paths = get_full_paths('brown bear', dataset_path)
            brown_rel_paths = get_rel_paths('brown bear', dataset_path)
            polar_full_paths = get_full_paths('polar bear', dataset_path)
            polar_rel_paths = get_rel_paths('polar bear', dataset_path)

            csv_file = os.path.join(dataset_path, 'data.csv')
            if os.path.exists(csv_file):
                os.remove(csv_file)

            write_to_csv(csv_file, brown_full_paths, brown_rel_paths, 'brown bear')
            write_to_csv(csv_file, polar_full_paths, polar_rel_paths, 'polar bear')
        except Exception as ex:
            logging.error(f"Couldn't create annotation: {ex.message}\n{ex.args}\n")


    def create_annotation(self)-> None:
        """function for manual annotation creation"""
        try:
            if self.dataset_path:
                self.annotation_path, _ = QFileDialog.getSaveFileName(self, 'Укажите файл аннотации', '', 'CSV Files (*.csv)')

                if self.annotation_path:
                    brown_full_paths = get_full_paths('brown bear', self.dataset_path)
                    brown_rel_paths = get_rel_paths('brown bear', self.dataset_path)
                    polar_full_paths = get_full_paths('polar bear', self.dataset_path)
                    polar_rel_paths = get_rel_paths('polar bear', self.dataset_path)

                    write_to_csv(self.annotation_path, brown_full_paths, brown_rel_paths, 'brown bear')
                    write_to_csv(self.annotation_path, polar_full_paths, polar_rel_paths, 'polar bear')

                    QMessageBox.about(self, "Успех", "Файл аннотации успешно создан.")
            else:
                QMessageBox.about(self, "Ошибка", "Пожалуйста, выберите директорию")
        except Exception as ex:
            logging.error(f"Couldn't create annotation: {ex.message}\n{ex.args}\n")


    def show_create_dataset_dialog(self)-> None:
        """function for creating a selection how to create a dataset"""
        try:
            if self.dataset_path:
                dialog = CreateDatasetDialog()
                result = dialog.exec()
                if result == QDialog.DialogCode.Accepted:
                    create_random_dataset = dialog.combo_box.currentText() == 'С рандомом'
                    self.destination_path = QFileDialog.getExistingDirectory(self, 'Выберите папку назначения')
                    if self.destination_path:
                        if create_random_dataset:
                            replace_images_and_randomize('brown bear', self.dataset_path, self.destination_path)
                            replace_images_and_randomize('polar bear', self.dataset_path, self.destination_path)
                        else:
                            replace_images_no_random('brown bear', self.dataset_path)
                            replace_images_no_random('polar bear', self.dataset_path)
                    QMessageBox.about(self, "Успех", "Датасет успешно создан.")
            else:
                QMessageBox.about(self, "Ошибка", "Пожалуйста, выберите директорию")
        except Exception as ex:
            logging.error(f"Couldn't create dataset: {ex.message}\n{ex.args}\n")
            

    def show_image(self, 
                   image_path: str) -> None:
        """function for displaying an image"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(300, 300))
        else:
            self.image_label.setText('Изображение не найдено')


    def show_next_bear(self, bear_class: str) -> None:
        """Function to switch between bear classes"""
        iterator = None

        if bear_class == 'brown bear':
            iterator = self.brown_iterator
        elif bear_class == 'polar bear':
            iterator = self.polar_iterator

        if self.dataset_path and iterator:
            next_path = next(iterator)
            if next_path:
                self.show_image(next_path)
        else:
            QMessageBox.about(self, "Ошибка", "Пожалуйста, выберите директорию")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())