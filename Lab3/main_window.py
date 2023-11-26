import sys
import os
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QLabel,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QGridLayout,
)
from PyQt6.QtGui import QPixmap
sys.path.insert(1,"Lab2")
import copy_dataset_no_random
import copy_dataset_random
from iterator import ElementIterator
import make_rel_abs_path
import return_element

class DatasetViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset_path = ""
        self.image_paths = []
        self.current_index = 0
        self.image_label = QLabel(self)
        self.polarbear_iterator = None
        self.brownbear_iterator = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        select_folder_button = QPushButton('Выбрать папку с dataset')
        select_folder_button.clicked.connect(self.select_dataset_folder)
        layout.addWidget(select_folder_button)

        next_polarbear_button = QPushButton('Следующий полярный медведь')
        next_polarbear_button.clicked.connect(self.show_next_polarbear)
        layout.addWidget(next_polarbear_button)

        next_brownbear_button = QPushButton('Следующая бурый медведь')
        next_brownbear_button.clicked.connect(self.show_next_brownbear)
        layout.addWidget(next_brownbear_button)

        copy_dataset_button = QPushButton('Копировать и изменить датасет')
        copy_dataset_button.clicked.connect(self.copy_and_modify_dataset)
        layout.addWidget(copy_dataset_button)

        copy_dataset_random_button = QPushButton('Копировать и изменить датасет (с рандомизацией)')
        copy_dataset_random_button.clicked.connect(self.copy_and_modify_dataset_with_random)
        layout.addWidget(copy_dataset_random_button)

        generate_paths_button = QPushButton('Создать CSV с путями')
        generate_paths_button.clicked.connect(self.generate_paths_csv)
        layout.addWidget(generate_paths_button)

        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.setWindowTitle('Dataset Viewer')

        self.polarbear_iterator = None
        self.brownbear_iterator = None

    def select_dataset_folder(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Выберите папку с dataset", os.getcwd())

        polarbear_folder = os.path.join(self.dataset_path, "polar bear")
        brownbear_folder = os.path.join(self.dataset_path, "brown bear")

        self.image_paths.extend([os.path.join(polarbear_folder, file) for file in os.listdir(polarbear_folder) if file.endswith('.jpg')])
        self.image_paths.extend([os.path.join(brownbear_folder, file) for file in os.listdir(brownbear_folder) if file.endswith('.jpg')])

        self.show_image()

    def copy_and_modify_dataset(self):
        script_path = "copy_dataset_no_random.py"
        os.system(f"python {script_path}")

    def copy_and_modify_dataset_with_random(self):
        script_path = "copy_dataset_random.py"
        os.system(f"python {script_path}")

    def generate_paths_csv(self):
        script_path = "make_rel_abs_path.py"
        os.system(f"python {script_path}")

    def show_next_polarbear(self):
        self.polarbear_iterator = ElementIterator('polar bear', 'paths.csv')
        next_polarbear_path = next(self.polarbear_iterator)
        if next_polarbear_path:
            self.display_image(next_polarbear_path)

    def show_next_brownbear(self):
        self.brownbear_iterator = ElementIterator('brown bear', 'paths.csv')
        next_brownbear_path = next(self.brownbear_iterator)
        if next_brownbear_path:
            self.display_image(next_brownbear_path)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaledToWidth(300))  # Adjust the width as needed
    def show_image(self):
        if self.image_paths and 0 <= self.current_index < len(self.image_paths):
            image_path = self.image_paths[self.current_index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaledToWidth(300))  # Adjust the width as needed

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DatasetViewer()
    viewer.show()
    sys.exit(app.exec())