import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QPushButton, 
                             QFileDialog,
                             QGridLayout,)
from PyQt5.QtCore import QSize
from make_csv import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.GUI()

    def GUI(self):
        self.setWindowTitle("בוריס")
        self.setMinimumSize(QSize(600, 400))

        # Path to dataset
        self.dataset_path = None

        # Button to select fold with dataset
        self.base_button = self.add_button("Select dataset path")
        self.base_button.clicked.connect(self.select_path)

        # Button to create annotation
        self.normal_button = self.add_button("Create annotation")
        self.normal_button.clicked.connect(self.make_normal_annotation)

        # Button to Next "img_class" element and Next class
        self.button_next_rose = self.add_button(f"Next {img_classes[0]}")
        self.button_next_tulip = self.add_button(f"Next {img_classes[1]}")


        # Make a grid
        self.layout_grid = QGridLayout(self)
        self.setLayout(self.layout_grid)


    def select_path(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select path")
        # self.dataset_path = os.path.basename(os.path.normpath(folder))

    def add_button(self, text : str, x : int = 0, y : int = 0) -> QPushButton:
        button = QPushButton(text, self)
        button.resize(button.sizeHint())
        button.move(x, y)
        # button.setFixedSize(QSize(size_x, size_y))

        return button
    
    def make_normal_annotation(self):
        try:
            fold = QFileDialog.getExistingDirectory(self, "Select save path")
            make_csv(os.path.join(fold, "dataset"), img_classes, self.dataset_path, "normal")
        except:
            print("error")

if __name__ == "__main__":

    # Получение настроечных данных
    with open("Lab3/setting.json", 'r') as setting_json:
        setting = json.load(setting_json)
    img_classes = setting["objects"]
    
    # Приложение
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())