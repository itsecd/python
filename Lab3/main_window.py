import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QWidget,
                             QPushButton, 
                             QFileDialog,
                             QGridLayout,
                             QLabel,
                             QTextEdit,
                             QLineEdit,
                             QHBoxLayout,
                             QVBoxLayout,
                             QScrollArea,
                             QMessageBox)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from make_csv import *
# from  import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Make a grid
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        # Path to dataset
        self.dataset_path = None

        # Button to select fold with dataset
        self.select_dataset = QPushButton("Select Dataset")
        self.select_dataset.setAutoDefault(True)
        self.select_dataset.setMaximumSize(400, 100)

        self.select_dataset.clicked.connect(self.select_path)

        self.dataset_path_label = QLineEdit()

        # Buttons to create annotation and datset
        self.create_annotation = QPushButton("Create Annotation")
        self.create_together = QPushButton("Create Together")
        self.create_random = QPushButton("Create Random")

        self.create_annotation.setDisabled(1)
        self.create_together.setDisabled(1)
        self.create_random.setDisabled(1)
        self.create_annotation.clicked.connect(self.make_normal)
        self.create_together.clicked.connect(self.make_together)
        self.create_random.clicked.connect(self.make_random)

        # Buttons to Next "img_class" element and Next class
        self.button_next_rose = QPushButton(f"Next {img_classes[0]}")
        self.button_next_tulip = QPushButton(f"Next {img_classes[1]}")
        self.button_next_rose.setDisabled(1)
        self.button_next_tulip.setDisabled(1)
        self.button_next_rose.clicked.connect()
        self.button_next_tulip.clicked.connect()

        # Test image in app
        self.image_ = QLabel()
        # pixmap = QPixmap("C:/Users/boris/Desktop/web/images/avatar.jpg")
        # self.image_.setPixmap(pixmap)

        self.scroll_area = QScrollArea() 
        self.scroll_area.setWidget(self.image_) 
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(700, 700)


        self.grid.setSpacing(10)
        self.grid.addWidget(self.select_dataset, 1, 1)
        self.grid.addWidget(self.dataset_path_label, 2, 1)
        self.grid.addWidget(self.create_annotation, 3, 1)
        self.grid.addWidget(self.create_together, 4, 1)
        self.grid.addWidget(self.create_random, 5, 1)
        self.grid.addWidget(self.scroll_area, 1, 2, 4, 2)
        self.grid.addWidget(self.button_next_rose, 5, 2)
        self.grid.addWidget(self.button_next_tulip, 5, 3)


        
        self.setWindowTitle("Application")
        self.show()


    def select_path(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select path")
        if self.dataset_path:
            self.dataset_path_label.setText(self.dataset_path)
            self.create_annotation.setDisabled(False)            
            self.create_together.setDisabled(False)            
            self.create_random.setDisabled(False)            
            self.button_next_rose.setDisabled(False)   
            self.button_next_tulip.setDisabled(False)            

        # self.dataset_path = os.path.basename(os.path.normpath(folder))

    # def add_button(self, text : str, x : int = 0, y : int = 0) -> QPushButton:
    #     button = QPushButton(text, self)
    #     button.resize(button.sizeHint())
    #     button.move(x, y)
    #     # button.setFixedSize(QSize(size_x, size_y))

    #     return button


    def make_normal(self):
        try:
            fold = QFileDialog.getExistingDirectory(self, "Select save path")
            make_csv(os.path.join(fold, "dataset"), img_classes, self.dataset_path, "normal", "")
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Всё плохо")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()


    def make_together(self):
        try:
            fold_data = QFileDialog.getExistingDirectory(self, "Select Path to Dataset")
            fold_csv = QFileDialog.getExistingDirectory(self, "Select Path to CSV")

            make_csv(os.path.join(fold_csv, "dataset_together"), img_classes, self.dataset_path, 
                                    "together", fold_data)
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Всё плохо")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()


    def make_random(self):
        try:
            fold_data = QFileDialog.getExistingDirectory(self, "Select Path to Dataset")
            fold_csv = QFileDialog.getExistingDirectory(self, "Select Path to CSV")

            make_csv(os.path.join(fold_csv, "dataset_random"), img_classes, self.dataset_path, 
                                    "random", fold_data)
        except:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("Всё плохо")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()

if __name__ == "__main__":

    # Получение настроечных данных
    with open("Lab3/setting.json", 'r') as setting_json:
        setting = json.load(setting_json)
    img_classes = setting["objects"]
    
    # Приложение
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())