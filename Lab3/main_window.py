import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, 
                             QWidget,
                             QPushButton, 
                             QFileDialog,
                             QGridLayout,
                             QLabel,
                             QScrollArea,
                             QMessageBox,
                             QPlainTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

sys.path.insert(1, "C:/Users/boris/Desktop/unik/python_university_laba/Lab2/")

from make_csv import make_csv
from iterator import ImgIterator


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

        #Path to save annotation
        self.fold = None

        #Iterators
        self.iter_rose = None
        self.iter_tulip = None

        # Button to select fold with dataset
        self.select_dataset = QPushButton("Select Dataset")
        self.select_dataset.setAutoDefault(True)
        self.select_dataset.setMinimumSize(200, 100)
        self.select_dataset.setMaximumSize(400, 200)
        self.select_dataset.setStyleSheet("""
            font: bold;
            font-size: 15px;
        """)

        self.select_dataset.clicked.connect(self.select_path)

        self.dataset_path_label = QPlainTextEdit()
        self.dataset_path_label.setReadOnly(True)
        self.dataset_path_label.setMinimumSize(200, 100)
        self.dataset_path_label.setMaximumSize(400, 200)
        self.dataset_path_label.setStyleSheet("""
            font-size: 20px;
        """)


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
        self.create_annotation.setMinimumSize(200, 100)
        self.create_together.setMinimumSize(200, 100)
        self.create_random.setMinimumSize(200, 100)
        self.create_annotation.setMaximumSize(400, 200)
        self.create_together.setMaximumSize(400, 200)
        self.create_random.setMaximumSize(400, 200)
        self.create_annotation.setStyleSheet("""
            font: bold;
            font-size: 15px;
        """)
        self.create_together.setStyleSheet("""
            font: bold;
            font-size: 15px;
        """)
        self.create_random.setStyleSheet("""
            font: bold;
            font-size: 15px;
        """)


        # Buttons to Next "img_class" element and Next class
        self.button_next_rose = QPushButton(f"Next {img_classes[0]}")
        self.button_next_tulip = QPushButton(f"Next {img_classes[1]}")
        self.button_next_rose.setDisabled(1)
        self.button_next_tulip.setDisabled(1)
        self.button_next_rose.clicked.connect(self.next_rose)
        self.button_next_tulip.clicked.connect(self.next_tulip)
        self.button_next_rose.setMinimumSize(200, 100)
        self.button_next_tulip.setMinimumSize(200, 100)
        self.button_next_rose.setStyleSheet("""
            font: bold;
            font-size: 20px;
        """)
        self.button_next_tulip.setStyleSheet("""
            font: bold;
            font-size: 20px;
        """)



        # Image in app
        self.image_ = QLabel()
        self.image_.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea() 
        self.scroll_area.setWidget(self.image_) 
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumSize(500, 500)
        # self.scroll_area.setMaximumSize(1000, 1000)

        # Add Widget in Grid 
        self.grid.setSpacing(0)
        self.grid.addWidget(self.select_dataset, 0, 1)
        self.grid.addWidget(self.dataset_path_label, 1, 1)
        self.grid.addWidget(self.create_annotation, 2, 1)
        self.grid.addWidget(self.create_together, 3, 1)
        self.grid.addWidget(self.create_random, 4, 1)
        self.grid.addWidget(self.scroll_area, 0, 2, 4, 2)
        self.grid.addWidget(self.button_next_rose, 4, 2)
        self.grid.addWidget(self.button_next_tulip, 4, 3)

        self.grid.setSpacing(3)
        
        self.setWindowTitle("Rose & Tulip")
        self.setGeometry(300, 300, 700, 600)
        # self.setMaximumSize(900, 800)
        self.show()


    def select_path(self):
        """
        Select path to dataset
        """
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select path")
        if self.dataset_path:
            self.dataset_path_label.setPlainText(self.dataset_path)
            self.create_annotation.setDisabled(False)            
            self.create_together.setDisabled(False)            
            self.create_random.setDisabled(False)                   


    def make_normal(self):
        """
        Make annotation to dataset
        """
        try:
            # self.fold = QFileDialog.getExistingDirectory(self, "Select save path")
            self.fold = QFileDialog.getSaveFileName(
                self,
                "Select path to csv file:",
                "",
                "CSV File(*.csv)",
            )[0]

            if self.fold == "": raise FileExistsError

            make_csv(self.fold, img_classes, self.dataset_path, "normal", "")
            self.iter_rose = ImgIterator(self.fold, "rose")
            self.iter_tulip = ImgIterator(self.fold, "tulip")
            self.button_next_rose.setDisabled(False)   
            self.button_next_tulip.setDisabled(False) 

            self.msg_ok() 
        except FileExistsError:
            self.msg_didnot_choose()
        except Exception as err:
            self.msg_error(err)


    def make_together(self):
        """
        Make annotation and copy dataset in another fold
        """
        try:
            fold_data = QFileDialog.getExistingDirectory(self, "Select Path to Dataset:")
            fold_csv = QFileDialog.getSaveFileName(
                self,
                "Select path to csv file:",
                "",
                "CSV File(*.csv)",
            )[0]

            if fold_data != "" and fold_csv != "":
                make_csv(fold_csv, img_classes, self.dataset_path, 
                                        "together", fold_data)
                self.msg_ok()
            else: 
                raise FileExistsError
        except FileExistsError:
            self.msg_didnot_choose()
        except Exception as err:
            self.msg_error(err)


    def make_random(self):
        """
        Make annotation and copy dataset in another fold
        """
        try:
            fold_data = QFileDialog.getExistingDirectory(self, "Select Path to Dataset:")
            fold_csv = QFileDialog.getSaveFileName(
                self,
                "Select path to csv file:",
                "",
                "CSV File(*.csv)",
            )[0]

            if fold_data != "" and fold_csv != "":
                make_csv(fold_csv, img_classes, self.dataset_path, 
                                        "random", fold_data)
                
                self.msg_ok()
            else:
                raise FileExistsError
        except FileExistsError:
            self.msg_didnot_choose()
        except Exception as err:
            self.msg_error(err)

    # Я НЕ ПОНИМАЮ ПОЧЕМУ ЭТОТ КОД НЕ РАБОТАЕТ(
    # ОН ПРОСТО НЕ ЗАПУСКАЕТСЯ И ВЫКИДВАЕТ ОШИБКУ
    # 
    # def next_image(self, type : str):
    #     try:
    #         if type == "rose":
    #             pixmap = QPixmap(next(self.iter_rose))
    #         else:
    #             pixmap = QPixmap(next(self.iter_tulip))
    #         self.image_.setPixmap(pixmap)
    #     except StopIteration:
    #         if type == "rose":
    #             self.iter_rose = ImgIterator(self.fold, type)
    #         else:
    #             self.iter_tulip = ImgIterator(self.fold, type)


    def next_rose(self):
        """
        Switch image to next
        """
        try:
            pixmap = QPixmap(next(self.iter_rose))
            self.image_.setPixmap(pixmap)
        except StopIteration:
            self.iter_rose = ImgIterator(self.fold, "rose")

    
    def next_tulip(self):
        """
        Switch image to next
        """
        try:
            pixmap = QPixmap(next(self.iter_tulip))
            self.image_.setPixmap(pixmap)
        except StopIteration:
            self.iter_tulip = ImgIterator(self.fold, "tulip")


    def msg_ok(self):
        """
        Create a good message window
        """
        QMessageBox.information(
            self,
            'Information',
            'Successfully created'
        )


    def msg_didnot_choose(self):
        """
        Create a error message window if person did not choose a fold
        """
        QMessageBox.critical(
            self,
            'Error',
            'You did not choose a fold.'
        )


    def msg_error(self, err : Exception):
        """
        Create a error message in other situations
        """
        QMessageBox.critical(
            self,
            'Error',
            f"{err}"
        )



if __name__ == "__main__":

    # Получение настроечных данных
    with open("Lab3/setting.json", 'r') as setting_json:
        setting = json.load(setting_json)
    img_classes = setting["objects"]
    
    # Приложение
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())