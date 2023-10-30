import os
from dotenv import load_dotenv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget,
                             QHBoxLayout, QVBoxLayout,
                             QGroupBox, QRadioButton,
                             QPushButton, QLabel, 
                             QListWidget, QLineEdit,
                             QFileDialog)
from write_reader import *


load_dotenv()

class ImageClient(QWidget):
    def __init__(self):
        super().__init__()
        self.storages : list[DataWriteReader] = []
        self.set_appears() #настраиваем окно
        self.initUI()      #обьявляются виджеты
        self.connects()    #подключение обработчиков событий
        self.show()        #показать окно
    def set_appears(self):
        self.setWindowTitle(os.getenv("TITLE"))
        self.resize((int(os.getenv("WIDTH"))),int(os.getenv("HEIGHT")))
        self.move(int(os.getenv("X_OFF")),int(os.getenv("Y_OFF")))  
    def initUI(self):
        self.label3 = QLabel(os.getenv("LIST_LABEL")) 
        self.label4 = QLabel(os.getenv("LIST_ITEM")) 
        self.open_btn = QPushButton(os.getenv("BUTTON_OPEN"))
        self.storage_list = QListWidget()
        self.file_list = QListWidget()
        self.image = QLabel()
        pixmap = QPixmap("D:/Coding/Python/python/dataset/brown_bear/0001.jpg")
        self.image.setPixmap(pixmap)
        self.image.resize(900,300)
        

        self.lv1 = QVBoxLayout()
        self.lv2 = QVBoxLayout()
        self.lv3 = QVBoxLayout()

        self.lh1 = QHBoxLayout()
        self.lh2 = QHBoxLayout()        

        self.lh2.addWidget(self.label3,alignment= Qt.AlignLeft)
        self.lh2.addWidget(self.open_btn,alignment= Qt.AlignLeft)

        self.lv1.addLayout(self.lh2)
        self.lv1.addWidget(self.storage_list,alignment= Qt.AlignLeft)

        self.lv2.addWidget(self.image)
        self.lv3.addWidget(self.label4)
        self.lv3.addWidget(self.file_list)

        self.lh1.addLayout(self.lv1)
        self.lh1.addLayout(self.lv2)
        self.lh1.addLayout(self.lv3)

        self.setLayout(self.lh1)

    def connects(self):
        self.open_btn.clicked.connect(self.open)
        self.storage_list.itemClicked.connect(self.selectionChanged)

    def selectionChanged(self, item):
        for i in self.storages[0].data_list:
            self.file_list.addItem(f'{i[DataWriteReader.key1]}')    

    def open(self):
        path = QFileDialog.getExistingDirectory()
        if path: 
            self.storages.append(DataWriteReader(path))
            self.storage_list.addItem(path)
            




app = QApplication([])  

window = ImageClient()

app.exec()