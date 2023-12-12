import os
import sys
from typing import List, Set
from functools import partial
from dotenv import load_dotenv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget,
                             QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, 
                             QListWidget, QFileDialog,
                             QTableWidget,QTableWidgetItem, 
                             QLayoutItem, QHeaderView)

sys.path.insert(0,os.path.join(sys.path[0].replace("Lab3", "Lab2")))
from src.write_reader import ClassInstanceIterator, DataWriteReader 


load_dotenv()

class ImageClient(QWidget):
    '''This is a class designed as a QT window for convenient work with images in datasets'''
    def __init__(self):
        super().__init__()
        self.storages : list[DataWriteReader] = []
        self.set_appears()
        self.initUI()
        self.connects()
        self.show()

    def set_appears(self) -> None:
        '''this method initializes the application window settings'''
        self.setWindowTitle(os.getenv("TITLE"))
        self.resize((int(os.getenv("WIDTH"))),int(os.getenv("HEIGHT")))
        self.move(int(os.getenv("X_OFF")),int(os.getenv("Y_OFF")))  

    def initUI(self) -> None:
        '''this method initializes all Qt widgets used in the application window'''
        self.label3 = QLabel(os.getenv("LIST_LABEL")) 
        self.label4 = QLabel(os.getenv("LIST_ITEM")) 
        self.open_btn = QPushButton(os.getenv("BUTTON_OPEN"))
        self.create_anotation_btn = QPushButton(os.getenv("BUTTON_ANOTATION"))
        self.read_anotation_btn = QPushButton(os.getenv("BUTTON_READ"))
        self.storage_list = QListWidget()
        self.file_list = QListWidget()
        self.class_label = QLabel(alignment= Qt.AlignCenter)
        self.image = QLabel(alignment= Qt.AlignCenter)
        self.image.setMinimumWidth(600)
        self.image.setFixedHeight(600)
        self.image.setWordWrap(True)
        self.table = QTableWidget(1, 3, self)
        self.table.setHorizontalHeaderLabels([os.getenv("COLUMN_1"), os.getenv("COLUMN_2"), os.getenv("COLUMN_3")])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch) 
        self.lv1 = QVBoxLayout()
        self.lv2 = QVBoxLayout()
        self.lv3 = QVBoxLayout()

        self.lh1 = QHBoxLayout()
        self.lh2 = QHBoxLayout()
        self.lh3 = QHBoxLayout()        

        self.lh2.addWidget(self.label3,alignment= Qt.AlignLeft)
        self.lv1.addLayout(self.lh2)
        self.lv1.addWidget(self.storage_list,alignment= Qt.AlignLeft)
        self.lv1.addWidget(self.open_btn)
        self.lv1.addWidget(self.create_anotation_btn)
        self.lv1.addWidget(self.read_anotation_btn)
        
        self.lv2.addWidget(self.class_label)    
        self.lv2.addWidget(self.image)
        self.lv2.addLayout(self.lh3)
        self.lv2.addWidget(self.table)
        self.lv3.addWidget(self.label4,alignment= Qt.AlignLeft)
        self.lv3.addWidget(self.file_list, alignment= Qt.AlignRight)

        self.lh1.addLayout(self.lv1)
        self.lh1.addLayout(self.lv2)
        self.lh1.addLayout(self.lv3)

        self.setLayout(self.lh1)

    def connects(self) -> None:
        '''this method slots for signals are declared'''
        self.open_btn.clicked.connect(self.open_dir)
        self.storage_list.itemClicked.connect(self.selectionDir)
        self.file_list.itemClicked.connect(self.selection_item)
        self.read_anotation_btn.clicked.connect(self.read_annotation)

    def selectionDir(self) -> None:
        '''this method fills the contents of the widgets when the active element is selected in the storageList'''
        self.file_list.clear()  
        index: int = self.storage_list.currentRow() 
        self.iter = ClassInstanceIterator(DataWriteReader.key3, None, *self.storages[index].data_list) 
        classes: Set[str] = set(self.create_table(index))  
        self.create_iters_btns(classes)
        self.create_anotation_btn.clicked.connect(partial(self.create_anotation, index))  

    def create_table(self, index: int) -> List[str]:
        '''this method fills the contents of the table, creating an annotation for the dataset'''
        classes: List[str] = []
        self.table.setRowCount(len(self.storages[index].data_list))     
        for index, row in enumerate(self.storages[index].data_list):
            self.file_list.addItem(f'{row[DataWriteReader.key1]}')            
            self.table.setItem(index, 0, QTableWidgetItem(row[DataWriteReader.key1]))
            self.table.setItem(index, 1, QTableWidgetItem(row[DataWriteReader.key2]))
            self.table.setItem(index, 2, QTableWidgetItem(row[DataWriteReader.key3]))
            classes.append(row[DataWriteReader.key3])
        return classes 
           
    def create_iters_btns(self, classes: List[str]) -> None:
        '''this method generates a button widget for each image class,\n
           and connects the corresponding slot to it
        '''
        self.destroy_iters_btns()
        for index, cl in enumerate(classes):
            btn_next = QPushButton(f"next {cl}")
            btn_next.clicked.connect(partial(self.get_next_image, cl))
            self.lh3.addWidget(btn_next)     

    def destroy_iters_btns(self) -> None:
        '''this method removes all button widgets from the layout'''
        while self.lh3.count():
            item: QLayoutItem = self.lh3.takeAt(0)
            widget: QWidget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.lh3.removeItem(item)

    def selection_item(self) -> None:
        '''this method loads an image when you select it from the "Items" widget list'''
        path: str = self.file_list.currentItem().text()
        self.load_img(path)           

    def open_dir(self) -> None:
        '''this method calls a file dialog to select the working directory,\n
           and creates an instance of the "DataWriteReader" class based on it
        '''
        path: str = QFileDialog.getExistingDirectory()
        if path: 
            self.storages.append(DataWriteReader(path))
            self.storage_list.addItem(path)
            

    def load_img(self, path: str) -> None:
        '''this method loads the image to display it on the widget'''
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)
        self.class_label.setText(f"<h1>{self.iter._ClassInstanceIterator__class_label}</h1>")        
            
    def get_next_image(self, cl: str) -> None:
        '''this method to iterate images over a selected image class'''
        self.iter.change_class_label(cl)
        path: str = next(self.iter)[DataWriteReader.key1]
        self.load_img(path)

    def create_anotation(self, index: int) -> None:
        '''this method saves the data of the current instance of "DataWriteReader" in "csv" format'''
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None,"Сохранить файл", "", "Data (*.csv)", options=options)
        self.storages[index].write_to_csv(f"{fileName}.csv")

    def read_annotation(self)-> None:
        '''this method reads data from a "csv" file\n
           and creates an instance of the "DataWriteReader" class based on it
        '''
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "Data (*.csv)", options=options)
        if fileName:
            self.storages.append(DataWriteReader(fileName).read_from_csv(fileName))
            self.storage_list.addItem(fileName)


class ImageClientApp(QApplication):
        
    def __init__(self, argv: List[str]):
        super().__init__(argv)

    def run(self) -> None:
        self.window = ImageClient()
        self.exec()