import os
import sys
import logging

from PyQt6.QtCore import QSize
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
sys.path.append("C:\\Users\\Yana\\Documents\\python-v6\\Lab2")
from csv_ import write_csv,make_list
from iterator import TagIterator
from copy_dataset import copy_dataset



logging.basicConfig(filename="py_log3.log", filemode="a", level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Pictures")
        self.setGeometry(0,0,900,900)
        self.setMaximumSize(900,900)
        widget=QWidget()
        buttons_layout=QVBoxLayout()
        layout=QGridLayout()

        self.dataset=os.path.abspath("Lab1\dataset")
        base=QLabel(f'Basic dataset:{self.dataset}',self)
        base.setFixedSize(QSize(350,50))
        buttons_layout.addWidget(base)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.tags=['tiger','leopard']
        self.iterator=None
        self.image_path=None

        #images gallery
        self.image=QLabel(self)
        self.image.setFixedSize(500,500)
        self.image.setScaledContents(True)

        #buttons
        self.button_annotation=self.add_button("Create annotation",300,50)
        self.button_сopy_dataset=self.add_button("Copy dataset",300,50)
        self.button_сopy_random=self.add_button("Copy dataset with random numbers",300,50)
        self.button_ntiger=self.add_button("Next tiger",300,50)
        self.button_nleopard=self.add_button("Next leopard",300,50)
        self.button_iterator=self.add_button("Iterator",300,50)
        self.button_exit=self.add_button("Exit",300,50)

        #widgets
        buttons_layout.addWidget(self.button_annotation)
        buttons_layout.addWidget(self.button_сopy_dataset)
        buttons_layout.addWidget(self.button_сopy_random)
        buttons_layout.addWidget(self.button_ntiger)
        buttons_layout.addWidget(self.button_nleopard)
        buttons_layout.addWidget(self.button_iterator)
        buttons_layout.addWidget(self.button_exit)
        buttons_layout.addStretch()

        layout.addLayout(buttons_layout,1,0)
        layout.addWidget(self.image,1,1)

        #app functions
        self.button_annotation.clicked.connect(self.create_annotation)
        #self.button_сopy_dataset.connect(self.copy_dataset)
        self.button_iterator.clicked.connect(self.path_to_csv)
        self.button_ntiger.clicked.connect(self.next_first_tag)
        self.button_ntiger.clicked.connect(self.next_second_tag)
        self.button_exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name:str, size_x:str, size_y:str):
        button=QPushButton(name,self)
        button.resize(button.sizeHint())
        button.setFixedSize(QSize(size_x,size_y))
        return button
          
    def create_annotation(self):
        try:
            directory = QFileDialog.getSaveFileName(self,"Select folder - ",
            "","CSV File(*.csv)",)[0]
            if directory == "":
                QMessageBox.information(None, "Path error")
                return
            l=make_list(self.dataset, self.tags)
            print('make list')
            write_csv(directory, l)
            QMessageBox.information(None, "Done", "Annotation created")
        except Exception as e:
            logging.error(f"Annotation not created:{e}")
    def path_to_csv(self):
        try:
            path=QFileDialog.getOpenFileName(self,"Select picture - ")[0]
            if(path==""):
                QMessageBox.information(None, "Path error")
                return
            self.iterator=TagIterator(path,self.tags[0],self.tags[1])
            print(path)
        except Exception as e:
            logging.error(f"Error with picture:{e}")
    
    def next_first_tag(self):
        if (self.iterator==None):
            QMessageBox.information(None, "Error when selecting picture")
            return
        tag=self.iterator.next_first_tag()
        self.image_path=tag
        self.image.update()
        self.image.setPixmap(QPixmap(tag))

    def next_second_tag(self):
        if (self.iterator==None):
            QMessageBox.information(None, "Error when selecting picture")
            return
        tag=self.iterator.next_second_tag()
        self.image_path=tag
        self.image.update()
        self.image.setPixmap(QPixmap(tag)) 
 



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()