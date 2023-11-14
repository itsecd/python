import sys
import os
import json
from PyQt5.QtWidgets import (QWidget, QToolTip,
    QPushButton, QApplication, QFileDialog)
from PyQt5.QtGui import QFont
#from ..Lab2.csv_annotation import write_in_file, make_list
#from ..Lab2.main import *
import csv_annotation
from new_name_copy import copy_in_new_directory
from random_of_copy import copy_with_random

with open(os.path.join("Lab2", "settings.json"), "r") as settings:
        settings = json.load(settings)


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))

        self.setToolTip('This is a <b>QWidget</b> widget')
        self.basic_path=None
        base_path = QPushButton('Basic path', self)
        base_path.resize(base_path.sizeHint())
        base_path.move(500, 500)
        base_path.setCheckable(True)
        base_path.clicked.connect(lambda:select_path(self))

        btn = QPushButton('Mode Normal', self)
        btn.resize(btn.sizeHint())
        btn.move(50, 50)
        btn.setCheckable(True)
        btn.clicked.connect(lambda:csv_annotation.write_in_file((os.path.join(settings["directory"], settings["folder"], settings["normal"])), (csv_annotation.make_list(self.basic_path, settings["classes"]))))
        rtn = QPushButton('Mode Copy', self)
        rtn.resize(rtn.sizeHint())
        rtn.move( 50, 200)
        rtn.setCheckable(True)
        rtn.clicked.connect(lambda:copy_in_new_directory(
        self.basic_path,
        settings["classes"],
        settings["main_folder"],
        (os.path.join(settings["directory"],
         settings["folder"], settings["new_name"]))))
        ttn = QPushButton('Random', self)
        ttn.resize(ttn.sizeHint())
        ttn.move(50, 350)
        ttn.setCheckable(True)
        ttn.clicked.connect(lambda:copy_with_random(
        self.basic_path,
        settings["classes"],
        self.basic_path,
        (os.path.join(settings["directory"],
         settings["folder"], settings["random"]))))


        self.setGeometry(400, 100, 1000, 1000)
        self.setWindowTitle('Tooltips')
        self.show()

        def select_path(self):
            file_path = QFileDialog.getExistingDirectory()
            path=os.path.basename(os.path.normpath(file_path))
            self.basic_path=path
            


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())