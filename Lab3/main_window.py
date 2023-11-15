import sys
import os
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow, 
                             QPushButton, 
                             QFileDialog)
from PyQt5.QtCore import QSize
from make_csv import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.GUI()

    def GUI(self):
        self.setWindowTitle("App")
        self.setFixedSize(QSize(600, 400))


        self.base_button = QPushButton("Select base path", self)
        self.base_button.clicked.connect(self.select_path)


    def select_path(self):
        folder = QFileDialog.getExistingDirectory(self, "Select path")
        self.base_path = os.path.basename(os.path.normpath(folder))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())