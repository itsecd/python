import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
sys.path.insert(1, '../Lab2')
from rev_iterator import RevIterator
from annotation import write_csv

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        text=QtWidgets.QLabel(self)
        write_csv()
         
        text.adjustSize()
        self.setGeometry(200, 200, 700, 700)
        self.show()

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    write_csv()
    iter=RevIterator('review.csv',1)
    print(iter.__next__())
        

    #app = QApplication(sys.argv)
       
    #ex = MyApp()
    #sys.exit(app.exec_())