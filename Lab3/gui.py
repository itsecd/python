import re
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDesktopWidget

sys.path.insert(1, '../Lab2')
from rev_iterator import RevIterator
from annotation import write_csv


class MyApp(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyApp,self).__init__()

        avail_width=QDesktopWidget().availableGeometry().width()
        avail_height=QDesktopWidget().availableGeometry().height()
        self.setGeometry(avail_width//5, avail_height//4, avail_width//2, avail_height//2)

        iter=RevIterator('review.csv',1)

        self.r_text=QtWidgets.QLabel(self)
        self.btn=QtWidgets.QPushButton(self)
        
        self.path_to_dataset=QtWidgets.QLineEdit(self)
        self.path_to_dataset.setFixedWidth(200)
        self.path_to_dataset.setPlaceholderText("Enter path to dataset")

        self.path_to_csv=QtWidgets.QLineEdit(self)
        self.path_to_csv.move(300,0)
        self.path_to_csv.setFixedWidth(200)
        self.path_to_csv.setPlaceholderText("Enter path to annotation")

        self.path_to_new_d=QtWidgets.QLineEdit(self)
        self.path_to_new_d.move(600,0)
        self.path_to_new_d.setFixedWidth(200)
        self.path_to_new_d.setPlaceholderText("Enter path to new dataset")

        self.btn.setText("next review")
        self.btn.move( avail_width//4-100,avail_height//3)
        self.btn.setFixedWidth(200)
        self.btn.clicked.connect(lambda: self.next_review(iter))

    def next_review(self,iter):
        path=iter.__next__() 
        if path==None: 
            return
        with open(path, "r", encoding='utf-8') as f:
            text = f.read()
        new_text=''
        for line in re.findall('(.{%s}|.+$)' % 120, text):
            new_text+=line+'\n'
        self.r_text.setText(new_text)
        self.r_text.move(100,50)
        self.r_text.setFixedHeight(250)
        self.r_text.adjustSize()
        
    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
 
    app = QApplication(sys.argv)
    win=MyApp()
    win.show()
    sys.exit(app.exec_())