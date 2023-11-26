import re
import sys
import os
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QApplication,
    QFileDialog,
    QComboBox,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(1, '../Lab2')
from rev_iterator import RevIterator
from annotation import write_csv


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()

        self.iter = None

        self.setWindowTitle("Reviews")
        self.setGeometry(0,0,900,800)
        self.setMaximumSize(900,800)


        self.open_old_dir_btn = QPushButton('Open directory with dataset')
        self.open_old_dir_btn.clicked.connect(self.get_old_dir)
        self.chosen_old_dir=QLabel(self)
        self.chosen_old_dir.setFixedHeight(18)

        self.open_new_dir_btn = QPushButton('Choose directory for new dataset')
        self.open_new_dir_btn.clicked.connect(self.get_new_dir)
        self.chosen_new_dir=QLabel(self)
        self.chosen_new_dir.setFixedHeight(18)

        self.path_to_csv = QPushButton('Choose directory for csv')
        self.path_to_csv.clicked.connect(self.get_csv)
        self.chosen_csv=QLabel(self)
        self.chosen_csv.setFixedHeight(18)

        self.create_annotation_btn=QPushButton('Do the task!')
        self.create_annotation_btn.clicked.connect(self.create_annotation)
        self.task=QLabel(self)
        self.task.setFixedHeight(18)

        self.label_ann=QComboBox(self)
        self.label_ann.addItems(['_', 'default annotation', 'copy to new dir', 'copy with random numbers'])
        self.label_ann.currentIndexChanged.connect( self.ann_changed )
        
        self.label_class=QComboBox(self)
        self.label_class.addItems(['_', '1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
        self.label_class.currentIndexChanged.connect( self.index_changed )
        
        self.review_text=QLabel(self)

        self.next_rev = QPushButton("next review")
        self.next_rev.clicked.connect(self.next_review)


        layout = QVBoxLayout()
        layout.addWidget(self.open_old_dir_btn)
        layout.addWidget(self.chosen_old_dir)
        layout.addWidget(self.open_new_dir_btn)
        layout.addWidget(self.chosen_new_dir)
        layout.addWidget(self.path_to_csv)
        layout.addWidget(self.chosen_csv)
        layout.addWidget(self.label_ann)
        layout.addWidget(self.create_annotation_btn)
        layout.addWidget(self.task)
        layout.addWidget(self.label_class)
        layout.addWidget(self.review_text)
        layout.addWidget(self.next_rev)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def ann_changed(self,i):
        self.ann_label=i-1

    def create_annotation(self):
        self.task.setText('Task was completed, you can see reviews!')
        write_csv(self.csv,self.ann_label,self.new_dir,self.old_dir)

    def index_changed(self,i):
        """
        called on dropdown change
        """
        self.index_class=i
        self.iter = RevIterator(self.csv, i)
        self.show_rev(self.iter.__next__())


    def show_rev(self, path):
        if path==None: 
            return
        with open(path, "r", encoding='utf-8') as f:
            text = f.read()
        new_text=''
        for line in re.findall('(.{%s}|.+$)' % 140, text):
            new_text+=line+'\n'
        self.review_text.setText(new_text)
        self.review_text.setFixedHeight(250)
        self.review_text.adjustSize()


    def next_review(self):
        path=self.iter.__next__() 
        self.show_rev(path)
    

    def get_csv(self):
         self.csv = QFileDialog.getSaveFileName(self,'Choose path to csv','','csv-file(*.csv)')[0]
         self.chosen_csv.setText(f'You chose: {self.csv}')
         print(self.csv)
    
    def get_old_dir(self):
        self.old_dir = QFileDialog.getExistingDirectory(self,'Open working directory')
        self.chosen_old_dir.setText(f'You chose: {self.old_dir}')
        print(self.old_dir)
    
    def get_new_dir(self):
        self.new_dir =QFileDialog.getExistingDirectory(self,'Choose directory')
        self.chosen_new_dir.setText(f'You chose: {self.new_dir}')
        print(self.new_dir)
        
    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
 
    app = QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())