from enum import Enum
import re
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QApplication,
    QFileDialog,
    QComboBox,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(1, '../Lab2')

from annotation import write_csv
from rev_iterator import RevIterator


class DirOrFile(Enum):
    """
    param for method get_file_or_dir()
    """
    NEW_DIR = 0
    OLD_DIR = 1
    CSV = 2


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.iter = None

        self.setWindowTitle("Reviews")
        self.setGeometry(0, 0, 900, 800)
        self.setMaximumSize(900, 800)

        self.open_old_dir_btn = QPushButton('Open directory with dataset')
        self.open_old_dir_btn.clicked.connect(lambda: self.get_file_or_dir(DirOrFile.OLD_DIR))
        self.chosen_old_dir = QLabel(self)
        self.chosen_old_dir.setFixedHeight(18)

        self.open_new_dir_btn = QPushButton('Choose directory for new dataset')
        self.open_new_dir_btn.clicked.connect(lambda: self.get_file_or_dir(DirOrFile.NEW_DIR))
        self.chosen_new_dir = QLabel(self)
        self.chosen_new_dir.setFixedHeight(18)

        self.path_to_csv = QPushButton('Choose directory for csv')
        self.path_to_csv.clicked.connect(lambda: self.get_file_or_dir(DirOrFile.CSV))
        self.chosen_csv = QLabel(self)
        self.chosen_csv.setFixedHeight(18)

        self.create_annotation_btn = QPushButton('Do the task!')
        self.create_annotation_btn.clicked.connect(self.create_annotation)
        self.task = QLabel(self)
        self.task.setFixedHeight(18)

        self.label_ann = QComboBox(self)
        self.label_ann.addItems(['_',
                                 'default annotation',
                                 'copy to new dir',
                                 'copy with random numbers'])
        self.label_ann.currentIndexChanged.connect(self.ann_changed)

        self.label_class = QComboBox(self)
        self.label_class.addItems(
            ['_', '1 star', '2 stars', '3 stars', '4 stars', '5 stars'])
        self.label_class.currentIndexChanged.connect(self.index_changed)

        self.review_text = QLabel(self)

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

    def create_annotation(self) -> None:
        """
        write csv and new dataset
        """
        try:
            write_csv(self.csv, self.ann_label, self.new_dir, self.old_dir)
            self.task.setText('Task was completed, you can see reviews!')
            QMessageBox.information(self, 'Done', f'Task was completed')
        except Exception as ex:
            QMessageBox.critical(self, 'Error', f'Error: {str(ex)}')

    def index_changed(self, i: int) -> None:
        """
        get amount of stars
        """
        self.index_class = i
        self.iter = RevIterator(self.csv, i)
        self.show_rev(self.iter.__next__())

    def ann_changed(self, i: int) -> None:
        """
        get label of annotation
        """
        self.ann_label = i - 1

    def show_rev(self, path: str) -> None:
        """
        show text of annotation, which from one long line was converted into several
        short ones for convienient  display in the app
        """
        if path == None:
            return
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        new_text = ''
        for line in re.findall('(.{%s}|.+$)' % 140, text):
            new_text += line + '\n'
        self.review_text.setText(new_text)
        self.review_text.setFixedHeight(250)
        self.review_text.adjustSize()

    def next_review(self) -> None:
        """
        get next review
        """
        path = self.iter.__next__()
        self.show_rev(path)

    def get_file_or_dir(self, label_of_file) -> None:
        """
        path to dataset or file
        """
        if label_of_file == DirOrFile.CSV:
            self.csv = QFileDialog.getSaveFileName(
                        self, 'Choose path to csv', '', 'csv-file(*.csv)')[0]
            self.chosen_csv.setText(f'You chose: {self.csv}')
            return
        self.dir = QFileDialog.getExistingDirectory(
            self, 'Open directory')
        if label_of_file == DirOrFile.OLD_DIR:
            self.old_dir=self.dir
            self.chosen_old_dir.setText(f'You chose: {self.dir}')
        else:
            self.new_dir = self.dir
            self.chosen_new_dir.setText(f'You chose: {self.dir}')

    def closeEvent(self, event) -> None:
        event.accept()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
