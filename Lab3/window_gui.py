import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton, QLabel, QFileDialog, QMessageBox, QTextBrowser

sys.path.insert(1, r"C:\Users\Ceh9\PycharmProjects\pythonProject\Lab2")
from create_annotation import create_annotation_file
from copy_dataset import copy_and_rename_dataset, generate_random_set
from iterator import ReviewIterator




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.review_iterator = None
        self.folder_path = ''
        self.annotation_path = ''
        self.review_path = ''
        self.class_label = ''
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dataset Processing App')
        self.setGeometry(300, 300, 200, 200)

        self.folder_label = QtWidgets.QLabel('Select Dataset Folder:')
        self.folder_path = QtWidgets.QLabel(self)
        self.browse_button = QtWidgets.QPushButton('Browse',self)
        self.browse_button.clicked.connect(self.browse_folder)

        self.create_annotation_button = QtWidgets.QPushButton('Create Annotation File',self)
        self.create_annotation_button.clicked.connect(self.create_annotation)

        self.copy_dataset_button = QtWidgets.QPushButton('Copy Dataset',self)
        self.copy_dataset_button.clicked.connect(self.copy_dataset)

        self.random_copy = QtWidgets.QCheckBox('Random name of new dataset?')

        self.need_annotation = QtWidgets.QCheckBox('New annotation for copied dataset?')

        self.next_good_button = QPushButton('Next Good Review', self)
        self.next_bad_button = QPushButton('Next Bad Review', self)
        self.next_good_button.clicked.connect(lambda: self.next('good'))
        self.next_bad_button.clicked.connect(lambda: self.next('bad'))

        self.txt_file = QLabel(self)
        self.text_label = QTextBrowser(self)
        self.text_label.setText("Review:")
        self.text_label.setMinimumSize(300, 200)
        self.text_label.setMaximumSize(1920,1080)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.folder_label)
        layout.addWidget(self.folder_path)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.create_annotation_button)
        layout.addWidget(self.copy_dataset_button)
        layout.addWidget(self.random_copy)
        layout.addWidget(self.need_annotation)
        layout.addWidget(self.next_good_button)
        layout.addWidget(self.next_bad_button)
        layout.addWidget(self.txt_file)
        layout.addWidget(self.text_label)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_folder(self)-> None:
        """
        Получаем директорию с датасетом и создаем итератор
        """
        self.folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if self.folder_path:
            dataset_iterator = self.get_dataset_files()
            dataset_files = list(dataset_iterator)
            self.review_iterator = ReviewIterator(dataset_files)

    def create_annotation(self) -> None:
        """
        Создаем аннотацию
        """
        folder_path = self.folder_path
        QtWidgets.QMessageBox.information(self, 'Select', 'Select Destination File And Name Of Annotation file')
        destination_file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select Destination File', filter='(*.csv)')

        if folder_path or destination_file is None:
            QMessageBox.warning(None, "Folder path or Destination file not selected", "No file selected for create annotation")
        else:
            create_annotation_file(folder_path, destination_file)
            self.annotation_path = os.path.join(folder_path, destination_file)
            QtWidgets.QMessageBox.information(self, 'Success', 'Annotation file created successfully.')

    def copy_dataset(self) -> None:
        """
        Копируем датасет и в зависимости от чек боксов мы называем его и создаем аннотацию
        """
        source_folder = self.folder_path.text()
        destination_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        destination_folder = os.path.join(destination_folder, "_dataset")
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)
        if source_folder or destination_folder is None:
            QMessageBox.warning(None, "Source or Destination folder not selected", "No file selected for copy")
        else:
            copy_and_rename_dataset(source_folder, destination_folder, destination_folder, self.random_copy.isChecked())
            if self.need_annotation.isChecked():
                annotation_file, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Select Destination File',filter='(*.csv)')
                if annotation_file is None:
                    QMessageBox.warning(None, "Annotation file name is empty", "No annotation file name")
                else:
                    create_annotation_file(destination_folder, str(annotation_file))
                    self.annotation_path = os.path.join(destination_folder, annotation_file)
            QtWidgets.QMessageBox.information(self, 'Success', 'Dataset copy successfully.')

    def next(self, review_type):
        """
         Выводим на экран ревью
        """
        if self.review_iterator is None:
            QMessageBox.warning(None, "File not selected", "No file selected for iteration")
            return

        if review_type == "good":
            element = self.review_iterator.next_good()
        elif review_type == "bad":
            element = self.review_iterator.next_bad()
        else:
            QMessageBox.warning(None, "Invalid value", "An invalid value has been selected")
            return

        review_path = element

        if review_path is None:
            QMessageBox.warning(None, "End of class", "No more files for this class")
            return

        self.text_label.update()

        with open(review_path, 'r', encoding='utf-8') as file:
            self.txt_file.setText(review_path)
            self.text_label.setText(file.read())

    def get_dataset_files(self):
        if self.folder_path:
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    yield os.path.join(root, file)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())