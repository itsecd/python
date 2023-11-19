"""Module providing a function printing python version 3.11.5."""
import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QRect
from functools import partial
sys.path.append("D:/python/Lab2")
import create_annotation
import copy_dataset_in_new_folder
import copy_dataset_random_names
import class_iterator


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Lab 2")
        self.resize(420, 200)
        self.move(0, 0)
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.setFont(font)


        self.folderpath_dataset = None
        self.folderpath_dataset_random = None
        self.annotation_to_iterate = None
        self.image = QLabel(self)
        self.image.setFixedSize(400, 400)

        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.image.setGeometry(QRect(9, 149, 441, 431))

        self.create_annotation_button = QPushButton("Create Annotation")
        self.create_annotation_button.clicked.connect(self.create_annotation)
        self.layout.addWidget(self.create_annotation_button)

        self.create_dataset_button = QPushButton("Create copy Dataset")
        self.create_dataset_button.clicked.connect(self.create_new_dataset)
        self.layout.addWidget(self.create_dataset_button)

        self.create_dataset_random_button = QPushButton("Create Dataset random names")
        self.create_dataset_random_button.clicked.connect(self.create_new_random_dataset)
        self.layout.addWidget(self.create_dataset_random_button)

        self.tiger_iterator = None
        self.next_button_tiger = QPushButton("Next tiger")
        self.next_button_tiger.clicked.connect(partial(self.next_tiger))
        self.layout.addWidget(self.next_button_tiger)

        self.leopard_iterator = None
        self.next_button_leopard = QPushButton("Next leopard")
        self.next_button_leopard.clicked.connect(partial(self.next_leopard))
        self.layout.addWidget(self.next_button_leopard)

        self.update_iterators_buttton = QPushButton("Update iterators")
        self.update_iterators_buttton.clicked.connect(partial(self.update_iterators_with_message))
        self.layout.addWidget(self.update_iterators_buttton)

        self.go_to_exit = QPushButton("Exit")
        self.go_to_exit.clicked.connect(self.close)
        self.layout.addWidget(self.go_to_exit)

        self.show()


    def create_annotation(self):
        '''This function create standart type if annotation'''
        self.folderpath_dataset = QFileDialog.getExistingDirectory(self, 'Select Folder to dataset')
        try:
            save_filepath, _ = QFileDialog.getSaveFileName(self, 'Save Annotation File', '', '')
            folder_path = os.path.dirname(save_filepath)
            name = os.path.basename(save_filepath).rsplit('.', 1)[0]
            if save_filepath & name != "":
                create_annotation.create_csv_annotation(self.folderpath_dataset, name ,folder_path)
                QMessageBox.information(self,
                                        "Success",
                                        "Annotation file created successfully!")
            else:
                QMessageBox.warning(self, "Error", f"Please select a correct folder next time")
        except Exception:
            QMessageBox.warning(self, "Error", f"Please select a correct folder\n {Exception}")
            self.folderpath_dataset = None


    def create_new_dataset(self):
        '''
        The function creates a dataset without included folders,
        as well as an annotation to it
        '''
        folder_new_dataset = QFileDialog.getExistingDirectory(
                None,
                "Select a folder",
                "",
                QFileDialog.ShowDirsOnly
                )
        try:
            save_filepath, _ = QFileDialog.getSaveFileName(self, 'Save Annotation File', '', '')
            folder_path = os.path.dirname(save_filepath)
            name = os.path.basename(save_filepath).rsplit('.', 1)[0]
            if save_filepath & name != "":
                copy_dataset_in_new_folder.copy_dataset_in_new_folder(
                    folder_new_dataset,
                    "dataset",
                    name,
                    folder_path
                    )
                QMessageBox.information(self, "Success", "Dataset created successfully!")
            else:
                QMessageBox.warning(self, "Error",
                                f"Please select a correct folder or annotation next time")
        except Exception:
            QMessageBox.warning(self, "Error", f"Please select a correct folder\n {Exception}")


    def create_new_random_dataset(self):
        '''
        The function creates a dataset without included folders and random names,
        as well as an annotation to it
        '''
        save_folderpath = QFileDialog.getExistingDirectory(
            None,
            "Select a folder for dataset",
            "",
            QFileDialog.ShowDirsOnly
            )
        try:
            save_filepath, _ = QFileDialog.getSaveFileName(self, 'Save annotation file', '', '')
            annotation_folder_path = os.path.dirname(save_filepath)
            name = os.path.basename(save_filepath).rsplit('.', 1)[0]
            if name != "" & save_filepath:
                copy_dataset_random_names.copy_dataset_in_new_folder(
                    save_folderpath,
                    "dataset",
                    name,
                    annotation_folder_path
                    )
                QMessageBox.information(self, "Success", "Dataset created successfully!")
            else:
                QMessageBox.warning(self, "Error",
                                f"Please select a correct annotation name next time")
        except Exception:
            QMessageBox.warning(self, "Error",
                                f"Please select a correct folder or annotation\n {Exception}")


    def next_tiger(self):
        '''Iterator for tiger'''
        if self.annotation_to_iterate:
            try:
                if not self.tiger_iterator:
                    self.tiger_iterator = class_iterator.PhotoIterator(self.annotation_to_iterate,
                                                                    "tiger")
                pixmap = QPixmap(next(self.tiger_iterator))
                self.image.update()
                self.image.setPixmap(pixmap)
                self.layout.addWidget(self.image)
            except Exception:
                QMessageBox.warning(self, "Error", f"Please select a correct folder\n {Exception}")
                self.update_iterators()
        else:
            self.annotation_to_iterate = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]


    def next_leopard(self):
        '''Iterator for leopard'''
        if self.annotation_to_iterate:
            try:
                if not self.leopard_iterator:
                    self.leopard_iterator = class_iterator.PhotoIterator(self.annotation_to_iterate,
                                                                        "leopard")
                pixmap = QPixmap(next(self.leopard_iterator))
                self.image.update()
                self.image.setPixmap(pixmap)
                self.layout.addWidget(self.image)
            except Exception:
                QMessageBox.warning(self, "Error", f"Please select a correct folder next time\n {Exception}")
                self.update_iterators()
        else:
            self.annotation_to_iterate = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]


    def update_iterators(self):
        '''Removes iterators and the path to the annotation'''
        self.annotation_to_iterate = None
        self.leopard_iterator = None
        self.tiger_iterator = None


    def update_iterators_with_message(self):
        '''Calls the update_iterators function and gives the user a message'''
        self.update_iterators()
        QMessageBox.information(self,
                                "Success",
                                "Iterators updated!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
