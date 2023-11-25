import sys
import os
import logging
from PyQt5 import QtWidgets, QtGui
sys.path.insert(1, 'Lab2')
from iterator import ImageIterator
from create_annotation import list_files_in_directory, generate_annotation, write_annotation_to_csv
from create_copy_dataset import copy_dataset

logging.basicConfig(level=logging.INFO)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Dataset Annotation App")

        # Create widgets
        self.create_annotation_button = QtWidgets.QPushButton("Создать аннотацию")

        # Connect buttons to functions
        self.create_annotation_button.clicked.connect(self.create_annotation)

        # Layout setup
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.create_annotation_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def create_annotation(self):
        dataset_folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку для создания аннотации:')

        if dataset_folder:
            annotation_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Выберите папку для сохранения аннотации:', '', 'All Files (*)')

            if annotation_path:
                try:
                    write_annotation_to_csv(dataset_folder, annotation_path)
                    QtWidgets.QMessageBox.information(
                        self, 'Success', 'Annotation file created successfully!')
                except Exception as ex:
                    logging.error(f"Failed to create annotation: {ex}")
                    QtWidgets.QMessageBox.critical(
                        self, 'Error', f'Failed to create annotation: {ex}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())