from PyQt6 import QtWidgets
import sys
sys.path.insert(0,"Lab2")
from sort_csv import split_into_two, sort_by_year, sort_by_week

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.init_ui()


    def init_ui(self):
        self.setWindowTitle("Dataset Organizer")

        self.label_folder = QtWidgets.QLabel("Select source folder:")
        self.button_select_folder = QtWidgets.QPushButton("Browse")
        self.button_select_folder.clicked.connect(self.get_source_folder)

        self.dataset1_folder = QtWidgets.QLabel("Select destination folder for sort by X and Y:")
        self.button_create_dataset1 = QtWidgets.QPushButton("Create Dataset 1")
        self.button_create_dataset1.clicked.connect(self.create_dataset1)

        self.dataset2_folder = QtWidgets.QLabel("Select destination folder for sort by years:")
        self.button_create_dataset2 = QtWidgets.QPushButton("Create Dataset 2")
        self.button_create_dataset2.clicked.connect(self.create_dataset2)

        self.dataset3_folder = QtWidgets.QLabel("Select destination folder for sort by weeks:")
        self.button_create_dataset3 = QtWidgets.QPushButton("Create Dataset 3")
        self.button_create_dataset3.clicked.connect(self.create_dataset3)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_folder)
        layout.addWidget(self.button_select_folder)
        layout.addWidget(self.dataset1_folder)
        layout.addWidget(self.button_create_dataset1)
        layout.addWidget(self.dataset2_folder)
        layout.addWidget(self.button_create_dataset2)
        layout.addWidget(self.dataset3_folder)
        layout.addWidget(self.button_create_dataset3)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def get_source_folder(self):
        source_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        return source_folder


    def get_destination_folder(self):
        dest_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        return dest_folder


    def create_dataset1(self):
        split_into_two(self.get_destination_folder())
    

    def create_dataset2(self):
        sort_by_year(self.get_destination_folder())


    def create_dataset3(self):
        sort_by_week(self.get_destination_folder())
        



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())