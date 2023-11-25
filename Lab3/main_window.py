from PyQt6 import QtWidgets
from typing import Optional
import sys
from datetime import datetime
sys.path.insert(0,"Lab2")
from sort_csv import split_into_two, sort_by_year, sort_by_week
from get_value_from_date import get_data_for_date


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.init_ui()


    def init_ui(self) -> None:
        """Initialize the main user interface."""
        self.setWindowTitle("Dataset Organizer")

        self.label_folder = QtWidgets.QLabel("Select source folder:")
        self.button_select_folder = QtWidgets.QPushButton("Browse")
        self.button_select_folder.clicked.connect(self.get_source_folder)

        self.dataset1_folder = QtWidgets.QLabel("Select destination folder for sort by X and Y:")
        self.button_create_dataset1 = QtWidgets.QPushButton("Create Dataset(X and Y)")
        self.button_create_dataset1.clicked.connect(self.create_dataset_xy)

        self.dataset2_folder = QtWidgets.QLabel("Select destination folder for sort by years:")
        self.button_create_dataset2 = QtWidgets.QPushButton("Create Dataset(years)")
        self.button_create_dataset2.clicked.connect(self.create_dataset_years)

        self.dataset3_folder = QtWidgets.QLabel("Select destination folder for sort by weeks:")
        self.button_create_dataset3 = QtWidgets.QPushButton("Create Dataset(weeks)")
        self.button_create_dataset3.clicked.connect(self.create_dataset_weeks)

        self.label = QtWidgets.QLabel('Input date (YYYY/MM/DD):')
        self.date_input = QtWidgets.QLineEdit()
        self.get_data_button = QtWidgets.QPushButton('Get data about dollar exchange rate')
        self.data_display = QtWidgets.QLabel()
        self.get_data_button.clicked.connect(self.get_data)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_folder)
        layout.addWidget(self.button_select_folder)
        layout.addWidget(self.dataset1_folder)
        layout.addWidget(self.button_create_dataset1)
        layout.addWidget(self.dataset2_folder)
        layout.addWidget(self.button_create_dataset2)
        layout.addWidget(self.dataset3_folder)
        layout.addWidget(self.button_create_dataset3)
        layout.addWidget(self.label)
        layout.addWidget(self.date_input)
        layout.addWidget(self.get_data_button)
        layout.addWidget(self.data_display)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def get_source_folder(self) -> Optional[str]:
        """Open a dialog to select a source folder."""
        source_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        return source_folder


    def get_destination_folder(self) -> Optional[str]:
        """Open a dialog to select a destination folder."""
        dest_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        return dest_folder


    def create_dataset_xy(self) -> None:
        """Create dataset 1 by splitting data."""
        split_into_two(self.get_destination_folder())
    

    def create_dataset_years(self) -> None:
        """Create dataset 2 by sorting data by years."""
        sort_by_year(self.get_destination_folder())


    def create_dataset_weeks(self) -> None:
        """Create dataset 3 by sorting data by weeks."""
        sort_by_week(self.get_destination_folder())
    

    def get_data(self) -> None:
        """Get data based on the input date."""
        input_date = self.date_input.text()
        date = datetime.strptime(input_date, '%Y/%m/%d')
        value = get_data_for_date(date)
        self.data_display.setText(f"Данные для даты {date} = {value}")

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())