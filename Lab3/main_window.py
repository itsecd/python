import sys
from datetime import datetime
from typing import Optional
from PyQt6 import QtWidgets
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
        self.setFixedSize(400, 400)

        self.label_folder = QtWidgets.QLabel("Select source folder:")
        self.button_select_folder = QtWidgets.QPushButton("Browse")
        self.button_select_folder.clicked.connect(self.get_source_folder)
        self.selected_folder_label = QtWidgets.QLabel("")

        self.combo_box = QtWidgets.QComboBox(self)
        self.combo_box.addItem("X and Y")
        self.combo_box.addItem("Years")
        self.combo_box.addItem("Weeks")

        self.sort_type = QtWidgets.QLabel("Select sorting type and then folder:")
        self.combo_box.activated.connect(self.on_combo_box_activated)

        self.label = QtWidgets.QLabel('Input date (YYYY/MM/DD):')
        self.date_input = QtWidgets.QLineEdit()
        self.get_data_button = QtWidgets.QPushButton('Get data about dollar exchange rate')
        self.data_display = QtWidgets.QLabel()
        self.get_data_button.clicked.connect(self.get_data)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_folder)
        layout.addWidget(self.button_select_folder)
        layout.addWidget(self.selected_folder_label)
        layout.addWidget(self.sort_type)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.label)
        layout.addWidget(self.date_input)
        layout.addWidget(self.get_data_button)
        layout.addWidget(self.data_display)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def get_source_folder(self) -> Optional[str]:
        source_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        if source_folder:
            self.selected_folder_label.setText(f"Folder: {source_folder}")
            return source_folder
        return None


    def get_destination_folder(self) -> Optional[str]:
        """Open a dialog to select a destination folder."""
        dest_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        return dest_folder


    def create_dataset(self, dataset_type: str) -> None:
        """Create datasets based on the dataset type."""
        dest_folder = self.get_destination_folder()
        if dest_folder:
            if dataset_type == "X and Y":
                split_into_two(dest_folder)
            elif dataset_type == "Years":
                sort_by_year(dest_folder)
            elif dataset_type == "Weeks":
                sort_by_week(dest_folder)
            
            QtWidgets.QMessageBox.information(self, "Success", "Dataset created successfully.")
        else:
            QtWidgets.QMessageBox.information(self, "Error", "You didn't choose a folder")
    

    def get_data(self) -> None:
        """Get data based on the input date."""
        input_date = self.date_input.text()
        date = datetime.strptime(input_date, '%Y/%m/%d')
        value = get_data_for_date(date)
        self.data_display.setText(f"Данные для даты {date} = {value}")

    
    def on_combo_box_activated(self) -> None:
        """Handle the activation of the combo box"""
        selected_text = self.combo_box.currentText()
        self.create_dataset(selected_text)

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    app.setActiveWindow(main_window)
    main_window.show()
    sys.exit(app.exec())