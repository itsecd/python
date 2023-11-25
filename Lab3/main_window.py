import os
import sys
import logging
sys.path.insert(0, "Lab2")
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLineEdit, QVBoxLayout, QWidget, QInputDialog, QLabel, QMessageBox
from main import main_function
from get_data import get_data_for_date


logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        self.setWindowTitle('Курс доллара по дате')
        
        self.source_folder = ""
        self.destination_folder = ""
        
        self.select_source_folder_btn = QPushButton('Выбрать папку исходного датасета', self)
        self.select_source_folder_btn.clicked.connect(self.select_source_folder)

        self.create_dataset_btn = QPushButton('Создать датасет', self)
        self.create_dataset_btn.clicked.connect(self.create_dataset)

        self.date_input = QLineEdit(self)
        self.date_input.setPlaceholderText('Введите дату в формате YYYY-MM-DD')

        self.get_data_btn = QPushButton('Получить данные', self)
        self.get_data_btn.clicked.connect(self.get_data)

        self.data_display = QLabel(self)
        self.data_display.setText("Данные по дате:")

        layout = QVBoxLayout()
        layout.addWidget(self.select_source_folder_btn)
        layout.addWidget(self.create_dataset_btn)
        layout.addWidget(self.date_input)
        layout.addWidget(self.get_data_btn)
        layout.addWidget(self.data_display)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


    def select_source_folder(self) -> None:
        """Opens a dialog for selecting the source folder and writes the selected folder path to the variable"""
        folder_path = QFileDialog.getExistingDirectory(self, 'Выберите папку исходного датасета')
        self.source_folder = folder_path


    def create_dataset(self) -> None:
        """Opens a dialog for selecting a mode, destination folder, and then calls the main_function with the selected mode"""
        modes = ['X_Y', 'years', 'weeks', 'find data']
        mode, ok = QInputDialog.getItem(self, 'Выбор режима', 'Выберите режим:', modes, 0, False)
        if ok and mode:
            self.destination_folder = QFileDialog.getExistingDirectory(self, 'Выберите папку назначения')
            while not self.destination_folder:
                QMessageBox.warning(self, 'Внимание', 'Выберите папку назначения.')
                self.destination_folder = QFileDialog.getExistingDirectory(self, 'Выберите папку назначения')
            args = {
                'mode': mode,
                'path_file': self.destination_folder,
                'output_x': 'X.csv',
                'output_y': 'Y.csv',
                'date': self.date_input.text()
            }
            main_function(args)
            


    def get_data(self) -> None:
        """Converts the date to the desired format, calls the get_data function and set text to the window"""
        date_str = self.date_input.text()
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {date_str}. Please enter the date in the format YYYY-MM-DD.")
            return
        args = {
            'mode': "find data",
            'path_file': os.path.join(self.source_folder, "dataset.csv"),
            'output_x': 'X.csv',
            'output_y': 'Y.csv',
            'date': date
        }
        data_for_date = get_data_for_date(date, args['path_file'], args['output_x'], args['output_y'])
        self.data_display.setText(f"Данные по дате {date}: {data_for_date}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())