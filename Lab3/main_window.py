import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap
sys.path.insert(0,"Lab2")
from make_rel_abs_path import get_full_paths, get_rel_paths, write_to_csv
from iterator import ElementIterator  # Импортируем ваш класс ElementIterator

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dataset Viewer')
        self.layout = QVBoxLayout()

        self.dataset_path = None
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.brown_bear_button = QPushButton('Следующий бурый медведь')
        self.polar_bear_button = QPushButton('Следующий полярный медведь')
        self.layout.addWidget(self.brown_bear_button)
        self.layout.addWidget(self.polar_bear_button)

        self.brown_bear_button.clicked.connect(self.show_next_brown_bear)
        self.polar_bear_button.clicked.connect(self.show_next_polar_bear)

        self.select_dataset_folder()

        self.setLayout(self.layout)

    def select_dataset_folder(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с датасетом')
        if self.dataset_path:
            self.create_csv(self.dataset_path)
            self.brown_iterator = ElementIterator('brown bear', f'{self.dataset_path}/data.csv')
            self.polar_iterator = ElementIterator('polar bear', f'{self.dataset_path}/data.csv')

    def create_csv(self, dataset_path):
        brown_full_paths = get_full_paths('brown bear', dataset_path)
        brown_rel_paths = get_rel_paths('brown bear', dataset_path)
        polar_full_paths = get_full_paths('polar bear', dataset_path)
        polar_rel_paths = get_rel_paths('polar bear', dataset_path)

        csv_file = os.path.join(dataset_path, 'data.csv')
        if os.path.exists(csv_file):
            os.remove(csv_file)

        write_to_csv(csv_file, brown_full_paths, brown_rel_paths, 'brown bear')
        write_to_csv(csv_file, polar_full_paths, polar_rel_paths, 'polar bear')

    def show_next_brown_bear(self):
        if self.dataset_path:
            next_path = next(self.brown_iterator)
            if next_path:
                self.show_image(next_path)

    def show_next_polar_bear(self):
        if self.dataset_path:
            next_path = next(self.polar_iterator)
            if next_path:
                self.show_image(next_path)

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(300, 300))
        else:
            self.image_label.setText('Изображение не найдено')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())