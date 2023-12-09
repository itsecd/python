import sys
import logging
import main
import realization
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout,)

logging.basicConfig(level=logging.INFO)


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setGeometry(800, 200, 200, 200)
        self.setWindowTitle("Lab4-var3")
        self.setStyleSheet('background-color: #2cf3e2;')

        main_widget = QWidget()
        box_layout = QVBoxLayout()
        layout = QGridLayout()
        src = QLabel(f"Выберете нужные действия", self)
        box_layout.addWidget(src)
        main_widget.setStyleSheet(
            "max-width: 1500%;" "margin: 0 0 0 5%;" "height: auto;" "position: center;" "padding: 5% 40% 5% 40%;")

        # установим кнопки
        self.frame = self.add_button("frame")
        self.balance_test = self.add_button("balance_test")
        self.filter = self.add_button("filter")
        self.max_filter = self.add_button("max_filter")
        self.grouping = self.add_button("grouping")
        self.draw_histogram = self.add_button("draw_histogram")
        self.exit = self.add_button("Выйти из программы")

        # форматируем виджеты по размеру окна
        box_layout.addWidget(self.frame)
        box_layout.addWidget(self.balance_test)
        box_layout.addWidget(self.filter)
        box_layout.addWidget(self.max_filter)
        box_layout.addWidget(self.grouping)
        box_layout.addWidget(self.draw_histogram)
        box_layout.addWidget(self.exit)
        box_layout.addStretch()  # кнопки вплотную
        layout.addLayout(box_layout, 0, 0)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # то, что будет на экране при нажатии кнопки создания аннотации, копии и рандома
        self.frame.clicked.connect(
            lambda: main.save_csv(frame, 'lab4/set/frame.csv'))
        self.balance_test.clicked.connect(lambda: main.save_csv(
            realization.balance_test(frame), 'lab4/set/balance_test.csv'))
        self.filter.clicked.connect(lambda: main.save_csv(
            realization.filter(frame, 1), 'lab4/set/filter.csv'))
        self.max_filter.clicked.connect(lambda: main.save_csv(realization.max_filter(
            frame, max_width, max_height,  1), 'lab4/set/max_filter.csv'))
        self.grouping.clicked.connect(lambda: main.save_csv(
            realization.grouping(frame), 'lab4/set/grouping.csv'))
        self.draw_histogram.clicked.connect(
            lambda: main.draw_histogram(main.histogram(frame, 0)))

        # то, что будет на экране при нажатии кнопки выход из программы
        self.exit.clicked.connect(self.close)

        self.show()

    def add_button(self, name: str) -> QPushButton:
        '''принимает название поля кнопки и ее размеры'''
        button = QPushButton(name, self)
        button.resize(button.sizeHint())
        button.setStyleSheet('background-color: #ee5300;')
        return button


if __name__ == "__main__":
    frame = realization.generate_frame(main.open_new_csv('lab4/set/dataset.csv'), "cat")
    max_width = frame['Width'].max()
    max_height = frame['Height'].max()
    print(frame[["Height", "Width", "Depth"]].describe(), "\n")
    print(realization.grouping(frame), "\n")
    app = QApplication(sys.argv)
    window = Window()
    app.exec()
