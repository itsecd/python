import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon, QPixmap, QAction
from PyQt6.QtCore import Qt, QCoreApplication, QEvent
from PyQt6.QtWidgets import QDesktopWidget
sys.path.insert(1, "Lab2")
from iterator import DirectoryIterator
from create_annotation import get_absolute_paths, get_relative_paths, write_annotation_to_csv

class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.initUI()
        self.initIterators()
        self.createActions()
        self.createMenuBar()
        self.createToolBar()

    def initUI(self) -> None:
        self.center()
        self.setWindowTitle('Cats & Dogs')
        self.setWindowIcon(QIcon('img/main_icon.png'))
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        cat_btn = QPushButton('Next Cat', self)
        dog_btn = QPushButton('Next Dog', self)

        pixmap = QPixmap('img/both.jpg')
        self.lbl = QLabel(self)
        self.lbl.setPixmap(pixmap)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hbox = QHBoxLayout()
        hbox.addSpacing(1)
        hbox.addWidget(cat_btn)
        hbox.addWidget(dog_btn)

        vbox = QVBoxLayout()
        vbox.addSpacing(1)
        vbox.addWidget(self.lbl)
        vbox.addLayout(hbox)

        self.centralWidget.setLayout(vbox)

        cat_btn.clicked.connect(self.nextCat)
        dog_btn.clicked.connect(self.nextDog)

        self.folderpath = ' '

        self.showMaximized()

    def initIterators(self) -> None:
        self.cats = DirectoryIterator('cat', 'dataset')
        self.dogs = DirectoryIterator('dog', 'dataset')

    def nextCat(self) -> None:
        lbl_size = self.lbl.size()
        next_image = next(self.cats)
        if next_image is not None:
            img = QPixmap(next_image).scaled(
                lbl_size, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl.setPixmap(img)
            self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.initIterators()
            self.nextCat()

    def nextDog(self) -> None:
        lbl_size = self.lbl.size()
        next_image = next(self.dogs)
        if next_image is not None:
            img = QPixmap(next_image).scaled(
                lbl_size, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl.setPixmap(img)
            self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            self.initIterators()
            self.nextDog()

    def center(self) -> None:
        widget_rect = self.frameGeometry()
        pc_rect = QDesktopWidget().availableGeometry().center()
        widget_rect.moveCenter(pc_rect)
        self.move(widget_rect.center())

    def createMenuBar(self) -> None:
        menuBar = self.menuBar()

        self.fileMenu = menuBar.addMenu('&File')
        self.fileMenu.addAction(self.exitAction)
        self.fileMenu.addAction(self.changeAction)

        self.annotMenu = menuBar.addMenu('&Annotation')
        self.annotMenu.addAction(self.createAnnotAction)

        self.dataMenu = menuBar.addMenu('&Dataset')
        self.dataMenu.addAction(self.createData2Action)

    def createToolBar(self) -> None:
        fileToolBar = self.addToolBar('File')
        fileToolBar.addAction(self.exitAction)

        annotToolBar = self.addToolBar('Annotation')
        annotToolBar.addAction(self.createAnnotAction)

    def createActions(self) -> None:
        self.exitAction = QAction(QIcon('img/exit.png'), '&Exit')
        self.exitAction.triggered.connect(QCoreApplication.instance().quit)

        self.changeAction = QAction(QIcon('img/change.png'), '&Change dataset')
        self.changeAction.triggered.connect(self.changeDataset)

        self.createAnnotAction = QAction(
            QIcon('img/csv.png'), '&Create annotation for current dataset')
        self.createAnnotAction.triggered.connect(self.createAnnotation)

        self.createData2Action = QAction(
            QIcon('img/new_dataset.png'), '&Create dataset2')
        self.createData2Action.triggered.connect(self.createDataset2)

        self.createData3Action = QAction(
            QIcon('img/new_dataset.png'), '&Create dataset3')
        self.createData3Action.triggered.connect(self.createDataset3)

    def createAnnotation(self) -> None:
        if 'dataset2' in str(self.folderpath):
            write_annotation_to_csv('dataset2/data.csv', get_absolute_paths('cat', 'dataset2'),
                                    get_relative_paths('cat', 'dataset2'), 'cat')
            write_annotation_to_csv('dataset2/data.csv', get_absolute_paths('dog', 'dataset2'),
                                    get_relative_paths('dog', 'dataset2'), 'dog')
        elif 'dataset3' in str(self.folderpath):
            write_annotation_to_csv('dataset3/data.csv', get_absolute_paths('cat', 'dataset3'),
                                    get_relative_paths('cat', 'dataset3'), 'cat')
            write_annotation_to_csv('dataset3/data.csv', get_absolute_paths('dog', 'dataset3'),
                                    get_relative_paths('dog', 'dataset3'), 'dog')
        elif 'dataset' in str(self.folderpath):
            write_annotation_to_csv('dataset/data.csv', get_absolute_paths('cat', 'dataset'),
                                    get_relative_paths('cat', 'dataset'), 'cat')
            write_annotation_to_csv('dataset/data.csv', get_absolute_paths('dog', 'dataset'),
                                    get_relative_paths('dog', 'dataset'), 'dog')

    def createDataset2(self) -> None:
        write_annotation_to_csv('dataset2/data.csv', get_absolute_paths('cat', 'dataset2'),
                                get_relative_paths('cat', 'dataset2'), 'cat')
        write_annotation_to_csv('dataset2/data.csv', get_absolute_paths('dog', 'dataset2'),
                                get_relative_paths('dog', 'dataset2'), 'dog')

        self.dataMenu.addAction(self.createData3Action)

    def createDataset3(self) -> None:
        write_annotation_to_csv('dataset3/data.csv', get_absolute_paths('cat', 'dataset3'),
                                get_relative_paths('cat', 'dataset3'), 'cat')
        write_annotation_to_csv('dataset3/data.csv', get_absolute_paths('dog', 'dataset3'),
                                get_relative_paths('dog', 'dataset3'), 'dog')

    def changeDataset(self) -> None:
        reply = QMessageBox.question(self, 'Warning', f'Are you sure you want to change current dataset?\nCurrent dataset: {str(self.folderpath)}',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.folderpath = QFileDialog.getExistingDirectory(
                self, 'Select Folder')
        else:
            pass

    def closeEvent(self, event: QEvent) -> None:
        reply = QMessageBox.question(self, 'Warning', 'Are you sure to quit?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
