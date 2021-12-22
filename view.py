import controller

import sys

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar
from PyQt5.QtWidgets import QLineEdit, QGridLayout
from PyQt5.QtWidgets import QPushButton, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFileDialog

class View(QMainWindow):
    """Test App View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()

        # window size and title
        self.setWindowTitle("FYDP GUI Test")
        self.setFixedSize(600,500)
        self._createMainWidget()
        self._createLayout()

        # self._createMenu()
        self._createToolBar()
        # self._createStatusBar()

    def _createMainWidget(self):
        widget = QWidget()
        self.setCentralWidget(widget)

    def _createLayout(self):
        layout = QGridLayout()
        self.centralWidget().setLayout(layout)
        self._populateGrid()

    def _populateGrid(self):
        l = self.centralWidget().layout()

        search = QLineEdit()
        search.setPlaceholderText('Search')
        l.addWidget(search, 0, 0)
        l.addWidget(QPushButton('Go'), 0, 1)

        filters = QComboBox()
        filters.addItems(["--No Filter--", "MRI", "CT", "X-ray", "Ultrasound"])
        l.addWidget(filters, 0, 2)
        l.addWidget(QPushButton('Apply Filters'), 0, 3)

        l.addWidget(QPushButton('Button (1, 0)'), 1, 0)
        l.addWidget(QPushButton('Button (1, 1)'), 1, 1)
        l.addWidget(QPushButton('Button (1, 2)'), 1, 2)
        l.addWidget(QPushButton('Button (2, 0)'), 2, 0)
        l.addWidget(QPushButton('Button (2, 1)'), 2, 1)
        l.addWidget(QPushButton('Button (2, 2)'), 2, 2)
        l.addWidget(QPushButton('Button (3, 0)'), 3, 0)
        l.addWidget(QPushButton('Button (3, 1) + 2 Columns Span'), 3, 1, 1, 2)

        self.import_button = QPushButton("Import File")
        l.addWidget(self.import_button, 4, 1, 1, 2)

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Exit', self.close)

    def _createToolBar(self):
        tools = QToolBar()
        self.addToolBar(tools)
        tools.addAction('Exit', self.close)


    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        self.setStatusBar(status)

    def show_directory(self):
        self.explorer = QFileDialog.getOpenFileName(self, 'Open File', '/Users/cathleenl/Documents/4A')[0]










