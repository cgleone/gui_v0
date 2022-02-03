import os
from time import sleep

from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont

import controller

import sys
# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWebEngineWidgets
#import PyQt5.QtGui.QAbstractItemView.NoEditTriggers


from PyQt5.QtCore import Qt, QStringListModel, QTextStream, QObject, pyqtSignal, QThread, QRect, QPersistentModelIndex, \
    QEvent, QModelIndex

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem



class HomeScreen(QWidget):
    def __init__(self):
        super().__init__()


       # self.setGeometry(100, 100, 100, 100)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("Select Patient")
        self.add_new_patient_button = QPushButton("Add New Patient")
        self.quit_button = QPushButton("Quit")
        h = 100
        w = 400

        self.select_button.setFixedSize(w, h)
        self.add_new_patient_button.setFixedSize(w, h)
        self.quit_button.setFixedSize(w, h)

        font = QFont()
        font.setPointSize(32)
        font.setBold(True)

        self.select_button.setFont(font)
        self.add_new_patient_button.setFont(font)
        self.quit_button.setFont(font)

        # self.select_button.setStyleSheet("background-color: white;"
        #                                  "border-radius: 10px;"
        #                                  "color: grey;")
        # self.add_new_patient_button.setStyleSheet("background-color: white;"
        #                                           "border-radius: 10px;"
        #                                           "color: grey;")
        # self.quit_button.setStyleSheet("background-color: white;"
        #                                "border-radius: 10px;"
        #                                "border-color: red;"
        #                                "border-width: 5px;"
        #                                "color: #708090;"
        #                                "padding: 4px")

        self.layout().addWidget(self.select_button)
        self.layout().addWidget(self.add_new_patient_button)
        self.layout().addWidget(self.quit_button)

        return

