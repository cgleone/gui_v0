

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt



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

