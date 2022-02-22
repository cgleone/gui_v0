import screens.supporting_classes as helpers
from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont, QCloseEvent
from PyQt5.QtWidgets import QWidget

from PyQt5.QtCore import Qt, QThread, QRect, pyqtSignal

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem, QTabWidget, QFrame


class CorrectLabelDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setFixedHeight(700)
        self.setFixedWidth(1200)
        self.filename = None
        self.report_name = None
        self.isPDF = None
        self.setWindowTitle("Correct Report Labels")
        self.dialog_layout = QHBoxLayout()
        self.setLayout(self.dialog_layout)
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.dialog_layout.addLayout(self.left_column)
        self.dialog_layout.addLayout(self.right_column)

        self.title_label = None
        self.current_report = None
        self.tigger_thing = None
        self.button_layout = None

        self.done_button = QPushButton("Apply Changes")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(("color: white; "
                                          "background-color: #8b0000; "
                                          "font: bold 14px;"
                                          "border-style: outset; "
                                          "border-width: 2px; "
                                          "border-radius: 10px;"
                                          "border-color: black; "
                                          "font: bold 14px; "
                                          "min-width: 10em; "
                                          "padding: 6px;"))
        self.done_button.setStyleSheet(("color: white; "
                                        "background-color: green; "
                                        "font: bold 14px;"
                                        "border-style: outset; "
                                        "border-width: 2px; "
                                        "border-radius: 10px;"
                                        "border-color: black; "
                                        "font: bold 14px; "
                                        "min-width: 10em; "
                                        "padding: 6px;"))

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addWidget(self.done_button)


    def report_clicked(self, path, report, isPDF):
        self.title_label = QLabel("Report: {}".format(report))
        self.title_label.setStyleSheet("font: bold 18px")
        self.tigger_thing = self.make_the_tigger_thing()

        if isPDF:
            self.current_report = self.show_pdf(path)
        else:
            self.current_report = self.show_image_report()

        self.left_column.addWidget(self.title_label, alignment=Qt.AlignCenter)
        self.left_column.addWidget(self.current_report)
        self.right_column.addWidget(self.tigger_thing, alignment=Qt.AlignCenter)
        self.right_column.addLayout(self.button_layout)

        self.exec()

    def reset_dialog(self):
        self.left_column.removeWidget(self.title_label)
        self.left_column.removeWidget(self.current_report)
        self.right_column.removeWidget(self.tigger_thing)
        self.right_column.removeItem(self.button_layout)

        self.current_report = None


    def show_pdf(self, path):
        viewer = helpers.ReportViewer(path)
        return viewer

    def show_image_report(self):
        image = QLabel()
        image.setPixmap(QPixmap(self.filename))
        image.setScaledContents(True)
        return image

    def make_the_tigger_thing(self):
        label = QLabel("Hello Tigger, please put\n your awesome label editor here")
        label.setStyleSheet("background-color: #c893fa; color: black; font: bold 14px;")
        label.setFixedWidth(250)
        label.setFixedHeight(200)
        label.setAlignment(Qt.AlignCenter)
        return label

