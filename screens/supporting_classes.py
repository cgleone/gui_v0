import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QDialog, QHBoxLayout, QLabel, QCheckBox, QPushButton
import PyQt5.QtWebEngineWidgets

from PyQt5.QtCore import Qt, QObject, pyqtSignal

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__)).rstrip('/screens')

class PDFReport(PyQt5.QtWebEngineWidgets.QWebEngineView):
    def load_pdf(self, filename):
        path = os.path.join(CURRENT_DIR, filename)
        url = PyQt5.QtCore.QUrl.fromLocalFile(path).toString()
        self.settings().setAttribute(
            PyQt5.QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)
        self.settings().setAttribute(
            PyQt5.QtWebEngineWidgets.QWebEngineSettings.PdfViewerEnabled, True)
        self.load(PyQt5.QtCore.QUrl.fromUserInput(url))

    def sizeHint(self):
        return PyQt5.QtCore.QSize(700, 600)


class ReportViewer(QWidget):
    def __init__(self, filename, parent=None):
        super(ReportViewer, self).__init__(parent)

        self.pdf = PDFReport()
        self.pdf.load_pdf(filename)

        lay = QVBoxLayout(self)
        lay.addWidget(self.pdf)


class Worker(QObject):
    finished = pyqtSignal()
    start = pyqtSignal()

    def run(self):
        """Long-running task."""
        self.start.emit()
        self.controller.thread_interior()
        #

        self.finished.emit()

    def set_controller(self, controller):
        self.controller = controller


class WarningDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.yes_button = QPushButton("Yes")
        self.no_button = QPushButton("No")
        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(30)
        self.setWindowFlag(Qt.CustomizeWindowHint)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

    def call_dialog(self, filename, multi_file):
        self.button_layout = QHBoxLayout()
        if multi_file:
            self.message = QLabel("Are you sure you want to delete all of the selected files? \nThis "
                             "action cannot be undone".format(filename))
        else:
            self.message = QLabel("Are you sure you want to delete the file '{}'? \nThis "
                             "action cannot be undone".format(filename))
        self.message.setStyleSheet("font: bold 14px")
        self.message.setAlignment(Qt.AlignCenter)

        self.checkbox = QCheckBox("Don't ask me this again during this session")

        self.button_layout.addWidget(self.no_button, alignment=Qt.AlignCenter)
        self.button_layout.addWidget(self.yes_button, alignment=Qt.AlignCenter)

        self.layout().addWidget(self.message, alignment=Qt.AlignCenter)
        self.layout().addWidget(self.checkbox, alignment=Qt.AlignCenter)
        self.layout().addLayout(self.button_layout)
        self.exec()

    def clear(self):
        self.layout().removeWidget(self.message)
        self.layout().removeWidget(self.checkbox)
        self.layout().removeItem(self.button_layout)