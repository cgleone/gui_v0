import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout
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