import os

from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont

from screens.patient_select import PatientSelectScreen
from screens.home import HomeScreen
from screens.report_screen import ReportScreen

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWebEngineWidgets
#import PyQt5.QtGui.QAbstractItemView.NoEditTriggers


from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QPersistentModelIndex, \
    QEvent, QModelIndex

from PyQt5.QtWidgets import QLabel, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QListWidget, QListWidgetItem, QStackedWidget, QTabWidget


class View(QMainWindow):
    """Test App View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()

        self.home = HomeScreen()
        self.patient_select = PatientSelectScreen()
        self.report_screen = ReportScreen()
        # window size and title
        self.setWindowTitle("EMR Report Viewer")
        self.setFixedSize(1000,700)

        # self.create_settings_dialog_for_later()

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.home)
        self.stacked_widget.addWidget(self.patient_select)
        self.stacked_widget.addWidget(self.report_screen)
        self.setCentralWidget(self.stacked_widget)

        self.go_to_home()

    def go_to_home(self):
        self.stacked_widget.setCurrentWidget(self.home)

    def go_to_patient_select(self):
        self.stacked_widget.setCurrentWidget(self.patient_select)

    def go_to_report_screen(self):
        self.stacked_widget.setCurrentWidget(self.report_screen)

    def set_table_row_count(self, row_count, table):
        table.setRowCount(row_count)