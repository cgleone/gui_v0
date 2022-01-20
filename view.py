import os
from time import sleep

import controller

import sys
# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
import PyQt5.QtWebEngineWidgets
#import PyQt5.QtGui.QAbstractItemView.NoEditTriggers


from PyQt5.QtCore import Qt, QStringListModel, QTextStream, QObject, pyqtSignal, QThread

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class View(QMainWindow):
    """Test App View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()

        # window size and title
        self.setWindowTitle("FYDP GUI Test")
        self.setFixedSize(1000,700)
        self._createMainWidget()
        self.create_buttons()
        self.create_user_inputs()

        self._createLayouts()

        # self._createMenu()
        # self._createStatusBar()

    def _createMainWidget(self):
        widget = QWidget()
        self.setCentralWidget(widget)

    def _createLayouts(self):
        self.vertical_main = QVBoxLayout()
        self.centralWidget().setLayout(self.vertical_main)

        self.table_grid = QGridLayout()
        self.title_layout = QHBoxLayout()
        self.search_layout = QHBoxLayout()

        self.populate_vertical_main()
        self.create_table_grid()
        self.populate_title_layout()
        self.populate_search_layout()

    def populate_vertical_main(self):
        self.vertical_main.addLayout(self.title_layout)
        self.vertical_main.addLayout(self.search_layout)
        self.vertical_main.addLayout(self.table_grid)

    def set_table_row_count(self, row_count):
        self.report_table.setRowCount(row_count)

    def create_table_grid(self):
        self.report_table = QTableWidget()
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.report_table.setColumnCount(5)

        self.report_table.setHorizontalHeaderLabels(['Date Added', 'File Name', 'Imaging Modality', 'Body Part', 'Notes'])
        self.report_table.horizontalHeader().setStretchLastSection(True)
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.report_table.verticalHeader().setVisible(False)
        self.table_grid.addWidget(self.report_table)


    def populate_report_table(self, report_data):
        if report_data is None:
            return
        for i in range(len(report_data)):
            row_data = report_data[i]
            for j in range(len(row_data)):
                cell_data = row_data[j][0][0]

                # if j == 1:
                #     self.report_table.setItem(i, j, custom_widgets.ClickableLabel(cell_data, i, click_event))
                # else:
                self.report_table.setItem(i, j, QTableWidgetItem(cell_data))



    def populate_title_layout(self):
       # self.title_layout.addWidget(QLabel("Patient Portal Demo"))
        self.title_layout.addWidget(QLabel("Patient: Cathleen Leone, 22F, 04/04/1999"), 3)
        self.title_layout.addWidget(self.logout_button, 1)

    def populate_search_layout(self):
        self.search_layout.addWidget(self.search_bar)
        self.search_layout.addWidget(self.go_button)
        self.search_layout.addWidget(self.filter_button)
        self.search_layout.addWidget(self.import_button)

    def create_buttons(self):
        self.filter_button = QPushButton("Filters")
        self.import_button = QPushButton("Import File")
        self.go_button = QPushButton("Go")
        self.dialog_button = QPushButton("Apply Filters")
        self.logout_button = QPushButton("Logout")

    def create_user_inputs(self):
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText('Search')

    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        self.setStatusBar(status)

    def show_directory(self):
        self.explorer = QFileDialog.getOpenFileName(self, 'Open File', '/Users/cathleenl/Documents/data_stuff')[0]

    def show_dialog(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("Select Filters")
        self.dialog_layout = QGridLayout()
        self.populate_dialog()
        self.dialog.setLayout(self.dialog_layout)
        self.dialog.exec()


    def populate_dialog(self):
        self.create_filter_options()
        self.dialog_layout.addWidget(QLabel("Imaging Modalities: "), 0, 0)
        self.dialog_layout.addWidget(QLabel("Body Parts: "), 0, 1)
        self.dialog_layout.addWidget(QLabel("Facilities/Institutions: "), 0, 2)

        max_rows = 1
        column = 0
        for category in [self.mod_options, self.bodypart_options, self.hospital_options]:
            row = 1
            for value in category.values():
                self.dialog_layout.addWidget(value, row, column)
                row = row + 1
                if row > max_rows: max_rows = row
            column = column + 1

        self.dialog_layout.addWidget(self.dialog_button, max_rows+1, 2)

    def close_dialog(self):
        self.dialog.close()

    def create_filter_options(self):

        # modalities
        self.mod_options = {"X-ray": QCheckBox("X-ray"), "MRI": QCheckBox("MRI"), "CT": QCheckBox("CT"),
                            "Ultrasound": QCheckBox("Ultrasound")}
        self.hospital_options = {"St. Mary's Hospital": QCheckBox("St. Mary's Hospital"),
                                 "Grand River Hospital": QCheckBox("Grand River Hospital")}
        self.bodypart_options = {"Head and Neck": QCheckBox("Head and Neck"), "Chest": QCheckBox("Chest"),
                                 "Abdomen": QCheckBox("Abdomen"), "Upper Limbs": QCheckBox("Upper Limbs"),
                                 "Lower Limbs": QCheckBox("Lower Limbs"), "Other": QCheckBox("Other")}

    def display_pdf(self, filename, report_name):

        viewer = ReportViewer(filename)
        viewer.show()

        dialog = QDialog()
        dialog.setWindowTitle(report_name)
        dialog_layout = QGridLayout()
        dialog.setLayout(dialog_layout)
        dialog_layout.addWidget(viewer)
        dialog.exec()

    def create_thread(self, controller):
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.worker.set_controller(controller)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.hello_testing)
        # self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        #self.thread.start()
        return self.thread

    def import_enabled(self, enable):
        self.import_button.setEnabled(enable)



class PDFReport(PyQt5.QtWebEngineWidgets.QWebEngineView):
    def load_pdf(self, filename):
        path = os.path.join(CURRENT_DIR, filename)
        url = PyQt5.QtCore.QUrl.fromLocalFile(path).toString()
        print(url)
        self.settings().setAttribute(
            PyQt5.QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)
        self.settings().setAttribute(
            PyQt5.QtWebEngineWidgets.QWebEngineSettings.PdfViewerEnabled, True)
        self.load(PyQt5.QtCore.QUrl.fromUserInput(url))

        #self.load(PyQt5.QtCore.QUrl.fromUserInput("%s?file=%s" % (PDFJS, url)))
        # self.load(PyQt5.QtCore.QUrl.fromUserInput(f'{PDFJS}?file={url}'))


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

