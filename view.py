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
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem, QTabWidget

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class View(QMainWindow):
    """Test App View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()

        # window size and title
        self.setWindowTitle("EMR Report Viewer")
        self.setFixedSize(1000,700)
        self._createMainWidget()
        self.create_buttons()
        self.create_user_inputs()
        self.create_dialog_for_later()
        # self.create_settings_dialog_for_later()

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
        self.filters_layout = QHBoxLayout()
        self.settings_layout = QHBoxLayout()

        self.populate_vertical_main()
        # self.create_table_grid()
        self.populate_title_layout()
        self.populate_search_layout()
        self.populate_settings_layout()

    def populate_vertical_main(self):
        self.vertical_main.addLayout(self.title_layout)
        self.vertical_main.addLayout(self.settings_layout)
        self.vertical_main.addLayout(self.search_layout)
        self.vertical_main.addLayout(self.filters_layout)
        self.vertical_main.addLayout(self.table_grid)

    def set_table_row_count(self, row_count):
        self.report_table.setRowCount(row_count)

    def create_table_grid(self, current_categories):
        self.report_table = QTableWidget()
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.report_table.setMouseTracking(True)

        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)

        self.create_table_columns(current_categories)

        self.report_table.horizontalHeader().setFont(header_font)
        self.report_table.verticalHeader().setVisible(False)
        self.table_grid.addWidget(self.report_table)

        self.current_hover = [0]
        self.report_table.cellEntered.connect(self.cell_hover)


    def create_table_columns(self, current_categories):
        self.column_count = len(current_categories)
        self.report_table.setColumnCount(self.column_count)
        # self.report_table.setHorizontalHeaderLabels(['Date Added', 'File Name', 'Imaging Modality', 'Body Part', 'Notes'])
        self.report_table.setHorizontalHeaderLabels(current_categories)
        self.report_table.horizontalHeader().setStretchLastSection(True)
        # self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.report_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def cell_hover(self, row):
        underlined = QFont()
        underlined.setUnderline(True)
        normal = QFont()
        self.report_table.setCursor(Qt.PointingHandCursor)

        col = 0
        for i in range(0, self.column_count):
            item = self.report_table.item(row, col)
            old_item = self.report_table.item(self.current_hover[0], col)
            if self.current_hover != [row]:
                old_item.setBackground(QBrush(QColor('white')))
                item.setBackground(QBrush(QColor('#E0EEEE')))
                old_item.setFont(normal)
                item.setFont(underlined)
            col = col + 1
        self.current_hover = [row]

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

    def populate_settings_layout(self):
        self.settings_layout.addStretch(1)
        self.settings_layout.addWidget(self.settings_button)

    def create_buttons(self):
        self.filter_button = QPushButton("Filters")
        self.import_button = QPushButton("Import File")
        self.go_button = QPushButton("Go")
        self.dialog_button = QPushButton("Apply Filters")
        self.logout_button = QPushButton("Logout")
        self.clear_filters_button = QPushButton("Clear Active Filters")
        self.dialog_clear_filters_button = QPushButton("Clear Filters")
        self.remove_filter_buttons = QButtonGroup()
        self.settings_button = QPushButton("User Preferences")
        self.apply_settings_button = QPushButton("Apply Settings")
        self.clear_display_name_group = QButtonGroup()
        self.reset_display_names = QPushButton("Reset All")

    def create_user_inputs(self):
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText('Search')

    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        self.setStatusBar(status)

    def show_directory(self):
        #self.explorer = QFileDialog.getOpenFileName(self, 'Open File', '/Users/cathleenl/Documents/data_stuff')[0]
        self.explorer = QFileDialog.getOpenFileName(self, 'Open File', '/Users\Wong\PycharmProjects\ocr_testing\pics')[0]

    def create_dialog_for_later(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("Select Filters")
        self.dialog_layout = QGridLayout()
        # self.populate_dialog()
        self.dialog.setLayout(self.dialog_layout)

    def show_dialog(self):
        self.dialog.exec()


    def populate_dialog(self, display_names):
        self.create_filter_options(display_names)
        self.dialog_layout.addWidget(QLabel("Imaging Modalities: "), 0, 0)
        self.dialog_layout.addWidget(QLabel("Body Parts: "), 0, 1)
        self.dialog_layout.addWidget(QLabel("Date of Exam: "), 0, 2)

        max_rows = 1
        column = 0
        for category in [self.mod_options, self.bodypart_options, self.date_options]:
            row = 1
            for value in category.values():
                self.dialog_layout.addWidget(value, row, column)
                row = row + 1
                if row > max_rows: max_rows = row
            column = column + 1

        self.dialog_layout.addWidget(self.dialog_clear_filters_button, max_rows+1, 1)
        self.dialog_layout.addWidget(self.dialog_button, max_rows+1, 2)

    def close_dialog(self):
        self.dialog.close()

    def create_filter_options(self, display_names):

        # modalities
        self.mod_options = {"X-ray": QCheckBox(display_names["X-ray"]), "MRI": QCheckBox(display_names["MRI"]),
                            "CT": QCheckBox(display_names["CT"]), "Ultrasound": QCheckBox(display_names["Ultrasound"])}
        self.date_options = {"<6mos": QCheckBox("< 6mos"),
                                 "6mos-1yr": QCheckBox("6mos - 1yr"), "1yr-5yrs": QCheckBox("1yr - 5yrs"),
                             ">5yrs": QCheckBox("> 5 yrs")}
        self.bodypart_options = {"Head and Neck": QCheckBox(display_names["Head and Neck"]),
                                 "Chest": QCheckBox(display_names["Chest"]), "Abdomen": QCheckBox("Abdomen"),
                                 "Upper Limbs": QCheckBox(display_names["Upper Limbs"]),
                                 "Lower Limbs": QCheckBox(display_names["Lower Limbs"]),
                                 "Other": QCheckBox(display_names["Other"])}

    def create_settings_dialog_for_later(self, display_names):
        self.settings_dialog = QDialog()
        self.settings_dialog.setWindowTitle("User Preferences")
        self.settings_dialog.setMinimumSize(500,710)
        self.settings_dialog_layout = QGridLayout()
        self.create_tabs()
        self.populate_table_columns_tab()
        self.populate_display_names_tab(display_names)
        self.settings_dialog_layout.addWidget(self.user_pref_tabs)
        self.settings_dialog_layout.addWidget(self.apply_settings_button)
        self.settings_dialog.setLayout(self.settings_dialog_layout)

    def create_tabs(self):
        self.user_pref_tabs = QTabWidget()
        self.table_colums_tab = QWidget()
        self.display_names_tab = QWidget()
        self.user_pref_tabs.addTab(self.table_colums_tab, "Table Columns")
        self.user_pref_tabs.addTab(self.display_names_tab, "Display Names")

    def populate_table_columns_tab(self):
        self.table_colums_tab.layout = QVBoxLayout()
        self.table_colums_tab.layout.addWidget(QLabel("Select Visible Categories: "))
        self.table_colums_tab.layout.addWidget(QLabel("(drag to reorder)"))
        self.table_colums_tab.layout.addWidget(self.category_list)
        self.table_colums_tab.setLayout(self.table_colums_tab.layout)

    def populate_display_names_tab(self, display_names):
        self.display_names_tab.layout = QGridLayout()
        self.create_display_name_table(display_names)
        self.display_names_tab.layout.addWidget(QLabel("Edit Display Names:"), 1, 1, 1, 1)
        self.display_names_tab.layout.addWidget(self.display_names_table,2,1, 1, 2)
        self.display_names_tab.layout.addWidget(self.reset_display_names, 3, 2)
        self.display_names_tab.setLayout(self.display_names_tab.layout)

    def create_display_name_table(self, display_names):
        self.display_names_table = QTableWidget()
        self.display_names_table.setColumnCount(2)
        self.display_names_table.setRowCount(len(display_names))
        header_font = QFont()
        header_font.setBold(True)
        self.display_names_table.horizontalHeader().setFont(header_font)
        self.display_names_table.setHorizontalHeaderLabels(['Display Name', ''])
        # self.display_names_table.verticalHeader().setVisible(False)
        self.display_names_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.display_names_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.display_names_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_display_names_table(display_names)

    def populate_display_names_table(self, display_names):
        self.display_names_table.setVerticalHeaderLabels(display_names.keys())
        row = 0
        for key in display_names.keys():
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(key)
            if key != display_names[key]:
                line_edit.setText(display_names[key])
            self.display_names_table.setCellWidget(row, 0, line_edit)
            xbutton = QPushButton("X")
            xbutton.setMaximumSize(60,80)
            self.display_names_table.setCellWidget(row, 1, xbutton)
            self.clear_display_name_group.addButton(xbutton, row)
            row = row+1

    #     row_data = [key, display_names[key]]
    #     for j in range(2):
    #         self.display_names_table.setItem(row, j, QTableWidgetItem(row_data[j]))




    def create_categories(self):
        self.create_category_options()
        self.category_list = QListWidget()
        for checkbox in self.category_options:
            item = QListWidgetItem(checkbox)
            item.setFlags(item.flags())
            if checkbox == "Exam Date" or checkbox == "File Name":
                self.category_list.addItem(item)
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked)
            else:
                item.setFlags(item.flags()|Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                self.category_list.addItem(item)
        self.category_list.setDragDropMode(self.category_list.InternalMove)

    def create_category_options(self):
        self.category_options = ["Exam Date", "File Name", "Imaging Modality", "Body Part",
                              "Institution", "Clinician", "Notes"]

    def show_settings_dialog(self):
        self.user_pref_tabs.setCurrentIndex(0)
        self.settings_dialog.show()

    def close_settings_dialog(self):
        self.settings_dialog.close()

    def populate_filters_layout(self, active_filters):
        self.filters_layout.addWidget(QLabel("Active Filters: "))
        for i in range(len(active_filters)):
            button = QPushButton(active_filters[i])
            self.remove_filter_buttons.addButton(button, i)
            self.filters_layout.addWidget(button)
        self.filters_layout.addWidget(self.clear_filters_button)

    def display_pdf(self, filename, report_name, row, col):
        item = self.report_table.item(row, col)
        item.setBackground(QBrush(QColor('white')))

        viewer = ReportViewer(filename)
        try:
            viewer.show()
        except:
            viewer.show()

        dialog = QDialog()
        dialog.setWindowTitle(report_name)
        dialog_layout = QGridLayout()
        dialog.setLayout(dialog_layout)
        dialog_layout.addWidget(viewer)
        try:
            dialog.exec()
        except:
            dialog.exec()

    def display_image_report(self, filename, report_name):
        dialog = QDialog()
        dialog.setWindowTitle(report_name)
        dialog_layout = QGridLayout()
        dialog.setLayout(dialog_layout)
        dialog.setFixedSize(800, 700)

        image = QLabel()
        image.setPixmap(QPixmap(filename))
        image.setScaledContents(True)
        dialog_layout.addWidget(image)

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


class TableWidget(QTableWidget):
    cellExited = pyqtSignal(int, int)
    itemExited = pyqtSignal(QTableWidgetItem)

    def __init__(self, rows, columns, parent=None):
        QTableWidget.__init__(self, rows, columns, parent)
        self._last_index = QPersistentModelIndex()
        self.viewport().installEventFilter(self)

    def eventFilter(self, widget, event):
        if widget is self.viewport():
            index = self._last_index
            if event.type() == QEvent.MouseMove:
                index = self.indexAt(event.pos())
            elif event.type() == QEvent.Leave:
                index = QModelIndex()
            if index != self._last_index:
                row = self._last_index.row()
                column = self._last_index.column()
                item = self.item(row, column)
                if item is not None:
                    self.itemExited.emit(item)
                self.cellExited.emit(row, column)
                self._last_index = QPersistentModelIndex(index)
        return QTableWidget.eventFilter(self, widget, event)