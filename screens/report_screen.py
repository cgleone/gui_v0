
import screens.supporting_classes as helpers
from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont
from PyQt5.QtWidgets import QWidget

from PyQt5.QtCore import Qt, QThread

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem


class ReportScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.patient_name = "Default"
        self.patient_label = QLabel("Patient: Default")

        self.create_buttons()
        self.create_user_inputs()
        self.create_dialog_for_later()
        self.create_layouts()

    def create_layouts(self):
        self.vertical_main = QVBoxLayout()
        self.setLayout(self.vertical_main)

        self.table_grid = QGridLayout()
        self.title_layout = QHBoxLayout()
        self.search_layout = QHBoxLayout()
        self.filters_layout = QHBoxLayout()
        self.settings_layout = QHBoxLayout()

        self.populate_vertical_main()
        self.populate_title_layout()
        self.populate_search_layout()
        self.populate_settings_layout()

    def populate_vertical_main(self):
        self.vertical_main.addLayout(self.title_layout)
        self.vertical_main.addLayout(self.settings_layout)
        self.vertical_main.addLayout(self.search_layout)
        self.vertical_main.addLayout(self.filters_layout)
        self.vertical_main.addLayout(self.table_grid)

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
                try:
                    old_item.setBackground(QBrush(QColor('white')))
                    old_item.setFont(normal)
                    item.setBackground(QBrush(QColor('#E0EEEE')))
                    item.setFont(underlined)
                except:
                    item.setBackground(QBrush(QColor('#E0EEEE')))
                    item.setFont(underlined)
                    print("problem caught")
            col = col + 1
        self.current_hover = [row]

    def populate_report_table(self, report_data):
        if report_data is None:
            return
        for i in range(len(report_data)):
            row_data = report_data[i]
            for j in range(len(row_data)):
                cell_data = row_data[j][0][0]
                self.report_table.setItem(i, j, QTableWidgetItem(cell_data))

    def populate_title_layout(self):
        # self.title_layout.addWidget(QLabel("Patient Portal Demo"))
        self.title_layout.addWidget(self.patient_label, 2)
        self.title_layout.addWidget(self.back_button, 1)
        self.title_layout.addWidget(self.main_menu_button, 1)

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
        self.back_button = QPushButton("Select A Different Patient")
        self.main_menu_button = QPushButton("Back to Main Menu")
        self.clear_filters_button = QPushButton("Clear Active Filters")
        self.dialog_clear_filters_button = QPushButton("Clear Filters")
        self.remove_filter_buttons = QButtonGroup()
        self.settings_button = QPushButton("User Preferences")
        self.apply_settings_button = QPushButton("Apply Settings")

    def create_user_inputs(self):
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText('Search')

    def show_directory(self):
        # self.explorer = QFileDialog.getOpenFileName(self, 'Open File', '/Users/cathleenl/Documents/data_stuff')[0]
        self.explorer = \
        QFileDialog.getOpenFileName(self, 'Open File', '/Users\Wong\PycharmProjects\ocr_testing\pics')[0]

    def create_dialog_for_later(self):
        self.dialog = QDialog()
        self.dialog.setWindowTitle("Select Filters")
        self.dialog_layout = QGridLayout()
        self.populate_dialog()
        self.dialog.setLayout(self.dialog_layout)

    def show_dialog(self):
        self.dialog.exec()

    def set_patient_name(self, name):
        self.patient_name = name
        self.patient_label.setText("Patient: {}".format(self.patient_name))

    def populate_dialog(self):
        self.create_filter_options()
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

        self.dialog_layout.addWidget(self.dialog_clear_filters_button, max_rows + 1, 1)
        self.dialog_layout.addWidget(self.dialog_button, max_rows + 1, 2)

    def close_dialog(self):
        self.dialog.close()

    def create_filter_options(self):

        # modalities
        self.mod_options = {"X-ray": QCheckBox("X-ray"), "MRI": QCheckBox("MRI"), "CT": QCheckBox("CT"),
                            "Ultrasound": QCheckBox("Ultrasound")}
        self.date_options = {"<6mos": QCheckBox("< 6mos"),
                             "6mos-1yr": QCheckBox("6mos - 1yr"), "1yr-5yrs": QCheckBox("1yr - 5yrs"),
                             ">5yrs": QCheckBox("> 5 yrs")}
        self.bodypart_options = {"Head and Neck": QCheckBox("Head and Neck"), "Chest": QCheckBox("Chest"),
                                 "Abdomen": QCheckBox("Abdomen"), "Upper Limbs": QCheckBox("Upper Limbs"),
                                 "Lower Limbs": QCheckBox("Lower Limbs"), "Other": QCheckBox("Other")}

    def create_settings_dialog_for_later(self):
        self.settings_dialog = QDialog()
        self.settings_dialog.setWindowTitle("User Preferences")
        self.settings_dialog_layout = QGridLayout()
        self.populate_settings_dialog()
        self.settings_dialog.setLayout(self.settings_dialog_layout)

    def populate_settings_dialog(self):
        self.settings_dialog_layout.addWidget(QLabel("Select Visible Categories: "), 0, 0)
        self.settings_dialog_layout.addWidget(QLabel("(drag to reorder)"), 1, 0)
        self.settings_dialog_layout.addWidget(self.category_list)
        self.settings_dialog_layout.addWidget(self.apply_settings_button)

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
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                self.category_list.addItem(item)
        self.category_list.setDragDropMode(self.category_list.InternalMove)

    def create_category_options(self):
        self.category_options = ["Exam Date", "File Name", "Imaging Modality", "Body Part",
                                 "Institution", "Clinician", "Notes"]

    def show_settings_dialog(self):
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

        viewer = helpers.ReportViewer(filename)
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
        self.worker = helpers.Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.worker.set_controller(controller)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)

        return self.thread

    def import_enabled(self, enable):
        self.import_button.setEnabled(enable)


