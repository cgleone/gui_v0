import math
import os

import screens.supporting_classes as helpers
from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont, QIcon
from PyQt5.QtWidgets import QWidget, QScrollArea, QGroupBox
from PIL import Image
from PyQt5.QtCore import Qt, QThread, QRect, QSize, QModelIndex

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem, QTabWidget, QFrame


class ReportScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.patient_name = "Default"
        self.patient_label = QLabel("Patient: Default")
        self.no_results = QLabel("No reports to show")
        self.correction_instructions = QLabel("Click on a report to edit its labels")
        self.correction_instructions.setStyleSheet("color: #8b0000; font: bold 18px;")
        self.no_results.setHidden(True)

        self.grey_widget = QLabel(" ")
        self.grey_widget.setStyleSheet("background-color: rgba(0, 0, 0, 10);")
        self.grey_widget.setFixedHeight(150)
        self.grey_widget.setFixedWidth(1000)

        self.create_buttons()
        self.create_user_inputs()
        self.create_dialog_for_later()
        self.create_layouts()
        self.cell_hover_colour = '#E0EEEE'

        self.thread_is_running = False
        self.in_select_mode = False
        self.dont_ask_again = False
        self.warning_dialog = helpers.WarningDialog()
        self.select_file_boxes = []

    def create_layouts(self):
        self.vertical_main = QVBoxLayout()
        self.setLayout(self.vertical_main)

        self.table_grid = QGridLayout()
        self.title_layout = QHBoxLayout()
        self.search_layout = QHBoxLayout()
        self.filters_layout = QHBoxLayout()
        self.settings_layout = QHBoxLayout()
        self.label_correction_layout = QHBoxLayout()
        self.label_correction_frame = QFrame()
        self.label_correction_frame.setLayout(self.label_correction_layout)

        self.populate_vertical_main()
        self.populate_title_layout()
        self.populate_search_layout()
        self.populate_settings_layout()
        self.populate_label_correction_layout()

        self.empty_grey_widgets = [QWidget(), QWidget(), QWidget(), QWidget()]
        for widget in self.empty_grey_widgets:
            widget.setFixedHeight(30)


    def populate_vertical_main(self):
        self.vertical_main.addLayout(self.title_layout)
        self.vertical_main.addLayout(self.settings_layout)
        self.vertical_main.addLayout(self.search_layout)
        self.vertical_main.addLayout(self.filters_layout)
        self.vertical_main.addWidget(self.no_results)
        self.vertical_main.addLayout(self.table_grid)
        self.vertical_main.addWidget(self.label_correction_frame)

    def create_table_grid(self, current_categories):
        self.report_table = QTableWidget()
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.report_table.setSelectionMode(QAbstractItemView.NoSelection)
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
        self.column_count = len(current_categories) + 1 # add one for the garbage cans
        self.report_table.setColumnCount(self.column_count)
        self.report_table.setHorizontalHeaderLabels(current_categories + [''])

        for i in range(len(current_categories)):
            column_header = current_categories[i]
            if column_header != "Notes" and column_header != "Institution" and column_header != "File Name":
                self.report_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
            elif column_header == "Notes":
                self.report_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Fixed)
                self.report_table.horizontalHeader().resizeSection(i, 250)
            elif column_header == "Institution":
                self.report_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Fixed)
                self.report_table.horizontalHeader().resizeSection(i, 250)
            else: # file name
                self.report_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        self.report_table.horizontalHeader().setSectionResizeMode(self.column_count-1, QHeaderView.Fixed)
        self.report_table.horizontalHeader().resizeSection(self.column_count-1, 40)
        self.report_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.report_table.horizontalHeader().resizeSection(0, 100)


    def cell_hover(self, row):

        underlined = QFont()
        underlined.setUnderline(True)
        normal = QFont()
        if not self.in_select_mode:
            self.report_table.setCursor(Qt.PointingHandCursor)
            columns_for_highlighting = self.column_count-1
        else:
            self.report_table.setCursor(Qt.ArrowCursor)
            columns_for_highlighting = self.column_count
            self.cell_hover_colour = '#c7c7c7'

        col = 0
        for i in range(0, columns_for_highlighting):
            item = self.report_table.item(row, col)
            old_item = self.report_table.item(self.current_hover[0], col)
            if self.current_hover != [row]:
                try:
                    old_item.setBackground(QBrush(QColor('white')))
                    old_item.setFont(normal)
                    item.setBackground(QBrush(QColor(self.cell_hover_colour)))
                    if not self.in_select_mode:
                        item.setFont(underlined)
                except:
                    item.setBackground(QBrush(QColor(self.cell_hover_colour)))
                    if not self.in_select_mode:
                        item.setFont(underlined)
                    print("problem caught")
            col = col + 1
        self.current_hover = [row]

    def clear_garbage_and_checks(self):
        self.select_file_boxes.clear()
        for button in self.garbage_can_buttons.buttons():
            self.garbage_can_buttons.removeButton(button)

    def populate_report_table(self, report_data):
        self.clear_garbage_and_checks()
        if report_data is None:
            return
        for i in range(len(report_data)):
            row_data = report_data[i]
            for j in range(len(row_data)):
                cell_data = row_data[j][0][0]
                self.report_table.setItem(i, j, QTableWidgetItem(cell_data))

            if self.in_select_mode:
                layout = QHBoxLayout()
                widget = QWidget()
                widget.setLayout(layout)
                checkbox = QCheckBox()
                layout.addWidget(checkbox)
                self.report_table.setCellWidget(i, len(row_data), widget)
                self.select_file_boxes.append(checkbox)

            else:
                garbage_can = QIcon(os.path.join('icons', 'delete.png'))
                del_button = QPushButton()
                del_button.setIcon(garbage_can)
                del_button.setStyleSheet("background-color: #bfbfbf; border-color: white; padding: 1px")
                self.report_table.setCellWidget(i, len(row_data), del_button)
                self.garbage_can_buttons.addButton(del_button, i)

    def populate_last_column(self):
        self.clear_garbage_and_checks()
        row_count = self.report_table.rowCount()
        column_count = self.report_table.columnCount()
        for i in range(row_count):
            # self.report_table.cellWidget(i, row_count).hide()
            # self.report_table.removeCellWidget(i, row_count)
            self.report_table.setItem(i, column_count-1, QTableWidgetItem(''))

            if self.in_select_mode:
                layout = QHBoxLayout()
                widget = QWidget()
                widget.setLayout(layout)
                checkbox = QCheckBox()
                layout.addWidget(checkbox)
                # checkbox.setStyleSheet("QCheckBox::indicator {border: 1px solid gray}")
                self.report_table.setCellWidget(i, column_count-1, widget)
                self.select_file_boxes.append(checkbox)

            else:
                garbage_can = QIcon(os.path.join('icons', 'delete.png'))
                del_button = QPushButton()
                del_button.setIcon(garbage_can)
                del_button.setStyleSheet("background-color: #bfbfbf; border-color: white; padding: 1px")
                self.report_table.setCellWidget(i, column_count-1, del_button)
                self.garbage_can_buttons.addButton(del_button, i)



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

    def populate_label_correction_layout(self):
        self.label_correction_layout.addWidget(self.multi_file_select_button, 0, alignment=Qt.AlignLeft)
        self.label_correction_layout.addWidget(self.label_correction_button, 1, alignment=Qt.AlignRight)

    def enter_label_correction_mode(self):
        self.label_correction_layout.removeWidget(self.label_correction_button)
        self.label_correction_layout.removeWidget(self.multi_file_select_button)
        self.label_correction_button.hide()
        self.label_correction_layout.addWidget(self.correction_instructions, 0, alignment=Qt.AlignLeft)
        self.label_correction_layout.addWidget(self.done_correction_button, 1, alignment=Qt.AlignRight)
        self.done_correction_button.show()
        self.correction_instructions.show()
        self.cell_hover_colour = '#fc9992'
        self.label_correction_frame.setStyleSheet("background-color: white")
        self.enable_actions(False)

    def enter_file_deletion_mode(self):
        self.label_correction_layout.removeWidget(self.label_correction_button)
        self.label_correction_layout.removeWidget(self.multi_file_select_button)
        self.label_correction_button.hide()
        self.multi_file_select_button.hide()
        self.label_correction_layout.addWidget(self.cancel_deletion_button, 0, alignment=Qt.AlignLeft)
        self.label_correction_layout.addWidget(self.delete_selected_button, 1, alignment=Qt.AlignRight)
        self.cancel_deletion_button.show()
        self.delete_selected_button.show()
        self.cell_hover_colour = '#fc9992'
        self.label_correction_frame.setStyleSheet("background-color: white")
        # for garbage_can in self.garbage_can_buttons.buttons():
        #     garbage_can.setEnabled(False)
        self.in_select_mode = True
        self.populate_last_column()
        self.update_delete_enabled_status()
        self.enable_actions(False)

    def update_delete_enabled_status(self):
        if any(box.isChecked() for box in self.select_file_boxes):
            self.delete_selected_button.setEnabled(True)
            self.delete_selected_button.setStyleSheet("color: white; "
                                                      "background-color: green; "
                                                      "font: bold 14px;"
                                                      "border-style: outset; "
                                                      "border-width: 2px; "
                                                      "border-radius: 10px;"
                                                      "border-color: black; "
                                                      "font: bold 14px; "
                                                      "min-width: 10em; "
                                                      "padding: 6px;")
        else:
            self.delete_selected_button.setEnabled(False)
            self.delete_selected_button.setStyleSheet("color: black; "
                                                      "background-color: #7d9479; "
                                                      "font: bold 14px;"
                                                      "border-style: outset; "
                                                      "border-width: 2px; "
                                                      "border-radius: 10px;"
                                                      "border-color: black; "
                                                      "font: bold 14px; "
                                                      "min-width: 10em; "
                                                      "padding: 6px;")

    def exit_file_deletion_mode(self):
        self.label_correction_layout.removeWidget(self.cancel_deletion_button)
        self.label_correction_layout.removeWidget(self.delete_selected_button)
        self.cancel_deletion_button.hide()
        self.delete_selected_button.hide()
        self.label_correction_layout.addWidget(self.multi_file_select_button, 0, alignment=Qt.AlignLeft)
        self.label_correction_layout.addWidget(self.label_correction_button, 1, alignment=Qt.AlignRight)
        self.label_correction_button.show()
        self.multi_file_select_button.show()
        self.cell_hover_colour = '#E0EEEE'
        self.report_table.setStyleSheet("")
        self.label_correction_frame.setStyleSheet("")
        for garbage_can in self.garbage_can_buttons.buttons():
            garbage_can.setEnabled(True)
        self.in_select_mode = False
        self.populate_last_column()
        self.enable_actions(True)

    def get_selected_file_indices(self):
        indices = []
        for i in range(len(self.select_file_boxes)):
            if self.select_file_boxes[i].isChecked():
                indices.append(i)
        return indices


    def exit_label_correction_mode(self):
        self.label_correction_layout.removeWidget(self.correction_instructions)
        self.label_correction_layout.removeWidget(self.done_correction_button)
        self.correction_instructions.hide()
        self.done_correction_button.hide()
        self.label_correction_layout.addWidget(self.multi_file_select_button, 0, alignment=Qt.AlignLeft)
        self.label_correction_layout.addWidget(self.label_correction_button, 1, alignment=Qt.AlignRight)
        self.label_correction_button.show()
        self.cell_hover_colour = '#E0EEEE'
        self.report_table.setStyleSheet("")
        self.label_correction_frame.setStyleSheet("")
        self.enable_actions(True)

    def enable_actions(self, enable):
        for layout in [self.title_layout, self.settings_layout, self.filters_layout, self.search_layout]:
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                try:
                    widget.setEnabled(enable)
                except:
                    random_nonsense = True

        if self.thread_is_running:
            self.import_enabled(False)





    def create_buttons(self):
        self.filter_button = QPushButton("Quick Search")
        self.import_button = QPushButton("Import File")
        self.go_button = QPushButton("Go")
        self.dialog_button = QPushButton("Apply Filters")
        self.back_button = QPushButton("Select A Different Patient")
        self.main_menu_button = QPushButton("Back to Main Menu")
        self.clear_filters_button = QPushButton("Clear Search")
        self.dialog_clear_filters_button = QPushButton("Clear Filters")
        self.remove_filter_buttons = QButtonGroup()
        self.settings_button = QPushButton("User Preferences")
        self.apply_settings_button = QPushButton("Apply Settings")
        self.clear_display_name_group = QButtonGroup()
        self.clear_modality_display_group = QButtonGroup()
        self.clear_bodypart_display_group = QButtonGroup()
        self.clear_institution_display_group = QButtonGroup()
        self.garbage_can_buttons = QButtonGroup()
        self.reset_display_names = QPushButton("Reset All")
        self.label_correction_button = QPushButton("Enter Label Correction Mode")
        self.label_correction_button.setStyleSheet("color: white; "
                                                   "background-color: #8b0000; "
                                                   "font: bold 14px;"
                                                   "border-style: outset; "
                                                   "border-width: 2px; "
                                                   "border-radius: 10px;"
                                                   "border-color: black; "
                                                   "font: bold 14px; "
                                                   "min-width: 10em; "
                                                   "padding: 6px;")
        self.multi_file_select_button = QPushButton("Select Files for Deletion")
        self.multi_file_select_button.setStyleSheet("color: white; "
                                                   "background-color: #8b0000; "
                                                   "font: bold 14px;"
                                                   "border-style: outset; "
                                                   "border-width: 2px; "
                                                   "border-radius: 10px;"
                                                   "border-color: black; "
                                                   "font: bold 14px; "
                                                   "min-width: 10em; "
                                                   "padding: 6px;")

        self.done_correction_button = QPushButton("Done")
        self.done_correction_button.setStyleSheet("color: white; "
                                                  "background-color: green; "
                                                  "font: bold 14px;"
                                                  "border-style: outset; "
                                                  "border-width: 2px; "
                                                  "border-radius: 10px;"
                                                  "border-color: black; "
                                                  "font: bold 14px; "
                                                  "min-width: 10em; "
                                                  "padding: 6px;")

        self.delete_selected_button = QPushButton("Delete Selected Files")
        self.delete_selected_button.setStyleSheet("color: white; "
                                                  "background-color: green; "
                                                  "font: bold 14px;"
                                                  "border-style: outset; "
                                                  "border-width: 2px; "
                                                  "border-radius: 10px;"
                                                  "border-color: black; "
                                                  "font: bold 14px; "
                                                  "min-width: 10em; "
                                                  "padding: 6px;")

        self.cancel_deletion_button = QPushButton("Cancel")
        self.cancel_deletion_button.setStyleSheet("color: white; "
                                                   "background-color: #8b0000; "
                                                   "font: bold 14px;"
                                                   "border-style: outset; "
                                                   "border-width: 2px; "
                                                   "border-radius: 10px;"
                                                   "border-color: black; "
                                                   "font: bold 14px; "
                                                   "min-width: 10em; "
                                                   "padding: 6px;")


        self.all_buttons = [self.filter_button, self.go_button, self.settings_button, self.reset_display_names,
                            self.dialog_button, self.back_button, self.main_menu_button, self.import_button]

    def set_table_color(self):
        self.report_table.setStyleSheet("")

        for row in range(self.report_table.rowCount()):
            for col in range(0, self.column_count - 1):
                item = self.report_table.item(row, col)
                item.setBackground(QBrush(QColor('white')))
        self.report_table.horizontalHeader().setStyleSheet("background-color: white")
        if not self.in_select_mode:
            self.report_table.setStyleSheet("border-style: outset; "
                                                      "border-width: 2px; "
                                                      "border-radius: 1px;"
                                                      "border-color: black;")


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
        # self.populate_dialog()
        self.dialog.setLayout(self.dialog_layout)

    def show_dialog(self):
        self.dialog.exec()

    def set_patient_name(self, name):
        self.patient_name = name
        self.patient_label.setText("Patient: {}".format(self.patient_name))

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
                                 "Chest": QCheckBox(display_names["Chest"]),
                                 "Abdomen": QCheckBox(display_names["Abdomen"]),
                                 "Upper Limbs": QCheckBox(display_names["Upper Limbs"]),
                                 "Lower Limbs": QCheckBox(display_names["Lower Limbs"]),
                                 "Other": QCheckBox(display_names["Other"])}

    def create_settings_dialog_for_later(self):
        self.settings_dialog = QDialog()
        self.settings_dialog.setWindowTitle("User Preferences")
        self.settings_dialog.setMinimumSize(500,500)
        self.settings_dialog_layout = QGridLayout()
        self.create_tabs()
        self.settings_dialog_layout.addWidget(self.user_pref_tabs)
        self.settings_dialog_layout.addWidget(self.apply_settings_button)
        self.settings_dialog.setLayout(self.settings_dialog_layout)

    def create_tabs(self):
        self.user_pref_tabs = QTabWidget()
        self.table_colums_tab = QWidget()
        self.display_names_tab = QTabWidget()
        self.user_pref_tabs.addTab(self.table_colums_tab, "Table Columns")
        self.user_pref_tabs.addTab(self.display_names_tab, "Display Names")

    def populate_table_columns_tab(self):
        self.table_colums_tab.layout = QVBoxLayout()
        self.table_colums_tab.layout.addWidget(QLabel("Select Visible Categories: "))
        self.table_colums_tab.layout.addWidget(QLabel("(drag to reorder)"))
        self.table_colums_tab.layout.addWidget(self.category_list)
        self.table_colums_tab.setLayout(self.table_colums_tab.layout)

    def create_display_name_tables(self, modalities, bodyparts, institutions):
        self.modality_display_table = QTableWidget()
        self.create_display_name_table(self.modality_display_table, modalities, self.clear_modality_display_group)
        self.bodypart_display_table = QTableWidget()
        self.create_display_name_table(self.bodypart_display_table, bodyparts, self.clear_bodypart_display_group)
        self.institutions_display_table = QTableWidget()
        self.create_display_name_table(self.institutions_display_table, institutions,
                                       self.clear_institution_display_group)

    def populate_display_names_tabular(self):
        self.display_names_tab.layout = QGridLayout()
        self.display_names_tab.layout.addWidget(QLabel("Edit Modality Display Names:"), 1, 1, 1, 1)
        self.display_names_tab.layout.addWidget(self.modality_display_table,2,1, 1, 2)
        self.display_names_tab.layout.addWidget(QLabel("Edit Body Part Display Names:"), 3, 1, 1, 1)
        self.display_names_tab.layout.addWidget(self.bodypart_display_table, 4, 1, 1, 2)
        self.display_names_tab.layout.addWidget(QLabel("Edit Institutions Display Names:"), 5, 1, 1, 1)
        self.display_names_tab.layout.addWidget(self.institutions_display_table, 6, 1, 1, 2)
        self.display_names_tab.layout.addWidget(self.reset_display_names, 7, 2)
        self.display_names_tab.setLayout(self.display_names_tab.layout)

    def create_displaynames_tabs(self):
        self.modality_tab = QWidget()
        self.bodyparts_tab = QWidget()
        self.institutions_tab = QWidget()
        self.display_names_tab.addTab(self.modality_tab, "Modalities")
        self.display_names_tab.addTab(self.bodyparts_tab, "Body Parts")
        self.display_names_tab.addTab(self.institutions_tab, "Institutions")

    def populate_display_names_tabs(self):
        self.populate_display_names_tab(self.modality_tab, self.modality_display_table)
        self.populate_display_names_tab(self.bodyparts_tab, self.bodypart_display_table)
        self.populate_display_names_tab(self.institutions_tab, self.institutions_display_table)

    def populate_display_names_tab(self, tab, table):
        tab.layout = QVBoxLayout()
        label = QLabel("Edit Display Names:")
        tab.layout.addWidget(label)
        tab.layout.setAlignment(label, Qt.AlignTop)
        tab.layout.addWidget(table)
        tab.layout.setAlignment(Qt.AlignTop)
        tab.layout.addStretch()
        tab.setLayout(tab.layout)

    def create_display_name_table(self, display_names_table, display_names, button_group):
        # display_names_table = QTableWidget()
        display_names_table.setColumnCount(2)
        display_names_table.setRowCount(len(display_names))
        header_font = QFont()
        header_font.setBold(True)
        display_names_table.horizontalHeader().setFont(header_font)
        display_names_table.setHorizontalHeaderLabels(['Display Name', ''])
        # self.display_names_table.verticalHeader().setVisible(False)
        display_names_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        display_names_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        display_names_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.populate_display_names_table(display_names_table, display_names, button_group)

    def resize_Table(self, table):
        # got this code from https://forum.qt.io/topic/26799/qtablewidget-shrink-to-fit/2
        table_width = 2 + table.verticalHeader().width()
        for i in range(table.columnCount()):
            table_width = table_width + table.columnWidth(i)

        table_height = 2 + table.horizontalHeader().height()
        for i in range(table.rowCount()):
            table_height = table_height + table.rowHeight(i)

        table.setMinimumHeight(table_height)
        table.setMaximumWidth(table_width)

    def populate_display_names_table(self, display_names_table, display_names, button_group):
        display_names_table.setRowCount(len(display_names))
        display_names_table.setVerticalHeaderLabels(display_names.keys())
        row = 0
        for key in display_names.keys():
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(key)
            if key != display_names[key]:
                line_edit.setText(display_names[key])
            display_names_table.setCellWidget(row, 0, line_edit)
            xbutton = QPushButton("X")
            xbutton.setMaximumSize(60,80)
            display_names_table.setCellWidget(row, 1, xbutton)
            button_group.addButton(xbutton, row)
            row = row+1
        display_names_table.resizeColumnsToContents()
        display_names_table.resizeRowsToContents()
        self.resize_Table(display_names_table)

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
        self.user_pref_tabs.setCurrentIndex(0)
        self.settings_dialog.show()

    def close_settings_dialog(self):
        self.settings_dialog.close()

    def populate_filters_layout(self, active_filters):
        self.filters_layout.addWidget(QLabel("Active Filters: "))
        # for i in range(len(active_filters)):
        #     button = QPushButton(active_filters[i])
        #     self.remove_filter_buttons.addButton(button, i)
        #     self.filters_layout.addWidget(button)

        for i in range(len(active_filters)):
            self.filters_layout.addWidget(QLabel(active_filters[i]))
        self.filters_layout.addWidget(self.clear_filters_button)

    def file_deletion_warning(self, filename, multi_file=False):
        self.warning_dialog.call_dialog(filename, multi_file)




    def display_pdf(self, filename, report_name, row, col):

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

    def display_image_report(self, path, report_name):
        dialog = QDialog()
        dialog.setWindowTitle(report_name)
        dialog_layout = QGridLayout()
        dialog.setLayout(dialog_layout)
        dialog.setFixedSize(800, 700)

        image = QLabel()
        image.setPixmap(QPixmap(path))
        image.setScaledContents(True)
        pillow = Image.open(path)
        w = pillow.width
        h = pillow.height
        frame_width = 750

        adjusted_h = int(math.ceil(h*frame_width/w))
        layout = QVBoxLayout()
        layout.addWidget(image)

        frame = QFrame()
        frame.setFixedSize(frame_width, adjusted_h)
        frame.setLayout(layout)

        if adjusted_h > 600:
            scroll = QScrollArea()
            scroll.setFixedSize(frame_width, min([adjusted_h, 600]))
            scroll.setWidgetResizable(False)
            scroll.setWidget(frame)
            dialog_layout.addWidget(scroll)
        else:
            frame.setFixedHeight(adjusted_h)
            dialog_layout.addWidget(frame)

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


