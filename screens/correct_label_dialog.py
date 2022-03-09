import math

import screens.supporting_classes as helpers
from PyQt5.QtGui import QPixmap, QBrush, QColor, QFont, QCloseEvent
from PyQt5.QtWidgets import QWidget, QScrollArea, QGridLayout
from PIL import Image
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, QRect, pyqtSignal, QDateTime

from PyQt5.QtWidgets import QGridLayout, QLabel, QToolBar, QStatusBar, QDialog, QTableWidgetItem, QHeaderView, \
    QLineEdit, QGridLayout, QTableWidget, QPushButton, QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, \
    QButtonGroup, QTreeWidget, QTreeWidgetItem, QAbstractItemView, QListWidget, QListWidgetItem, QTabWidget, QFrame, \
    QComboBox, QDateEdit, QSpacerItem, QSizePolicy, QCompleter


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
        self.label_corrections_layout = None
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

        self.modalities = ["X-ray", "MRI", "CT", "Ultrasound"]
        self.body_parts = ["Head and Neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"]
        self.label_corrections_layout = QGridLayout()
        self.populate_label_corrections_layout()


    def report_clicked(self, path, report, isPDF, report_labels, current_institutions):
        self.title_label = QLabel("Report: {}".format(report))
        self.title_label.setStyleSheet("font: bold 18px")
        self.set_initial_information(report_labels)
        self.set_institution_completer(current_institutions)

        if isPDF:
            self.current_report = self.show_pdf(path)
        else:
            self.current_report = self.show_image_report(path)

        self.left_column.addWidget(self.title_label, alignment=Qt.AlignCenter)
        self.left_column.addWidget(self.current_report, alignment=Qt.AlignTop)
        self.right_column.addLayout(self.label_corrections_layout)
        self.right_column.addLayout(self.button_layout)

        self.exec()

    def reset_dialog(self):
        self.left_column.removeWidget(self.title_label)
        self.left_column.removeWidget(self.current_report)
        self.right_column.removeItem(self.label_corrections_layout)
        self.right_column.removeItem(self.button_layout)

        self.current_report = None


    def show_pdf(self, path):
        viewer = helpers.ReportViewer(path)
        return viewer

    def show_image_report(self, path):
        image = QLabel()
        image.setPixmap(QPixmap(path))
        image.setScaledContents(True)
        pillow = Image.open(path)
        w = pillow.width
        h = pillow.height
        frame_width = 700

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
            return scroll
        else:
            frame.setFixedHeight(adjusted_h)
            return frame


    def populate_label_corrections_layout(self):
        self.create_correction_form_widgets()

        spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.label_corrections_layout.addItem(spacer)
        self.label_corrections_layout.addWidget(QLabel("Fix incorrect labels and apply changes:"),1,1)
        self.label_corrections_layout.addWidget(QLabel("Modality:"), 2, 1)
        self.label_corrections_layout.addWidget(self.modality_dropdown, 2,2)
        self.label_corrections_layout.addWidget(QLabel("Body Part:"), 3, 1)
        self.label_corrections_layout.addWidget(self.bodypart_dropdown, 3, 2)
        self.label_corrections_layout.addWidget(QLabel("Institution:"), 4, 1)
        self.label_corrections_layout.addWidget(self.institution_text, 4, 2)
        self.label_corrections_layout.addWidget(QLabel("Clinician:"), 5, 1)
        self.label_corrections_layout.addWidget(self.clinician_text, 5, 2)
        self.label_corrections_layout.addWidget(QLabel("Examination Date:"), 6,1)
        self.label_corrections_layout.addWidget(self.date_picker)
        self.label_corrections_layout.addItem(spacer)

    def create_correction_form_widgets(self):
        self.modality_dropdown = self.create_dropdown(self.modalities)
        self.bodypart_dropdown = self.create_dropdown(self.body_parts)
        self.institution_text = QLineEdit()
        self.clinician_text = QLineEdit()
        self.date_picker = QDateEdit(calendarPopup=True)

    def set_initial_information(self, report_labels):
        self.modality_dropdown.setCurrentText(report_labels[0])
        self.bodypart_dropdown.setCurrentText(report_labels[1])
        self.institution_text.setText(report_labels[2])
        self.clinician_text.setText(report_labels[3])
        self.date_picker.setDateTime(datetime.strptime(report_labels[4], '%Y-%m-%d'))

    def set_institution_completer(self, current_institutions):
        self.completer = QCompleter(current_institutions)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.institution_text.setCompleter(self.completer)


    def create_dropdown(self, items):
        dropdown = QComboBox()
        dropdown.addItems(items)
        return dropdown



