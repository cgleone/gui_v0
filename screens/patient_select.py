
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import QWidget, QTableWidgetItem, QHeaderView, QTableWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt


class PatientSelectScreen(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)
        top_layout = QHBoxLayout()
        self.layout().addLayout(top_layout)
        self.layout().setAlignment(Qt.AlignCenter)
        self.layout().setSpacing(50)
        #self.layout().setContentsMargins(0, 50, 0, 0)

        self.create_patient_table()

        self.back_button = QPushButton("Back to Main Menu")
        self.back_button.setFixedSize(200, 30)
        self.add_new_patient_button = QPushButton("Add New Patient")
        self.add_new_patient_button.setFixedSize(200, 30)
        top_layout.setSpacing(600)
        top_layout.addWidget(self.back_button, Qt.AlignLeft)
        top_layout.addWidget(self.add_new_patient_button, Qt.AlignRight)

    def create_patient_table(self):
        self.patient_table = QTableWidget()
        self.patient_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.patient_table.setMouseTracking(True)

        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)

        self.create_table_columns()

        self.patient_table.horizontalHeader().setFont(header_font)
        self.patient_table.verticalHeader().setVisible(False)
        self.layout().addWidget(self.patient_table)

        self.current_hover = [0]
        self.patient_table.cellEntered.connect(self.cell_hover)

    def create_table_columns(self):
        patient_categories = ["Patient ID", "Last Name", "Given Name(s)", "Age", "D.O.B", "Sex"]
        self.column_count = len(patient_categories)
        self.patient_table.setColumnCount(self.column_count)
        self.patient_table.setHorizontalHeaderLabels(patient_categories)
        # self.patient_table.horizontalHeader().setStretchLastSection(True)
        # self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.patient_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # self.patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def cell_hover(self, row):
        underlined = QFont()
        underlined.setUnderline(True)
        normal = QFont()
        self.patient_table.setCursor(Qt.PointingHandCursor)

        col = 0
        for i in range(0, self.column_count):
            item = self.patient_table.item(row, col)
            old_item = self.patient_table.item(self.current_hover[0], col)
            if self.current_hover != [row]:
                old_item.setBackground(QBrush(QColor('white')))
                item.setBackground(QBrush(QColor('#E0EEEE')))
                old_item.setFont(normal)
                item.setFont(underlined)
            col = col + 1
        self.current_hover = [row]

    def populate_table(self, patient_data):
        self.patient_table.setRowCount(len(patient_data))
        print(patient_data)
        if patient_data is None:
            return
        for i in range(len(patient_data)):
            row_data = patient_data[i]
            for j in range(len(row_data)):
                cell_data = row_data[j]
                self.patient_table.setItem(i, j, QTableWidgetItem(cell_data))