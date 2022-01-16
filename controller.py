import model
import view


class Controller:

    def __init__(self, view, model):

        self.view = view
        self.model = model
        self.connect_signals()

        # when we go to a patient's report screen:
        self.populate_report_info()



    def connect_signals(self):
        """Connect signals and slots."""
        self.view.import_button.clicked.connect(self.import_file)
        self.view.filter_button.clicked.connect(self.filter_selection)
        self.view.dialog_button.clicked.connect(self.apply_filters)

    def import_file(self):
        self.view.show_directory()
        self.model.import_report(self.view.explorer)
        self.populate_report_info()

    def filter_selection(self):
        self.view.show_dialog()

    def apply_filters(self):
        self.model.set_filters(self.view.mod_options, self.view.bodypart_options, self.view.hospital_options)
        self.view.close_dialog()

    def populate_report_info(self):
        reports = self.model.get_reports_to_display()
        rows = self.model.db_connection.get_row_count(patient=self.model.current_patient_ID)
        self.view.set_table_row_count(rows)
        self.view.populate_report_table(reports)

        # populate report info
        # populate report link