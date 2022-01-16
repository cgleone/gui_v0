import shutil
from OCR import ocr_main
import os
from database_connector import DB_Connection
from temp_nlp import generate_random_tags

# temp patient id
patient_id = 22
#
# list_of_files = []
# class File():
#     def __init__(self):
#
#
#         self.file_name
#         self.OCR_output
#         self.tags = []
#         list_of_files.append(self)
#

class Model():

    def __init__(self):
        self.db_connection = DB_Connection()

        self.current_patient_ID = None
        self.current_filters = None

        self.set_current_patient_ID(22)

    def set_current_patient_ID(self, ID):
        self.current_patient_ID = ID

    def import_report(self, path):
        filename = path.split('/')[-1]
        shutil.copy(path, 'reports/'+filename)
        shutil.copy(path, 'OCR/reports_temp/'+filename)
        self.call_ocr(filename)
        # self.update_database_reports()
        # self.call_nlp()
        # self.update_database_labels()

    def call_ocr(self, filename):
        report_text = ocr_main.run_ocr(filename)
        #os.remove('OCR/reports_temp/'+filename)
        self.save_ocr_result(filename, report_text)
        filename = filename.split('.')[0]

        text_path = 'report_texts/' + filename + '.txt'
        file_path = 'reports/' + filename + '.pdf'

        report_id = self.db_connection.add_report(patient_id, filename, file_path, text_path)
        labels = generate_random_tags()
        label_args = [patient_id, report_id] + labels
        self.db_connection.add_labels(label_args)

    def save_ocr_result(self, report_name, result):
        filename = report_name.split('.')[0]
        text_path = 'report_texts/' + filename + '.txt'
        file_path = 'reports/' + filename + '.pdf'
        file = open(text_path, "w+")
        file.write(result)
        file.close()

    def set_filters(self, modalities, bodyparts, hospitals):

        checked_filters = []
        for category in [modalities, bodyparts, hospitals]:
            for key in category.keys():
                if category[key].isChecked():
                    checked_filters.append(key)

        self.current_filters = checked_filters

    def get_reports_to_display(self):
        report_IDs = self.db_connection.get_report_IDs(self.current_patient_ID)
        if report_IDs is None:
            return None

        display_data = []

        if self.current_filters is None:
            for id in report_IDs:
                display = [self.db_connection.get_report_date(id), self.db_connection.get_report_name(id),
                           self.db_connection.get_report_modality(id), self.db_connection.get_report_bodypart(id),
                           [["None"]]]
                display_data.append(display)

        return display_data


