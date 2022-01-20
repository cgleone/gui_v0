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
        self.current_display_data_with_IDs = None

    def set_current_patient_ID(self, ID):
        self.current_patient_ID = ID

    def import_report(self, path):
        filename = path.split('/')[-1]
        id = str(self.db_connection.generate_report_id())
        shutil.copy(path, self.get_unique_report_paths(filename, id)[0])
        shutil.copy(path, 'OCR/reports_temp/'+filename)
        self.call_ocr(filename, id)
        self.call_nlp(id)

    def call_ocr(self, filename, id):
        report_text = ocr_main.run_ocr(filename)
        self.save_ocr_result(filename, id, report_text)

    def get_unique_report_paths(self, report_name, id):
        filename = report_name.split('.')[0]
        extension = report_name.split('.')[1]
        file_path = 'reports/' + filename + "_" + id + "." + extension
        text_path = 'report_texts/' + filename + "_" + id + '.txt'
        return file_path, text_path

    def save_ocr_result(self, report_name, id, result):
        file_path, text_path = self.get_unique_report_paths(report_name, id)
        file = open(text_path, "w+")
        file.write(result)
        file.close()
        self.db_connection.add_report(patient_id, id, report_name.split('.')[0], file_path, text_path)

    def call_nlp(self, report_id):
        labels = generate_random_tags()
        label_args = [patient_id, report_id] + labels
        self.db_connection.add_labels(label_args)

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
        data_with_IDs = []

        if self.current_filters is None:
            for id in report_IDs:
                display = [self.db_connection.get_report_date(id), self.db_connection.get_report_name(id),
                           self.db_connection.get_report_modality(id), self.db_connection.get_report_bodypart(id),
                           [["None"]]]
                display_with_ID = [self.db_connection.get_report_date(id), self.db_connection.get_report_name(id),
                           self.db_connection.get_report_modality(id), self.db_connection.get_report_bodypart(id),
                           [["None"]], [[id]]]

                display_data.append(display)
                data_with_IDs.append(display_with_ID)

        report_IDs.reverse()
        display_data.reverse()
        data_with_IDs.reverse()
        self.current_display_data_with_IDs = data_with_IDs
        return display_data

    def view_report(self, row, col):
        row_data = self.current_display_data_with_IDs[row]
        name = row_data[1][0][0]
        file_ID = row_data[-1][0][0]
        filepath = self.db_connection.get_report_path(file_ID)[0][0]
        if filepath.split('.')[-1] == 'pdf':
            print("this file is {} and is a pdf".format(name))
            isPDF = True
        else:
            print("this file is {} and is not a pdf".format(name))
            isPDF = False
        return filepath, isPDF, name


    # def filter_by_label(self, desired_labels, category=None):
    #     if category is None:
    #         self.get_category(desired_labels)


    # def get_category(self, labels):
    #     categories = []
    #     for label in labels:


    def set_category_dict(self):
        self.category_dict = {"Modality": ["MRI", "CT", "Ultrasound", "X-ray"],
                              "Bodypart": ["Head and Neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"],
                              "Institution": ["Hospital", ]}



