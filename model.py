import shutil
from OCR import ocr_main
import os
from database_connector import DB_Connection
from temp_nlp import generate_random_tags
import pandas as pd

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
        self.current_columns = None

        self.set_current_patient_ID(22)
        self.current_display_data_with_IDs = None

        self.filter_options = {"modalities": ["X-ray", "MRI", "CT", "Ultrasound"],
                               "bodyparts": ["Head and neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"],
                               "exam_date":["<6mos", "6mos-1yr", "1yr-5yrs", ">5yrs"]}

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

    def set_filters(self, modalities, bodyparts, dates):

        # for category in [modalities, bodyparts, dates]:
        #     for key in category.keys():
        #         if category[key].isChecked():
        #             checked_filters.append(key)

        checked_modalities = []
        for key in modalities.keys():
            if modalities[key].isChecked():
                checked_modalities.append(key)
        checked_modalities = self.get_checked_datatype(checked_modalities, "modalities")

        checked_bodyparts = []
        for key in bodyparts.keys():
            if bodyparts[key].isChecked():
                checked_bodyparts.append(key)
        checked_bodyparts = self.get_checked_datatype(checked_bodyparts, "bodyparts")

        checked_dates = []
        for key in dates.keys():
            if dates[key].isChecked():
                checked_dates.append(key)
        # checked_dates = self.get_checked_datatype(checked_dates, "exam_date")

        checked_filters = {"modality": checked_modalities, "bodypart": checked_bodyparts, "exam_date": checked_dates}

        self.current_filters = checked_filters

    def get_checked_datatype(self, list, category):
        if len(list) == 0:
            return tuple(self.filter_options[category])
        elif len(list) == 1:
            single_filter = "('{}')".format(list[0])
            return single_filter
        else:
            return tuple(list)

    def get_reports_to_display(self, filtered_IDs=None):
        if filtered_IDs is None:
            report_IDs = self.db_connection.get_report_IDs(self.current_patient_ID, self.current_filters)
            if report_IDs is None:
                return None
        else:
            report_IDs = filtered_IDs

        display_data = []
        data_with_IDs = []

        if self.current_columns is None:
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
        print("display data: {}".format(display_data))
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


    def set_category_dict(self):
        self.category_dict = {"Modality": ["MRI", "CT", "Ultrasound", "X-ray"],
                              "Bodypart": ["Head and Neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"],
                              "Institution": ["Hospital", ]}


    def begin_search(self, user_query):
        # do some logic to break it into pieces
        # step 1 - check for institutions / clinicians
        # step 2 - check for other labels
        # step 3 - look through OCR text

        return self.get_institution_ids(user_query)

    def apply_search_labels(self, labels):
        all_ids = []
        for label in labels:
            ids = self.db_connection.search_by_label(label)
            all_ids = all_ids + ids
        print(all_ids)
        return all_ids


    def get_institution_ids(self, user_query):

        # assume user query exactly matches institution short form from list
        self.read_csv()
        short_forms = self.all_institutions['Short forms']
        desired_institutions = []
        for i in range(len(short_forms)):
            if user_query == short_forms[i]:
                inst = self.all_institutions['Names'][i]
                desired_institutions.append(inst)
        print(desired_institutions)
        ids = self.apply_search_labels(desired_institutions)
        return ids

    def read_csv(self):
        self.all_institutions = pd.read_csv('institution_list.csv')
        print(self.all_institutions)


