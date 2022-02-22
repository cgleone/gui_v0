import pickle
import shutil
from OCR import ocr_main
import os
from database_connector import DB_Connection
from temp_nlp import generate_random_tags
import pandas as pd
from PyQt5.QtCore import Qt
import datetime
from models.minimal_model import MinimalModel





class Model():

    def __init__(self):
        self.db_connection = DB_Connection()
        self.nlp_model = MinimalModel()
        self.set_nlp_params()

        self.current_patient_ID = None
        self.current_physician_ID = None
        self.current_filters = {}
        self.current_categories = None
        self.filter_options = None
        self.date_filters = None
        self.db_functions = None
        self.current_report_IDs = None
        self.display_names = None
        self.modality_display_names = None
        self.bodypart_display_names = None
        self.current_institutions = {}

        self.set_current_patient_ID(22)
        self.set_current_physician_ID(1)
        self.set_filter_options()
        self.set_date_filters()
        self.current_display_data_with_IDs = None
        self.set_default_categories()
        self.set_db_functions()
        self.set_display_names()
        self.set_current_institutions()

        self.in_label_correction_mode = False

        self.short_form_dictionary = {"X-ray": ["x-ray", "xray"],
                                      "MRI": ["mr"],
                                      "Ultrasound": ["doppler", "us", "sonogram"],
                                      "CT": ["cat", "scan"],
                                      "Upper Limbs": ["arm", "arms", "wrist", "hand", "elbow", "shoulder", "forearm"
                                                      "humerus", "radius", "ulna"],
                                      "Lower Limbs": ["leg", "legs","foot", "ankle","toe", "hip", "femur", "knee",
                                                      "patella", "acl", "tibia", "fibula", ],
                                      "Chest": ["heart", "cardiac", "lung", "pulmonary", "sternum", "rib", "ribs",
                                                "diaphragm", "clavicle"],
                                      "Head and neck": ["head","brain", "carotid", "frontal", "parietal", "temporal",
                                                        "occipital", "sinus", "nose", "dental", "mandible", "occular",
                                                        "eye", "mouth", "ear", "pituitary", "neck"],
                                      "Abdomen": ["liver", "stomach", "belly", "kidney", "gallbladder", "spleen",
                                                  "pancreas", "intestine", "colon", "appendix", "uterus", "ovaries",
                                                  "ovarian", "fetal", "pregnancy", "pelvic", "pelvis"]}

        self.unimportant_words = ["and", "the", "to", "or", "a", "report", "transcript", "at", "for"]

        self.date_terms = {"01": ["january", "jan"], "02": ["february", "feb"], "03": ["march", "mar"],
                           "04": ["april","apr"], "05": ["may"], "06": ["june", "jun"], "07": ["july", "jul"],
                           "08": ["august", "aug"], "09": ["september","sept"], "10": ["october", "oct"],
                           "11": ["november", "nov"], "12": ["december", "dec"]}


        self.clinicians_in_db = self.update_clinician_list()
        self.read_csv()

    def set_nlp_params(self):
        with open('models/minimal_model_data/parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
        self.nlp_model.set_parameters(parameters)

    def update_clinician_list(self):
        clinicians = self.db_connection.get_all_clinicians()
        clin_list = list(set(self.get_untupled_label_list(clinicians)))
        print(clin_list)
        return clin_list

    def set_current_patient_ID(self, ID):
        self.current_patient_ID = ID

    def set_current_physician_ID(self, ID):
        self.current_physician_ID = ID

    def get_patient_name(self):
        first = self.db_connection.get_patient_first_name(self.current_patient_ID)[0][0]
        last = self.db_connection.get_patient_last_name(self.current_patient_ID)[0][0]
        print(first)
        print(last)
        return first + ' ' + last

    def set_filter_options(self):
        self.filter_options = {"modalities": ["X-ray", "MRI", "CT", "Ultrasound"],
                               "bodyparts": ["Head and neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs",
                                             "Other"],
                               "exam_date": ["<6mos", "6mos-1yr", "1yr-5yrs", ">5yrs"]}

    def set_date_filters(self):
        self.date_filters = {"<6mos": "> date_sub(now(), interval 6 month)",
                             "6mos-1yr": "between date_sub(now(), interval 1 year) "
                                         "AND date_sub(now(), interval 6 month)",
                             "1yr-5yrs": "between date_sub(now(), interval 5 year) AND date_sub(now(), interval 1 year)",
                             ">5yrs": "< date_sub(now(), interval 5 year)"}

    def set_default_categories(self):
        self.current_categories = ["Exam Date", "File Name", "Imaging Modality", "Body Part", "Notes"]

    def set_db_functions(self):
        self.db_functions = {"Exam Date": self.db_connection.get_report_date,
                             "File Name": self.db_connection.get_report_name,
                             "Imaging Modality": self.db_connection.get_report_modality,
                             "Body Part": self.db_connection.get_report_bodypart,
                             "Institution": self.db_connection.get_report_institution,
                             "Clinician": self.db_connection.get_report_clinician,
                             "Notes": self.db_connection.get_report_notes}

    def set_display_names(self):
        physician_preferences = self.db_connection.get_physician_preferences(self.current_physician_ID)
        self.display_names = {'X-ray': physician_preferences[1],
                              'MRI': physician_preferences[2],
                              'CT': physician_preferences[3],
                              'Ultrasound': physician_preferences[4],
                              'Head and Neck': physician_preferences[5],
                              'Chest': physician_preferences[6],
                              'Abdomen': physician_preferences[7],
                              'Upper Limbs': physician_preferences[8],
                              'Lower Limbs': physician_preferences[9],
                              'Other': physician_preferences[10]}

        self.modality_display_names = {'X-ray': physician_preferences[1],
                                       'MRI': physician_preferences[2],
                                       'CT': physician_preferences[3],
                                       'Ultrasound': physician_preferences[4]}
        self.bodypart_display_names = {'Head and Neck': physician_preferences[5],
                                       'Chest': physician_preferences[6],
                                       'Abdomen': physician_preferences[7],
                                       'Upper Limbs': physician_preferences[8],
                                       'Lower Limbs': physician_preferences[9],
                                       'Other': physician_preferences[10]}

    def set_current_institutions(self):
        institutions = self.db_connection.get_institutions_in_db(self.current_physician_ID)
        for institute in institutions:
            self.current_institutions.update({institute[0]: institute[1]})

    def import_report(self, path):
        filename = path.split('/')[-1]
        id = str(self.db_connection.generate_report_id())
        shutil.copy(path, self.get_unique_report_paths(filename, id)[0])
        shutil.copy(path, 'OCR/reports_temp/'+filename)
        self.call_ocr(filename, id)
        # self.call_nlp(id)
        self.call_fake_nlp(id)
        self.deal_with_institution(id)

    def call_ocr(self, filename, id):
        report_text = ocr_main.run_ocr(filename)
        self.save_ocr_result(filename, id, report_text)

    def call_fake_nlp(self, report_id):
        labels = generate_random_tags()
        # mod_display = self.get_display_name(labels[0])
        # bp_display = self.get_display_name(labels[1])
        # label_args = [self.current_patient_ID, report_id] + labels + [mod_display, bp_display]
        label_args = [self.current_patient_ID, report_id] + labels
        self.db_connection.add_labels(label_args)

    def deal_with_institution(self, report_id):
        institution = self.db_connection.get_report_institution(report_id)[0][0]
        is_Existing_Institution = institution in self.current_institutions.keys()
        if not is_Existing_Institution:
            self.db_connection.add_institution(institution, self.current_physician_ID)
            self.set_current_institutions()

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
        self.db_connection.add_report(self.current_patient_ID, id, report_name.split('.')[0], file_path, text_path, "")
        self.update_clinician_list()

    def call_nlp(self, report_id):
        text_file = self.db_connection.get_text_path(report_id)[0][0]
        file = open(text_file, 'r')
        text = file.read()
        file.close()

        nlp_data = self.nlp_model.predict({0: text})[0]
        labels = [nlp_data['modality'], nlp_data['body_part'], nlp_data['clinic_name'],
                  nlp_data['dr_name'], nlp_data['date_of_procedure']]
        # mod_display = self.get_display_name(labels[0])
        # bp_display = self.get_display_name(labels[1])
        # label_args = [self.current_patient_ID, report_id] + labels + [mod_display, bp_display]
        label_args = [self.current_patient_ID, report_id] + labels
        self.db_connection.add_labels(label_args)

    def set_filters(self, modalities, bodyparts, dates):
        checked_modalities = []
        active_filters=[]
        for key in modalities.keys():
            if modalities[key].isChecked():
                checked_modalities.append(key)
                active_filters.append(self.display_names[key])
        # active_filters = active_filters + checked_modalities
        checked_modalities = self.get_checked_datatype(checked_modalities, "modalities")

        checked_bodyparts = []
        for key in bodyparts.keys():
            if bodyparts[key].isChecked():
                checked_bodyparts.append(key)
                active_filters.append(self.display_names[key])
        # active_filters = active_filters + checked_bodyparts
        checked_bodyparts = self.get_checked_datatype(checked_bodyparts, "bodyparts")

        checked_dates = []
        checked_dates_values = []
        for key in dates.keys():
            if dates[key].isChecked():
                checked_dates.append(key)
                checked_dates_values.append(dates[key].text())
        active_filters = active_filters + checked_dates_values

        checked_filters = {"modality": checked_modalities, "bodypart": checked_bodyparts, "exam_date": checked_dates}

        self.current_filters = checked_filters
        return active_filters

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def reset_current_filters(self):
        self.current_filters = None

    def reset_filter_checkboxes(self, filter_checkboxes):
        for i in range(len(filter_checkboxes)):
            for key in filter_checkboxes[i].keys():
                if filter_checkboxes[i][key].isChecked():
                    filter_checkboxes[i][key].setChecked(False)

    def uncheck_filter(self, filter_to_remove, filter_checkboxes):
        for i in range(len(filter_checkboxes)):
            for key, value in filter_checkboxes[i].items():
                valtext = value.text()
                filtertext = filter_to_remove.text()
                if value.text() == filter_to_remove.text():
                    filter_checkboxes[i][key].setChecked(False)

    def get_checked_datatype(self, list, category=None):
        if len(list) == 0:
            return tuple(self.filter_options[category])
        elif len(list) == 1:
            single_filter = "('{}')".format(list[0])
            return single_filter
        else:
            return tuple(list)

    def get_filtered_ids(self):
        mod_bp_ids = self.get_mod_bp_ids()

        if len(mod_bp_ids) == 0 or len(self.current_filters["exam_date"]) == 0:
            filtered_IDs = mod_bp_ids
            return filtered_IDs

        else:
            date_IDs = self.get_date_ids(mod_bp_ids)
            filtered_IDs = date_IDs
            return filtered_IDs


    def get_mod_bp_ids(self):
        mod_bp_ids = []
        query_values = (self.current_patient_ID, self.current_filters["modality"], self.current_filters["bodypart"])
        mod_bp_ids_tuple = self.db_connection.get_mod_bd_IDs(query_values)
        for id in mod_bp_ids_tuple:
            mod_bp_ids.append(id[0])
        return mod_bp_ids

    def get_date_ids(self, mod_bp_ids):
        total_ids = []
        for option in self.current_filters["exam_date"]:
            date_query = self.date_filters[option]
            query_values = (self.current_patient_ID, self.get_checked_datatype(mod_bp_ids), date_query)
            date_IDs = self.db_connection.get_filtered_date_IDs(query_values)
            total_ids = total_ids + date_IDs
        return total_ids

    def get_patient_data(self):
        prepped_data = []
        data = self.db_connection.get_patient_info()
        for patient in data:
            prepped_patient = list(patient)
            prepped_patient[0] = str(patient[0])
            dob = prepped_patient[4]
            dob = dob.strftime("%Y-%m-%d")
            prepped_patient[4] = dob
            prepped_data.append(prepped_patient)

        return prepped_data

    def get_reports_to_display(self, filtered_IDs=None):
        if filtered_IDs is None:
            report_IDs = self.db_connection.get_report_IDs(self.current_patient_ID)
            self.current_report_IDs = report_IDs
            if report_IDs is None:
                return None
        else:
            report_IDs = filtered_IDs
            self.current_report_IDs = report_IDs

        display_data = []
        data_with_IDs = []

        for id in report_IDs:
            display = []
            display_with_ID = []
            for category in self.current_categories:
                label = self.db_functions[category](id)
                try:
                    if label[0][0] in self.display_names.keys():
                        data = self.display_names[label[0][0]]
                        data = [(data,)]
                    elif label[0][0] in self.current_institutions.keys():
                        data = self.current_institutions[label[0][0]]
                        data = [(data,)]
                    else:
                        data = label
                    display.append(data)
                    display_with_ID.append(data)
                except:
                    display.append(" ")
                    display_with_ID.append(" ")

            display_with_ID.append([[id]])

            display_data.append(display)
            data_with_IDs.append(display_with_ID)


        report_IDs.reverse()
        display_data.reverse()
        data_with_IDs.reverse()
        self.current_display_data_with_IDs = data_with_IDs
        print("display data: {}".format(display_data))
        self.current_report_IDs.reverse()
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


    def determine_checked_categories(self, category_list):
        checked_categories=[]
        for i in range(category_list.count()):
            category = category_list.item(i)
            if category.checkState() == Qt.Checked:
                checked_categories.append(category.text())
        self.current_categories = checked_categories

    def update_view_category_list(self, category_list):
        for i in range(category_list.count()):
            category = category_list.item(i)
            if category.text() in self.current_categories:
                category.setCheckState(Qt.Checked)

    def determine_display_names(self, table, display_name_dict):
        rows = table.rowCount()
        for i in range(rows):
            category = table.verticalHeaderItem(i).text()
            display_name = table.cellWidget(i,0).text()
            if display_name != '':
                self.display_names[category] = display_name
                display_name_dict[category] = display_name
            else:
                self.display_names[category] = category
                display_name_dict[category] = category
            query_values = (category, self.display_names[category], self.current_physician_ID)
            self.db_connection.update_display_name_db(query_values)
        # self.update_display_columns_db()

    def determine_institution_display_names(self, table):
        rows = table.rowCount()
        for i in range(rows):
            formal_name = table.verticalHeaderItem(i).text()
            display_name = table.cellWidget(i,0).text()
            if display_name != '':
                self.current_institutions[formal_name] = display_name
            else:
                self.current_institutions[formal_name] = formal_name
            id_institutions = self.db_connection.get_id_institutions(self.current_physician_ID, formal_name)[0][0]
            self.db_connection.update_institution_display_name_db(self.current_institutions[formal_name], id_institutions)

    def update_display_columns_db(self, report_ids = None):
        if report_ids is None:
            report_ids = self.db_connection.get_all_report_ids()
        for report_id in report_ids:
            print(report_id)
            try:
                modality = self.db_connection.get_report_modality(report_id)[0][0]
                bodypart = self.db_connection.get_report_bodypart(report_id)[0][0]
                modality_display = self.display_names[modality]
                bodypart_display = self.display_names[bodypart]
            except:
                modality_display = ""
                bodypart_display = ""

            query_values = (modality_display, bodypart_display, report_id[0])
            self.db_connection.update_label_table_display_name_column(query_values)

    def get_display_name(self, category):
        display_name = self.display_names[category]
        return display_name

    def update_filter_checkmark_display_text(self, checkmarks):
        for dict in checkmarks:
            for key in dict.keys():
                dict[key].setText(self.display_names[key])

    def reset_single_display_name(self, button_pressed, display_name_group, display_names_table):
        row = display_name_group.id(button_pressed)
        display_names_table.cellWidget(row, 0).clear()

    def reset_all_display_names(self, display_names_table):
        for i in range(display_names_table.rowCount()):
            display_names_table.cellWidget(i, 0).clear()

    def clear_searchbar(self, searchbar):
        searchbar.clear()



    def set_category_dict(self):
        self.category_dict = {"Modality": ["MRI", "CT", "Ultrasound", "X-ray"],
                              "Bodypart": ["Head and Neck", "Chest", "Abdomen", "Upper Limbs", "Lower Limbs", "Other"],
                              "Institution": ["Hospital", ]}


# ----------- BELOW THIS POINT, ALL CODE IS FOR SEARCH BAR IMPLEMENTATION -----------


    def search_labels_by_partial_sf_query(self, partial_query):
        labels_present = []
        if any(partial_query.lower() in map(str.lower, sf_list) for sf_list in self.short_form_dictionary.values()):
            labels_present = self.get_full_label_name(partial_query.lower())
        if partial_query.lower() in map(str.lower, self.short_form_dictionary.keys()):
            labels_present = labels_present + [partial_query]
        # get any institution labels:
        labels_present = labels_present + self.get_institution_labels(partial_query)
        return labels_present

    def get_full_label_name(self, value):
        full_names = []  # could maybe be more than one answer (ideally just 1 though)
        for label in self.short_form_dictionary.keys():
            if value in self.short_form_dictionary[label]:
                full_names.append(label)
        return full_names

    def apply_search_labels(self, labels):
        category_set = list(set([label.type for label in labels]))
        label_values = [label.value for label in labels]

        print("category set: {}".format(category_set))
        print("label values: {}".format(label_values))

        category_count = len(category_set)
        unique_label_values = len(label_values)  # we already know that all values in label_values are unique because we created it that way

        if category_count == 1:
            # apply all labels inclusively since they're all the same category of labels
            # example: ['Upper Limbs', 'Lower Limbs'] should show all reports tagged for either upper or lower limbs
            ids = []
            for label in labels:
                ids += self.db_connection.search_by_label(label.value, label.type)

        elif category_count == unique_label_values:
            # one label of each type -- this is "easy", just make all labels required
            # example: ['CT', 'Head and Neck'] or ['MRI', 'Grand River Hospital', 'Lower Limbs']
            # only return reports that match ALL of the labels

            query_variable_string = ""
            for i in range(len(labels)):
                label_obj = labels[i]
                if i < len(labels)-1:  # if we're not on the last item of the list
                    query_variable_string += " " + label_obj.type + "=" + "'" + label_obj.value + "' AND"
                else: # we're on the last item, so don't put an "AND" at the end
                    query_variable_string += " " + label_obj.type + "=" + "'" + label_obj.value + "'"

            ids = self.db_connection.search_with_super_variable_query(query_variable_string)


        elif unique_label_values > category_count:
            # there's more than one label in at least one category, ex. ['Upper Limbs', 'Lower Limbs', 'X-ray']
            # should return any xray of either the upper or lower limbs in the above example
            query_variable_string = ""
            # create "OR" sections first, i.e. deal with labels of the same category
            labels_by_category_dict = {}
            for category in category_set:
                labels_by_category_dict[category] = []
            for label in labels:
                labels_by_category_dict[label.type].append(label.value)
            print(labels_by_category_dict)

            # we should now have a dict (for the above example) that is:
            # {"bodypart": ['Upper Limbs', 'Lower Limbs'], "modality": ['X-ray']
            sub_queries = []
            for category in labels_by_category_dict.keys():
                sub_query = " ("
                for i in range(len(labels_by_category_dict[category])):
                    value = labels_by_category_dict[category][i]
                    if i < len(labels_by_category_dict[category])-1:
                        sub_query += " " + category + "=" + "'" + value + "' OR"
                    else:
                        sub_query += " " + category + "=" + "'" + value + "')"
                sub_queries.append(sub_query)

            total_query_var = ""
            for i in range(len(sub_queries)):
                if i < len(sub_queries)-1:
                    total_query_var += sub_queries[i] + " AND"
                else:
                    total_query_var += sub_queries[i]

            ids = self.db_connection.search_with_super_variable_query(total_query_var)

        else:
            print("No labels found - reached else statement in apply_search_labels()")
            ids = []

        return ids

    def search(self, user_query):

        all_current_label_options = self.get_untupled_label_list(self.db_connection.get_all_labels())

        labels_searched_for, date_in_search = self.label_search_main(user_query, all_current_label_options)
        if len(labels_searched_for) == 0 and date_in_search == [[], []]:
            return [[], []]
        elif len(labels_searched_for) > 0:
            print("labels searched for: {}".format(labels_searched_for))
            labels_with_types = self.assign_label_type(labels_searched_for)
            ids_from_labels = self.apply_search_labels(labels_with_types)
            print("ids from labels: {}".format(ids_from_labels))
            if date_in_search != [[], []]:
                final_ids = self.search_by_date(ids_from_labels, date_in_search[0], date_in_search[1])
            else:
                final_ids = ids_from_labels
            return [final_ids, labels_searched_for]
        else:
            all_ids = self.db_connection.get_report_IDs(self.current_patient_ID)
            final_ids = self.search_by_date(all_ids, date_in_search[0], date_in_search[1])
            return [final_ids, []]
       # ids = self.apply_search_labels(desired_institutions)


    def label_search_main(self, user_query, label_options):
        labels_searched_for = []
        if self.is_exact_label_match(user_query, label_options):  # check for exact matches to search criteria
            labels_searched_for.append(user_query)
            date_in_search = [[], []]
        else: # no exact match for label, but there might be a match for a short form of a label
            label_info = self.identify_short_form_search_labels(user_query)
            date_in_search = label_info[1]
            labels_searched_for = labels_searched_for + label_info[0]


        lower_case_label_list = [label.lower() for label in labels_searched_for]

        return list(set(lower_case_label_list)), date_in_search   # make sure there are no duplicate labels by returning "set()"

    def assign_label_type(self, labels):

        label_types = []
        for label in labels:
            if label in map(str.lower, self.filter_options["modalities"]):
                l = Label("modality", label)
                label_types.append(l)
            elif label in map(str.lower, self.filter_options["bodyparts"]):
                l = Label("bodypart", label)
                label_types.append(l)
            elif label in map(str.lower, self.all_institutions["Names"]):
                l = Label("institution", label)
                label_types.append(l)
            elif label in map(str.lower, self.clinicians_in_db):
                l = Label("clinician", label)
                label_types.append(l)

        return label_types

    def identify_short_form_search_labels(self, user_query):
        all_present_sf_labels = []
        exact_match_segments, user_query = self.get_exact_match_segments(user_query)
        if user_query.isspace():
            return exact_match_segments

        months = []
        years = []

        date_in_search = [[], []]
        if " " in user_query: # check each "word" of the search individually
            partial_queries = user_query.split(' ')
            for query in partial_queries:
                if query not in self.unimportant_words:
                    all_present_sf_labels = all_present_sf_labels + self.search_labels_by_partial_sf_query(query)
                    months, years = self.get_dates_in_query(query, months, years)
                    if len(months) or len(years):
                        date_in_search = [months, years]
        else: # check whole label if there's only one "word"
            all_present_sf_labels = all_present_sf_labels + self.search_labels_by_partial_sf_query(user_query)
            months, years = self.get_dates_in_query(user_query, months, years)
            if len(months) or len(years):
                date_in_search = [months, years]

        all_present_sf_labels = all_present_sf_labels + exact_match_segments

        return all_present_sf_labels, date_in_search

    def get_dates_in_query(self, query, months, years):
        month = self.month_in_query(query)
        year = self.year_in_query(query)
        if month:
            months.append(month)
        if year:
            years.append(year)

        return months, years

    def get_exact_match_segments(self, query):
        exact_match_segments = []
        for label_list in [self.short_form_dictionary.keys(), self.all_institutions['Names'], self.clinicians_in_db]:
            for label in label_list:
                if " " in label:
                    if label.lower() in query.lower():
                        exact_match_segments.append(label)
                        query = query.lower().replace(label.lower(), '')  # replace the exact match with nothingness (removing the match from the query - tigger)
        return [exact_match_segments, query]

    def is_exact_label_match(self, query, labels):
        thing = query.lower()
        idk = map(str.lower, labels)
        if query.lower() in map(str.lower, labels):
            return True
        else:
            return False

    def get_untupled_label_list(self, tupled_list):
        new_list = []
        for tuple in tupled_list:
            if tuple[0] is not None:
                new_list.append(tuple[0])
        new_list = list(dict.fromkeys(new_list)) # remove repeated items
        return new_list

    def get_institution_labels(self, user_query):

        # assume user query exactly matches institution short form from list
        short_forms = self.all_institutions['Short forms']
        desired_institutions = []
        for i in range(len(short_forms)):
            test = short_forms[i]
            blah = user_query.lower()
            print(test)
            bluh = test.lower()
            if user_query.lower() == test.lower():
                inst = self.all_institutions['Names'][i]
                desired_institutions.append(inst)

        return desired_institutions

    def month_in_query(self, query):
        for month in self.date_terms.values():
            for month_term in month:
                if month_term == query.lower():
                    return month_term

        return None

    def year_in_query(self, query):
        now = datetime.datetime.now()
        if query.isdigit():
            if int(query) <= int(now.year) and int(query) >= 1920:
                return query
        return None

    def search_by_date(self, ids, months, years):
        if months == [] and years == []:
            return []
        new_ids = []
        month_codes = []
        for month in months:
            month_codes.append(self.get_month_code(month))

        for report_id in ids:
            report_date = self.db_connection.get_report_date(report_id)[0][0]
            if (report_date.split('-')[1] in month_codes or len(month_codes) == 0)\
                    and (report_date[0:4] in years or len(years) == 0):
                new_ids.append(report_id)

        return new_ids

    def get_month_code(self, month):
        month_code = None
        for month_option in self.date_terms.keys():
            if month.lower() in self.date_terms[month_option]:
                month_code = month_option
        return month_code


    def read_csv(self):
        self.all_institutions = pd.read_csv('institution_list.csv')
       # print(self.all_institutions)

    def link_search_and_filters(self, labels, checkmark_categories):
        if labels != []:
            for category in checkmark_categories:
                for checkmark in category.items():
                    for label in labels:
                        if label == checkmark[0].casefold():
                            checkmark[1].setChecked(True)




class Label:
    def __init__(self, type, value):
        self.type = type
        self.value = value
