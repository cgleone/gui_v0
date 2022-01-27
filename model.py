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
        self.filter_options = None
        self.date_filters = None

        self.set_current_patient_ID(22)
        self.set_filter_options()
        self.set_date_filters()
        self.current_display_data_with_IDs = None


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
                                                  "ovarian", "fetal", "pregnancy"]}

        self.unimportant_words = ["and", "the", "to", "or", "a", "report", "transcript", "at", "for"]
        self.read_csv()


    def set_current_patient_ID(self, ID):
        self.current_patient_ID = ID

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
        checked_modalities = []
        active_filters=[]
        for key in modalities.keys():
            if modalities[key].isChecked():
                checked_modalities.append(key)
        active_filters = active_filters + checked_modalities
        checked_modalities = self.get_checked_datatype(checked_modalities, "modalities")

        checked_bodyparts = []
        for key in bodyparts.keys():
            if bodyparts[key].isChecked():
                checked_bodyparts.append(key)
        active_filters = active_filters + checked_bodyparts
        checked_bodyparts = self.get_checked_datatype(checked_bodyparts, "bodyparts")

        checked_dates = []
        for key in dates.keys():
            if dates[key].isChecked():
                checked_dates.append(key)
        active_filters = active_filters + checked_dates

        checked_filters = {"modality": checked_modalities, "bodypart": checked_bodyparts, "exam_date": checked_dates}

        self.current_filters = checked_filters
        return active_filters

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

    def get_reports_to_display(self, filtered_IDs=None):
        if filtered_IDs is None:
            report_IDs = self.db_connection.get_report_IDs(self.current_patient_ID)
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

        active_type_count = len(category_set)
        unique_label_values = len(label_values)  # we already know that all values in label_values are unique because we created it that way

        if active_type_count == 1:
            # apply all labels inclusively since they're all the same category of labels
            # example: ['Upper Limbs', 'Lower Limbs'] should show all reports tagged for either upper or lower limbs
            ids = []
            for label in labels:
                ids += self.db_connection.search_by_label(label.value, label.type)

        elif active_type_count == unique_label_values:
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


        elif unique_label_values > active_type_count:
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
        # do some logic to break it into pieces
        # step 1 - check for institutions / clinicians
        # step 2 - check for other labels
        # step 3 - look through OCR text

        all_current_label_options = self.get_untupled_label_list(self.db_connection.get_all_labels())
        labels_searched_for = self.label_search_main(user_query, all_current_label_options)
        print("labels searched for: {}".format(labels_searched_for))
        labels_with_types = self.assign_label_type(labels_searched_for)
        ids_from_labels = self.apply_search_labels(labels_with_types)
        print("ids from labels: {}".format(ids_from_labels))
        return ids_from_labels
       # ids = self.apply_search_labels(desired_institutions)


    def label_search_main(self, user_query, label_options):
        labels_searched_for = []
        if self.is_exact_label_match(user_query, label_options):  # check for exact matches to search criteria
            labels_searched_for.append(user_query)
        else: # no exact match for label, but there might be a match for a short form of a label
            labels_searched_for = labels_searched_for + self.identify_short_form_search_labels(user_query)

        return list(set(labels_searched_for))  # make sure there are no duplicate labels by returning "set()"

    def assign_label_type(self, labels):
        # label_types = {"mod": [], "body": [], "inst": [], "clin": []}
        # for label in labels:
        #     if label in self.filter_options["modalities"]:
        #         label_types["mod"].append(label)
        #     elif label in self.filter_options["bodyparts"]:
        #         label_types["body"].append(label)
        #     elif label in self.all_institutions["Names"]:
        #         label_types["inst"].append(label)
        #     else:
        #         label_types["clin"].append(label)

        label_types = []
        for label in labels:
            if label.lower() in map(str.lower, self.filter_options["modalities"]):
                l = Label("modality", label)
                label_types.append(l)
            elif label.lower() in map(str.lower, self.filter_options["bodyparts"]):
                l = Label("bodypart", label)
                label_types.append(l)
            elif label.lower() in map(str.lower, self.all_institutions["Names"]):
                l = Label("institution", label)
                label_types.append(l)
            else:
                l = Label("clinician", label)
                label_types.append(l)

        return label_types


    def identify_short_form_search_labels(self, user_query):
        all_present_sf_labels = []
        if " " in user_query: # check each "word" of the search individually
            partial_queries = user_query.split(' ')
            for query in partial_queries:
                if query not in self.unimportant_words:
                    all_present_sf_labels = all_present_sf_labels + self.search_labels_by_partial_sf_query(query)
        else: # check whole label if there's only one "word"
            all_present_sf_labels = all_present_sf_labels + self.search_labels_by_partial_sf_query(user_query)

        return all_present_sf_labels

    def is_exact_label_match(self, query, labels):
        if query.lower() in map(str.lower, labels):
            return True
        else:
            return False

    def get_untupled_label_list(self, tupled_list):
        new_list = []
        for tuple in tupled_list:
            new_list.append(tuple[0])
        new_list = list(dict.fromkeys(new_list)) # remove repeated items
        return new_list

    def get_institution_labels(self, user_query):

        # assume user query exactly matches institution short form from list
        short_forms = self.all_institutions['Short forms']
        desired_institutions = []
        for i in range(len(short_forms)):
            if user_query == short_forms[i]:
                inst = self.all_institutions['Names'][i]
                desired_institutions.append(inst)

        return desired_institutions

    def read_csv(self):
        self.all_institutions = pd.read_csv('institution_list.csv')
       # print(self.all_institutions)

    def clear_filters_layout(self, filters_layout):
        for i in reversed(range(filters_layout.count())):
            widget_to_remove = filters_layout.itemAt(i).widget()
            filters_layout.removeWidget(widget_to_remove)
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
            for key in filter_checkboxes[i].keys():
                if key == filter_to_remove.text():
                    filter_checkboxes[i][key].setChecked(False)

class Label:
    def __init__(self, type, value):
        self.type = type
        self.value = value
