
import mysql.connector as mysql

class DB_Connection:

    def __init__(self):

        # self.db = mysql.connect(
        #     host="localhost",
        #     user="root",
        #     passwd="#Darren89candiesEW!",
        #     database="ReportData")

        self.db = mysql.connect(
            host="localhost",
            user="root",
            passwd="FYDP2022",
            database="reportdata")

        self.cursor = self.db.cursor()


    def add_patient(self, info):
        return

    def add_report(self, patient, report_id, name, og_path, txt_path, notes):

       # report_id = self.generate_report_id()
        query = "INSERT INTO reports VALUES (%s, %s, %s, %s, %s, %s)"
        data = (patient, report_id, name, og_path, txt_path, notes)
        self.cursor.execute(query, data)
        self.db.commit()
       # return report_id

    def add_labels(self, info):
        query = "INSERT INTO labels VALUES (%s, %s, %s, %s, %s, %s, %s)"
        self.cursor.execute(query, info)
        self.db.commit()
        return

    def generate_report_id(self):
        # if self.get_row_count() == 0:
        #     return 1

        query = "SELECT MAX(Report_ID) FROM reports"
        self.cursor.execute(query)
        max_id = self.cursor.fetchall()[0][0]
        return max_id + 1

    def get_row_count(self, patient=None):
        if id is None:
            self.cursor.execute("SELECT COUNT(*) FROM reports")
            return self.cursor.fetchall()[0][0]

        query = "SELECT COUNT(*) FROM reports WHERE Patient_ID='%s'"
        self.cursor.execute(query % patient)
        return self.cursor.fetchall()[0][0]

    def get_report_IDs(self, patient_ID):
        query = "SELECT Report_ID FROM reports WHERE Patient_ID='%s'"
        self.cursor.execute(query % patient_ID)
        ids = self.cursor.fetchall()
        return ids

    def get_mod_bd_IDs(self, values):
        query = "SELECT Report_ID FROM labels WHERE Patient_ID='%s' and modality in %s and bodypart in %s"
        self.cursor.execute(query % values)
        return self.cursor.fetchall()

    def get_filtered_date_IDs(self, values):
        query = "SELECT Report_ID FROM labels WHERE (Patient_ID='%s' and Report_ID in %s and exam_date %s)"
        self.cursor.execute(query % values)
        return self.cursor.fetchall()

    def get_report_date(self, report_ID):
        query = "SELECT Exam_Date FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        print(report_ID)
        datetime = self.cursor.fetchone()
        # print(datetime)
        try:
            date_string = datetime[-1].strftime("%Y-%m-%d")
        except:
            return None
        return [(date_string,),]

    def get_report_modality(self, report_ID):
        query = "SELECT Modality FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_modality_display(self, values):
        query = "SELECT %s FROM physician_preferences WHERE physician_id='%s'"
        self.cursor.execute(query % values)
        return self.cursor.fetchall()

    def get_report_bodypart(self, report_ID):
        query = "SELECT Bodypart FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_bodypart_display(self, report_ID):
        query = "SELECT Bodypart_display FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_institution(self, report_ID):
        query = "SELECT Institution FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_clinician(self, report_ID):
        query = "SELECT Clinician FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_path(self, report_ID):
        query = "SELECT Report_file FROM reports WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_text_path(self, report_ID):
        query = "SELECT Report_text FROM reports WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_name(self, report_ID):
        query = "SELECT Report_name FROM reports WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_notes(self, report_ID):
        query = "SELECT Notes FROM reports WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def search_by_label(self, label, specified_category=None):
        if specified_category:
            query = "SELECT Report_ID FROM labels WHERE " + specified_category + "=" + "'" + label + "'"
            self.cursor.execute(query)
        else:
            query = "SELECT Report_ID FROM labels WHERE Institution='%s' OR Modality='%s' OR Bodypart='%s' OR Clinician='%s'"
            self.cursor.execute(query % (label, label, label, label))
        return self.cursor.fetchall()

    def search_with_super_variable_query(self, query_after_where):
        query = "SELECT Report_ID FROM labels WHERE" + query_after_where
        print("query: {}".format(query))
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_patient_info(self):
        query = "SELECT * FROM patients"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_all_labels(self):
        self.cursor.execute("SELECT Institution FROM labels")
        labels = self.cursor.fetchall()
        self.cursor.execute("SELECT Modality FROM labels")
        labels = labels + self.cursor.fetchall()
        self.cursor.execute("SELECT Bodypart FROM labels")
        labels = labels + self.cursor.fetchall()
        self.cursor.execute("SELECT Clinician FROM labels")
        labels = labels + self.cursor.fetchall()
        return labels

    def get_all_clinicians(self):
        self.cursor.execute("SELECT Clinician FROM labels")
        return self.cursor.fetchall()

    def get_physician_preferences(self, id):
        query = "SELECT * FROM physician_preferences WHERE physician_id = '%s'"
        self.cursor.execute(query % id)
        preferences = self.cursor.fetchall()
        return preferences[0]

    def update_display_name_db(self, values):
        query = "UPDATE physician_preferences SET `%s` = '%s' WHERE physician_id = '%s'"
        self.cursor.execute(query % values)
        self.db.commit()

    def update_institution_display_name_db(self, display_name, institution_id):
        query = "UPDATE institutions SET display_name = \"{}\" WHERE id_institutions = {}".format(display_name, institution_id)
        self.cursor.execute(query)
        self.db.commit()

    def get_id_institutions(self, physician_id, formal_name):
        # query = "SELECT id_institutions FROM institutions WHERE physician_id = %s and formal_name = \"%s\""
        query = "SELECT id_institutions FROM institutions WHERE physician_id = {} and formal_name = \"{}\"".format(physician_id, formal_name)
        print(query)
        # self.cursor.execute(query % values)
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_all_report_ids(self):
        query = "SELECT Report_ID FROM reports"
        self.cursor.execute(query)
        ids = self.cursor.fetchall()
        return ids

    def update_label_table_display_name_column(self, values):
        query = "UPDATE labels SET Modality_display = '%s', Bodypart_display = '%s' WHERE report_ID = '%s'"
        self.cursor.execute(query % values)
        self.db.commit()

    def get_patient_last_name(self, patient_ID):
        query = "SELECT Last_name FROM patients WHERE patient_ID='%s'"
        self.cursor.execute(query % patient_ID)
        return self.cursor.fetchall()

    def get_patient_first_name(self, patient_ID):
        query = "SELECT First_name FROM patients WHERE patient_ID='%s'"
        self.cursor.execute(query % patient_ID)
        return self.cursor.fetchall()

    def get_institutions_in_db(self, physician_ID):
        query = "SELECT formal_name, display_name FROM institutions WHERE physician_id = %s"
        self.cursor.execute(query % physician_ID)
        return self.cursor.fetchall()

    def add_institution(self, institution, physician_ID):
        query = "INSERT INTO institutions (physician_id, formal_name, display_name) VALUES({}, \"{}\", \"{}\")".format(physician_ID, institution, institution)
        self.cursor.execute(query)
        self.db.commit()

    def delete_report_from_db(self, report_id):
        query = "DELETE FROM reports WHERE report_id=%s"
        self.cursor.execute(query % report_id)
        self.db.commit()
        query = "DELETE FROM labels WHERE report_id=%s"
        self.cursor.execute(query % report_id)
        self.db.commit()


    def remove_things_that_dont_exist(self):
        self.cursor.execute("SELECT report_id FROM labels")
        ids = self.cursor.fetchall()
        for id in ids:
            query = "SELECT 1 FROM reports WHERE report_id=%s"
            self.cursor.execute(query % id)
            result = self.cursor.fetchall()
            if not result:
                query = "DELETE FROM labels WHERE report_id=%s"
                self.cursor.execute(query % id)
                self.db.commit()

# test_db = DB_Connection()
# test_db.remove_things_that_dont_exist()
# test_db.get_report_IDs(4)
# test_db.get_report_date(4)
# print(test_db.get_patient_info())