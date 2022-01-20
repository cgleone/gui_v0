
import mysql.connector as mysql


class DB_Connection:

    def __init__(self):

        self.db = mysql.connect(
            host="localhost",
            user="root",
            passwd="FYDP2022",
            database="ReportData")

        self.cursor = self.db.cursor()

    def add_patient(self, info):
        return

    def add_report(self, patient, report_id, name, og_path, txt_path):

       # report_id = self.generate_report_id()
        query = "INSERT INTO reports VALUES (%s, %s, %s, %s, %s)"
        data = (patient, report_id, name, og_path, txt_path)
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
        rows = self.cursor.fetchall()
        print(rows)
        return rows

    def get_report_date(self, report_ID):
        query = "SELECT Exam_Date FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_modality(self, report_ID):
        query = "SELECT Modality FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def get_report_bodypart(self, report_ID):
        query = "SELECT Bodypart FROM labels WHERE Report_ID='%s'"
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

    def get_report_name(self, report_ID):
        query = "SELECT Report_name FROM reports WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        return self.cursor.fetchall()

    def search_by_label(self, column, label_value):
        query = "SELECT Report_ID FROM labels WHERE '%s'='%s'"
        self.cursor.execute(query % column, label_value)
        return self.cursor.fetchall()




# test_db = DB_Connection()
# test_db.get_report_IDs(4)
# test_db.get_report_date(4)