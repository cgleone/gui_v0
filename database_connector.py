
import mysql.connector as mysql

class DB_Connection:

    def __init__(self):

        # self.db = mysql.connect(
        #     host="localhost",
        #     user="root",
        #     passwd="FYDP2022",
        #     database="ReportData")

        self.db = mysql.connect(
            host="localhost",
            user="root",
            passwd="FYDP2022",
            database="reportdata")

        self.cursor = self.db.cursor()

        self.date_filters = {"<6mos": "> date_sub(now(), interval 6 month)",
                             "6mos-1yr": "between date_sub(now(), interval 1 year) "
                                         "AND date_sub(now(), interval 6 month)",
                             "1yr-5yrs": "between date_sub(now(), interval 5 year) AND date_sub(now(), interval 1 year)",
                             ">5yrs": "< date_sub(now(), interval 5 year)"}

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

    def get_report_IDs(self, patient_ID, filters):
        if filters is None:
            query = "SELECT Report_ID FROM reports WHERE Patient_ID='%s'"
            self.cursor.execute(query % patient_ID)
            ids = self.cursor.fetchall()
        else:
            mod_bp_ids=[]
            query = "SELECT Report_ID FROM labels WHERE modality in %s and bodypart in %s"
            values = (filters["modality"], filters["bodypart"])
            self.cursor.execute(query % values)
            reports = self.cursor.fetchall()
            for report in reports:
                mod_bp_ids.append(report[0])
            if len(mod_bp_ids)==0:
                return []
            else:
                if len(filters["exam_date"]) ==0: return mod_bp_ids
                report_ids=[]
                for option in filters["exam_date"]:
                    query = "SELECT Report_ID FROM labels WHERE (Patient_ID='%s' and Report_ID in %s and exam_date %s)"
                    values = (patient_ID, tuple(mod_bp_ids), self.date_filters[option])
                    self.cursor.execute(query % values)
                    reports = self.cursor.fetchall()
                    for report in reports:
                        report_ids.append(report)
                ids=report_ids


        return ids

    def get_report_date(self, report_ID):
        query = "SELECT Exam_Date FROM labels WHERE Report_ID='%s'"
        self.cursor.execute(query % report_ID)
        datetime = self.cursor.fetchone()[-1]
        date_string = datetime.strftime("%Y-%m-%d")
        return [(date_string,),]

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

    def search_by_label(self, label):
        query = "SELECT Report_ID FROM labels WHERE Institution='%s' OR Modality='%s' OR Bodypart='%s' OR Clinician='%s'"
        self.cursor.execute(query % (label, label, label, label))
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



# test_db = DB_Connection()
# test_db.get_report_IDs(4)
# test_db.get_report_date(4)