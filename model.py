import shutil
from OCR import ocr_main
import os

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
        self.hello = "hello"

    def import_report(self, path):
        filename = path.split('/')[-1]
        shutil.copy(path, 'reports/'+filename)
        shutil.copy(path, 'OCR/reports_temp/'+filename)
        self.call_ocr(filename)

    def call_ocr(self, filename):
        report_text = ocr_main.run_ocr(filename)
        os.remove('OCR/reports_temp/'+filename)
        self.save_ocr_result(filename, report_text)

    def save_ocr_result(self, report_name, result):
        filename = report_name.split('.')[0]
        file = open('report_texts/' + filename + '.txt', "w+")
        file.write(result)
        file.close()


