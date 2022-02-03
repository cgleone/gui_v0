import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
import time
import OCR.preprocessing as pre
from datetime import datetime
from PIL import Image
import shutil

def get_text(file):

    path = 'OCR/reports_temp/' + file

    img = cv2.imread(path)
    img = pre.deskew(img) # always leave on
    img = pre.greyscale(img) # always leave on
    img = pre.rescale(img, 1.5)

    # NOW MAKE IT A PILLOW
    pillow_path = save_for_pillowing(img, file)
    pil_img = get_pil_img(pillow_path)
    pil_img = pre.sharpen(pil_img, 3)
    pil_img = pre.brighten(pil_img, 1.5)
    pil_img = pre.contrast(pil_img, 2.5)

    # BACK TO OPENCV NOW
    img = np.array(pil_img)  # alrighty done it's a opencv now

    img = pre.thresholding(img) # always leave on
    img = pre.gaussian_blur(img, 3)

    converted = pytesseract.image_to_string(img)

    os.remove(path)  # removes the prepped image
    return converted


def convert_pdf(path):
    new_name = get_new_name(path)
    images = convert_from_path(path, paths_only=True, output_folder='OCR/reports_temp', fmt='jpeg', output_file=new_name)
    return images


def save_for_pillowing(img, filename):
    path = 'OCR/reports_temp/' + filename
    cv2.imwrite(path, img)
    return path


def get_pil_img(path):
    return Image.open(path)


def get_new_name(path):
    split_path = path.split('/')
    pdf_name = split_path[len(split_path)-1]
    new_name = pdf_name.rstrip('.pdf')
    return new_name


def prep_image(file):

    og_path = 'OCR/reports_temp/' + file

    if file.endswith('.pdf'):
        jpg_list = convert_pdf(og_path)
        os.remove(og_path)

        if len(jpg_list) == 1:
            jpg_path = jpg_list
            if '\\' in jpg_path[0]:
                return jpg_path.split('\\')[-1]
            else:
                return jpg_path.split('/')[-1]

        else:  # it's a multiple page document
            returned_paths = []
            for path in jpg_list:
                if '\\' in path:
                    returned_paths.append(path.split('\\')[-1])
                else:
                    returned_paths.append(path.split('/')[-1])
            return returned_paths

    else:
        jpg_path = og_path
        if '\\' in jpg_path:
            return [jpg_path.split('\\')[-1]]
        else:
            return [jpg_path.split('/')[-1]]


def run_ocr(file_name):
    print("filename: ")
    print(file_name)
    prepped_pages = prep_image(file_name)

    print(prepped_pages)
    full_text = ""
    for page in prepped_pages:
        page_text = get_text(page)
        full_text = full_text + page_text + '\n\n'
    return full_text


def write_test_result(name, text):
    path = 'testing_texts/'+name+'.txt'
    f = open(path, 'w+')
    f.write(text)
    f.close()


# ----- the code below can be used to run just the ocr component for testing
# ----- must create the a folder in OCR/ called 'testing_texts'

# filename = "532 Assignment 1"
# path = "532 Assignment 1.pdf"
# shutil.copy(path, 'reports_temp/'+path)
#
# text = run_ocr(path)
# write_test_result(filename, text)
# print(text)

