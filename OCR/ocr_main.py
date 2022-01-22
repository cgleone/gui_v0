import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import os
import time
import OCR.preprocessing as pre
from datetime import datetime
from PIL import Image

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
    image = convert_from_path(path, paths_only=True, output_folder='OCR/reports_temp', fmt='jpeg', output_file=new_name)
    return image[0]


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
        jpg_path = convert_pdf(og_path)
        os.remove(og_path)

    else:
        jpg_path = og_path

    if '\\' in jpg_path:
        return jpg_path.split('\\')[-1]
    else:
        return jpg_path.split('/')[-1]


def run_ocr(file_name):
    print("filename: ")
    print(file_name)
    prepped = prep_image(file_name)
    return get_text(prepped)
