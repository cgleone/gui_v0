import numpy as np
import cv2
import pytesseract
import re

from PIL import ImageEnhance
from statistics import mean

method_list = []


# ------------------------ preprocessing methods ---------------------------------

def recursive_deskew(img, angle, deskew_angles):

    new_image = rotation_iteration(img, angle)
    new_angle, updated_skew_angles = get_skew_angle(new_image, deskew_angles)

    if abs(new_angle) < 3:
        return new_image, updated_skew_angles
    else:
        return recursive_deskew(new_image, new_angle, updated_skew_angles)



def deskew(image):
    #

    img_for_contours = image.copy()
    skew_angle_list = []
    angle_from_axis, skew_angle_list = get_skew_angle(img_for_contours, skew_angle_list)
    if angle_from_axis != 0:
        new_image, skew_angle_list = recursive_deskew(img_for_contours, angle_from_axis, skew_angle_list)
        axis_aligned_img = rotate(image, sum(skew_angle_list))
    else:
        axis_aligned_img = image

    angle_by_text = get_text_rotation(axis_aligned_img)
    fully_aligned_img = rotate(axis_aligned_img, angle_by_text)


    method_list.append("Deskewed from an angle of {}".format(sum(skew_angle_list)))

    return fully_aligned_img


def dilate(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    method_list.append("Dilation with kernel size of {}x{}".format(kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    method_list.append("Erosion with kernel size of {}x{}".format(kernel_size, kernel_size))
    return cv2.erode(image, kernel, iterations=1)


def greyscale(img):
    method_list.append("Greyscale")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thresholding(img):
    method_list.append("Thresholding")
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def gaussian_blur(img, kernel):
    method_list.append("Gaussian Blur with a {}x{} kernel".format(kernel, kernel))
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def median_blur(img, kernel):
    method_list.append("Median Blur with a {}x{} kernel".format(kernel, kernel))
    return cv2.medianBlur(img, kernel)


def averaging_blur(img, kernel):
    method_list.append("Averaging Blur with a {}x{} kernel".format(kernel, kernel))
    return cv2.blur(img, (kernel, kernel))


def bilateral_Filter(img):
    method_list.append("Bilateral")
    return cv2.bilateralFilter(img, 9, 75, 75)


def canny(image):
    method_list.append("Canny")
    return cv2.Canny(image, 100, 200)


def rescale(img, factor):
    new_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    method_list.append("Rescaling  by fx={} and fy={}".format(factor, factor))
    return new_img


def closing(img, kernel):
    img = dilate(img, kernel)
    img = erode(img, kernel)
    return img


def sharpen(pil_img, factor):
    sharp_enhancer = ImageEnhance.Sharpness(pil_img)
    new_img = sharp_enhancer.enhance(factor)
    method_list.append("Sharpened by factor of {}".format(factor))
    return new_img



def brighten(pil_img, factor):
    bright_enhancer = ImageEnhance.Brightness(pil_img)
    new_img = bright_enhancer.enhance(factor)
    method_list.append("Brightened by factor of {}".format(factor))
    return new_img



def contrast(pil_img, factor):
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    new_img = contrast_enhancer.enhance(factor)
    method_list.append("Contrast changed by factor of {}".format(factor))
    return new_img



# def sharpen(img):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     new_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
#     return new_img


# --------------------- methods called by preprocessing methods ---------------------------------------

# got this code from https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
def get_skew_angle(image, deskew_angles):
    # Prep image, copy, convert to gray scale, blur, and threshold
    new_image = image.copy()
    image_area = new_image.size

    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=5)


    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for contour in contours:
        cv2.drawContours(dilate, contour, -1, (0, 255, 0), 3)

    contour_angles = []
    original_contour_angles = []
    contour_areas_and_angles = []

    for c in contours:
        contour_angle = cv2.minAreaRect(c)[-1]
        original_contour_angles.append(contour_angle)
        if abs(contour_angle) <= 45:
            contour_angle = contour_angle
        elif contour_angle > 45 and contour_angle <= 90:
            contour_angle = 90 - contour_angle
        elif contour_angle < -45 and contour_angle >= -90:
            contour_angle = -(90+contour_angle)
        else:
            print("Nope you did it wrong.")


        contour_areas_and_angles.append([cv2.contourArea(c), contour_angle])
        if cv2.contourArea(c) > (0.1*image_area):
            contour_angles.append(contour_angle)

    percent_of_image_size = []
    important_angles = []
    for item in contour_areas_and_angles:
        percent_total_area = (item[0]/image_area)*100
        percent_of_image_size.append([percent_total_area, item[1]])
        if percent_total_area > 0.1:
            important_angles.append(item[1])

    rounded_important_angles = []
    for angle in important_angles:
        if angle < 45 and angle > 44:
            angle = 44
        elif angle > 45 and angle < 46:
            angle = 46
        elif angle > 0 and angle < 1:
            angle = 0
        elif angle: # not equal to zero
            angle = round(angle, 0)

        if angle:
            rounded_important_angles.append(angle)
    #
    modes = []
    for angle in rounded_important_angles:
        if rounded_important_angles.count(angle) > 1:
            if modes.count(angle) == 0:
                modes.append(angle)

    if len(modes) == 0:
        if rounded_important_angles.count(angle+1)>0:
            if modes.count(angle) == 0 and modes.count(angle+1) == 0:
                modes.append(angle+1)
        elif rounded_important_angles.count(angle - 1) > 0:
            if modes.count(angle) == 0 and modes.count(angle - 1) == 0:
                modes.append(angle-1)

    if len(modes) == 0:
        angle = 0
    elif len(modes) == 1:
        angle = modes[0]
    else:
        angle = mean(modes)

    if angle < -45:
        angle = 90 + angle

    deskew_angles.append(angle)
    return angle, deskew_angles


def rotation_iteration(img, angle):

    if abs(angle) <= 45:
        new_img = rotate(img, angle)
    elif angle > 45:
        new_img = rotate(img, 90 - angle)
    elif angle < -45:
        new_img = rotate(img, -(90 + angle))
    else:
        print("Just stop, you're confused.")
        new_img = img

    return new_img

def rotate(img, angle):
    if angle:
        img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return img


def get_text_rotation(img):
    osd = pytesseract.image_to_osd(img)
    angle = float(re.search('(?<=Rotate: )\d+', osd).group(0))
    return angle
