import cv2
import numpy as np


def normalize(src_img: np.uint8):
    '''
    Takes the lowest value, sets it to 0, the highest to 255, and scales everything else linearly in between. Good if the image is fairly well distributed between min and max.
    :param src_img: source 16 bit image
    :return: src_img that has been normalized from 0 - 255
    '''

    return cv2.normalize(src_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def equalize_histogram(src_img: np.uint8):
    '''
    Takes the image, and spreads it out across the range of values. Saturates the extremes and alters the relative values of pixels.
    :param src_img: source 16 bit image
    :return: src_img whose values have been spread
    '''
    src_img_8bits = (src_img / src_img.max()) * 255
    return cv2.equalizeHist(src_img, None)
