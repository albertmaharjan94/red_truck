from __future__ import division

import cv2
import numpy as np
import numexpr as ne




_BLUE = 'blue'
_ORANGE = 'orange'
_YELLOW = 'yellow'
_GREEN = 'green'

_COLORS = {_BLUE: (255, 0, 0), _ORANGE: (0, 165, 255), _YELLOW: (0, 255, 255), _GREEN: (0, 255, 0) }

_HSV_COLOR_RANGES = {
    # _BLUE: (np.array([100, 100, 100], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8)),
    _ORANGE: (np.array([5, 100, 100], dtype=np.uint8), np.array([15, 250, 250], dtype=np.uint8)),
    _YELLOW: (np.array([26, 50, 50], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
    _GREEN: (np.array([32, 50, 50], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8))
}

_COLORS_OBJECT = {
    _BLUE: (255, 0, 0), _ORANGE: (0, 165, 255), _YELLOW: (0, 255, 255), _GREEN: (0, 255, 0)
}

_HSV_COLOR_RANGES_OBJECT = {
    _BLUE: (np.array([101, 100, 100], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8)),
    _ORANGE: (np.array([8, 100, 100], dtype=np.uint8), np.array([15, 250, 250], dtype=np.uint8)),
    _YELLOW: (np.array([26, 50, 50], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
    _GREEN: (np.array([32, 50, 50], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8))
}

def predominant_rgb_color(img, ymin, xmin, ymax, xmax):
    crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[ymin:ymax, xmin:xmax]
    best_color, highest_pxl_count = None, -1
    for color, r in _HSV_COLOR_RANGES.items():
        lower, upper = r
        pxl_count = np.count_nonzero(cv2.inRange(crop, lower, upper))
        if pxl_count > highest_pxl_count:
            best_color = color
            highest_pxl_count = pxl_count
    return _COLORS[best_color]



def predominant_rgb_color_object(img, ymin, xmin, ymax, xmax):
    crop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[int(ymin/2):int(ymax/2), xmin:xmax]
    best_color, highest_pxl_count = None, -1
    for color, r in _HSV_COLOR_RANGES_OBJECT.items():
        lower, upper = r
        pxl_count = np.count_nonzero(cv2.inRange(crop, lower, upper))
        if pxl_count > highest_pxl_count:
            best_color = color
            highest_pxl_count = pxl_count
    return _COLORS_OBJECT[best_color]


def add_rectangle_with_text(image, ymin, xmin, ymax, xmax, color, text):
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)
    cv2.putText(image, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)


def resize_width_keeping_aspect_ratio(image, desired_width, interpolation=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    r = desired_width / w
    dim = (desired_width, int(h * r))
    return cv2.resize(image, dim, interpolation=interpolation)
