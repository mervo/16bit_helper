
import argparse

import cv2
import numpy as np

from convert_16bit_to_8bit import normalize, equalize_histogram, adaptive_equalize_histogram

# Following https://pyimagesearch.com/2022/10/17/thermal-vision-measuring-your-first-temperature-from-an-image-with-python-and-opencv/ tutorial 
def temp_convert(x_mouse,y_mouse,gray16_image):
    temperature_pointer = gray16_image[y_mouse, x_mouse]
    temperature_pointer = (temperature_pointer / 100) - 273.15

    return temperature_pointer

def draw_temp(x_mouse,y_mouse,temp, gray8_image):
    cv2.circle(gray8_image, (x_mouse, y_mouse), 2, (255, 255, 255), -1)

    # write temperature
    cv2.putText(gray8_image, "{0:.1f} Degree Celsius".format(temp), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    

if __name__ == "__main__":
   # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-f", "--file", nargs='?', help="Path of the image file", default="uc-ImgRaw2023.06.13.19.21.12.0162.v02_16b.001.tiff")
    ap.add_argument("-m", "--mode", nargs='?', help="16-8bit Conversion Mode \nn:normalise\ne:equalisation histogram \nc:CLAHE", default="uc-ImgRaw2023.06.13.19.21.12.0162.v02_16b.001.tiff")
    args = ap.parse_args()

    gray16_image = cv2.imread(args.file, cv2.IMREAD_ANYDEPTH)

    # create mouse global coordinates
    x_mouse = 0
    y_mouse = 0                 
    
    # mouse events function
    def mouse_events(event, x, y, flags, param):
        # mouse movement event
        if event == cv2.EVENT_MOUSEMOVE:
        # update global mouse coordinates
            global x_mouse
            global y_mouse
            x_mouse = x
            y_mouse = y
          
            temp = temp_convert(x_mouse,y_mouse,gray16_image)
            if args.mode.lower() == 'h':
                gray8_frame = equalize_histogram(gray16_image)
            elif args.mode.lower() == 'c':
                gray8_frame = adaptive_equalize_histogram(gray16_image)
            else: 
                 gray8_frame = normalize(gray16_image)
            draw_temp(x_mouse,y_mouse,temp,gray8_frame)
            # show the thermal frame
            cv2.imshow("gray8", gray8_frame)
                    
    # set up mouse events and prepare the thermal frame display
    cv2.imshow('gray8', gray16_image)
    cv2.setMouseCallback('gray8', mouse_events)
    cv2.waitKey(0)