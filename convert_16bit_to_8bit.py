import cv2
import numpy as np


def transform_dim(img):
    return np.expand_dims(img,axis=2)

def normalize(src_img: np.uint16,min_val=0,max_val=255, transform_img=True):
    '''
    Takes the lowest value, sets it to 0, the highest to 255, and scales everything else linearly in between. Good if the image is fairly well distributed between min and max.
    :param 
        src_img         : source 16 bit image
        transform_dim   : if the source image does not have 3 channels, transformation (addition of channels) would be required
                          defaulted to True
        min_val         : lower limit of normalise value (defaulted to 0)
        max_val         : upper limit of normalise value (defaulted to 255)
    :return: src_img that has been normalized from 0 - 255
    :disadv: If the values are at the extreme, it will be lost 
    '''
    
    norm_img = cv2.normalize(src_img, None, min_val, max_val, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if transform_img:
        return transform_dim(norm_img)
    else: 
        return norm_img


def equalize_histogram(src_img: np.uint16,  transform_img=True):
    '''
    Takes the image, and spreads it out across the range of values. Saturates the extremes and alters the relative values of pixels.
    :param 
        src_img         : source 16 bit image
        transform_img   : if the source image does not have 3 channels, transformation (addition of channels) would be required
                          defaulted to True
    :return: src_img whose values have been spread
    :disadv: Increase noise 
    '''
    src_img_8bits = normalize(src_img)
    hist_image = cv2.equalizeHist(src_img_8bits)

    if transform_img:
        return transform_dim(hist_image)
    else: 
        return hist_image


def adaptive_equalize_histogram(src_img: np.uint16, clip_limit=4 , tile_grid=(8,8),transform_img=True):
    '''
    Improve Histogram Equalization by boxing up the image (Adjust intensities based on neighbouring pixels)
    :param 
        src_img         : source 16 bit image
        clip_limit      : threshold for contrast limiting
        tile_grid       : tile size 
        transform_img   : if the source image does not have 3 channels, transformation (addition of channels) would be required
                          defaulted to True
    :return: src_img whose values have been spread
    :disadv: Halo at random areas of the output 
    '''

    if len(tile_grid) != 2 :
        raise ValueError("tile_grid has to be a tuple")

    if tile_grid[0] <= 0 or tile_grid[1] <= 0:
        raise ValueError("tile_grid has to be above 0")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    equalized = clahe.apply(src_img) 

    if transform_img:
        transformed_img = transform_dim(equalized)

        '''
        https://stackoverflow.com/questions/48331211/how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet
        https://stackoverflow.com/questions/65117678/cv2-imshow-function-displays-correct-image-but-while-saving-it-using-cv2-imwri
        unsigned 16bit will be automatically divided by 256 by cv2.imshow() but cv2.imwrite() doesn't do it thus need to manually divide the transformation by 256 and convert to uint8 
        '''
       
        gray8_frame = transformed_img/256
        gray8_frame = gray8_frame.astype(np.uint8)

        return gray8_frame
    else : 
        return equalized


def simplest_cb(img, percent=1):
    '''
    https://www.ipol.im/pub/art/2011/llmps-scb/article.pdf 
    https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc 

    To brighten the image after conversion 
    :param 
        percent     : percentage to be saturate 
    :return: brighten src img
    :disadv: will not work if there's not enough contrast in the image. Works better for CLAHE-ed images  
    '''
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)
