import os
import time

import cv2
import numpy as numpy
from PIL import ImageGrab
from tqdm import tqdm


def take_screenshot(bbox, size=(160,90), name=0):
    
    img = ImageGrab.grab(bbox=bbox)
    
    img = np.array(img)
    img = cv2.resize(img,size)

    cv2.imshow('img',img)
    cv2.waitKey(1)

    cv2.imwrite('raw\\'+str(name)+'_'+str(time.time())+'.png',img)

    