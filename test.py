import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import threading
import time

import cv2
import d3dshot
from PIL import ImageGrab, Image
import mss
import numpy as np
import pyautogui
import keyboard
from keras.models import load_model
import keras.backend as K

import interface
import pyvjoy
pyautogui.FAILSAFE = False


class joy_keyboard():
    def __init__(self):
        self.i = 0
        self.kdict = ['q', 'r', 'z', 's']
        self.k = self.kdict[-1]
        self.idt = 1/30
        self.dt = self.idt
        self.pred = 0

    def iterate(self):
        pyautogui.press(self.k)

    def get_key(self, direct, prev, dt):
        if prev>0:
            kp = self.kdict[1]
        elif prev<0:
            kp = self.kdict[0]
        else:
            kp = self.kdict[-1]

        if direct>0:
            self.k = self.kdict[1]
        elif direct<0:
            self.k = self.kdict[0]
        else:
            self.k = self.kdict[-1]

        # self.dt = np.absolute(direct)*dt
        # self.i = int(np.absolute(direct)*10)
        

def dir_loss(y_true, y_pred):
    # return K.sqrt(K.square(y_true-y_pred))
    return K.sqrt(K.square(y_true-y_pred))

class joy_controller():
    def __init__(self, n):
        self.vjoyobj = pyvjoy.VJoyDevice(n)
        self.vjoyobj.data.wAxisX = int(32767/2)
        self.vjoyobj.data.wAxisY = int(32767/2)

    def iterate(self, dire):
        self.vjoyobj.data.wAxisX = int(32767*(dire+1)/2)
        self.vjoyobj.update()
        

model = load_model('lightv3_mix.h5', custom_objects={"dir_loss":dir_loss})
bbox = (0,33,514,421)
dico=[-2,-1,0,1,2]
lab_dico = [3, 7, 11]
lab_dickey = ["q", "z", "d"]
key_av = []
prev = 0

# d = d3dshot.create(capture_output="numpy", frame_buffer_size=120)
sct = mss.mss()
canvas = interface.ui(name="autonomous_driving")

# kj = joy_keyboard()
vj = joy_controller(1)


while(1):
    
    sct_img = sct.grab(bbox)
    raw = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))

    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)/255
    img = cv2.resize(raw, (320,240))
    img = img[100:, :, :]
    img = cv2.resize(img, (160,120))

    x = model.predict(np.expand_dims(img, axis=0))[0]
    av = 0
    for it, nyx in enumerate(x):
        av+=nyx*dico[it]
    dire = av/2

    c = cv2.line(np.copy(img), (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+dire*30), img.shape[0]-50), color=[1, 0, 0], thickness=4)

    canvas.stack(screens=[raw, c])
    canvas.show()

    # kj.get_key(dire, prev, dt)
    # kj.iterate()
    # prev = dire
    
    # vj.iterate(av/2)

    lab_key = keyboard.read_key()
    if len(key_av) < 5 :
        key_av.append(lab_dico[lab_dickey.index(lab_key)])
    else:
        del key_av[0]
        key_av.append(lab_dico[lab_dickey.index(lab_key)])

    lab = int(np.average(key_av))

    cv2.imwrite('C:\\Users\\maxim\\img_trackmania\\'+str(lab)+'_'+str(time.time())+'.png', img*255)
