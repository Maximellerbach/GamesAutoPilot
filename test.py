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
        self.kdict = ['left', 'right', 'up', 'down']
        self.k = self.kdict[-1]
        self.idt = 1/30
        self.dt = self.idt
        self.pred = 0

    def iterate(self):
        pyautogui.press(self.k)

    def get_key(self, direct, prev):
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
        

model = load_model('C:\\Users\\maxim\\AutonomousCar\\test_model\\convolution\\lightv4_mix.h5', custom_objects={"dir_loss":dir_loss})
bbox = (0,33,800,450+33) # 514, 387 
dico=[-2,-1,0,1,2]
lab_dico = [3, 5, 7, 9, 11]
lab_dickey = ["left", "up", "right"]
key_av = []
recording = False
prev = 0

# d = d3dshot.create(capture_output="numpy", frame_buffer_size=120)
sct = mss.mss()

raw = interface.screen(0)
c = interface.screen(0)

canvas = interface.ui(screens=[raw, c], name="autonomous_driving")

# kj = joy_keyboard()

vj = joy_controller(1)
vj.vjoyobj.data.wAxisZRot = int(32767)//2
vj.vjoyobj.update()

while(1):
    
    sct_img = sct.grab(bbox)
    raw.img = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))

    raw.img = cv2.cvtColor(raw.img, cv2.COLOR_RGB2BGR)/255
    img = cv2.resize(raw.img, (320,240))
    img = img[100:, :, :]
    img = cv2.resize(img, (160,120))

    x = model.predict(np.expand_dims(img, axis=0))[0]
    av = 0
    for it, nyx in enumerate(x):
        av+=nyx*dico[it]
    dire = av/1.25

    c.img = cv2.line(np.copy(img), (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+dire*30), img.shape[0]-50), color=[1, 0, 0], thickness=4)

    canvas.stack()
    canvas.show()

    # kj.get_key(dire, prev)
    # kj.iterate()
    # prev = dire
    if keyboard.is_pressed('i'):
        if recording == False:
            recording = True
        else:
            recording = False
        print("recording: ", recording)
        
        time.sleep(0.5)

    if recording:
        lab_keys = []
        for k in lab_dickey:
            lab_keys.append(keyboard.is_pressed(k))
        lab = 0
        wl = []
        for it, ks in enumerate(lab_keys):
            lab+= ks*(it*2)
            if ks == True:
                wl.append(ks*it*2)

        if np.average(lab_keys) != 0:
            mean = int(np.average(wl))
            to_save = lab_dico[mean]
            cv2.imwrite('C:\\Users\\maxim\\img_trackmania\\'+str(to_save)+'_'+str(time.time())+'.png', img*255)
    else:
        # vj.vjoyobj.data.wAxisZRot = int((32767/2)+(32767/2*x[2]))
        vj.iterate(dire)
        