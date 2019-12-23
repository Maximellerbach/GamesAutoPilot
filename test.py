import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # force tensorflow/keras to use the cpu instead of gpu (already used by the game)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import threading
import time

import cv2
import numpy as np

from PIL import Image
import mss

import pyvjoy
import keyboard
import pyautogui
pyautogui.FAILSAFE = False

from keras.models import load_model
import keras.backend as K

import interface


def dir_loss(y_true, y_pred):
    """
    custom loss function for the models 
    (only use if you have the same models as me)
    """
    # return K.sqrt(K.square(y_true-y_pred))
    return K.sqrt(K.square(y_true-y_pred))


class joy_keyboard():
    """
    class to use keyboard to encode direction to the game
    """
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
        

class joy_controller():
    """
    class to use pyvjoy to simulate a joystick
    """
    def __init__(self, n):
        self.vjoyobj = pyvjoy.VJoyDevice(n)
        self.vjoyobj.data.wAxisX = int(32767/2)
        self.vjoyobj.data.wAxisY = int(32767/2)   
        self.vjoyobj.data.wAxisZRot = int(32767/2)
        self.vjoyobj.update()

    def iterate(self, dire, speed):
        self.vjoyobj.data.wAxisX = int(32767*(dire+1)/2)
        self.vjoyobj.data.wAxisZRot = int(32767/2)+int(32767/2*x[2])
        self.vjoyobj.update()
        

if __name__ == "__main__":
    bbox = (0,33,800,450+33) # set here the coordinates of your game/screen to capture

    dico=[-2,-1,0,1,2]
    lab_dico = [3, 5, 7, 9, 11]
    lab_dickey = ["left", "up", "right"] # keys to be listen when recording

    recording = False
    prev = 0

    model = load_model('C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\lightv4_mix.h5', custom_objects={"dir_loss":dir_loss}) # set here your path tou your model
    sct = mss.mss()

    raw = interface.screen(0)
    c = interface.screen(0)

    canvas = interface.ui(screens=[raw, c], name="autonomous_driving")

    # kj = joy_keyboard()
    vj = joy_controller(1)

    while(1): # controle/recording loop

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

        if keyboard.is_pressed('i'): # press "i" to switch from rec/control mode
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
            vj.vjoyobj.data.wAxisZRot = int((32767/2)+(32767/2*x[2]))
            vj.iterate(dire, max(x[2]))
            
            # kj.get_key(dire, prev)
            # kj.iterate()
            # prev = dire
        