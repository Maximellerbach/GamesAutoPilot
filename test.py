import math
import os
import threading
import time

import cv2
import keras.backend as K
import keyboard
import mss
import numpy as np
import pyautogui
import pyvjoy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.losses import mse
from keras.models import load_model
from PIL import Image

import interface

pyautogui.FAILSAFE = False

config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
config.log_device_placement = True  # to log device placement (on which device the operation ran)
set_session(sess) # set this TensorFlow session as the default

def round_list(iterable, n=3):
    rounded = []
    for i in iterable:
        rounded.append(round(i, n))
    return rounded

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
        pyautogui.keyDown(self.k)
        time.sleep(0.1)
        pyautogui.keyUp(self.k)

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
        self.vjoyobj.data.wAxisZRot = int(32767/2)+int(32767/2*speed)
        self.vjoyobj.update()
        

if __name__ == "__main__":
    bbox = (0,33,800,450+33) # (0,33,1024,768+33) # set here the coordinates of your game/screen to capture

    aggressive_coef = 0.5
    dico=[-2, -1, 0, 1, 2] # adjust for a custom driving sensivity
    lab_dico = [3, 5, 7, 9, 11]
    lab_dickey = ["left", "up", "right"] # keys to be listen when recording

    recording = False
    avlist = []
    memory_size = 30
    prev = 0

    model = load_model('C:\\Users\\maxim\\github\\AutonomousCar\\test_model\\convolution\\lightv6_mix.h5', compile=False) # set here your path tou your model
    sct = mss.mss()

    raw = interface.screen(0, stackx=False)
    c = interface.screen(0, stackx=False)

    font = cv2.FONT_HERSHEY_COMPLEX

    pred_txt = interface.text("", font=font, fontsize=0.5, stackx=True)
    dire_txt = interface.text("", font=font, fontsize=0.5, stackx=False)
    key_txt = interface.text("", font=font, fontsize=0.5, stackx=False)
    autonomous_txt = interface.text("", font=font, fontsize=0.5, stackx=False)

    keys = [0]*3

    canvas = interface.ui(screens=[raw, c], texts=[pred_txt, dire_txt, key_txt, autonomous_txt], name="autonomous_driving")
    # kj = joy_keyboard()
    vj = joy_controller(1)

    while(1): # control/recording loop

        sct_img = sct.grab(bbox)
        raw.img = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))

        raw.img = cv2.cvtColor(raw.img, cv2.COLOR_RGB2BGR)/255
        img = cv2.resize(raw.img, (320,240))
        img = img[100:, :, :]
        img = cv2.resize(img, (160,120))
        
        '''
        filled = [[0, 0.125, 0.75, 0.125, 0]]*(memory_size-len(avlist))+avlist
        x = model.predict([np.expand_dims(img, axis=0), np.expand_dims(filled, axis=0)])[0]
        if len(avlist)<memory_size:
            avlist.append(x)
        else:
            avlist.append(x)
            del avlist[0]
        '''
        x = model.predict(np.expand_dims(img, axis=0))[0]
        pred_txt.string = "predicted: "+str(round_list(x))

        av = 0
        for it, nyx in enumerate(x):
            av+=nyx*dico[it]
        
        if av>=0:
            dire = av**aggressive_coef
        else:
            dire = -np.absolute(av)**aggressive_coef
        
        dire_txt.string = "predicted angle: "+str(round_list([dire]))

        for k in lab_dickey:
            keys[lab_dickey.index(k)] = keyboard.is_pressed(k)
        key_txt.string = "manual steering: "+str(keys)
        autonomous_txt.string = "autonomous: "+str(not(any(keys)==True))

        c.img = cv2.line(np.copy(img), (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+dire*30), img.shape[0]-50), color=[1, 0, 0], thickness=4)

        
        ### 

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
            if any(keys)==True:
                vj.iterate(0, 0)
            else:
                vj.iterate(dire, 0) # max(x[1:3])
            
            # kj.get_key(dire, prev)
            # kj.iterate()
            # prev = dire

        canvas.update()
        canvas.show(factor=(1,1))
