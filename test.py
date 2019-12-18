import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import threading
import time

import cv2
import d3dshot
import numpy as np
import pyautogui
from keras.models import load_model
import keras.backend as K

import interface
import pyvjoy
pyautogui.FAILSAFE = False


class joy_keyboard():
    def __init__(self):
        self.i = 0
        self.kdict = ['q', 'd', 'z', 's', '']
        self.k = self.kdict[-1]
        self.idt = 1/30
        self.dt = self.idt
        self.pred = 0

    def iterate(self):
        pyautogui.keyDown(self.k)
        time.sleep(self.dt)
        pyautogui.keyUp(self.k)
        self.i -= 1

    def get_key(self, direct, prev, dt):
        
        if prev>0:
            kp = 'd'
        elif prev<0:
            kp = 'q'
        else:
            kp = ''

        if direct>0:
            self.k = 'd'
        elif direct<0:
            self.k = 'q'
        else:
            self.k = ''

        if kp != self.k:
            self.i += 1
            pyautogui.keyUp(kp)
        else:
            self.i += 1
        
        self.dt = np.absolute(direct)*dt
        # self.i = int(np.absolute(direct)*10)
        

def dir_loss(y_true, y_pred):
    # return K.sqrt(K.square(y_true-y_pred))
    return K.sqrt(K.square(y_true-y_pred))


last = 1
av = []
model = load_model('lightv3_mix.h5', custom_objects={"dir_loss":dir_loss})
bbox = (0,33,514,421)
dico=[-2,-1,0,1,2]
average = 0
x = [0,0,0,0,0]
dt = 0
prev = 0

d = d3dshot.create(capture_output="numpy")
canvas = interface.ui(name="autonomous_driving")

# j = joy_keyboard()
# t = threading.Thread(target=j.iterate())
# t.start()

vj = pyvjoy.VJoyDevice(1)
vj.data.wAxisX = int(32767/2)
vj.data.wAxisY = int(32767/2)

while(1):
    st = time.time()
    raw = d.screenshot(region=bbox)
    # img = img[bbox[0]:bbox[1], bbox[2]:bbox[3],:]
    
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)/255
    img = cv2.resize(raw, (320,240))
    img = img[100:, :, :]
    img = cv2.resize(img, (160,120))

    x = model.predict(np.expand_dims(img, axis=0))[0]
    average = 0
    for it, nyx in enumerate(x):
        average+=nyx*dico[it]

    if len(av)<5:
        av.append(average/2)
    else:
        av.append(average/2)
        del av[0]

    # dire = np.average(av)
    dire = average/2

    c = cv2.line(np.copy(img), (img.shape[1]//2, img.shape[0]), (int(img.shape[1]/2+dire*30), img.shape[0]-50), color=[1, 0, 0], thickness=4)

    canvas.update(screens=[raw, c])
    canvas.show()

    et = time.time()
    dt = et-st
    
    # t.join()
    # j.get_key(dire, prev, dt)
    # t = threading.Thread(target=j.iterate())
    # t.start()
    # prev = dire
    
    vj.data.wAxisX = int(32767*(dire+1)/2)
    vj.update()
    
    # cv2.imwrite('test_img\\'+str(dire)+'_'+str(time.time())+'.png', img*255)
