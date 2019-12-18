import cv2
import numpy as np

#in this program, I make custom functions


def image_process(img, size = (160,90), shape = (160,90,1), filter = True, gray = True, color = 'yellow'):
    
    img = cv2.resize(img,size)

    if gray == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img,shape)
        
    return img



def get_label(path, os = 'win', flip=True, before=True, dico=[3,5,7,9,11], rev=[11,9,7,5,3]):
    label = []

    if os == 'win':
        slash ='\\'
    elif os == 'linux':
        slash ='/'
    
    name = path.split(slash)[-1]
    name = int(name.split('_')[0])

    label.append(dico[name])
    if flip == True:
        label.append(rev[name])

        
    return(label)

def get_crop(img, cut = 30, width= 160, height= 120):
    
    w,h = width,height-cut 
    x,y = 0,cut

    img = img[y:y+h,x:x+w]

    return(img)
