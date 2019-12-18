import cv2 
import time

name = 'F:\\Forza Horizon 3 30_01_2019 15_08_40.mp4'

cap = cv2.VideoCapture(name)
i = 0

while(1): 
    ret, frame = cap.read()
    
    '''
    cv2.imshow('img',frame)
    cv2.waitKey(1)
    '''
    frame = cv2.resize(frame,(160,90))
    print(i)
    i+=1
    cv2.imwrite('extract\\'+str(time.time())+'.png',frame)