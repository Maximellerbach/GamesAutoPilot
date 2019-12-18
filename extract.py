import cv2
import d3dshot
import numpy as np

import interface
import vjoy


def extract_loop(bbox, capture):
    raw = capture.screenshot(region=bbox)

    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)/255
    img = cv2.resize(raw, (320,240))
    img = img[100:, :, :]
    img = cv2.resize(img, (160,120))


    for i in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        pygame.event.pump()

        name = joystick.get_name()
        axes = joystick.get_numaxes()
        for y in range(2):
            axis = joystick.get_axis(y)

            print(name, axes, axis)

    # cv2.imwrite('extracted_img\\'+str(dire)+'_'+str(time.time())+'.png', img*255)

d = d3dshot.create(capture_output="numpy")
pygame.init()

bbox = (0,33,514,421)
while(1):
    extract_loop(bbox, d)