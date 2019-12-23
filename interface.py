import cv2
import numpy as np

class screen():
    """
    screen class
    """
    def __init__(self, img):
        self.img = img

class ui():
    """
    custom class for combining multiple screens into one screen
    """
    def __init__(self, screens=[screen(np.zeros((1,1)))], name="img", dt=1):
        self.name = name
        self.dt = dt
        self.screens = screens
        self.stacks = [(0, 0, 0, 0)]

    def stack(self, stackx=True, offx=0, offy=0):
        xmax = []
        ymax = []

        for s in self.screens:
            ymax.append(s.img.shape[0])
            xmax.append(s.img.shape[1])

        if stackx == True:
            xmax = sum(xmax)
            ymax = max(ymax)
        else:
            ymax = sum(ymax)
            xmax = max(xmax)

        img = np.zeros((ymax, xmax, 3))

        offx = 0
        offy = 0

        for s in self.screens:
            dim = s.img.shape
            img[offy:offy+dim[0], offx:offx+dim[1], :] = s.img
            self.stacks.append((0, dim[0], offx, offx+dim[1]))

            if stackx == True:
                offx += dim[1]
            else:
                offy += dim[0]

        self.canv = img

    def show(self):
        cv2.imshow(self.name, self.canv)
        cv2.waitKey(self.dt)

