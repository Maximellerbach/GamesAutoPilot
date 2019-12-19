import cv2
import numpy as np

class ui():
    def __init__(self, name="img", dt=1):
        self.name = name
        self.dt = dt
        self.stacks = [(0, 0, 0, 0)]

    def stack(self, screens=[np.zeros((512,512))], stackx=True):
        self.screens = screens
        self.n_screens = len(screens)

        xmax = []
        ymax = []

        for s in self.screens:
            ymax.append(s.shape[0])
            xmax.append(s.shape[1])

        xmax = sum(xmax)
        ymax = max(ymax)

        img = np.zeros((ymax, xmax, 3))

        offx = 0
        for s in self.screens:
            dim = s.shape
            img[:dim[0], offx:offx+dim[1], :] = s
            self.stacks.append((0, dim[0], offx, offx+dim[1]))

            if stackx == True:
                offx += dim[1]

        self.canv = img

    def show(self):
        cv2.imshow(self.name, self.canv)
        cv2.waitKey(self.dt)

