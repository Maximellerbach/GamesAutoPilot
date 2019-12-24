import cv2
import numpy as np

class screen():
    """
    screen class
    """
    def __init__(self, img, stackx=False):
        """ img : bgr or rgb image """
        self.img = img
        self.stackx = stackx

class text():
    """
    text class
    """
    def __init__(self, txt, font = cv2.FONT_HERSHEY_SIMPLEX, fontsize=1, color = (255, 255, 255), stackx=False):
        self.string = txt
        self.font = font
        self.fontsize = fontsize
        self.color = color
        self.stackx = stackx

    def get_dim(self):
        """
        get the size of the bounding box of a given text
        returns (y, x, margin)
        """
        size = cv2.getTextSize(self.string, self.font, self.fontsize, 1)
        return size[0][1], size[0][0], size[1]


class ui():
    """
    custom class for combining multiple screens/texts into one displayed image
    """
    def __init__(self, screens=[screen(np.zeros((1,1)))], texts=[text("")], name="img", dt=1):
        self.name = name
        self.dt = dt
        self.screens = screens
        self.stacks = [(0, 0, 0, 0)]
        self.texts = texts

    def update(self):
        if len(self.stacks)>0:
            xmax = []
            ymax = []

            for s in self.screens:
                y, x, _ = s.img.shape
                ymax.append(y)
                xmax.append(x)

            for t in self.texts:
                y, x, m = t.get_dim()
                ymax.append(y+m)
                xmax.append(x)

            if len(xmax)>0 and len(ymax)>0:
                xmax = sum(xmax)
                ymax = sum(ymax)

        self.stacks = [(0, 0, 0, 0)]
        img = np.zeros((ymax, xmax, 3))

        for s in self.screens:
            py, pyh, px, pxw = self.stacks[-1]
            y, x, _ = s.img.shape
                
            if s.stackx==False:
                pxw = px
                py = pyh
                img[py:py+y, pxw:pxw+x, :] = s.img
                self.stacks.append((py, py+y, pxw, pxw+x))

            else:
                pyh = py
                px = pxw
                img[pyh:pyh+y, px:px+x, :] = s.img
                self.stacks.append((pyh, pyh+y, px, px+x))


        for t in self.texts:
            py, pyh, px, pxw = self.stacks[-1]
            y, x, m = t.get_dim()

            if t.stackx==False:
                pxw = px
                py = pyh
                cv2.putText(img, t.string, (pxw, py+y+m//2), t.font, t.fontsize, t.color)    
                self.stacks.append((py, py+y+m, pxw, pxw+x))

            else:
                pyh = py
                px = pxw
                cv2.putText(img, t.string, (px, pyh+y+m//2), t.font, t.fontsize, t.color)
                self.stacks.append((pyh, pyh+y+m, px, px+x))
        
        ymax = 0
        xmax = 0
        for st in self.stacks:
            if st[1]>ymax:
                ymax = st[1]
            if st[3]>xmax:
                xmax = st[3]

        self.canv = img[:ymax, :xmax, :]

    def show(self):
        cv2.imshow(self.name, self.canv)
        cv2.waitKey(self.dt)


if __name__ == "__main__":
    t = text("olala")
    i = screen(np.zeros((200, 200, 3)))
    c = ui(screens=[i], texts=[t], dt=0)

    c.update()
    c.show()
