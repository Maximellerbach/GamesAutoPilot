import pyvjoy


class vjoy_controller():
    """
    class to use pyvjoy to simulate a joystick
    """

    def __init__(self, n):
        self.vjoyobj = pyvjoy.VJoyDevice(n)
        self.default()

    def default(self):
        self.vjoyobj.data.wAxisX = int(32767/2)
        self.vjoyobj.data.wAxisY = int(32767/2)
        self.vjoyobj.data.wAxisZRot = int(32767/2)
        self.vjoyobj.update()

    def iterate(self, steering, throttle):
        self.vjoyobj.data.wAxisX = int(32767*(steering+1)/2)
        self.vjoyobj.data.wAxisZRot = int(32767 * throttle)
        self.vjoyobj.update()
