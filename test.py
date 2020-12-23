import os
import time

import cv2
import d3dshot
import numpy as np
import pyvjoy
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json

import xinput

physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)

joysticks = xinput.XInputJoystick.enumerate_devices()
print('found %d devices' % (len(joysticks)))

joy = joysticks[0]

controller_values = {
    'l_thumb_x': 0.0,
    'l_thumb_y': 0.0,
    'right_trigger': 0.0,
    'left_trigger': 0.0,
    'r_thumb_x': 0.0,
    'r_thumb_y': 0.0,
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0
}


@joy.event
def on_button(button, pressed):
    # print('button: %s pressed: %d' % (button, pressed))
    controller_values[button] = pressed


@joy.event
def on_axis(axis, value):
    # print('axis: %s value: %f' % (axis, value))
    controller_values[axis] = value


class vjoy_controller():
    """
    class to use pyvjoy to simulate a joystick
    """

    def __init__(self, n):
        self.vjoyobj = pyvjoy.VJoyDevice(n)
        self.vjoyobj.data.wAxisX = int(32767/2)
        self.vjoyobj.data.wAxisY = int(32767/2)
        self.vjoyobj.data.wAxisZRot = int(32767/2)
        self.vjoyobj.update()

    def iterate(self, steering, throttle):
        self.vjoyobj.data.wAxisX = int(32767*(steering+1)/2)
        self.vjoyobj.data.wAxisZRot = int(32767 * throttle)
        self.vjoyobj.update()


class autonomous_driving():
    def __init__(self, model_name, Dataset, region=(0, 33, 800, 450+33)):
        self.Dataset = Dataset
        self.model = architectures.load_model(model_name)
        architectures.apply_predict_decorator(self.model)

        self.region = region
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=5)
        self.d.display = self.d.displays[0]
        self.d.capture()

        self.frame = np.zeros((160, 120, 3), dtype=np.float32)

        # init virtual joy
        self.vjoy = vjoy_controller(1)

        # init recording folder
        self.dos_save = os.getcwd()+"\\recorded\\"
        if not os.path.isdir(self.dos_save):
            os.mkdir(self.dos_save)

    def get_latest_frame(self):
        frame = self.d.get_latest_frame()
        if frame is not None:
            self.frame = cv2.cvtColor(
                frame[
                    self.region[1]:self.region[3],
                    self.region[0]:self.region[2]
                ],
                cv2.COLOR_RGB2BGR)

    def prepare_frame(self, frame):
        return cv2.resize(frame, (160, 120))

    def predict(self, frame, annotation_list={}, input_components=[]):
        to_pred = Dataset.make_to_pred_annotations(
            [frame], [annotation_list], input_components)

        predicted, dt = self.model.predict(to_pred)
        predicted = predicted[0]
        print(predicted)

        return predicted['direction'][0], predicted['throttle'][0]

    def predict_and_drive(self):
        self.get_latest_frame()
        frame = self.prepare_frame(self.frame)
        steering, throttle = self.predict(frame)
        self.vjoy.iterate(steering, throttle)

    def labelize(self):
        if controller_values[15]:  # X button
            self.get_latest_frame()
            to_save = self.prepare_frame(self.frame)
            self.save_frame(to_save, controller_values)

    def save_frame(self, frame, controller_values):
        steering = controller_values['l_thumb_x'] * 2
        throttle = controller_values['right_trigger'] - \
            controller_values['left_trigger']

        Dataset.save_img_and_annotation(
            frame,
            {
                'direction': float(steering),
                'throttle': float(throttle),
                'time': time.time()
            },
            self.dos_save)


if __name__ == '__main__':
    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])

    autocar = autonomous_driving(
        "C:\\Users\\maxim\\GITHUB\\AutonomousCar\\test_model\\models\\trackmania.h5",
        Dataset)

    while(True):
        joy.dispatch_events()

        # autocar.labelize()
        autocar.predict_and_drive()

    cv2.destroyAllWindows()
