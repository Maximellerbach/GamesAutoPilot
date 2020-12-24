import os
import time

import cv2
import d3dshot
import numpy as np
import tensorflow
from custom_modules import architectures
from custom_modules.datasets import dataset_json
from custom_modules.vis import vis_lab

import xinput
from udpReceiver import UDP_receiver


physical_devices = tensorflow.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tensorflow.config.experimental.set_memory_growth(gpu_instance, True)
# tensorflow.config.set_visible_devices([], 'GPU')

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
axis_th = 0.1


@joy.event
def on_button(button, pressed):
    # print('button: %s pressed: %d' % (button, pressed))
    controller_values[button] = pressed


@joy.event
def on_axis(axis, value):
    # print('axis: %s value: %f' % (axis, value))
    controller_values[axis] = 0 if abs(value) < axis_th else value


class controller():
    def __init__(self, joystick=True):
        self.mode = 'joystick' if joystick else 'keyboard'

        if joystick:
            print("using virtual joystick")
            import direct_vjoy
            self.controller = direct_vjoy.vjoy_controller(1)

        else:
            print("using virtual keyboard")
            import direct_keyboard
            self.controller = direct_keyboard.keyboard_controller()

    def default(self):
        self.controller.default()

    def iterate(self, steering, throttle):
        self.controller.iterate(steering, throttle)


class autonomous_driving():
    def __init__(self, model_name, Dataset, region=(0, 200+33, 800, 450+33), tbbox=(705, 420, 775, 460), dos_save="", use_joystick=True, input_components=[]):
        # init Dataset and model
        self.Dataset = Dataset
        self.model = architectures.load_model(model_name, compile=False)
        architectures.apply_predict_decorator(self.model)
        self.input_components = input_components

        # init udp connection to fh4
        self.speedDetector = UDP_receiver()
        self.speed = 0

        # init recording
        self.region = region

        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=5)
        self.d.display = self.d.displays[0]
        self.d.capture()

        self.frame = np.zeros((10, 10, 3), dtype=np.float32)
        self.speed_frame = np.zeros((10, 10, 3), dtype=np.float32)

        # init virtual controller
        self.controller = controller(joystick=use_joystick)

        # init recording folder
        if dos_save != "":
            self.dos_save = dos_save
        else:
            self.dos_save = os.getcwd()+"\\recorded\\"
        if not os.path.isdir(self.dos_save):
            os.mkdir(self.dos_save)

        self.running = True
        self.annotations = {
            'direction': 0,
            'speed': 0,
            'throttle': 0,
        }

    def get_latest_frame(self):
        frame = self.d.get_latest_frame()
        if frame is not None:
            self.frame = cv2.cvtColor(
                frame[
                    self.region[1]:self.region[3],
                    self.region[0]:self.region[2]
                ],
                cv2.COLOR_RGB2BGR)

            self.annotations['speed'] = self.speedDetector.get_speed()

    def prepare_frame(self):
        return cv2.resize(self.frame, (160, 120))

    def predict(self, frame, annotation_list, input_components):
        to_pred = Dataset.make_to_pred_annotations(
            [frame], [annotation_list], input_components)

        predicted, dt = self.model.predict(to_pred)
        print(predicted)
        for key in predicted[0].keys():
            self.annotations[key] = predicted[0][key]

        vis_lab.vis_all(self.Dataset, input_components, frame, predicted)

    def predict_and_drive(self):
        frame = self.prepare_frame()

        if controller_values[15]:  # X button
            self.save_frame(frame, controller_values)
        else:
            self.predict(frame, self.annotations, self.input_components)
            self.controller.iterate(self.annotations['direction'], self.annotations['throttle'])

    def labelize(self):
        if controller_values[15]:  # X button
            to_save = self.prepare_frame()
            self.save_frame(to_save, controller_values)

    def get_controller_values(self):
        self.annotations['direction'] = controller_values['l_thumb_x'] * 2
        self.annotations['throttle'] = controller_values['right_trigger'] - \
            controller_values['left_trigger']

    def save_frame(self, frame, controller_values):
        self.get_controller_values()

        Dataset.save_img_and_annotation(
            frame,
            {  # make sure we cast everything to regular (python) float instead of numpy
                'direction': float(self.annotations['direction']),
                'speed': float(self.annotations['speed']),
                'throttle': float(self.annotations['throttle']),
                'time': time.time()
            },
            self.dos_save)

    def main_loop(self):
        time.sleep(1)
        labelizing = True

        while(self.running):
            joy.dispatch_events()

            if controller_values[7] == 1:  # left stick button
                labelizing = not labelizing
                self.controller.default()
                print("changing mode")
                time.sleep(1)

            self.get_latest_frame()

            if labelizing:
                self.labelize()
            else:
                self.predict_and_drive()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    Dataset = dataset_json.Dataset(
        ['direction', 'speed', 'throttle', 'time'])
    input_components = [1]

    # dos_save = "C:\\Users\\maxim\\random_data\\forza\\"+str(time.time())+"\\"
    dos_save = ""

    autocar = autonomous_driving(
        "C:\\Users\\maxim\\GITHUB\\AutonomousCar\\test_model\\models\\forza3.h5",
        Dataset, dos_save=dos_save, use_joystick=False, input_components=input_components)

    autocar.main_loop()
