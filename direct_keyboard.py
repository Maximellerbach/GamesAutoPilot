import threading
import time

import keyboard


class keyboard_controller():
    F_key = "z"
    L_key = "q"
    R_key = "d"
    B_key = "s"

    def __init__(self):
        self.do_drive = False
        self.pwm_timing = 0.2

        self.steering = 0
        self.throttle = 0
        self.keyboard = 0

        threading.Thread(target=self.steering_pwm).start()

    def default(self):
        self.steering = 0
        self.throttle = 0
        self.do_drive = False

    def steering_pwm(self):
        while(True):
            if self.do_drive:
                down_sleep_time = (abs(self.steering)**1.5)*self.pwm_timing
                if self.steering > 0:
                    print("going right", down_sleep_time)
                    keyboard.press(self.R_key)
                    time.sleep(down_sleep_time)
                    keyboard.release(self.R_key)
                    time.sleep(self.pwm_timing-down_sleep_time)

                elif self.steering < 0:
                    print("going left", down_sleep_time)
                    keyboard.press(self.L_key)
                    time.sleep(down_sleep_time)
                    keyboard.release(self.L_key)
                    time.sleep(self.pwm_timing-down_sleep_time)
            else:
                time.sleep(0.5)

    def throttle_pwm(self):
        down_sleep_time = (abs(self.steering)**1.5)*self.pwm_timing
        if self.throttle > 0:
            keyboard.press(self.F_key)
            time.sleep(abs(self.throttle)*self.pwm_timing)
            keyboard.release(self.F_key)
            time.sleep(self.pwm_timing-down_sleep_time)

        elif self.throttle < 0:
            keyboard.press(self.R_key)
            time.sleep(abs(self.throttle)*self.pwm_timing)
            keyboard.release(self.R_key)
            time.sleep(self.pwm_timing-down_sleep_time)

    def iterate(self, steering, throttle):
        self.do_drive = True
        self.steering = steering
        self.throttle = throttle
