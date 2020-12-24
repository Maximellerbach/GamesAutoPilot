import socket
import threading
import struct


class UDP_receiver():
    FH4BufSize = 324

    data_format = {
        'Speed': (256, 4, 'f'),
        'Power': (260, 4, 'f'),
        'Torque': (264, 4, 'f'),
        'TireTempFrontLeft': (268, 4, 'f'),
        'TireTempFrontRight': (272, 4, 'f'),
        'TireTempRearLeft': (276, 4, 'f'),
        'TireTempRearRight': (280, 4, 'f'),
        'Boost': (284, 4, 'f'),
        'Fuel': (288, 4, 'f'),
        'DistanceTraveled': (292, 4, 'f'),
        'BestLap': (296, 4, 'f'),
        'LastLap': (300, 4, 'f'),
        'CurrentLap': (304, 4, 'f'),
        'CurrentRaceTime': (308, 4, 'f'),
        'LapNumber': (312, 2, 'H'),
        'RacePosition': (314, 1, 'B'),
        'Accel': (315, 1, 'B'),
        'Brake': (316, 1, 'B'),
        'Clutch': (317, 1, 'B'),
        'HandBrake': (318, 1, 'B'),
        'Gear': (319, 1, 'B'),
        'Steer': (320, 1, 'b'),
    }

    values = {
        'Speed': 0,
        'Power': 0,
        'Torque': 0,
        'TireTempFrontLeft': 0,
        'TireTempFrontRight': 0,
        'TireTempRearLeft': 0,
        'TireTempRearRight': 0,
        'Boost': 0,
        'Fuel': 0,
        'DistanceTraveled': 0,
        'BestLap': 0,
        'LastLap': 0,
        'CurrentLap': 0,
        'CurrentRaceTime': 0,
        'LapNumber': 0,
        'RacePosition': 0,
        'Accel': 0,
        'Brake': 0,
        'Clutch': 0,
        'HandBrake': 0,
        'Gear': 0,
        'Steer': 0,
    }

    def __init__(self, UDP_IP="192.168.1.56", UDP_PORT=1642):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))

        # recvform blocks thread until we get data - so freezes window. We might not want that I guess ;)
        threading.Thread(target=self.listen).start()

    def listen(self):
        self.sock.settimeout(3)
        while True:
            data, addr = self.sock.recvfrom(self.FH4BufSize)

            for key in self.data_format.keys():
                bit_index = self.data_format[key][0]
                length = self.data_format[key][1]
                data_type = self.data_format[key][2]

                self.values[key] = struct.unpack(
                    data_type,
                    data[bit_index: bit_index+length]
                )[0] # getting a tuple so extracting the value from the tuple

    def get_speed(self):
        return self.values['Speed']


if __name__ == "__main__":
    UDP_receiver()
