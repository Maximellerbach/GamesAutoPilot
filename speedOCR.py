import cv2
import pytesseract


class speed_OCR():
    def __init__(self, tbbox):
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

        self.tbbox = tbbox
        self.speed_frame = np.zeros((40, 60, 3), dtype=np.float32)
        self.speed = 0

        self.dos_save = "C:\\Users\\maxim\\GITHUB\\GamesAutoPilot\\number_recognition\\images\\"

    def extract_frame(self, frame):
        self.speed_frame = frame[
            self.tbbox[1]:self.tbbox[3],
            self.tbbox[0]:self.tbbox[2]
        ]

        cv2.imshow('speed_frame', self.speed_frame)
        cv2.waitKey(1)

        # self.get_speed()

    def prepare_frame(self):
        frame = cv2.resize(self.speed_frame, (60, 40))
        return frame

    def get_speed(self):
        frame = self.prepare_frame()
        text_speed = pytesseract.image_to_string(
            frame,
            config='--psm 6 -c tessedit_char_whitelist=0123456789'
        ).replace(" ", "")

        if len(text_speed) == 5:
            self.speed = int(text_speed[:3])

    def save_frame(self):
        cv2.imwrite(self.dos_save+str(time.time())+".png", self.speed_frame)
