import cv2
import numpy as np

class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(int(self.frame_len))
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return ret, bgr


if __name__ == '__main__':
    filename = "akiyo_cif.yuv"
    size = (288, 352)
    cap = VideoCaptureYUV(filename, size)
    index = 100 # target_frame
    i = 1
    while 1:
        ret, frame = cap.read()
        if i > index or not ret:
            break
        elif i == index or i == index - 1 or i == index - 2 or i == index - 3 or i == index - 4:
            cv2.imwrite("akiyo_"+str(i)+'.png', frame)
        i += 1
    print("Extracted Done!")